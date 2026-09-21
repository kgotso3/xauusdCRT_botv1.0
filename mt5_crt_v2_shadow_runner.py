from __future__ import annotations

import argparse
import json
import sys
import time as time_module
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import pandas as pd

from mt5_crt_m5_model_test import broker_pip_size, c3_bounds, confirmed_pivots, evaluate_m5_model
from mt5_crt_m5_performance_test import add_performance_fields
from mt5_robust_history import load_m5_chunked


NY = ZoneInfo("America/New_York")
PRIMARY_COHORT = {"XAUUSD", "US500", "US30"}
LIVE_SYMBOL_CANDIDATES: dict[str, list[str]] = {
    "XAUUSD": ["GOLD", "XAUUSD"],
    "US500": ["US500Cash", "US500", "SPX500"],
    "US30": ["US30Cash", "US30", "WS30", "DJ30"],
}
DATED_MONTH_CODES = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
AMBIGUOUS_STATUSES = {"ENTRY_BAR_AMBIGUOUS", "PRE_1R_AMBIGUOUS", "POST_1R_AMBIGUOUS"}


def normalise_symbol(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def resolve_live_symbols() -> dict[str, str]:
    available = mt5.symbols_get()
    if not available:
        raise RuntimeError(f"MT5 returned no symbols. last_error={mt5.last_error()}")

    names = [s.name for s in available]
    normalised = {name: normalise_symbol(name) for name in names}
    mapping: dict[str, str] = {}

    for logical, candidates in LIVE_SYMBOL_CANDIDATES.items():
        chosen: str | None = None

        for candidate in candidates:
            target = normalise_symbol(candidate)
            exact = [name for name in names if normalised[name] == target]
            if exact:
                chosen = exact[0]
                break

        if chosen is None:
            for candidate in candidates:
                target = normalise_symbol(candidate)
                matches = [
                    name
                    for name in names
                    if normalised[name].startswith(target)
                    and not any(code in normalised[name] for code in DATED_MONTH_CODES)
                ]
                if matches:
                    chosen = matches[0]
                    break

        if chosen is not None:
            mt5.symbol_select(chosen, True)
            mapping[logical] = chosen

    return mapping


def api_request_json(url: str, method: str = "GET", headers: dict[str, str] | None = None, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    if body is not None:
        request_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc


def fetch_pending(api_url: str, secret: str, limit: int) -> list[dict]:
    query = urllib.parse.urlencode({"limit": int(limit)})
    data = api_request_json(
        f"{api_url.rstrip('/')}/shadow/pending?{query}",
        headers={"X-Shadow-Secret": secret},
    )
    setups = data.get("setups", [])
    return setups if isinstance(setups, list) else []


def post_outcome(api_url: str, secret: str, setup_id: int, values: dict) -> None:
    payload = {"secret": secret, **values}
    api_request_json(
        f"{api_url.rstrip('/')}/shadow/{int(setup_id)}/outcome",
        method="POST",
        payload=payload,
    )


def session_day_from_setup(setup: dict) -> date:
    text = str(setup.get("time_ny", ""))
    if len(text) < 10:
        raise ValueError(f"Invalid time_ny: {text!r}")
    return date.fromisoformat(text[:10])


def c3_is_complete(setup: dict, grace_minutes: int) -> bool:
    session_day = session_day_from_setup(setup)
    _, end = c3_bounds(session_day, str(setup["scan_type"]).upper())
    return datetime.now(NY) >= end + timedelta(minutes=grace_minutes)


def build_crt_series(setup: dict, mt5_symbol: str) -> pd.Series:
    c2_close = float(setup["c2_close"])
    return pd.Series({
        "logical_symbol": str(setup["label"]),
        "mt5_symbol": mt5_symbol,
        "date_ny": session_day_from_setup(setup).isoformat(),
        "session": str(setup["scan_type"]).upper(),
        "direction": str(setup["direction"]).upper(),
        "c1_high": float(setup["c1_high"]),
        "c1_low": float(setup["c1_low"]),
        # The frozen M5 evaluator records C2 high/low for audit output but does
        # not use them in raid/MSS/FVG/entry/management logic. TradingView V2
        # currently sends C2 close only, so use it as a non-decision placeholder.
        "c2_high": c2_close,
        "c2_low": c2_close,
        "c2_close": c2_close,
    })


def evaluate_setup(setup: dict, mt5_symbol: str, min_c3_bars: int, sl_buffer_pips: float) -> tuple[str, dict] | None:
    session_day = session_day_from_setup(setup)
    m5 = load_m5_chunked(mt5_symbol, session_day, session_day)
    if m5.empty:
        print(f"  #{setup['id']} {setup['label']}: M5 history unavailable; leave pending")
        return None

    pivot_lows, pivot_highs = confirmed_pivots(m5, left=2, right=2)
    pip_size = broker_pip_size(mt5_symbol)
    crt = build_crt_series(setup, mt5_symbol)
    result = evaluate_m5_model(
        crt,
        m5,
        pivot_lows,
        pivot_highs,
        pip_size,
        sl_buffer_pips,
        min_c3_bars,
    )

    status = str(result.get("trade_status", ""))
    c3_bars = int(result.get("c3_bars", 0) or 0)
    if status in {"NO_C3_DATA", "INSUFFICIENT_C3"}:
        print(f"  #{setup['id']} {setup['label']}: {status} ({c3_bars} bars); leave pending")
        return None

    perf = add_performance_fields(pd.DataFrame([result]), m5).iloc[0]
    retest_entry = bool(result.get("retest_entry", False))

    if not retest_entry:
        outcome = "NO_ENTRY"
        model_r = None
    elif bool(result.get("ambiguous", False)) or status in AMBIGUOUS_STATUSES:
        outcome = "AMBIGUOUS"
        model_r = None
    else:
        final_outcome = str(perf.get("final_outcome", ""))
        if final_outcome not in {"STOP", "BE", "TP2", "TIMEOUT"}:
            print(f"  #{setup['id']} {setup['label']}: unclassified final outcome {final_outcome!r}; leave pending")
            return None
        outcome = final_outcome
        model_value = perf.get("model_r")
        model_r = None if pd.isna(model_value) else float(model_value)

    two_r = result.get("two_r")
    values = {
        "outcome": outcome,
        "model_r": model_r,
        "entry_time_ny": result.get("entry_time_ny"),
        "entry_price": result.get("entry"),
        "stop_price": result.get("stop"),
        "tp2_price": None if two_r is None or pd.isna(two_r) else float(two_r),
        "notes": (
            f"AUTO_SHADOW; mt5_symbol={mt5_symbol}; trade_status={status}; c3_bars={c3_bars}; "
            f"raid={bool(result.get('raid', False))}; mss={bool(result.get('mss', False))}; "
            f"fvg={bool(result.get('fvg', False))}; aligned={bool(setup.get('v2_eligible', False))}"
        ),
    }
    return outcome, values


def process_once(args: argparse.Namespace, symbol_map: dict[str, str]) -> tuple[int, int, int]:
    pending = fetch_pending(args.api_url, args.secret, args.limit)
    print(f"Pending shadow setups: {len(pending)}")

    completed = 0
    waiting = 0
    errors = 0

    for setup in pending:
        logical = str(setup.get("label", "")).upper()
        if logical not in PRIMARY_COHORT:
            continue

        if not c3_is_complete(setup, args.grace_minutes):
            waiting += 1
            continue

        mt5_symbol = symbol_map.get(logical)
        if not mt5_symbol:
            print(f"  #{setup.get('id')} {logical}: no live MT5 symbol mapping")
            errors += 1
            continue

        try:
            evaluated = evaluate_setup(setup, mt5_symbol, args.c3_min_bars, args.sl_buffer_pips)
            if evaluated is None:
                waiting += 1
                continue
            outcome, values = evaluated
            post_outcome(args.api_url, args.secret, int(setup["id"]), values)
            tag = "V2" if bool(setup.get("v2_eligible", False)) else "V1-only"
            r_text = "n/a" if values["model_r"] is None else f"{float(values['model_r']):+.3f}R"
            print(f"  #{setup['id']} {logical} {tag}: {outcome} {r_text}")
            completed += 1
        except Exception as exc:
            print(f"  #{setup.get('id')} {logical}: ERROR {exc}")
            errors += 1

    return completed, waiting, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prospective CRT V2 shadow runner. Reads pending TradingView CRTs, evaluates the frozen M5 model in MT5, and records research outcomes. No orders are sent."
    )
    parser.add_argument("--api-url", required=True, help="Deployed CRT webhook base URL, e.g. https://example.onrender.com")
    parser.add_argument("--secret", required=True, help="Same TV_WEBHOOK_SECRET used by the deployed webhook")
    parser.add_argument("--terminal", default=None, help="Optional path to terminal64.exe")
    parser.add_argument("--c3-min-bars", type=int, default=36)
    parser.add_argument("--sl-buffer-pips", type=float, default=10.0)
    parser.add_argument("--grace-minutes", type=int, default=10, help="Wait this long after C3 end before evaluating")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--watch", action="store_true", help="Keep polling instead of running one pass")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()

    if not 1 <= args.c3_min_bars <= 48:
        parser.error("--c3-min-bars must be between 1 and 48")
    if args.sl_buffer_pips <= 0:
        parser.error("--sl-buffer-pips must be positive")
    if args.grace_minutes < 0:
        parser.error("--grace-minutes must be >= 0")
    if args.poll_seconds < 60:
        parser.error("--poll-seconds must be >= 60")

    initialized = mt5.initialize(args.terminal) if args.terminal else mt5.initialize()
    if not initialized:
        print(f"ERROR: Could not initialize MT5. last_error={mt5.last_error()}")
        return 2

    try:
        symbol_map = resolve_live_symbols()
        print("CRT Scanner V2 - Prospective MT5 shadow runner")
        print("ANALYSIS ONLY: no order-send functions are used.")
        print("MT5 mappings:", ", ".join(f"{k}={v}" for k, v in symbol_map.items()) or "none")

        missing = sorted(PRIMARY_COHORT - set(symbol_map))
        if missing:
            print("WARNING: unresolved primary symbols:", ", ".join(missing))

        while True:
            stamp = datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"\n[{stamp}] shadow pass")
            try:
                completed, waiting, errors = process_once(args, symbol_map)
                print(f"Completed={completed} | Waiting={waiting} | Errors={errors}")
            except Exception as exc:
                print(f"ERROR: shadow pass failed: {exc}")
                if not args.watch:
                    return 1

            if not args.watch:
                break
            time_module.sleep(args.poll_seconds)

        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
