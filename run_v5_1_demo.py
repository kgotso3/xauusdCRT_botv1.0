from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from mt5.connection import connect, disconnect
from mt5.market_data import get_rates, latest_tick, resolve_symbol
from mt5.v5_1_demo_execution import (
    DemoRiskConfig,
    account_is_demo_only,
    build_limit_requests,
    hard_risk_gate,
    submit_demo_requests,
)
from shadow.v5_engine import _candidate_from_latest_causal


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {
            "version": "V5.1",
            "last_completed_m5": None,
            "signals": {},
            "daily_r": 0.0,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _signal_id(symbol: str, candidate: dict) -> str:
    return f"{symbol}|V5.1|{candidate['confirmation_time_utc']}|{candidate['direction']}|OTE079"


def _print_request(req: dict) -> None:
    safe = {k: v for k, v in req.items() if k not in {"login", "password"}}
    print(json.dumps(safe, indent=2, sort_keys=True, default=str))


def run(args) -> None:
    cfg = DemoRiskConfig(
        risk_fraction=args.risk,
        max_open_setups=1,
        daily_loss_limit_r=args.daily_loss_r,
        max_consecutive_losses=args.max_consecutive_losses,
    )
    state_path = Path(args.state)
    state = _load_state(state_path)
    settings = connect()
    try:
        account_is_demo_only()
        symbol = resolve_symbol(settings.symbol)
        account = mt5.account_info()
        if account is None:
            raise RuntimeError(f"Unable to read account: {mt5.last_error()}")

        mode = "DEMO EXECUTION ENABLED" if args.execute_demo else "DRY RUN"
        print("CRT V5.1 — CORRECTED CAUSAL DEMO RUNNER")
        print(f"Mode: {mode}")
        print(f"Symbol: {symbol} | OTE 0.79 | setup risk: {cfg.risk_fraction:.3%}")
        print("HARD LOCK: non-demo MT5 accounts are rejected before order validation/submission.")
        print("Two legs: 50% volume to 1R and 50% volume to 2R.\n")

        while True:
            tick = latest_tick(symbol)
            m5_tail = get_rates(symbol, "M5", 10, completed_only=True, asof=tick["time"])
            latest_completed_m5 = pd.Timestamp(m5_tail.iloc[-1]["time"]).isoformat()

            if state.get("last_completed_m5") != latest_completed_m5:
                h1 = get_rates(symbol, "H1", 400, completed_only=True, asof=tick["time"])
                m15 = get_rates(symbol, "M15", 1200, completed_only=True, asof=tick["time"])
                m5 = get_rates(symbol, "M5", 3000, completed_only=True, asof=tick["time"])
                # V5.1 cohort starts when this runner first sees a corrected signal;
                # unlike V5.0, no historical backfill is accepted.
                start = state.get("started_at_utc") or pd.Timestamp(tick["time"]).isoformat()
                state.setdefault("started_at_utc", start)
                candidate = _candidate_from_latest_causal(h1, m15, m5, start)

                if candidate is not None:
                    candidate = dict(candidate)
                    candidate["retest_expiry_utc"] = (
                        pd.Timestamp(candidate["confirmation_time_utc"])
                        + pd.to_timedelta(2, unit="h")
                    ).isoformat()
                    sid = _signal_id(symbol, candidate)
                    if sid not in state["signals"]:
                        record = {
                            **candidate,
                            "signal_id": sid,
                            "status": "DETECTED",
                            "dry_run": not args.execute_demo,
                            "order_tickets": [],
                        }
                        state["signals"][sid] = record
                        _save_state(state_path, state)

                        print(f"\nNEW V5.1 SETUP {candidate['direction']} confirmation={candidate['confirmation_time_utc']}")
                        print(
                            f"entry={candidate['entry']:.3f} stop={candidate['stop']:.3f} "
                            f"tp1={candidate['tp1']:.3f} tp2={candidate['tp2']:.3f}"
                        )

                        hard_risk_gate(symbol, float(state.get("daily_r", 0.0)), cfg)
                        requests, meta = build_limit_requests(symbol, record, cfg)
                        record["risk_meta"] = meta
                        record["requests"] = requests
                        record["status"] = "VALIDATED"

                        print("\nBROKER-VALIDATED MT5 REQUESTS")
                        for req in requests:
                            _print_request(req)

                        results = submit_demo_requests(requests, dry_run=not args.execute_demo)
                        if args.execute_demo:
                            record["order_tickets"] = [int(getattr(r, "order", 0)) for r in results]
                            record["status"] = "PENDING_ORDERS_PLACED"
                            print(f"DEMO ORDERS PLACED: {record['order_tickets']}")
                        else:
                            record["status"] = "DRY_RUN_VALIDATED"
                            print("DRY RUN PASSED: order_check accepted both requests; order_send() was NOT called.")
                        _save_state(state_path, state)

                state["last_completed_m5"] = latest_completed_m5
                _save_state(state_path, state)

            if args.once:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nV5.1 runner stopped by user. State preserved.")
        _save_state(state_path, state)
    finally:
        disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="V5.1 corrected causal GOLD demo execution runner")
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--daily-loss-r", type=float, default=2.0)
    p.add_argument("--max-consecutive-losses", type=int, default=3)
    p.add_argument("--poll-seconds", type=int, default=5)
    p.add_argument("--state", default="data/demo/v5_1_demo_state.json")
    p.add_argument("--once", action="store_true")
    p.add_argument(
        "--execute-demo",
        action="store_true",
        help="Actually submit pending orders. Still hard-blocked unless MT5 account is DEMO.",
    )
    run(p.parse_args())
