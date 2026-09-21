from __future__ import annotations

from datetime import date, timedelta

import MetaTrader5 as mt5
import pandas as pd

from mt5_crt_frequency_test import NY, UTC, normalize_symbol, probe_history, utc_bounds


# Keep the historical test on the same instrument family used in the validated 2026 run.
# In particular, do not silently substitute dated index contracts such as US100-DEC26.
FROZEN_SYMBOL_CANDIDATES: dict[str, list[str]] = {
    "XAUUSD": ["GOLD", "XAUUSD"],
    "NAS100": ["US100Cash"],
    "US500": ["US500Cash"],
    "US30": ["US30Cash"],
    "EURUSD": ["EURUSD"],
    "GBPUSD": ["GBPUSD"],
    "USDJPY": ["USDJPY"],
    "USDCAD": ["USDCAD"],
    "AUDUSD": ["AUDUSD"],
    "BTCUSD": ["BTCUSD", "BTCUSDT"],
}


def _date_chunks(start_day: date, end_day: date, days: int = 31):
    cur = start_day
    while cur <= end_day:
        chunk_end = min(cur + timedelta(days=days - 1), end_day)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def _copy_rates_chunked(symbol: str, timeframe: int, start_day: date, end_day: date) -> pd.DataFrame:
    if not mt5.symbol_select(symbol, True):
        return pd.DataFrame()

    # Prime/synchronise the series before older-range requests.
    mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)

    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _date_chunks(start_day, end_day):
        start_utc, end_utc = utc_bounds(chunk_start, chunk_end)
        rates = mt5.copy_rates_range(symbol, timeframe, start_utc, end_utc)
        if rates is None or len(rates) == 0:
            continue
        frames.append(pd.DataFrame(rates))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time"], keep="first").sort_values("time").reset_index(drop=True)
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_ny"] = df["time_utc"].dt.tz_convert("America/New_York")
    return df


def load_m15_chunked(symbol: str, start_day: date, end_day: date) -> pd.DataFrame:
    df = _copy_rates_chunked(symbol, mt5.TIMEFRAME_M15, start_day, end_day)
    if df.empty:
        return df
    df["ny_date"] = df["time_ny"].dt.date
    df["minute_of_day"] = df["time_ny"].dt.hour * 60 + df["time_ny"].dt.minute
    return df[(df["ny_date"] >= start_day) & (df["ny_date"] <= end_day)].copy()


def load_m5_chunked(symbol: str, start_day: date, end_day: date) -> pd.DataFrame:
    # Preserve the original model's three-day causal pivot context and two-day PM-C3 tail.
    context_start = start_day - timedelta(days=3)
    context_end = end_day + timedelta(days=2)
    return _copy_rates_chunked(symbol, mt5.TIMEFRAME_M5, context_start, context_end)


def _matching_terminal_names(available_names: list[str], requested: str) -> list[str]:
    target = normalize_symbol(requested)
    exact = [name for name in available_names if normalize_symbol(name) == target]
    if exact:
        return exact

    # Allow broker suffixes/prefix decorations, but only around the frozen instrument name.
    return [
        name for name in available_names
        if normalize_symbol(name).startswith(target)
        and not any(code in normalize_symbol(name) for code in ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))
    ]


def resolve_frozen_symbols(start_day: date, end_day: date, max_candidates: int = 12):
    available = mt5.symbols_get()
    if not available:
        raise RuntimeError(f"MT5 returned no symbols. last_error={mt5.last_error()}")

    names = [s.name for s in available]
    mapping: dict[str, str] = {}
    missing: list[str] = []
    diagnostics: list[dict] = []

    for logical, preferred_names in FROZEN_SYMBOL_CANDIDATES.items():
        probes: list[dict] = []
        seen: set[str] = set()

        for preferred in preferred_names:
            for candidate in _matching_terminal_names(names, preferred):
                if candidate in seen:
                    continue
                seen.add(candidate)
                probe = probe_history(candidate, start_day, end_day)
                probe["logical_symbol"] = logical
                probe["preferred_name"] = preferred
                probe["frozen_universe"] = True
                probes.append(probe)

        probes.sort(key=lambda p: (int(p["bars"]), int(p["selected_ok"])), reverse=True)
        if probes and int(probes[0]["bars"]) > 0:
            chosen = probes[0]["candidate"]
            mapping[logical] = chosen
            mt5.symbol_select(chosen, True)
        else:
            chosen = None
            missing.append(logical)

        for probe in probes:
            probe["chosen"] = probe["candidate"] == chosen
            diagnostics.append(probe)

        if not probes:
            diagnostics.append({
                "logical_symbol": logical,
                "candidate": "NO FROZEN SYMBOL MATCH",
                "preferred_name": ";".join(preferred_names),
                "selected_ok": False,
                "bars": 0,
                "first_bar_ny": None,
                "last_bar_ny": None,
                "last_error": "",
                "frozen_universe": True,
                "chosen": False,
            })

    return mapping, missing, pd.DataFrame(diagnostics)
