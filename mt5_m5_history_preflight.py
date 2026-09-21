from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 is not installed in this Python environment.")
    print("Run: python -m pip install MetaTrader5 pandas openpyxl tzdata")
    raise SystemExit(1)

import pandas as pd

from mt5_crt_frequency_test import NY, UTC, SYMBOL_ALIASES, parse_date, resolve_symbols, utc_bounds


def month_chunks(start_day: date, end_day: date):
    """Yield inclusive calendar chunks of at most 31 days."""
    cursor = start_day
    while cursor <= end_day:
        chunk_end = min(cursor + timedelta(days=30), end_day)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def probe_m5(symbol: str, start_day: date, end_day: date) -> dict:
    if not mt5.symbol_select(symbol, True):
        return {
            "bars": 0,
            "first_ny": None,
            "last_ny": None,
            "status": "SELECT_FAIL",
            "chunks_ok": 0,
            "chunks_empty": 0,
            "first_available_chunk": None,
            "last_available_chunk": None,
            "last_error": str(mt5.last_error()),
        }

    # Prime/synchronise the M5 series using the most recent bar.
    mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1)

    frames: list[pd.DataFrame] = []
    chunks_ok = 0
    chunks_empty = 0
    first_available_chunk = None
    last_available_chunk = None
    errors: list[str] = []

    for chunk_start, chunk_end in month_chunks(start_day, end_day):
        start_utc, end_utc = utc_bounds(chunk_start, chunk_end)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_utc, end_utc)

        if rates is None or len(rates) == 0:
            chunks_empty += 1
            err = mt5.last_error()
            if err and str(err) not in {"(1, 'Success')", "(1, \"Success\")"}:
                errors.append(f"{chunk_start}->{chunk_end}: {err}")
            continue

        chunks_ok += 1
        label = f"{chunk_start}->{chunk_end}"
        if first_available_chunk is None:
            first_available_chunk = label
        last_available_chunk = label
        frames.append(pd.DataFrame(rates))

    if not frames:
        return {
            "bars": 0,
            "first_ny": None,
            "last_ny": None,
            "status": "NO_M5",
            "chunks_ok": chunks_ok,
            "chunks_empty": chunks_empty,
            "first_available_chunk": None,
            "last_available_chunk": None,
            "last_error": " | ".join(errors[-3:]),
        }

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    first_utc = datetime.fromtimestamp(int(df.iloc[0]["time"]), tz=UTC)
    last_utc = datetime.fromtimestamp(int(df.iloc[-1]["time"]), tz=UTC)
    first_ny = first_utc.astimezone(NY)
    last_ny = last_utc.astimezone(NY)

    # Allow a few days around calendar boundaries for weekends/holidays.
    start_ok = first_ny.date() <= start_day + timedelta(days=7)
    end_ok = last_ny.date() >= end_day - timedelta(days=7)

    if start_ok and end_ok:
        status = "FULL"
    elif not start_ok and end_ok:
        status = "PARTIAL_START"
    elif start_ok and not end_ok:
        status = "PARTIAL_END"
    else:
        status = "PARTIAL_BOTH"

    return {
        "bars": int(len(df)),
        "first_ny": first_ny.isoformat(timespec="minutes"),
        "last_ny": last_ny.isoformat(timespec="minutes"),
        "status": status,
        "chunks_ok": chunks_ok,
        "chunks_empty": chunks_empty,
        "first_available_chunk": first_available_chunk,
        "last_available_chunk": last_available_chunk,
        "last_error": " | ".join(errors[-3:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether MT5 has enough M5 history for CRT robustness backtests.")
    parser.add_argument("--start", type=parse_date, required=True)
    parser.add_argument("--end", type=parse_date, required=True)
    parser.add_argument("--terminal", default=None)
    parser.add_argument("--max-candidates", type=int, default=12)
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must be on or after --start")

    ok = mt5.initialize(args.terminal) if args.terminal else mt5.initialize()
    if not ok:
        print(f"ERROR: mt5.initialize() failed: {mt5.last_error()}")
        return 2

    try:
        mapping, missing, resolver_diag = resolve_symbols(args.start, args.end, args.max_candidates)

        print("MT5 M5 HISTORY PREFLIGHT - CHUNKED")
        print(f"Requested: {args.start} -> {args.end}")
        print("Resolver uses M15 coverage; M5 is probed in <=31-day chunks to avoid large-range false negatives.")
        print("=" * 132)
        print(f"{'Logical':<10} {'MT5 Symbol':<18} {'M5 bars':>10} {'First M5 NY':<23} {'Last M5 NY':<23} {'OK':>4} {'Empty':>5} Status")
        print("-" * 132)

        rows = []
        for logical in SYMBOL_ALIASES:
            mt5_symbol = mapping.get(logical)
            if not mt5_symbol:
                row = {
                    "logical_symbol": logical,
                    "mt5_symbol": "",
                    "bars": 0,
                    "first_ny": None,
                    "last_ny": None,
                    "status": "NO_M15_MAPPING",
                    "chunks_ok": 0,
                    "chunks_empty": 0,
                    "first_available_chunk": None,
                    "last_available_chunk": None,
                    "last_error": "",
                }
            else:
                probe = probe_m5(mt5_symbol, args.start, args.end)
                row = {"logical_symbol": logical, "mt5_symbol": mt5_symbol, **probe}
            rows.append(row)

            print(
                f"{logical:<10} {row['mt5_symbol']:<18} {row['bars']:>10,} "
                f"{str(row['first_ny'] or '-'): <23} {str(row['last_ny'] or '-'): <23} "
                f"{row['chunks_ok']:>4} {row['chunks_empty']:>5} {row['status']}"
            )

        df = pd.DataFrame(rows)
        df.to_csv("mt5_m5_history_preflight.csv", index=False)

        full = int((df["status"] == "FULL").sum())
        none = int((df["status"] == "NO_M5").sum())
        partial = int(df["status"].astype(str).str.startswith("PARTIAL").sum())

        print("\nSUMMARY")
        print(f"  FULL coverage:    {full}/10")
        print(f"  PARTIAL coverage: {partial}/10")
        print(f"  NO M5 coverage:   {none}/10")
        print("  Saved: mt5_m5_history_preflight.csv")

        print("\nINTERPRETATION")
        print("  FULL          = M5 reaches both requested boundaries (allowing weekends/holidays).")
        print("  PARTIAL_START = recent M5 exists, but the requested starting history is unavailable.")
        print("  NO_M5         = even the smaller chunk requests returned no M5 bars.")
        print("  The CSV includes first_available_chunk / last_available_chunk for diagnosis.")

        if full < len(SYMBOL_ALIASES):
            print("\nNEXT STEP:")
            print("  Do NOT treat missing years as zero-performance years.")
            print("  If PARTIAL_START appears, we will start the robustness test from the earliest common M5 date.")
            print("  If NO_M5 still appears despite your 2026 test working, rerun a 2026-only preflight to isolate terminal synchronisation.")
            return 1

        print("\nM5 coverage passed for all 10 symbols.")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
