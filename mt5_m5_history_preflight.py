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


def probe_m5(symbol: str, start_day: date, end_day: date) -> dict:
    if not mt5.symbol_select(symbol, True):
        return {
            "bars": 0,
            "first_ny": None,
            "last_ny": None,
            "status": "SELECT_FAIL",
            "last_error": str(mt5.last_error()),
        }

    start_utc, end_utc = utc_bounds(start_day, end_day)

    # A tiny current-bar request helps MT5 initialise/synchronise the series
    # before the historical range request on some terminals.
    mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_utc, end_utc)

    if rates is None or len(rates) == 0:
        return {
            "bars": 0,
            "first_ny": None,
            "last_ny": None,
            "status": "NO_M5",
            "last_error": str(mt5.last_error()),
        }

    first_utc = datetime.fromtimestamp(int(rates[0]["time"]), tz=UTC)
    last_utc = datetime.fromtimestamp(int(rates[-1]["time"]), tz=UTC)
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
        "bars": int(len(rates)),
        "first_ny": first_ny.isoformat(timespec="minutes"),
        "last_ny": last_ny.isoformat(timespec="minutes"),
        "status": status,
        "last_error": "",
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

        print("MT5 M5 HISTORY PREFLIGHT")
        print(f"Requested: {args.start} -> {args.end}")
        print("Resolver uses M15 coverage; this check validates M5 coverage separately.")
        print("=" * 104)
        print(f"{'Logical':<10} {'MT5 Symbol':<18} {'M5 bars':>10} {'First M5 NY':<23} {'Last M5 NY':<23} Status")
        print("-" * 104)

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
                    "last_error": "",
                }
            else:
                probe = probe_m5(mt5_symbol, args.start, args.end)
                row = {"logical_symbol": logical, "mt5_symbol": mt5_symbol, **probe}
            rows.append(row)

            print(
                f"{logical:<10} {row['mt5_symbol']:<18} {row['bars']:>10,} "
                f"{str(row['first_ny'] or '-'): <23} {str(row['last_ny'] or '-'): <23} {row['status']}"
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

        if full < len(SYMBOL_ALIASES):
            print("\nACTION REQUIRED BEFORE A FULL ROBUSTNESS TEST:")
            print("  In MT5: Tools -> Options -> Charts -> Max bars in chart")
            print("  Set it to a very high value (for example 1,000,000, or Unlimited if offered).")
            print("  Restart MT5, keep the XM account logged in, then rerun this preflight.")
            print("  The Python API can only access bars that the terminal/broker makes available.")
            return 1

        print("\nM5 coverage passed for all 10 symbols.")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
