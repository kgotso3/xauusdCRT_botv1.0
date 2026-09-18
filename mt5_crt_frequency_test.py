from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 is not installed in this Python environment.")
    print("Run: python -m pip install MetaTrader5 pandas openpyxl")
    raise SystemExit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is not installed in this Python environment.")
    print("Run: python -m pip install pandas openpyxl")
    raise SystemExit(1)


NY = ZoneInfo("America/New_York")
UTC = timezone.utc

# Logical scanner symbol -> common MT5/XM aliases.
SYMBOL_ALIASES: dict[str, list[str]] = {
    "XAUUSD": ["XAUUSD", "GOLD"],
    "NAS100": ["US100Cash", "US100", "NAS100", "USTEC", "NASDAQ"],
    "US500": ["US500Cash", "US500", "SPX500", "SP500"],
    "US30": ["US30Cash", "US30", "WS30", "DJ30", "DOW30"],
    "EURUSD": ["EURUSD"],
    "GBPUSD": ["GBPUSD"],
    "USDJPY": ["USDJPY"],
    "USDCAD": ["USDCAD"],
    "AUDUSD": ["AUDUSD"],
    "BTCUSD": ["BTCUSD", "BTCUSDT"],
}


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Date must use YYYY-MM-DD format") from exc


def normalize_symbol(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def symbol_score(mt5_name: str, aliases: list[str]) -> int:
    """Return a conservative score for matching an MT5 broker symbol to an alias."""
    name = normalize_symbol(mt5_name)
    best = -1
    for alias in aliases:
        a = normalize_symbol(alias)
        if name == a:
            best = max(best, 1000)
        elif name.startswith(a):
            # Handles broker suffixes such as EURUSDm, BTCUSD#, US100Cash.a.
            best = max(best, 900 - max(0, len(name) - len(a)))
        elif a in name:
            best = max(best, 700 - max(0, len(name) - len(a)))
    return best


def resolve_symbols() -> tuple[dict[str, str], list[str]]:
    available = mt5.symbols_get()
    if not available:
        raise RuntimeError(f"MT5 returned no symbols. last_error={mt5.last_error()}")

    names = [s.name for s in available]
    mapping: dict[str, str] = {}
    missing: list[str] = []

    for logical, aliases in SYMBOL_ALIASES.items():
        scored = sorted(
            ((symbol_score(name, aliases), name) for name in names),
            key=lambda x: (x[0], -len(x[1])),
            reverse=True,
        )
        if scored and scored[0][0] >= 700:
            mapping[logical] = scored[0][1]
            mt5.symbol_select(scored[0][1], True)
        else:
            missing.append(logical)

    return mapping, missing


def load_m15(symbol: str, start_day: date, end_day: date) -> pd.DataFrame:
    start_local = datetime.combine(start_day, time(0, 0), tzinfo=NY)
    end_local = datetime.combine(end_day + timedelta(days=1), time(0, 0), tzinfo=NY)

    rates = mt5.copy_rates_range(
        symbol,
        mt5.TIMEFRAME_M15,
        start_local.astimezone(UTC),
        end_local.astimezone(UTC),
    )

    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_ny"] = df["time_utc"].dt.tz_convert("America/New_York")
    df["ny_date"] = df["time_ny"].dt.date
    df["minute_of_day"] = df["time_ny"].dt.hour * 60 + df["time_ny"].dt.minute

    return df[(df["ny_date"] >= start_day) & (df["ny_date"] <= end_day)].copy()


def window(df_day: pd.DataFrame, start_hour: int, end_hour: int) -> pd.DataFrame:
    start_min = start_hour * 60
    end_min = end_hour * 60
    return df_day[(df_day["minute_of_day"] >= start_min) & (df_day["minute_of_day"] < end_min)].sort_values("time_ny")


def evaluate_session(
    logical: str,
    mt5_symbol: str,
    session_day: date,
    session_name: str,
    c1: pd.DataFrame,
    c2: pd.DataFrame,
    min_bars: int,
) -> dict | None:
    # Four hours of M15 data normally contains 16 bars. Requiring coverage prevents
    # weekend/stale-session data from being treated as a valid opportunity.
    if len(c1) < min_bars or len(c2) < min_bars:
        return None

    c1_high = float(c1["high"].max())
    c1_low = float(c1["low"].min())
    c2_high = float(c2["high"].max())
    c2_low = float(c2["low"].min())
    c2_close = float(c2.iloc[-1]["close"])

    swept_high = c2_high > c1_high
    swept_low = c2_low < c1_low
    exactly_one = swept_high != swept_low
    close_inside = c1_low < c2_close < c1_high
    valid = exactly_one and close_inside

    if swept_high and swept_low:
        sweep = "BOTH"
    elif swept_high:
        sweep = "HIGH"
    elif swept_low:
        sweep = "LOW"
    else:
        sweep = "NONE"

    direction = "BULLISH" if valid and swept_low else "BEARISH" if valid and swept_high else "-"

    if valid:
        reason = "VALID"
    elif swept_high and swept_low:
        reason = "BOTH_SIDES"
    elif not swept_high and not swept_low:
        reason = "NO_SWEEP"
    elif not close_inside:
        reason = "SWEEP_CLOSE_OUTSIDE"
    else:
        reason = "INVALID"

    return {
        "logical_symbol": logical,
        "mt5_symbol": mt5_symbol,
        "date_ny": session_day.isoformat(),
        "session": session_name,
        "c1_bars": int(len(c1)),
        "c2_bars": int(len(c2)),
        "c1_high": c1_high,
        "c1_low": c1_low,
        "c2_high": c2_high,
        "c2_low": c2_low,
        "c2_close": c2_close,
        "swept_high": bool(swept_high),
        "swept_low": bool(swept_low),
        "sweep": sweep,
        "close_inside": bool(close_inside),
        "valid_crt": bool(valid),
        "direction": direction,
        "result": reason,
    }


def analyze_symbol(
    logical: str,
    mt5_symbol: str,
    df: pd.DataFrame,
    sessions: str,
    min_bars: int,
) -> pd.DataFrame:
    records: list[dict] = []

    for session_day, day_df in df.groupby("ny_date", sort=True):
        if sessions in {"both", "am"}:
            rec = evaluate_session(
                logical,
                mt5_symbol,
                session_day,
                "AM",
                window(day_df, 1, 5),
                window(day_df, 5, 9),
                min_bars,
            )
            if rec is not None:
                records.append(rec)

        if sessions in {"both", "pm"}:
            rec = evaluate_session(
                logical,
                mt5_symbol,
                session_day,
                "PM",
                window(day_df, 13, 17),
                window(day_df, 17, 21),
                min_bars,
            )
            if rec is not None:
                records.append(rec)

    return pd.DataFrame(records)


def build_summary(details: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    rows: list[dict] = []

    for logical in SYMBOL_ALIASES:
        mapped = mapping.get(logical)
        if mapped is None:
            rows.append({
                "Symbol": logical,
                "MT5 Symbol": "NOT FOUND",
                "Eligible Scans": 0,
                "Valid CRT": 0,
                "Valid %": 0.0,
                "AM Valid": 0,
                "PM Valid": 0,
                "Bullish": 0,
                "Bearish": 0,
                "Both-side Sweeps": 0,
                "Sweep Close Outside": 0,
                "No Sweep": 0,
            })
            continue

        part = details[details["logical_symbol"] == logical] if not details.empty else pd.DataFrame()
        eligible = len(part)
        valid_count = int(part["valid_crt"].sum()) if eligible else 0

        rows.append({
            "Symbol": logical,
            "MT5 Symbol": mapped,
            "Eligible Scans": eligible,
            "Valid CRT": valid_count,
            "Valid %": round((valid_count / eligible * 100.0), 2) if eligible else 0.0,
            "AM Valid": int(((part["session"] == "AM") & part["valid_crt"]).sum()) if eligible else 0,
            "PM Valid": int(((part["session"] == "PM") & part["valid_crt"]).sum()) if eligible else 0,
            "Bullish": int((part["direction"] == "BULLISH").sum()) if eligible else 0,
            "Bearish": int((part["direction"] == "BEARISH").sum()) if eligible else 0,
            "Both-side Sweeps": int((part["result"] == "BOTH_SIDES").sum()) if eligible else 0,
            "Sweep Close Outside": int((part["result"] == "SWEEP_CLOSE_OUTSIDE").sum()) if eligible else 0,
            "No Sweep": int((part["result"] == "NO_SWEEP").sum()) if eligible else 0,
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["Valid CRT", "Valid %", "Eligible Scans"], ascending=[False, False, False]).reset_index(drop=True)
    summary.insert(0, "Rank", range(1, len(summary) + 1))
    return summary


def print_mapping(mapping: dict[str, str], missing: list[str]) -> None:
    print("\nMT5 SYMBOL MAPPING")
    print("-" * 52)
    for logical in SYMBOL_ALIASES:
        print(f"{logical:<10} -> {mapping.get(logical, 'NOT FOUND')}")
    if missing:
        print("\nMissing symbols will remain in the output with zero eligible scans.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank the 10 CRT Scanner V1 markets by historical valid 4H CRT frequency using MT5 M15 data."
    )
    parser.add_argument("--start", type=parse_date, required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=parse_date, required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--sessions", choices=["both", "am", "pm"], default="both", help="CRT sessions to test")
    parser.add_argument("--min-bars", type=int, default=16, help="Minimum M15 bars required in each 4H C1/C2 window (default: 16)")
    parser.add_argument("--output", default="results/crt_frequency", help="Output directory")
    parser.add_argument("--terminal", default=None, help="Optional path to terminal64.exe")
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must be on or after --start")
    if not 1 <= args.min_bars <= 16:
        parser.error("--min-bars must be between 1 and 16")

    print("CRT Scanner V1 - MT5 historical frequency test")
    print(f"Period: {args.start} to {args.end}")
    print(f"Sessions: {args.sessions.upper()} | Minimum bars/window: {args.min_bars}/16")
    print("Rule: C2 sweeps exactly ONE side of C1 and C2 closes back inside C1.")
    print("Time model: synthetic 4H windows in America/New_York, built from M15 bars.\n")

    initialized = mt5.initialize(args.terminal) if args.terminal else mt5.initialize()
    if not initialized:
        print(f"ERROR: Could not initialize MT5. last_error={mt5.last_error()}")
        print("Open MT5, log in to the account, then rerun this command.")
        return 2

    try:
        mapping, missing = resolve_symbols()
        print_mapping(mapping, missing)

        all_details: list[pd.DataFrame] = []

        print("\nDOWNLOADING / ANALYZING M15 DATA")
        print("-" * 52)
        for logical in SYMBOL_ALIASES:
            mt5_symbol = mapping.get(logical)
            if not mt5_symbol:
                print(f"{logical:<10} skipped - symbol not found")
                continue

            df = load_m15(mt5_symbol, args.start, args.end)
            if df.empty:
                print(f"{logical:<10} {mt5_symbol:<18} 0 bars - no history returned")
                continue

            result = analyze_symbol(logical, mt5_symbol, df, args.sessions, args.min_bars)
            if not result.empty:
                all_details.append(result)

            print(f"{logical:<10} {mt5_symbol:<18} {len(df):>8,} M15 bars | {len(result):>4} eligible scans")

        details = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
        summary = build_summary(details, mapping)

        print("\nCRT FREQUENCY RANKING")
        print("=" * 92)
        display_cols = ["Rank", "Symbol", "MT5 Symbol", "Eligible Scans", "Valid CRT", "Valid %", "AM Valid", "PM Valid"]
        print(summary[display_cols].to_string(index=False))

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_csv = out_dir / "crt_frequency_ranking.csv"
        details_csv = out_dir / "crt_frequency_details.csv"
        summary.to_csv(summary_csv, index=False)
        details.to_csv(details_csv, index=False)

        xlsx_path = out_dir / "crt_frequency_ranking.xlsx"
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                summary.to_excel(writer, sheet_name="Ranking", index=False)
                details.to_excel(writer, sheet_name="All Scans", index=False)
                if not details.empty:
                    details[details["session"] == "AM"].to_excel(writer, sheet_name="AM", index=False)
                    details[details["session"] == "PM"].to_excel(writer, sheet_name="PM", index=False)
                    details[details["valid_crt"]].to_excel(writer, sheet_name="Valid CRT", index=False)
            excel_msg = str(xlsx_path)
        except (ImportError, ModuleNotFoundError):
            excel_msg = "not written (install openpyxl)"

        print("\nSaved:")
        print(f"  {summary_csv}")
        print(f"  {details_csv}")
        print(f"  Excel: {excel_msg}")
        print("\nNOTE: This test ranks CRT occurrence frequency only. It does not yet measure profitability or C3 outcomes.")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
