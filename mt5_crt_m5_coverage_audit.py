from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from mt5_crt_frequency_test import SYMBOL_ALIASES, analyze_symbol, parse_date
from mt5_crt_m5_model_test import c3_bounds
from mt5_robust_history import load_m15_chunked, load_m5_chunked, resolve_frozen_symbols


PRIMARY_COHORT = ["US30", "US500", "XAUUSD"]


def c3_bar_count(m5: pd.DataFrame, session_day: date, session: str) -> int:
    start, end = c3_bounds(session_day, session)
    return int(((m5["time_ny"] >= start) & (m5["time_ny"] < end)).sum())


def coverage_summary(setups: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if setups.empty:
        return pd.DataFrame()

    grouped = setups.groupby(group_cols, dropna=False)
    out = grouped.agg(
        Valid_CRT=("coverage_ok", "size"),
        Covered_CRT=("coverage_ok", "sum"),
        Avg_C3_Bars=("c3_bars", "mean"),
        Min_C3_Bars=("c3_bars", "min"),
        Max_C3_Bars=("c3_bars", "max"),
    ).reset_index()
    out["Coverage %"] = (out["Covered_CRT"] / out["Valid_CRT"] * 100.0).round(1)
    out["Avg_C3_Bars"] = out["Avg_C3_Bars"].round(1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit M5 C3 history coverage so CRT robustness results are not biased by missing lower-timeframe data."
    )
    parser.add_argument("--start", type=parse_date, required=True)
    parser.add_argument("--end", type=parse_date, required=True)
    parser.add_argument("--min-bars", type=int, default=12, help="Minimum M15 bars in C1/C2")
    parser.add_argument("--c3-min-bars", type=int, default=36, help="Minimum M5 bars required for a usable C3")
    parser.add_argument("--sessions", choices=["both", "am", "pm"], default="both")
    parser.add_argument("--coverage-threshold", type=float, default=90.0, help="Coverage percent required for a month to be considered clean")
    parser.add_argument("--output", default="results/crt_m5_coverage_audit")
    parser.add_argument("--terminal", default=None)
    parser.add_argument("--max-candidates", type=int, default=12)
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must be on or after --start")
    if not 1 <= args.min_bars <= 16:
        parser.error("--min-bars must be between 1 and 16")
    if not 1 <= args.c3_min_bars <= 48:
        parser.error("--c3-min-bars must be between 1 and 48")
    if not 0.0 <= args.coverage_threshold <= 100.0:
        parser.error("--coverage-threshold must be between 0 and 100")

    initialized = mt5.initialize(args.terminal) if args.terminal else mt5.initialize()
    if not initialized:
        print(f"ERROR: Could not initialize MT5. last_error={mt5.last_error()}")
        return 2

    try:
        mapping, missing, resolver_diag = resolve_frozen_symbols(args.start, args.end, args.max_candidates)
        if missing:
            print("WARNING: missing frozen-universe symbols:", ", ".join(missing))

        rows: list[dict] = []

        print("CRT Scanner V1 - M5 coverage bias audit")
        print(f"Period: {args.start} -> {args.end}")
        print(f"Coverage rule: C3 must contain >= {args.c3_min_bars}/48 M5 bars")
        print("-" * 112)

        for logical in SYMBOL_ALIASES:
            mt5_symbol = mapping.get(logical)
            if not mt5_symbol:
                print(f"{logical:<10} skipped - no frozen MT5 symbol")
                continue

            m15 = load_m15_chunked(mt5_symbol, args.start, args.end)
            if m15.empty:
                print(f"{logical:<10} {mt5_symbol:<18} no M15 history")
                continue

            scans = analyze_symbol(logical, mt5_symbol, m15, args.sessions, args.min_bars)
            valid = scans[scans["valid_crt"]].copy() if not scans.empty else pd.DataFrame()
            if valid.empty:
                print(f"{logical:<10} {mt5_symbol:<18} 0 valid CRTs")
                continue

            m5 = load_m5_chunked(mt5_symbol, args.start, args.end)
            if m5.empty:
                print(f"{logical:<10} {mt5_symbol:<18} valid CRT={len(valid):>4} | M5 EMPTY")
                for _, crt in valid.iterrows():
                    session_day = date.fromisoformat(str(crt["date_ny"]))
                    rows.append({
                        "logical_symbol": logical,
                        "mt5_symbol": mt5_symbol,
                        "date_ny": session_day.isoformat(),
                        "month": session_day.strftime("%Y-%m"),
                        "session": str(crt["session"]),
                        "direction": str(crt["direction"]),
                        "c3_bars": 0,
                        "coverage_ok": False,
                    })
                continue

            covered = 0
            for _, crt in valid.iterrows():
                session_day = date.fromisoformat(str(crt["date_ny"]))
                count = c3_bar_count(m5, session_day, str(crt["session"]))
                ok = count >= args.c3_min_bars
                covered += int(ok)
                rows.append({
                    "logical_symbol": logical,
                    "mt5_symbol": mt5_symbol,
                    "date_ny": session_day.isoformat(),
                    "month": session_day.strftime("%Y-%m"),
                    "session": str(crt["session"]),
                    "direction": str(crt["direction"]),
                    "c3_bars": count,
                    "coverage_ok": ok,
                })

            pct = covered / len(valid) * 100.0
            first_m5 = m5.iloc[0]["time_ny"]
            last_m5 = m5.iloc[-1]["time_ny"]
            print(
                f"{logical:<10} {mt5_symbol:<18} CRT={len(valid):>4} | covered={covered:>4} "
                f"({pct:>5.1f}%) | M5 {first_m5} -> {last_m5}"
            )

        setups = pd.DataFrame(rows)
        if setups.empty:
            print("ERROR: No valid CRT setups were available for the audit.")
            return 1

        symbol_summary = coverage_summary(setups, ["logical_symbol"])
        monthly = coverage_summary(setups, ["month", "logical_symbol"])
        session_summary = coverage_summary(setups, ["logical_symbol", "session"])

        primary_monthly = monthly[monthly["logical_symbol"].isin(PRIMARY_COHORT)].copy()
        if primary_monthly.empty:
            common_months = pd.DataFrame(columns=["month", "Symbols", "Min Coverage %", "All >= threshold"])
        else:
            pm = primary_monthly.copy()
            pm["passes"] = pm["Coverage %"] >= args.coverage_threshold
            common_months = pm.groupby("month").agg(
                Symbols=("logical_symbol", "nunique"),
                Min_Coverage=("Coverage %", "min"),
                All_Pass=("passes", "all"),
            ).reset_index()
            common_months["All >= threshold"] = (common_months["Symbols"] == len(PRIMARY_COHORT)) & common_months["All_Pass"]
            common_months = common_months.rename(columns={"Min_Coverage": "Min Coverage %"})
            common_months = common_months[["month", "Symbols", "Min Coverage %", "All >= threshold"]]

        print("\nSYMBOL COVERAGE SUMMARY")
        print("=" * 88)
        print(symbol_summary.to_string(index=False))

        print("\nPRIMARY COHORT MONTHLY COVERAGE")
        print("=" * 100)
        if primary_monthly.empty:
            print("No primary-cohort monthly rows.")
        else:
            print(primary_monthly[["month", "logical_symbol", "Valid_CRT", "Covered_CRT", "Coverage %", "Avg_C3_Bars"]].to_string(index=False))

        clean = common_months[common_months["All >= threshold"]] if not common_months.empty else pd.DataFrame()
        print(f"\nCOMMON PRIMARY-COHORT MONTHS >= {args.coverage_threshold:g}% COVERAGE")
        print("=" * 88)
        if clean.empty:
            print("None")
        else:
            print(clean.to_string(index=False))
            print(f"\nEarliest clean common month: {clean.iloc[0]['month']}")
            print(f"Latest clean common month:   {clean.iloc[-1]['month']}")

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        setups_csv = out_dir / "coverage_setups.csv"
        symbol_csv = out_dir / "coverage_by_symbol.csv"
        monthly_csv = out_dir / "coverage_by_symbol_month.csv"
        session_csv = out_dir / "coverage_by_symbol_session.csv"
        common_csv = out_dir / "coverage_primary_common_months.csv"
        resolver_csv = out_dir / "symbol_resolver_diagnostics.csv"

        setups.to_csv(setups_csv, index=False)
        symbol_summary.to_csv(symbol_csv, index=False)
        monthly.to_csv(monthly_csv, index=False)
        session_summary.to_csv(session_csv, index=False)
        common_months.to_csv(common_csv, index=False)
        resolver_diag.to_csv(resolver_csv, index=False)

        xlsx_path = out_dir / "crt_m5_coverage_audit.xlsx"
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                symbol_summary.to_excel(writer, sheet_name="By Symbol", index=False)
                monthly.to_excel(writer, sheet_name="By Symbol Month", index=False)
                session_summary.to_excel(writer, sheet_name="By Symbol Session", index=False)
                common_months.to_excel(writer, sheet_name="Primary Common Months", index=False)
                setups.to_excel(writer, sheet_name="All CRT Setups", index=False)
                resolver_diag.to_excel(writer, sheet_name="Resolver", index=False)
            excel_msg = str(xlsx_path)
        except (ImportError, ModuleNotFoundError):
            excel_msg = "not written (install openpyxl)"

        print("\nSaved:")
        print(f"  {setups_csv}")
        print(f"  {symbol_csv}")
        print(f"  {monthly_csv}")
        print(f"  {session_csv}")
        print(f"  {common_csv}")
        print(f"  {resolver_csv}")
        print(f"  Excel: {excel_msg}")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
