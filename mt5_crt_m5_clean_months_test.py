from __future__ import annotations

import argparse
import calendar
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from mt5_crt_frequency_test import parse_date
from mt5_crt_m5_performance_test import performance_stats


PRIMARY_COHORT = ["US30", "US500", "XAUUSD"]


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def month_bounds(month_text: str) -> tuple[date, date]:
    year, month = map(int, month_text.split("-"))
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last_day)


def load_month_outputs(month_text: str, month_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    trades_path = month_dir / "crt_m5_performance_trades.csv"
    ranking_path = month_dir / "crt_m5_performance_ranking.csv"
    if not trades_path.exists() or not ranking_path.exists():
        return None

    trades = pd.read_csv(trades_path)
    ranking = pd.read_csv(ranking_path)

    if "Clean Month" not in trades.columns:
        trades.insert(0, "Clean Month", month_text)
    else:
        trades["Clean Month"] = month_text

    if "Clean Month" not in ranking.columns:
        ranking.insert(0, "Clean Month", month_text)
    else:
        ranking["Clean Month"] = month_text

    return trades, ranking


def run_month(
    month_text: str,
    args: argparse.Namespace,
    script_path: Path,
    month_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    start_day, end_day = month_bounds(month_text)
    month_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_existing:
        existing = load_month_outputs(month_text, month_dir)
        if existing is not None:
            print(f"\nREUSING CLEAN MONTH {month_text}: existing completed outputs")
            return existing

    cmd = [
        sys.executable,
        str(script_path),
        "--start", start_day.isoformat(),
        "--end", end_day.isoformat(),
        "--min-bars", str(args.min_bars),
        "--c3-min-bars", str(args.c3_min_bars),
        "--sl-buffer-pips", str(args.sl_buffer_pips),
        "--sessions", args.sessions,
        "--output", str(month_dir),
        "--max-candidates", str(args.max_candidates),
    ]
    if args.terminal:
        cmd.extend(["--terminal", args.terminal])

    print(f"\nRUNNING CLEAN MONTH {month_text}: {start_day} -> {end_day}")
    print("-" * 96)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_path = month_dir / "console_output.txt"
    log_path.write_text((proc.stdout or "") + ("\nSTDERR:\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")

    if proc.stdout:
        for line in proc.stdout.splitlines():
            if line.startswith(tuple(PRIMARY_COHORT)) or line.startswith("M5 C3 COVERAGE") or line.startswith("WARNING") or line.startswith("ERROR"):
                print(line)

    if proc.returncode != 0:
        print(f"WARNING: {month_text} failed with exit code {proc.returncode}. See {log_path}")
        return None

    result = load_month_outputs(month_text, month_dir)
    if result is None:
        print(f"WARNING: {month_text} missing expected output files. See {month_dir}")
        return None
    return result


def entry_mask(df: pd.DataFrame) -> pd.Series:
    if "retest_entry" not in df.columns:
        return pd.Series(False, index=df.index)
    return parse_bool_series(df["retest_entry"])


def stats_row(label: str, part: pd.DataFrame) -> dict:
    entries = part[entry_mask(part)].copy() if not part.empty else pd.DataFrame()
    ambiguous = int((entries.get("final_outcome", pd.Series(dtype=str)).astype(str) == "AMBIGUOUS").sum()) if not entries.empty else 0
    return {
        "Scope": label,
        "Entries": len(entries),
        "Ambiguous": ambiguous,
        **performance_stats(entries),
    }


def symbol_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol in PRIMARY_COHORT:
        part = trades[trades["logical_symbol"] == symbol].copy()
        rows.append(stats_row(symbol, part))
    return pd.DataFrame(rows)


def monthly_cohort_summary(trades: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    rows = []
    for month in months:
        part = trades[
            (trades["Clean Month"] == month)
            & (trades["logical_symbol"].isin(PRIMARY_COHORT))
        ].copy()
        stats = stats_row("US30+US500+XAUUSD", part)
        row = {"Month": month, **stats}
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay the frozen CRT M5 model only on coverage-safe primary-cohort months from the coverage audit."
    )
    parser.add_argument(
        "--coverage-csv",
        default="results/crt_m5_coverage_audit/coverage_primary_common_months.csv",
        help="coverage_primary_common_months.csv produced by mt5_crt_m5_coverage_audit.py",
    )
    parser.add_argument("--min-bars", type=int, default=12)
    parser.add_argument("--c3-min-bars", type=int, default=36)
    parser.add_argument("--sl-buffer-pips", type=float, default=10.0)
    parser.add_argument("--sessions", choices=["both", "am", "pm"], default="both")
    parser.add_argument("--output", default="results/crt_m5_clean_months")
    parser.add_argument("--terminal", default=None)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing per-month CSV outputs instead of rerunning MT5 when they already exist.",
    )
    args = parser.parse_args()

    if not 1 <= args.min_bars <= 16:
        parser.error("--min-bars must be between 1 and 16")
    if not 1 <= args.c3_min_bars <= 48:
        parser.error("--c3-min-bars must be between 1 and 48")
    if args.sl_buffer_pips <= 0:
        parser.error("--sl-buffer-pips must be positive")

    coverage_path = Path(args.coverage_csv)
    if not coverage_path.exists():
        print(f"ERROR: Coverage CSV not found: {coverage_path}")
        return 2

    coverage = pd.read_csv(coverage_path)
    required = {"month", "All >= threshold"}
    missing_cols = required - set(coverage.columns)
    if missing_cols:
        print(f"ERROR: Coverage CSV missing columns: {', '.join(sorted(missing_cols))}")
        return 2

    clean = coverage[parse_bool_series(coverage["All >= threshold"])].copy()
    clean = clean.sort_values("month")
    months = clean["month"].astype(str).tolist()
    if not months:
        print("ERROR: No clean common months found in coverage CSV.")
        return 1

    script_path = Path(__file__).with_name("mt5_crt_m5_performance_chunked_test.py")
    if not script_path.exists():
        print(f"ERROR: Could not find {script_path}")
        return 2

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("CRT Scanner V1 - Coverage-safe clean-month robustness replay")
    print("Primary cohort: US30 + US500 + XAUUSD")
    print("Frozen strategy rules are unchanged.")
    print("Included months are read directly from the coverage audit.")
    print("Clean months:", ", ".join(months))
    if args.reuse_existing:
        print("Reuse mode: existing completed month outputs will be loaded when available.")

    trade_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    completed_months: list[str] = []

    for month in months:
        result = run_month(month, args, script_path, out_dir / month)
        if result is None:
            continue
        trades, ranking = result
        trade_frames.append(trades)
        ranking_frames.append(ranking)
        completed_months.append(month)

    if not trade_frames:
        print("ERROR: No clean month completed successfully.")
        return 1

    trades_all = pd.concat(trade_frames, ignore_index=True)
    rankings_all = pd.concat(ranking_frames, ignore_index=True)
    primary_trades = trades_all[trades_all["logical_symbol"].isin(PRIMARY_COHORT)].copy()

    monthly_summary = monthly_cohort_summary(primary_trades, completed_months)
    symbols = symbol_summary(primary_trades)
    cohort = pd.DataFrame([stats_row("US30+US500+XAUUSD", primary_trades)])

    print("\nCLEAN-MONTH PRIMARY COHORT - MONTH BY MONTH")
    print("=" * 126)
    show_month_cols = [
        "Month", "Entries", "Trades", "Ambiguous", "Expectancy R",
        "Total R", "Profit Factor", "Max DD R", "TP2 %", "Timeout %",
    ]
    print(monthly_summary[show_month_cols].to_string(index=False))

    print("\nCLEAN-MONTH PRIMARY SYMBOLS - AGGREGATE")
    print("=" * 118)
    show_symbol_cols = [
        "Scope", "Entries", "Trades", "Ambiguous", "Expectancy R",
        "Total R", "Profit Factor", "Max DD R", "TP2 %", "Timeout %",
    ]
    print(symbols[show_symbol_cols].to_string(index=False))

    print("\nCLEAN-MONTH PRIMARY COHORT - AGGREGATE")
    print("=" * 118)
    print(cohort[show_symbol_cols].to_string(index=False))

    positive_months = int((monthly_summary["Expectancy R"] > 0).sum())
    tested_months = len(monthly_summary)
    print("\nMONTHLY CONSISTENCY")
    print("=" * 88)
    print(f"Completed clean months: {tested_months}/{len(months)}")
    if tested_months:
        print(f"Positive-expectancy months: {positive_months}/{tested_months} ({positive_months / tested_months * 100.0:.1f}%)")
        print(f"Worst monthly expectancy: {monthly_summary['Expectancy R'].min():.3f}R")
        print(f"Best monthly expectancy:  {monthly_summary['Expectancy R'].max():.3f}R")
    else:
        print("Positive-expectancy months: 0/0")

    monthly_csv = out_dir / "clean_month_primary_cohort_monthly.csv"
    symbol_csv = out_dir / "clean_month_primary_symbols.csv"
    cohort_csv = out_dir / "clean_month_primary_cohort.csv"
    trades_csv = out_dir / "clean_month_primary_trades.csv"
    ranking_csv = out_dir / "clean_month_all_rankings.csv"
    selected_csv = out_dir / "clean_months_selected.csv"

    monthly_summary.to_csv(monthly_csv, index=False)
    symbols.to_csv(symbol_csv, index=False)
    cohort.to_csv(cohort_csv, index=False)
    primary_trades.to_csv(trades_csv, index=False)
    rankings_all.to_csv(ranking_csv, index=False)
    clean.to_csv(selected_csv, index=False)

    xlsx_path = out_dir / "crt_m5_clean_months_results.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            clean.to_excel(writer, sheet_name="Selected Months", index=False)
            monthly_summary.to_excel(writer, sheet_name="Monthly Cohort", index=False)
            symbols.to_excel(writer, sheet_name="Primary Symbols", index=False)
            cohort.to_excel(writer, sheet_name="Primary Cohort", index=False)
            primary_trades.to_excel(writer, sheet_name="Primary Trades", index=False)
            rankings_all.to_excel(writer, sheet_name="All Monthly Rankings", index=False)
        excel_msg = str(xlsx_path)
    except (ImportError, ModuleNotFoundError):
        excel_msg = "not written (install openpyxl)"

    print("\nSaved:")
    print(f"  {monthly_csv}")
    print(f"  {symbol_csv}")
    print(f"  {cohort_csv}")
    print(f"  {trades_csv}")
    print(f"  {ranking_csv}")
    print(f"  {selected_csv}")
    print(f"  Excel: {excel_msg}")
    print("\nNOTE: This test changes only sample inclusion by using coverage-safe months. Strategy rules remain frozen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
