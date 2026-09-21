from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from mt5_crt_frequency_test import SYMBOL_ALIASES, parse_date
from mt5_crt_m5_performance_test import performance_stats


PRIMARY_COHORT = ["US30", "US500", "XAUUSD"]


def period_specs(start_year: int, end_day: date) -> list[tuple[str, date, date]]:
    specs: list[tuple[str, date, date]] = []
    for year in range(start_year, end_day.year):
        specs.append((str(year), date(year, 1, 1), date(year, 12, 31)))

    specs.append((f"{end_day.year}_YTD", date(end_day.year, 1, 1), end_day))
    specs.append(("ALL", date(start_year, 1, 1), end_day))
    return specs


def run_period(
    label: str,
    start_day: date,
    end_day: date,
    args: argparse.Namespace,
    script_path: Path,
    period_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    period_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script_path),
        "--start", start_day.isoformat(),
        "--end", end_day.isoformat(),
        "--min-bars", str(args.min_bars),
        "--c3-min-bars", str(args.c3_min_bars),
        "--sl-buffer-pips", str(args.sl_buffer_pips),
        "--sessions", args.sessions,
        "--output", str(period_dir),
        "--max-candidates", str(args.max_candidates),
    ]
    if args.terminal:
        cmd.extend(["--terminal", args.terminal])

    print(f"\nRUNNING {label}: {start_day} -> {end_day}")
    print("-" * 88)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    combined_output = (proc.stdout or "") + ("\nSTDERR:\n" + proc.stderr if proc.stderr else "")
    log_path = period_dir / "console_output.txt"
    log_path.write_text(combined_output, encoding="utf-8")

    if proc.stdout:
        for line in proc.stdout.splitlines():
            if (
                line.startswith(tuple(SYMBOL_ALIASES.keys()))
                or "FROZEN MODEL PERFORMANCE" in line
                or "10-SYMBOL PORTFOLIO" in line
                or line.startswith("All 10 symbols")
                or line.startswith("WARNING")
                or line.startswith("ERROR")
            ):
                print(line)

    no_m5_count = combined_output.count("no M5 history")
    history_unavailable = no_m5_count >= max(1, len(SYMBOL_ALIASES) // 2)

    if proc.returncode != 0:
        if history_unavailable:
            print(
                f"WARNING: {label} skipped because MT5 does not currently expose enough M5 history "
                f"({no_m5_count}/{len(SYMBOL_ALIASES)} symbols reported no M5 history)."
            )
            print(f"         See {log_path}")
            return None

        raise RuntimeError(
            f"Frozen performance test failed for {label} with exit code {proc.returncode}. "
            f"See {log_path}"
        )

    ranking_path = period_dir / "crt_m5_performance_ranking.csv"
    trades_path = period_dir / "crt_m5_performance_trades.csv"
    session_path = period_dir / "crt_m5_session_breakdown.csv"
    portfolio_path = period_dir / "crt_m5_portfolio_summary.csv"

    missing = [p for p in [ranking_path, trades_path, session_path, portfolio_path] if not p.exists()]
    if missing:
        if history_unavailable:
            print(f"WARNING: {label} skipped because history outputs are incomplete.")
            return None
        raise RuntimeError(f"Missing expected output(s) for {label}: {', '.join(str(p) for p in missing)}")

    ranking = pd.read_csv(ranking_path)
    trades = pd.read_csv(trades_path)
    sessions = pd.read_csv(session_path)
    portfolio = pd.read_csv(portfolio_path)

    if ranking.empty or int(ranking.get("Trades", pd.Series(dtype=int)).sum()) == 0:
        print(f"WARNING: {label} produced zero clean trades and will be excluded from robustness aggregation.")
        return None

    for df in (ranking, trades, sessions, portfolio):
        df.insert(0, "Period", label)
        df.insert(1, "Period Start", start_day.isoformat())
        df.insert(2, "Period End", end_day.isoformat())

    return ranking, trades, sessions, portfolio


def cohort_stats(trades: pd.DataFrame, period: str, symbols: list[str]) -> dict:
    part = trades[
        (trades["Period"] == period)
        & (trades["logical_symbol"].isin(symbols))
        & (trades["retest_entry"].astype(bool))
    ].copy()

    ambiguous = int((part["final_outcome"] == "AMBIGUOUS").sum()) if not part.empty else 0
    stats = performance_stats(part)
    return {
        "Period": period,
        "Cohort": "+".join(symbols),
        "Entries": len(part),
        "Ambiguous": ambiguous,
        **stats,
    }


def build_consistency(ranking_all: pd.DataFrame) -> pd.DataFrame:
    yearly = ranking_all[ranking_all["Period"] != "ALL"].copy()
    full = ranking_all[ranking_all["Period"] == "ALL"].copy()
    rows: list[dict] = []

    for symbol in SYMBOL_ALIASES:
        yp = yearly[yearly["Symbol"] == symbol].copy()
        fp = full[full["Symbol"] == symbol].copy()

        tested = int((yp["Trades"] > 0).sum()) if not yp.empty else 0
        positive = int(((yp["Trades"] > 0) & (yp["Expectancy R"] > 0)).sum()) if not yp.empty else 0
        pf_gt_1 = int(((yp["Trades"] > 0) & (yp["Profit Factor"] > 1)).sum()) if not yp.empty else 0
        exp_values = yp.loc[yp["Trades"] > 0, "Expectancy R"] if not yp.empty else pd.Series(dtype=float)

        if not fp.empty:
            fr = fp.iloc[0]
            full_exp = float(fr["Expectancy R"])
            full_total = float(fr["Total R"])
            full_pf = float(fr["Profit Factor"])
            full_dd = float(fr["Max DD R"])
            full_trades = int(fr["Trades"])
        else:
            full_exp = full_total = full_pf = full_dd = 0.0
            full_trades = 0

        rows.append({
            "Symbol": symbol,
            "Periods Tested": tested,
            "Positive Exp Periods": positive,
            "PF>1 Periods": pf_gt_1,
            "Positive Exp %": round(positive / tested * 100.0, 2) if tested else 0.0,
            "Avg Period Exp R": round(float(exp_values.mean()), 3) if len(exp_values) else 0.0,
            "Median Period Exp R": round(float(exp_values.median()), 3) if len(exp_values) else 0.0,
            "Worst Period Exp R": round(float(exp_values.min()), 3) if len(exp_values) else 0.0,
            "Best Period Exp R": round(float(exp_values.max()), 3) if len(exp_values) else 0.0,
            "Full Trades": full_trades,
            "Full Expectancy R": round(full_exp, 3),
            "Full Total R": round(full_total, 2),
            "Full Profit Factor": round(full_pf, 3),
            "Full Max DD R": round(full_dd, 2),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["Positive Exp Periods", "Full Expectancy R", "Full Profit Factor", "Full Trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    out.insert(0, "Consistency Rank", range(1, len(out) + 1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen CRT M5 model across independent calendar periods and aggregate robustness statistics."
    )
    parser.add_argument("--start-year", type=int, default=2024, help="First full calendar year to test (default 2024)")
    parser.add_argument("--end", type=parse_date, required=True, help="End date for the final YTD and ALL tests")
    parser.add_argument("--min-bars", type=int, default=12)
    parser.add_argument("--c3-min-bars", type=int, default=36)
    parser.add_argument("--sl-buffer-pips", type=float, default=10.0)
    parser.add_argument("--sessions", choices=["both", "am", "pm"], default="both")
    parser.add_argument("--output", default="results/crt_m5_robustness")
    parser.add_argument("--terminal", default=None)
    parser.add_argument("--max-candidates", type=int, default=12)
    args = parser.parse_args()

    if args.start_year < 2000 or args.start_year > args.end.year:
        parser.error("--start-year must be <= the year in --end")
    if not 1 <= args.min_bars <= 16:
        parser.error("--min-bars must be between 1 and 16")
    if not 1 <= args.c3_min_bars <= 48:
        parser.error("--c3-min-bars must be between 1 and 48")
    if args.sl_buffer_pips <= 0:
        parser.error("--sl-buffer-pips must be positive")

    script_path = Path(__file__).with_name("mt5_crt_m5_performance_test.py")
    if not script_path.exists():
        print(f"ERROR: Could not find {script_path}")
        return 2

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("CRT Scanner V1 - Multi-period robustness test")
    print(f"Calendar start: {args.start_year} | Final date: {args.end}")
    print(f"Frozen rules: min-bars={args.min_bars}, C3-min-bars={args.c3_min_bars}, SL buffer={args.sl_buffer_pips:g} pips")
    print("No strategy parameters are changed between periods.")

    ranking_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    session_frames: list[pd.DataFrame] = []
    portfolio_frames: list[pd.DataFrame] = []
    completed_labels: list[str] = []
    skipped_rows: list[dict] = []

    specs = period_specs(args.start_year, args.end)

    for label, start_day, end_day in specs:
        period_dir = out_dir / label
        result = run_period(label, start_day, end_day, args, script_path, period_dir)
        if result is None:
            skipped_rows.append({
                "Period": label,
                "Period Start": start_day.isoformat(),
                "Period End": end_day.isoformat(),
                "Status": "SKIPPED_HISTORY_UNAVAILABLE",
            })
            continue

        ranking, trades, sessions, portfolio = result
        ranking_frames.append(ranking)
        trade_frames.append(trades)
        session_frames.append(sessions)
        portfolio_frames.append(portfolio)
        completed_labels.append(label)

    if not ranking_frames:
        print("\nERROR: No period had enough M5 history to produce a valid robustness result.")
        print("Run mt5_m5_history_preflight.py after increasing MT5 Max bars in chart.")
        return 1

    ranking_all = pd.concat(ranking_frames, ignore_index=True)
    trades_all = pd.concat(trade_frames, ignore_index=True)
    sessions_all = pd.concat(session_frames, ignore_index=True)
    portfolios_all = pd.concat(portfolio_frames, ignore_index=True)

    consistency = build_consistency(ranking_all)
    primary_periods = pd.DataFrame([
        cohort_stats(trades_all, label, PRIMARY_COHORT)
        for label in completed_labels
    ])

    primary_symbol_periods = ranking_all[ranking_all["Symbol"].isin(PRIMARY_COHORT)].copy()

    print("\nROBUSTNESS BY PERIOD - PRIMARY SYMBOLS")
    print("=" * 126)
    show_cols = [
        "Period", "Symbol", "Trades", "Expectancy R", "Total R",
        "Profit Factor", "Max DD R", "TP2 %", "Timeout %",
    ]
    print(primary_symbol_periods[show_cols].to_string(index=False))

    print("\nPRIMARY COHORT: US30 + US500 + XAUUSD")
    print("=" * 118)
    cohort_cols = [
        "Period", "Entries", "Trades", "Ambiguous", "Expectancy R",
        "Total R", "Profit Factor", "Max DD R", "TP2 %", "Timeout %",
    ]
    print(primary_periods[cohort_cols].to_string(index=False))

    print("\nCROSS-PERIOD CONSISTENCY")
    print("=" * 148)
    consistency_cols = [
        "Consistency Rank", "Symbol", "Periods Tested", "Positive Exp Periods",
        "Positive Exp %", "Avg Period Exp R", "Worst Period Exp R",
        "Full Trades", "Full Expectancy R", "Full Total R",
        "Full Profit Factor", "Full Max DD R",
    ]
    print(consistency[consistency_cols].to_string(index=False))

    if skipped_rows:
        print("\nSKIPPED PERIODS - MT5 HISTORY UNAVAILABLE")
        print("=" * 88)
        print(pd.DataFrame(skipped_rows).to_string(index=False))

    ranking_csv = out_dir / "robustness_symbol_periods.csv"
    sessions_csv = out_dir / "robustness_session_periods.csv"
    portfolio_csv = out_dir / "robustness_portfolio_periods.csv"
    consistency_csv = out_dir / "robustness_consistency.csv"
    primary_csv = out_dir / "robustness_primary_cohort.csv"
    trades_csv = out_dir / "robustness_all_trades.csv"
    skipped_csv = out_dir / "robustness_skipped_periods.csv"

    ranking_all.to_csv(ranking_csv, index=False)
    sessions_all.to_csv(sessions_csv, index=False)
    portfolios_all.to_csv(portfolio_csv, index=False)
    consistency.to_csv(consistency_csv, index=False)
    primary_periods.to_csv(primary_csv, index=False)
    trades_all.to_csv(trades_csv, index=False)
    pd.DataFrame(skipped_rows).to_csv(skipped_csv, index=False)

    xlsx_path = out_dir / "crt_m5_robustness_results.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            consistency.to_excel(writer, sheet_name="Consistency", index=False)
            primary_periods.to_excel(writer, sheet_name="Primary Cohort", index=False)
            primary_symbol_periods.to_excel(writer, sheet_name="Primary Symbols", index=False)
            ranking_all.to_excel(writer, sheet_name="All Symbol Periods", index=False)
            sessions_all.to_excel(writer, sheet_name="AM PM Periods", index=False)
            portfolios_all.to_excel(writer, sheet_name="10 Symbol Portfolio", index=False)
            trades_all.to_excel(writer, sheet_name="All Trades", index=False)
            pd.DataFrame(skipped_rows).to_excel(writer, sheet_name="Skipped Periods", index=False)
        excel_msg = str(xlsx_path)
    except (ImportError, ModuleNotFoundError):
        excel_msg = "not written (install openpyxl)"

    print("\nSaved:")
    print(f"  {ranking_csv}")
    print(f"  {sessions_csv}")
    print(f"  {portfolio_csv}")
    print(f"  {consistency_csv}")
    print(f"  {primary_csv}")
    print(f"  {trades_csv}")
    print(f"  {skipped_csv}")
    print(f"  Excel: {excel_msg}")
    print("\nNOTE: Skipped periods are not treated as zero-performance periods; they are excluded because the terminal did not expose enough M5 history.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
