from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.session_crt_split import SplitTargetConfig, run_split_target_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest V4 H1 session CRT with 50/50 1R + 2R exits")
    parser.add_argument("--h1", required=True, help="Path to H1 historical CSV")
    parser.add_argument("--max-hold", type=int, default=24, help="Maximum future H1 bars to hold")
    parser.add_argument("--output-dir", default="data/research/v4_split", help="Output directory")
    args = parser.parse_args()

    h1_path = Path(args.h1)
    if not h1_path.exists():
        raise FileNotFoundError(f"H1 CSV not found: {h1_path}")

    h1 = pd.read_csv(h1_path)
    cfg = SplitTargetConfig(max_holding_bars=args.max_hold)
    summary, trades, _ = run_split_target_backtest(h1, output_dir=args.output_dir, split_config=cfg)

    print("CRT V4 — 50/50 SPLIT TARGET BACKTEST")
    print("H1 session CRT | TP1=1R | TP2=2R | runner keeps original stop")
    print(f"Trade rows: {len(trades)}")
    print()
    print(summary.to_string(index=False))
    print()
    print("Interpretation:")
    if summary.empty or int(summary.iloc[0]["trades"]) == 0:
        print("No qualifying CRT trades were found in the supplied H1 history.")
    else:
        total_r = float(summary.iloc[0]["total_r"])
        expectancy = float(summary.iloc[0]["expectancy_r"])
        growth = "POSITIVE" if total_r > 0 else "NEGATIVE" if total_r < 0 else "FLAT"
        print(f"Overall growth: {growth} | Total: {total_r:.2f}R | Expectancy: {expectancy:.4f}R/trade")
    print()
    print(f"Reports: {Path(args.output_dir) / 'v4_split_summary.csv'}")
    print(f"         {Path(args.output_dir) / 'v4_split_trades.csv'}")
    print(f"         {Path(args.output_dir) / 'v4_split_equity.csv'}")
    print("Research-only backtest. No MT5 order execution is enabled.")


if __name__ == "__main__":
    main()
