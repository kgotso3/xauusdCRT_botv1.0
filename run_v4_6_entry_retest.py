from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from backtesting.v4_6_entry_retest import EntryRetestConfig, run_v4_6_entry_retest


def main() -> None:
    p = argparse.ArgumentParser(description="V4.6 causal FVG/OTE retracement entry study")
    p.add_argument("--trades", required=True, help="V4.2 causal trades CSV")
    p.add_argument("--m5", required=True, help="Raw M5 OHLC CSV")
    p.add_argument("--retest-bars", type=int, default=24, help="Maximum M5 bars allowed for a retracement fill")
    p.add_argument("--output-dir", default="data/research/v4_6_entry_retest")
    a = p.parse_args()

    trades_path = Path(a.trades)
    m5_path = Path(a.m5)
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {trades_path}")
    if not m5_path.exists():
        raise FileNotFoundError(f"M5 CSV not found: {m5_path}")

    cfg = EntryRetestConfig(retest_window_bars=a.retest_bars)
    summary, rolling, results = run_v4_6_entry_retest(
        pd.read_csv(trades_path),
        pd.read_csv(m5_path),
        output_dir=a.output_dir,
        config=cfg,
    )

    print("CRT V4.6 — CAUSAL ENTRY MECHANICS REDESIGN")
    print(f"M5 retracement window: {a.retest_bars} bars | TP1=1R | TP2=2R | conservative pre-fill stop invalidation")
    print(f"Retest setup rows: {len(results)}\n")
    print("ENTRY MODEL SUMMARY")
    print(summary.to_string(index=False) if not summary.empty else "No valid retest setups.")
    print(f"\nReports: {Path(a.output_dir) / 'v4_6_entry_retest_summary.csv'}")
    print(f"         {Path(a.output_dir) / 'v4_6_entry_retest_rolling.csv'}")
    print(f"         {Path(a.output_dir) / 'v4_6_entry_retest_trades.csv'}")
    print("Research-only. No MT5 execution or strategy promotion is enabled.")


if __name__ == "__main__":
    main()
