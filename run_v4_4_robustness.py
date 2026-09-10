from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from backtesting.v4_4_robustness import RobustnessConfig, run_v4_4_robustness


def main():
    p = argparse.ArgumentParser(description="V4.4 rolling robustness and attribution study")
    p.add_argument("--trades", required=True, help="V4.2 causal trades CSV")
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--step-days", type=int, default=30)
    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--output-dir", default="data/research/v4_4_robustness")
    a = p.parse_args()
    path = Path(a.trades)
    if not path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {path}")
    cfg = RobustnessConfig(window_days=a.window_days, step_days=a.step_days, min_trades_per_window=a.min_trades)
    summary, windows, attribution, segments = run_v4_4_robustness(pd.read_csv(path), a.output_dir, cfg)
    print("CRT V4.4 — ROLLING ROBUSTNESS & ATTRIBUTION")
    print(f"Rolling window: {a.window_days} days | step: {a.step_days} days | minimum trades/window: {a.min_trades}\n")
    print("ROBUSTNESS SUMMARY")
    print(summary.to_string(index=False))
    print("\nSTAGE ATTRIBUTION")
    print(attribution.to_string(index=False) if not attribution.empty else "No attribution rows.")
    print("\nFULL_M5 SEGMENTS")
    print(segments.to_string(index=False) if not segments.empty else "No FULL_M5 segments.")
    print(f"\nReports: {Path(a.output_dir) / 'v4_4_robustness_summary.csv'}")
    print(f"         {Path(a.output_dir) / 'v4_4_rolling_windows.csv'}")
    print(f"         {Path(a.output_dir) / 'v4_4_stage_attribution.csv'}")
    print(f"         {Path(a.output_dir) / 'v4_4_full_m5_segments.csv'}")
    print("Research-only. No strategy promotion or MT5 execution is enabled.")


if __name__ == "__main__":
    main()
