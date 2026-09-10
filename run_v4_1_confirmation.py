from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from backtesting.session_crt_split import SplitTargetConfig
from backtesting.v4_1_confirmation import run_v4_1_confirmation_study


def main():
    p = argparse.ArgumentParser(description="V4.1 unrestricted H1 CRT confirmation study")
    p.add_argument("--h1", required=True)
    p.add_argument("--m15", required=True)
    p.add_argument("--max-hold", type=int, default=24)
    p.add_argument("--output-dir", default="data/research/v4_1_unrestricted")
    a = p.parse_args()
    h1p, m15p = Path(a.h1), Path(a.m15)
    if not h1p.exists(): raise FileNotFoundError(f"H1 CSV not found: {h1p}")
    if not m15p.exists(): raise FileNotFoundError(f"M15 CSV not found: {m15p}")
    summary, breakdown, trades = run_v4_1_confirmation_study(
        pd.read_csv(h1p), pd.read_csv(m15p), a.output_dir,
        SplitTargetConfig(max_holding_bars=a.max_hold),
    )
    print("CRT V4.1 — UNRESTRICTED H1 CONFIRMATION STUDY")
    print("No killzone barrier | every H1 range/sweep/reclaim eligible | 50/50 TP1=1R TP2=2R")
    print(f"Raw setup rows: {len(trades)}\n")
    print(summary.to_string(index=False))
    print("\nBreakdowns saved by New York hour, BUY/SELL direction, premium/discount correctness, and year.")
    print(f"Reports: {Path(a.output_dir) / 'v4_1_unrestricted_summary.csv'}")
    print(f"         {Path(a.output_dir) / 'v4_1_unrestricted_breakdown.csv'}")
    print(f"         {Path(a.output_dir) / 'v4_1_unrestricted_trades.csv'}")
    print("Research-only. No MT5 execution enabled.")

if __name__ == "__main__":
    main()
