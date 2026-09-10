from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from backtesting.v4_3_edge_discovery import EdgeDiscoveryConfig, run_v4_3_edge_discovery


def main():
    p = argparse.ArgumentParser(description="V4.3 causal edge discovery with frozen holdout")
    p.add_argument("--trades", default="data/research/v4_2_causal/v4_2_causal_trades.csv")
    p.add_argument("--variant", default="FULL_M5_CAUSAL", choices=["RAW_CAUSAL","MSS_CAUSAL","MSS_FVG_CAUSAL","FULL_MODEL_CAUSAL","FULL_M5_CAUSAL"])
    p.add_argument("--holdout-start", default="2026-07-01")
    p.add_argument("--min-dev-trades", type=int, default=50)
    p.add_argument("--min-pf", type=float, default=1.10)
    p.add_argument("--min-positive-years", type=int, default=2)
    p.add_argument("--output-dir", default="data/research/v4_3_edge")
    a = p.parse_args()
    path = Path(a.trades)
    if not path.exists():
        raise FileNotFoundError(f"V4.2 trades CSV not found: {path}")
    cfg = EdgeDiscoveryConfig(
        holdout_start=a.holdout_start,
        min_dev_trades=a.min_dev_trades,
        min_dev_profit_factor=a.min_pf,
        min_positive_years=a.min_positive_years,
    )
    candidates, holdout, overall = run_v4_3_edge_discovery(pd.read_csv(path), a.output_dir, a.variant, cfg)
    print("CRT V4.3 — CAUSAL EDGE DISCOVERY")
    print(f"Variant: {a.variant} | holdout starts: {a.holdout_start}")
    print("Development data selects rules; holdout data is validation only.\n")
    print("PERIOD SUMMARY")
    print(overall.to_string(index=False))
    selected = candidates.loc[candidates["selected"]]
    print(f"\nCandidate rules tested: {len(candidates)} | selected on development: {len(selected)}")
    if not selected.empty:
        print("\nTOP DEVELOPMENT RULES")
        cols = ["rule_name","trades","total_r","expectancy_r","profit_factor","positive_dev_years"]
        print(selected[cols].head(15).to_string(index=False))
    else:
        print("No development rules passed the promotion thresholds.")
    print("\nHOLDOUT VALIDATION")
    if holdout.empty:
        print("No selected rule reached holdout; nothing promoted.")
    else:
        print(holdout.head(15).to_string(index=False))
    out = Path(a.output_dir)
    print(f"\nReports: {out / 'v4_3_candidates.csv'}")
    print(f"         {out / 'v4_3_holdout_validation.csv'}")
    print(f"         {out / 'v4_3_period_summary.csv'}")
    print("Research-only. No strategy promotion or MT5 execution is enabled.")

if __name__ == "__main__":
    main()
