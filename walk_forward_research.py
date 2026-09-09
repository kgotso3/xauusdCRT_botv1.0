from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.walk_forward import evaluate_candidate_rules


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward validation for CRT V2 candidates")
    parser.add_argument("--dataset", required=True, help="Path to CRT occurrence CSV")
    parser.add_argument("--min-period-samples", type=int, default=5)
    parser.add_argument("--output", default="data/research/walk_forward")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    summary, periods = evaluate_candidate_rules(df, min_period_samples=args.min_period_samples)

    stem = dataset_path.stem
    summary_path = output_dir / f"{stem}_candidate_summary.csv"
    periods_path = output_dir / f"{stem}_candidate_periods.csv"
    summary.to_csv(summary_path, index=False)
    periods.to_csv(periods_path, index=False)

    print("CRT V2 WALK-FORWARD CANDIDATES")
    print(f"Dataset: {dataset_path}")
    print("Target focus: 1R and 1.5R")
    print(f"Candidate summary CSV: {summary_path}")
    print(f"Candidate periods CSV: {periods_path}")

    if summary.empty:
        print("No candidate rows generated.")
        return

    display_cols = [
        "rule", "samples", "hit_1r", "hit_1_5r", "expectancy_1r", "expectancy_1_5r",
        "usable_months", "positive_1_5r_months", "positive_1_5r_month_pct", "worst_month_expectancy_1_5r",
        "avg_mfe_r", "avg_mae_r",
    ]
    print("\nCANDIDATE SUMMARY")
    print(summary[display_cols].to_string(index=False))

    print("\nInterpretation rule: prefer candidates with positive expectancy at 1R/1.5R, enough samples, and consistency across months. Small-sample candidates are research leads, not deployable rules.")


if __name__ == "__main__":
    main()
