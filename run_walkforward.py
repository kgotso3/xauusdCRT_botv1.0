from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.walkforward import SplitConfig, build_walkforward_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronological V2 CRT validation for 1R and 1.5R")
    parser.add_argument("--dataset", required=True, help="Path to CRT occurrence CSV")
    parser.add_argument("--min-split-samples", type=int, default=8)
    parser.add_argument("--output", default="data/research/walkforward")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise RuntimeError("Research dataset is empty")

    report, periods = build_walkforward_report(
        df,
        SplitConfig(min_samples_per_split=args.min_split_samples),
    )

    stem = dataset_path.stem
    report_path = output_dir / f"{stem}_candidate_validation.csv"
    period_path = output_dir / f"{stem}_period_stability.csv"
    report.to_csv(report_path, index=False)
    periods.to_csv(period_path, index=False)

    print("V2 WALK-FORWARD VALIDATION")
    print("Targets: 1R and 1.5R only")
    print("Chronological split: 60% TRAIN / 20% VALIDATION / 20% TEST")
    print("Killzones (New York local time): ASIA 20:00-00:00 | LONDON 02:00-05:00 | NEW_YORK 08:00-11:00")

    stable = report[report["sample_stable"]].copy() if not report.empty else report
    print(f"Candidates evaluated: {len(report)} | sample-stable: {len(stable)}")
    if not stable.empty:
        cols = [
            "candidate", "samples_total",
            "train_hit_1r", "train_hit_1_5r",
            "validation_hit_1r", "validation_hit_1_5r",
            "test_hit_1r", "test_hit_1_5r",
            "validation_exp_1r", "validation_exp_1_5r",
            "test_exp_1r", "test_exp_1_5r", "test_positive_both",
        ]
        print("\nTOP VALIDATED CANDIDATES")
        print(stable[cols].head(15).to_string(index=False))

    print(f"\nCandidate report: {report_path}")
    print(f"Period stability: {period_path}")
    print("\nImportant: TEST is an untouched chronological holdout. Do not tune candidate rules using TEST results.")


if __name__ == "__main__":
    main()
