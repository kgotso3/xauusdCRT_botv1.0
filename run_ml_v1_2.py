from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.v1_2 import MLV12Config, train_ml_v1_2


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRT ML V1.2 with expanded causal features")
    parser.add_argument("--dataset", required=True, help="Updated CRT occurrence dataset CSV")
    parser.add_argument("--models", default="data/models", help="Model output directory")
    parser.add_argument("--report", default="data/research/ml_v1_2", help="Report output directory")
    parser.add_argument("--select-k", type=int, default=28, help="TRAIN-only selected transformed features")
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    selected, candidates = train_ml_v1_2(
        dataset,
        output_dir=args.models,
        report_dir=args.report,
        config=MLV12Config(select_k=args.select_k),
    )

    print("CRT V2 — ML V1.2")
    print("Expanded causal features + target-specific TRAIN-only feature selection")
    print("Candidate models: Logistic Regression | Random Forest | Extra Trees | Gradient Boosting")
    print("Model selection: VALIDATION ROC-AUC, Brier tie-break | TEST audit-only")

    if candidates.empty:
        print("\nNo candidate models were trained.")
        return

    candidate_cols = [
        "target", "model", "train_samples", "validation_samples",
        "validation_positive_rate", "validation_roc_auc", "validation_brier",
        "validation_precision", "validation_recall",
    ]
    print("\nCANDIDATE REPORT")
    print(candidates[candidate_cols].sort_values(["target", "validation_roc_auc"], ascending=[True, False]).to_string(index=False))

    selected_cols = [
        "target", "selected_model", "train_samples", "validation_samples", "test_samples",
        "validation_positive_rate", "validation_roc_auc", "validation_brier",
        "test_positive_rate", "test_roc_auc", "test_brier", "test_precision", "test_recall",
        "model_path",
    ]
    print("\nSELECTED MODEL REPORT")
    print(selected[selected_cols].to_string(index=False))

    report_path = Path(args.report)
    print(f"\nCandidate CSV: {report_path / 'ml_v1_2_candidates.csv'}")
    print(f"Selected CSV:  {report_path / 'ml_v1_2_selected.csv'}")
    print("TEST remains audit-only. Do not tune V1.2 from TEST results; use a future forward holdout for subsequent rule/model changes.")


if __name__ == "__main__":
    main()
