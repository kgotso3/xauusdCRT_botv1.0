from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.v1 import train_ml_v1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRT V2 ML V1 models for 1R and 1.5R")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models-output", default="data/models")
    parser.add_argument("--report-output", default="data/research/ml/ml_v1_report.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    report = train_ml_v1(df, output_dir=args.models_output)
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    print("CRT V2 — ML V1")
    print("Separate targets: 1R and 1.5R")
    print("Candidate models: Logistic Regression vs Random Forest")
    print("Model selection: VALIDATION only | TEST: report-only audit")
    if report.empty:
        print("No models trained.")
    else:
        cols = [
            "target", "selected_model", "train_samples", "validation_samples", "test_samples",
            "validation_positive_rate", "validation_roc_auc", "validation_brier",
            "test_positive_rate", "test_roc_auc", "test_brier", "test_precision", "test_recall",
            "model_path",
        ]
        print("\nMODEL REPORT")
        print(report[cols].to_string(index=False))
    print(f"\nReport CSV: {report_path}")
    print("Do not promote a model to XM demo solely because TEST looks attractive. Compare it with the deterministic benchmark and then forward-test.")


if __name__ == "__main__":
    main()
