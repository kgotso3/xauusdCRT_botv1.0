from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backtesting.outcomes import realized_r
from features.v2_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGETS, build_causal_v2_features
from ml.thresholds import ThresholdConfig, choose_threshold, scan_validation_thresholds, threshold_metrics
from ml.v1 import MLConfig, chronological_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only threshold research for CRT V2 ML V1")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models-dir", default="data/models")
    parser.add_argument("--output", default="data/research/ml_thresholds")
    parser.add_argument("--min-validation-trades", type=int, default=25)
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    df = chronological_split(build_causal_v2_features(dataset), MLConfig())
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for target_name, target_col in TARGETS.items():
        target_r = 1.0 if target_name == "1R" else 1.5
        target_df = df.dropna(subset=[target_col]).copy()
        validation = target_df[target_df["ml_split"] == "VALIDATION"].copy()
        test = target_df[target_df["ml_split"] == "TEST"].copy()

        model_candidates = list(Path(args.models_dir).glob(f"crt_v2_{target_name.replace('.', '_')}_*.joblib"))
        if not model_candidates:
            raise FileNotFoundError(f"No saved model found for {target_name} in {args.models_dir}")
        model_path = sorted(model_candidates)[-1]
        model = joblib.load(model_path)

        val_prob = model.predict_proba(validation[feature_cols])[:, 1]
        test_prob = model.predict_proba(test[feature_cols])[:, 1]

        cfg = ThresholdConfig(min_validation_trades=args.min_validation_trades)
        scan = scan_validation_thresholds(validation, val_prob, target_r=target_r, config=cfg)
        scan_path = output_dir / f"{target_name.replace('.', '_')}_validation_threshold_scan.csv"
        scan.to_csv(scan_path, index=False)

        chosen = choose_threshold(scan)
        if chosen is None:
            summary_rows.append({
                "target": target_name,
                "model_path": str(model_path),
                "selected_threshold": np.nan,
                "status": "NO_ELIGIBLE_THRESHOLD",
            })
            continue

        val_metrics = threshold_metrics(validation, val_prob, target_r=target_r, threshold=chosen)
        test_metrics = threshold_metrics(test, test_prob, target_r=target_r, threshold=chosen)
        row = {
            "target": target_name,
            "model_path": str(model_path),
            "selected_threshold": chosen,
            "status": "RESEARCH_ONLY",
        }
        row.update({f"validation_{k}": v for k, v in val_metrics.items() if k != "threshold"})
        row.update({f"test_{k}": v for k, v in test_metrics.items() if k != "threshold"})
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "ml_threshold_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("CRT V2 — ML THRESHOLD RESEARCH")
    print("Threshold selection: VALIDATION only | TEST: audit only")
    print("Timeout accounting: neither target nor stop within 24H = 0R research timeout")
    print("Threshold grid: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75")
    print("\nSUMMARY")
    if summary.empty:
        print("No threshold results generated.")
    else:
        cols = [c for c in [
            "target", "selected_threshold", "status",
            "validation_trades", "validation_coverage", "validation_hit_rate",
            "validation_expectancy_r", "validation_net_r", "validation_max_drawdown_r",
            "test_trades", "test_coverage", "test_hit_rate",
            "test_expectancy_r", "test_net_r", "test_max_drawdown_r",
        ] if c in summary.columns]
        print(summary[cols].to_string(index=False))
    print(f"\nSummary CSV: {summary_path}")
    print("Do not promote thresholds from TEST. The current TEST period has already been inspected and is audit-only.")


if __name__ == "__main__":
    main()
