from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features.regime import build_causal_regime_features
from ml.v3_stability import PromotionGate, write_v3_1_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="CRT V3.1 stability diagnostics and promotion-gate reporting")
    parser.add_argument("--folds", default="data/research/ml_v3/v3_walkforward_folds.csv")
    parser.add_argument("--dataset", required=True, help="CRT occurrence dataset; post-2026-09-01 rows are excluded internally from regime diagnostics")
    parser.add_argument("--output-dir", default="data/research/ml_v3_1")
    parser.add_argument("--min-mean-auc", type=float, default=0.55)
    parser.add_argument("--min-median-auc", type=float, default=0.55)
    parser.add_argument("--min-positive-folds", type=int, default=4)
    parser.add_argument("--catastrophic-auc", type=float, default=0.45)
    parser.add_argument("--max-catastrophic-folds", type=int, default=0)
    parser.add_argument("--max-auc-std", type=float, default=0.08)
    args = parser.parse_args()

    folds = pd.read_csv(args.folds)
    raw = pd.read_csv(args.dataset)
    regime_dataset = build_causal_regime_features(raw)

    gate = PromotionGate(
        min_mean_auc=args.min_mean_auc,
        min_median_auc=args.min_median_auc,
        min_positive_folds=args.min_positive_folds,
        max_catastrophic_folds=args.max_catastrophic_folds,
        catastrophic_auc=args.catastrophic_auc,
        max_auc_std=args.max_auc_std,
    )
    stability, candidates, selected, regimes = write_v3_1_reports(
        folds,
        regime_dataset=regime_dataset,
        output_dir=args.output_dir,
        gate=gate,
    )

    print("CRT V3.1 — STABILITY DIAGNOSTICS & PROMOTION GATE")
    print("Development-only diagnostics. Frozen forward cohort remains excluded from tuning.")
    print("\nPROMOTION GATE REPORT")
    if stability.empty:
        print("No stability rows available.")
    else:
        cols = [
            "target", "folds", "mean_selected_auc", "median_selected_auc",
            "selected_auc_std", "auc_ci95_low", "auc_ci95_high",
            "mean_selected_brier", "positive_auc_folds", "catastrophic_folds",
            "best_model_mode", "promotion_status",
        ]
        print(stability[cols].to_string(index=False))

    print("\nTOP CANDIDATE CONSISTENCY")
    if candidates.empty:
        print("No candidate rows available.")
    else:
        print(candidates.groupby("target", group_keys=False).head(4).to_string(index=False))

    print("\nReports:")
    for name in (
        "v3_1_selected_folds.csv",
        "v3_1_candidate_consistency.csv",
        "v3_1_promotion_gate.csv",
        "v3_1_regime_breakdown.csv",
    ):
        print(Path(args.output_dir) / name)

    print("\nPromotion status authorizes only final model freeze research. It does NOT authorize shadow, demo, or live MT5 order execution.")


if __name__ == "__main__":
    main()
