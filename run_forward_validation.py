from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.forward_validation import DEFAULT_CUTOFF, forward_metrics, score_forward_occurrences, upsert_prediction_journal


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen forward validation for CRT ML V1 and V1.2")
    parser.add_argument("--dataset", required=True, help="CRT occurrence dataset including post-cutoff rows")
    parser.add_argument("--cutoff", default=str(DEFAULT_CUTOFF), help="Forward-only UTC cutoff; default 2026-09-01")
    parser.add_argument("--journal", default="data/research/forward/forward_predictions.csv")
    parser.add_argument("--metrics", default="data/research/forward/forward_metrics.csv")
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    cutoff = pd.Timestamp(args.cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")

    scored = score_forward_occurrences(dataset, cutoff=cutoff)
    journal = upsert_prediction_journal(scored, args.journal)
    metrics = forward_metrics(journal)

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)

    print("CRT FROZEN FORWARD VALIDATION")
    print(f"Cutoff: {cutoff.isoformat()}")
    print("Models: V1 + V1.2 | no retraining | no threshold tuning")
    print("Fresh outcomes stay PENDING until target/stop resolves or the 24H horizon completes.")
    print(f"New scored rows this run: {len(scored)}")
    print(f"Journal rows total:       {len(journal)}")
    print(f"Journal: {args.journal}")
    print(f"Metrics: {args.metrics}")

    if metrics.empty:
        print("\nNo forward metrics available yet.")
        return

    cols = [
        "model_version", "target", "forward_predictions", "pending_predictions", "labelled_predictions",
        "positive_rate", "roc_auc", "brier", "threshold_frozen", "selected_trades",
        "selected_coverage", "selected_hit_rate", "selected_expectancy_r", "selected_net_r",
    ]
    print("\nFORWARD METRICS")
    print(metrics[cols].to_string(index=False))
    print("\nInterpretation rule: do not change model structure or thresholds from these forward outcomes until the planned forward sample gate is reached.")


if __name__ == "__main__":
    main()
