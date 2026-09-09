from __future__ import annotations

import argparse

import pandas as pd

from ml.v3_walkforward import V3WalkForwardConfig, run_v3_walkforward


def main() -> None:
    parser = argparse.ArgumentParser(description="CRT V3 regime-aware chronological walk-forward ML research")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--validation-size", type=int, default=300)
    parser.add_argument("--step-size", type=int, default=300)
    parser.add_argument("--select-k", type=int, default=32)
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    cfg = V3WalkForwardConfig(
        train_size=args.train_size,
        validation_size=args.validation_size,
        step_size=args.step_size,
        select_k=args.select_k,
    )
    summary, folds = run_v3_walkforward(dataset, config=cfg)

    print("CRT V3 — REGIME-AWARE WALK-FORWARD ML")
    print("Causal regime features | chronological folds | no use of frozen forward cohort for tuning")
    print(f"Fold candidate rows: {len(folds)}")
    if summary.empty:
        print("No valid folds. Increase dataset size or reduce train/validation window sizes.")
        return
    print("\nWALK-FORWARD SUMMARY")
    print(summary.to_string(index=False))
    print("\nReports:")
    print("data\\research\\ml_v3\\v3_walkforward_folds.csv")
    print("data\\research\\ml_v3\\v3_walkforward_summary.csv")
    print("\nFold models are research artifacts only. Do not connect them to MT5 execution.")


if __name__ == "__main__":
    main()
