from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.v3_3_comparison import V33Config, run_v3_3_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="V3.3 completed-session liquidity fixed-RF comparison")
    parser.add_argument("--dataset", required=True, help="Regenerated CRT occurrence CSV containing completed-session liquidity fields")
    parser.add_argument("--output-dir", default="data/research/ml_v3_3")
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--validation-size", type=int, default=300)
    parser.add_argument("--step-size", type=int, default=300)
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    cfg = V33Config(train_size=args.train_size, validation_size=args.validation_size, step_size=args.step_size)
    summary, folds = run_v3_3_comparison(dataset, output_dir=args.output_dir, config=cfg)

    print("CRT V3.3 — COMPLETED-SESSION LIQUIDITY REFINEMENT")
    print("Fixed Random Forest | 1R only | development-only | forward cohort excluded")
    print(f"Fold rows: {len(folds)}")
    print("\nFIXED-MODEL SESSION LIQUIDITY COMPARISON")
    print(summary.to_string(index=False))
    print("\nReports:")
    print(Path(args.output_dir) / "v3_3_session_rf_folds.csv")
    print(Path(args.output_dir) / "v3_3_session_rf_summary.csv")
    print("\nPROMOTE_TO_FINAL_FREEZE authorizes only frozen-model research. Shadow, demo, and live execution remain disabled.")


if __name__ == "__main__":
    main()
