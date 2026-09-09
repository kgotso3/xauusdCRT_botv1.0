from __future__ import annotations

import argparse
import pandas as pd

from ml.v3_2_comparison import run_v3_2_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="CRT V3.2 targeted 1R feature refinement with fixed Random Forest comparison")
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    summary, folds = run_v3_2_comparison(dataset)

    print("CRT V3.2 — TARGETED 1R FEATURE REFINEMENT")
    print("Fixed Random Forest | development-only | forward cohort excluded")
    print(f"Fold rows: {len(folds)}")
    print()
    print("FIXED-MODEL STABILITY COMPARISON")
    if summary.empty:
        print("No valid folds produced.")
    else:
        print(summary.to_string(index=False))
    print()
    print("Reports:")
    print(r"data\research\ml_v3_2\v3_2_fixed_rf_folds.csv")
    print(r"data\research\ml_v3_2\v3_2_fixed_rf_summary.csv")
    print()
    print("PROMOTE_TO_FINAL_FREEZE authorizes only final frozen-model research. Shadow, demo, and live execution remain disabled.")


if __name__ == "__main__":
    main()
