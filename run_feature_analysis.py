from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.feature_analysis import analyze_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze causal V2 CRT features for 1R and 1.5R")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="data/research/features")
    parser.add_argument("--min-bin-samples", type=int, default=30)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    importance, bins = analyze_features(df, min_bin_samples=args.min_bin_samples)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    importance_path = out_dir / "feature_rankings.csv"
    bins_path = out_dir / "feature_bins.csv"
    importance.to_csv(importance_path, index=False)
    bins.to_csv(bins_path, index=False)

    print("V2 CAUSAL FEATURE ANALYSIS")
    for target in ["1R", "1.5R"]:
        print(f"\nTOP FEATURES — {target}")
        part = importance[importance["target"] == target].head(12)
        print(part.to_string(index=False) if not part.empty else "No rows")
        print(f"\nTOP POSITIVE BINS — {target}")
        b = bins[bins["target"] == target].head(12)
        print(b.to_string(index=False) if not b.empty else "No rows")

    print(f"\nFeature rankings: {importance_path}")
    print(f"Feature bins:     {bins_path}")
    print("Analysis uses only the first 80% chronological development period; the final 20% is excluded from feature discovery.")


if __name__ == "__main__":
    main()
