from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from features.v2_features import build_causal_v2_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe V2 feature dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="data/research/features/GOLD_crt_v2_features.csv")
    parser.add_argument("--volatility-window", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    features = build_causal_v2_features(df, volatility_window=args.volatility_window)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)

    print("V2 CAUSAL FEATURE DATASET")
    print(f"Rows: {len(features)}")
    print(f"First signal: {features['signal_time_utc'].min()}")
    print(f"Last signal:  {features['signal_time_utc'].max()}")
    print(f"Output: {out_path}")
    print("Rolling volatility percentile is causal: current/past observations only.")


if __name__ == "__main__":
    main()
