from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.deterministic_benchmark import build_deterministic_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transparent V2 deterministic benchmark")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="data/research/benchmark")
    parser.add_argument("--min-split-samples", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    report = build_deterministic_benchmark(df, min_split_samples=args.min_split_samples)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "deterministic_benchmark.csv"
    report.to_csv(path, index=False)

    print("V2 DETERMINISTIC BENCHMARK")
    for target in ["1R", "1.5R"]:
        print(f"\nTOP RULES — {target}")
        part = report[(report["target"] == target) & report["sample_stable"]].head(10)
        cols = [
            "rule", "samples_total", "train_hit_rate", "validation_hit_rate", "test_hit_rate",
            "train_expectancy", "validation_expectancy", "test_expectancy",
        ]
        print(part[cols].to_string(index=False) if not part.empty else "No stable rules")

    print(f"\nBenchmark CSV: {path}")
    print("Benchmark rules are fixed and transparent; they are not selected from TEST results.")


if __name__ == "__main__":
    main()
