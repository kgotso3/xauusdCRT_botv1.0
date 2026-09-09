from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.segmentation import build_segmentation_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CRT research dataset")
    parser.add_argument("--dataset", required=True, help="Path to CRT occurrence CSV")
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum samples per segment")
    parser.add_argument("--output", default="data/research/segments", help="Output directory")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise RuntimeError("Research dataset is empty")

    for col in ["signal_time_utc", "decision_time_utc", "entry_time_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    print("CRT DATA COVERAGE")
    if "signal_time_utc" in df.columns:
        print(f"Occurrences: {len(df)}")
        print(f"First signal: {df['signal_time_utc'].min()}")
        print(f"Last signal: {df['signal_time_utc'].max()}")
    print(f"NY-window occurrences: {int(df['in_ny_08_13'].sum())}")

    tables = build_segmentation_tables(df, min_samples=args.min_samples)
    stem = dataset_path.stem
    for name, table in tables.items():
        path = output_dir / f"{stem}_{name}.csv"
        table.to_csv(path, index=False)

    print("\nTOP 2R SEGMENTS")
    candidates = []
    for name, table in tables.items():
        if table.empty:
            continue
        top = table.iloc[0].to_dict()
        top["segment_table"] = name
        candidates.append(top)
    ranked = pd.DataFrame(candidates)
    if not ranked.empty:
        ranked = ranked.sort_values(["hit_2r", "samples"], ascending=[False, False])
        display_cols = [c for c in ["segment_table", "direction", "ny_hour", "alignment_count", "day_of_week", "rsi_band", "volatility_regime", "sweep_atr_band", "samples", "hit_1r", "hit_2r", "hit_3r", "avg_mfe_r", "avg_mae_r"] if c in ranked.columns]
        print(ranked[display_cols].head(10).to_string(index=False))

    print(f"\nSegment CSVs saved to: {output_dir}")


if __name__ == "__main__":
    main()
