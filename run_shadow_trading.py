from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.shadow_trading import ShadowModelSpec, score_shadow_dataset, upsert_shadow_journal


def main() -> None:
    parser = argparse.ArgumentParser(description="CRT V3 shadow trading — journaling only, no MT5 order execution")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True, help="Path to a frozen V3 research model artifact")
    parser.add_argument("--target", choices=["1R", "1.5R"], required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--start", default=None, help="Optional UTC start timestamp for shadow cohort")
    parser.add_argument("--journal", default="data/research/shadow/v3_shadow_journal.csv")
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    start = None
    if args.start:
        start = pd.Timestamp(args.start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")

    target_r = 1.0 if args.target == "1R" else 1.5
    spec = ShadowModelSpec(target=args.target, model_path=args.model, threshold=args.threshold, target_r=target_r)
    scored = score_shadow_dataset(dataset, spec, start=start)
    selected = scored[scored["selected"]].copy() if not scored.empty else scored
    journal = upsert_shadow_journal(selected, args.journal)

    print("CRT V3 — SHADOW TRADING")
    print("MODE: JOURNAL ONLY | no MT5 order execution dependency")
    print(f"Model: {Path(args.model)}")
    print(f"Target: {args.target} | Threshold: {args.threshold:.2f}")
    if start is not None:
        print(f"Shadow cohort start: {start.isoformat()}")
    print(f"Scored occurrences: {len(scored)}")
    print(f"Selected shadow setups this run: {len(selected)}")
    print(f"Shadow journal rows total: {len(journal)}")
    print(f"Journal: {args.journal}")


if __name__ == "__main__":
    main()
