from __future__ import annotations

import argparse
from pathlib import Path

from backtesting.coverage import coverage_table, load_history_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect true H1/M15/M5 historical coverage")
    parser.add_argument("--h1", required=True)
    parser.add_argument("--m15", required=True)
    parser.add_argument("--m5", required=True)
    parser.add_argument("--output", default="data/research/coverage")
    args = parser.parse_args()

    histories = {
        "H1": load_history_csv(args.h1),
        "M15": load_history_csv(args.m15),
        "M5": load_history_csv(args.m5),
    }
    table = coverage_table(histories)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "history_coverage.csv"
    table.to_csv(out_path, index=False)

    print("HISTORICAL COVERAGE DIAGNOSTICS")
    print(table.to_string(index=False))
    if not table.empty and table["common_start"].notna().any():
        print(f"\nCommon H1/M15/M5 start: {table['common_start'].dropna().iloc[0]}")
        print(f"Common H1/M15/M5 end:   {table['common_end'].dropna().iloc[0]}")
    print(f"\nCoverage CSV: {out_path}")


if __name__ == "__main__":
    main()
