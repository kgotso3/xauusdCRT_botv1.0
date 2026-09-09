from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from backtesting.historical import load_history, save_history
from backtesting.research import ResearchConfig, build_crt_occurrence_dataset
from mt5.connection import connect, disconnect
from mt5.market_data import resolve_symbol


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a causal CRT occurrence research dataset")
    parser.add_argument("--start", required=True, help="ISO date/time, e.g. 2023-01-01")
    parser.add_argument("--end", required=True, help="ISO date/time, e.g. 2026-09-01")
    parser.add_argument("--max-hold", type=int, default=24, help="Maximum H1 outcome window")
    parser.add_argument("--output", default="data/research", help="Output directory")
    args = parser.parse_args()

    start = _parse_utc(args.start)
    end = _parse_utc(args.end)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = connect()
    try:
        symbol = resolve_symbol(settings.symbol)
        print(f"Loading historical MT5 data for {symbol}: {start.isoformat()} -> {end.isoformat()}")

        h1 = load_history(symbol, "H1", start, end)
        m15 = load_history(symbol, "M15", start, end)
        m5 = load_history(symbol, "M5", start, end)

        stamp = f"{start.date()}_{end.date()}"
        save_history(h1, str(output_dir / f"{symbol}_H1_{stamp}.csv"))
        save_history(m15, str(output_dir / f"{symbol}_M15_{stamp}.csv"))
        save_history(m5, str(output_dir / f"{symbol}_M5_{stamp}.csv"))

        dataset = build_crt_occurrence_dataset(
            h1,
            m15,
            m5,
            ResearchConfig(max_holding_bars=args.max_hold),
        )
        dataset_path = output_dir / f"{symbol}_crt_occurrences_{stamp}.csv"
        dataset.to_csv(dataset_path, index=False)

        print("\nCRT RESEARCH DATASET")
        print(f"Symbol: {symbol}")
        print(f"H1 bars: {len(h1)} | M15 bars: {len(m15)} | M5 bars: {len(m5)}")
        print(f"CRT occurrences: {len(dataset)}")
        if not dataset.empty:
            buys = int((dataset["direction"] == "BUY").sum())
            sells = int((dataset["direction"] == "SELL").sum())
            ny = int(dataset["in_ny_08_13"].sum())
            print(f"BUY/SELL occurrences: {buys}/{sells}")
            print(f"Occurrences inside NY 08:00-13:00: {ny}")
            for col, label in [
                ("hit_1_0r", "1R"),
                ("hit_1_5r", "1.5R"),
                ("hit_2_0r", "2R"),
                ("hit_3_0r", "3R"),
            ]:
                if col in dataset.columns:
                    print(f"Hit {label} before stop: {dataset[col].mean() * 100:.2f}%")
            print(f"Average MFE: {dataset['mfe_r'].mean():.2f}R")
            print(f"Average MAE: {dataset['mae_r'].mean():.2f}R")
        print(f"Dataset CSV: {dataset_path}")
        print("\nLeakage rule: all features are frozen at H1 decision time; only outcome-label columns use future bars.")
    finally:
        disconnect()


if __name__ == "__main__":
    main()
