from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtesting.engine import BacktestConfig, run_backtest
from backtesting.historical import load_history, save_history
from mt5.connection import connect, disconnect
from mt5.market_data import resolve_symbol


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical CRT backtest on XM MT5 data")
    parser.add_argument("--start", required=True, help="ISO date/time, e.g. 2026-03-01")
    parser.add_argument("--end", required=True, help="ISO date/time, e.g. 2026-09-01")
    parser.add_argument("--rr", type=float, default=2.0, help="Reward:risk target, default 2.0")
    parser.add_argument("--score", type=int, default=70, help="Minimum signal score, default 70")
    parser.add_argument("--max-hold", type=int, default=24, help="Maximum H1 holding bars")
    parser.add_argument("--output", default="data/backtests", help="Output directory")
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

        config = BacktestConfig(
            minimum_score=args.score,
            reward_risk=args.rr,
            max_holding_bars=args.max_hold,
        )
        trades, metrics = run_backtest(h1, m15, m5, config)
        trades_path = output_dir / f"{symbol}_trades_{stamp}.csv"
        trades.to_csv(trades_path, index=False)

        print("\nBACKTEST SUMMARY")
        print(f"Symbol: {symbol}")
        print(f"H1 bars: {len(h1)} | M15 bars: {len(m15)} | M5 bars: {len(m5)}")
        print(f"Trades: {metrics['trades']}")
        print(f"Wins/Losses: {metrics['wins']}/{metrics['losses']}")
        print(f"Win rate: {metrics['win_rate'] * 100:.2f}%")
        print(f"Net R: {metrics['net_r']:.2f}R")
        print(f"Average/Expectancy: {metrics['average_r']:.3f}R")
        pf = metrics['profit_factor']
        print(f"Profit factor: {'inf' if pd.isna(pf) is False and pf == float('inf') else f'{pf:.2f}'}")
        print(f"Max drawdown: {metrics['max_drawdown_r']:.2f}R")
        print(f"Trades CSV: {trades_path}")
        print(f"Raw history directory: {output_dir}")
        print("\nAssumptions: next-H1-open entry, structural H1 stop, conservative stop-first on ambiguous same-bar exits, one position at a time, no commissions/slippage yet.")
    finally:
        disconnect()


if __name__ == "__main__":
    main()
