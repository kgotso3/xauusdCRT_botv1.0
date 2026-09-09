from __future__ import annotations

from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from mt5.market_data import TIMEFRAME_MAP, ensure_symbol


def load_history(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load MT5 historical OHLCV bars for a UTC date range.

    ``start`` and ``end`` must be timezone-aware. Returned bars are sorted in
    chronological order and use UTC timestamps.
    """
    ensure_symbol(symbol)
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    if start >= end:
        raise ValueError("start must be before end")

    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    rates = mt5.copy_rates_range(symbol, TIMEFRAME_MAP[timeframe], start_utc, end_utc)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"No {timeframe} history returned for {symbol} between "
            f"{start_utc.isoformat()} and {end_utc.isoformat()}: {mt5.last_error()}"
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def save_history(df: pd.DataFrame, path: str) -> None:
    """Persist historical bars to CSV for reproducible research/backtests."""
    df.to_csv(path, index=False)
