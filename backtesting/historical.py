from __future__ import annotations

from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd

from mt5.market_data import TIMEFRAME_MAP, ensure_symbol


CHUNK_DAYS = {
    "H1": 365,
    "M15": 120,
    "M5": 45,
}


def _fetch_chunk(symbol: str, timeframe: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, TIMEFRAME_MAP[timeframe], start_utc, end_utc)
    if rates is None or len(rates) == 0:
        code, message = mt5.last_error()
        raise RuntimeError(
            f"No {timeframe} history returned for {symbol} between "
            f"{start_utc.isoformat()} and {end_utc.isoformat()}: ({code}, {message!r})"
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def load_history(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load MT5 historical OHLCV bars for a UTC date range.

    Long lower-timeframe ranges are fetched in smaller chunks because the MT5
    terminal can reject very large ``copy_rates_range`` requests. Returned bars
    are concatenated, sorted chronologically, deduplicated by open time, and use
    UTC timestamps throughout.
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
    chunk_days = CHUNK_DAYS.get(timeframe, 90)

    frames: list[pd.DataFrame] = []
    cursor = start_utc
    while cursor < end_utc:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_utc)
        frames.append(_fetch_chunk(symbol, timeframe, cursor, chunk_end))
        cursor = chunk_end

    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def save_history(df: pd.DataFrame, path: str) -> None:
    """Persist historical bars to CSV for reproducible research/backtests."""
    df.to_csv(path, index=False)
