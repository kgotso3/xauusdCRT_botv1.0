from __future__ import annotations

from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}


def ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol {symbol!r} was not found in MT5.")
    if not info.visible and not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not enable symbol {symbol!r} in Market Watch.")


def get_rates(symbol: str, timeframe: str, count: int = 300) -> pd.DataFrame:
    ensure_symbol(symbol)
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe], 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No {timeframe} data returned for {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def latest_tick(symbol: str) -> dict:
    ensure_symbol(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick available for {symbol}: {mt5.last_error()}")
    return {
        "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "spread": float(tick.ask - tick.bid),
    }
