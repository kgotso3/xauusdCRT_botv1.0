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


def _norm_symbol(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())


def resolve_symbol(preferred: str) -> str:
    """Resolve a logical gold symbol to the broker's actual MT5 symbol name.

    Exact matches are preferred. For XAUUSD/GOLD we then try common broker
    aliases and suffix variants, while avoiding unrelated equity names that
    merely contain the word 'Gold'.
    """
    preferred = preferred.strip()
    info = mt5.symbol_info(preferred)
    if info is not None:
        if not info.visible and not mt5.symbol_select(preferred, True):
            raise RuntimeError(f"Could not enable symbol {preferred!r} in Market Watch.")
        return preferred

    symbols = mt5.symbols_get()
    if symbols is None:
        raise RuntimeError(f"Unable to enumerate MT5 symbols: {mt5.last_error()}")

    target = _norm_symbol(preferred)
    gold_alias = target in {"XAUUSD", "GOLD"}
    candidates: list[tuple[int, str]] = []

    for symbol in symbols:
        name = symbol.name
        norm = _norm_symbol(name)
        score = None

        if norm == target:
            score = 0
        elif gold_alias and norm in {"XAUUSD", "GOLD"}:
            score = 1
        elif gold_alias and norm.startswith("XAUUSD"):
            score = 2
        elif gold_alias and norm.startswith("GOLD") and not norm.startswith("GOLDMAN"):
            score = 3
        elif norm.startswith(target):
            score = 4

        if score is not None:
            candidates.append((score, name))

    if not candidates:
        raise RuntimeError(
            f"Symbol {preferred!r} was not found and no safe broker alias could be resolved."
        )

    candidates.sort(key=lambda item: (item[0], len(item[1]), item[1]))
    resolved = candidates[0][1]
    info = mt5.symbol_info(resolved)
    if info is None:
        raise RuntimeError(f"Resolved symbol {resolved!r} is unavailable.")
    if not info.visible and not mt5.symbol_select(resolved, True):
        raise RuntimeError(f"Could not enable resolved symbol {resolved!r} in Market Watch.")
    return resolved


def ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol {symbol!r} was not found in MT5.")
    if not info.visible and not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Could not enable symbol {symbol!r} in Market Watch.")


def get_rates(
    symbol: str,
    timeframe: str,
    count: int = 300,
    completed_only: bool = True,
) -> pd.DataFrame:
    """Return MT5 rates in chronological order.

    MT5 bar position 0 is the current forming candle. Strategy calculations use
    start position 1 by default so signals are based only on completed candles.
    """
    ensure_symbol(symbol)
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    start_pos = 1 if completed_only else 0
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME_MAP[timeframe], start_pos, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No {timeframe} data returned for {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def latest_tick(symbol: str) -> dict:
    ensure_symbol(symbol)
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        raise RuntimeError(f"No tick available for {symbol}: {mt5.last_error()}")
    point = float(info.point or 0.0)
    spread_price = float(tick.ask - tick.bid)
    spread_points = spread_price / point if point > 0 else 0.0
    return {
        "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
        "bid": float(tick.bid),
        "ask": float(tick.ask),
        "spread": spread_price,
        "spread_points": float(spread_points),
    }
