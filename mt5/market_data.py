from __future__ import annotations

from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1,
}

TIMEFRAME_DURATION = {
    "M1": timedelta(minutes=1),
    "M5": timedelta(minutes=5),
    "M15": timedelta(minutes=15),
    "H1": timedelta(hours=1),
}


def _norm_symbol(name: str) -> str:
    return "".join(ch for ch in name.upper() if ch.isalnum())


def resolve_symbol(preferred: str) -> str:
    """Resolve a logical instrument name to the broker's actual MT5 symbol name."""
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
    nasdaq_aliases = {"NASDAQ", "NASDAQ100", "NAS100", "USTEC", "US100", "US100CASH"}
    nasdaq_alias = target in nasdaq_aliases
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
        elif nasdaq_alias and norm in {"US100CASH", "US100", "NAS100", "USTEC", "NASDAQ100"}:
            score = 1
        elif nasdaq_alias and any(norm.startswith(alias) for alias in ("US100CASH", "US100", "NAS100", "USTEC", "NASDAQ100")):
            score = 2
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
    asof: datetime | None = None,
) -> pd.DataFrame:
    """Return MT5 rates in chronological order with a completion safety check.

    MT5 position 0 is the current forming candle, so completed-only requests
    begin at position 1. When ``asof`` is supplied (normally the latest tick
    timestamp), every returned bar is additionally required to have fully
    closed by that timestamp. This blocks execution if MT5 timestamps appear
    to be in the future or otherwise inconsistent.
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
    df = df.sort_values("time").reset_index(drop=True)

    if completed_only and asof is not None:
        if asof.tzinfo is None:
            raise ValueError("asof must be timezone-aware")
        asof_utc = asof.astimezone(timezone.utc)
        duration = TIMEFRAME_DURATION[timeframe]
        close_times = df["time"] + duration
        valid = close_times <= pd.Timestamp(asof_utc)
        if not bool(valid.all()):
            bad = df.loc[~valid, "time"].iloc[-1]
            raise RuntimeError(
                f"MT5 {timeframe} timestamp safety check failed: bar {bad} has not "
                f"completed by latest tick time {asof_utc.isoformat()}. Trade blocked."
            )

    return df


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
