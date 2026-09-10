from __future__ import annotations

import math

import MetaTrader5 as mt5


def _floor_to_step(value: float, step: float) -> float:
    if step <= 0:
        raise ValueError("volume_step must be positive")
    steps = math.floor((value + 1e-12) / step)
    return round(steps * step, 8)


def calculate_volume(
    account_balance: float,
    risk_fraction: float,
    entry: float,
    stop: float,
    tick_size: float,
    tick_value: float,
    volume_min: float,
    volume_max: float,
    volume_step: float,
) -> float:
    """Legacy deterministic sizing helper retained for unit tests/backtests."""
    distance = abs(entry - stop)
    if distance <= 0 or tick_size <= 0 or tick_value <= 0 or volume_step <= 0:
        raise ValueError("Invalid entry/stop/tick/volume parameters")
    if not 0 < risk_fraction < 1:
        raise ValueError("risk_fraction must be between 0 and 1")

    risk_money = account_balance * risk_fraction
    loss_per_lot = (distance / tick_size) * tick_value
    raw = risk_money / loss_per_lot
    if raw < volume_min:
        return 0.0

    stepped = _floor_to_step(raw, volume_step)
    return min(volume_max, stepped)


def _symbol_loss_per_lot(info, entry: float, stop: float) -> float:
    """Fallback stop-loss estimate from the broker symbol contract fields."""
    tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
    tick_value_loss = float(getattr(info, "trade_tick_value_loss", 0.0) or 0.0)
    tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
    value = tick_value_loss if tick_value_loss > 0 else tick_value
    distance = abs(float(entry) - float(stop))
    if distance <= 0 or tick_size <= 0 or value <= 0:
        return 0.0
    return float((distance / tick_size) * value)


def calculate_mt5_volume(
    symbol: str,
    direction: str,
    account_equity: float,
    risk_fraction: float,
    entry: float,
    stop: float,
) -> tuple[float, float, float]:
    """Size a position using MT5's P/L calculator with a contract-spec fallback.

    Returns (volume, risk_budget, estimated_loss_at_stop). A zero volume means
    the broker minimum lot would exceed the configured risk budget.

    Some broker GOLD symbols can return zero from ``order_calc_profit`` for a
    hypothetical stop calculation. When that happens this function falls back
    to ``trade_tick_size`` and ``trade_tick_value_loss``/``trade_tick_value``.
    This function never sends an order.
    """
    if not 0 < risk_fraction < 1:
        raise ValueError("risk_fraction must be between 0 and 1")
    if account_equity <= 0:
        raise ValueError("account_equity must be positive")
    if entry <= 0 or stop <= 0 or entry == stop:
        raise ValueError("entry and stop must be positive and different")

    direction = direction.upper()
    if direction == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
    elif direction == "SELL":
        order_type = mt5.ORDER_TYPE_SELL
    else:
        raise ValueError("direction must be BUY or SELL")

    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Unable to read symbol information for {symbol}.")

    min_volume = float(info.volume_min)
    max_volume = float(info.volume_max)
    step = float(info.volume_step)
    if min_volume <= 0 or max_volume <= 0 or step <= 0:
        raise RuntimeError(f"Invalid broker volume specification for {symbol}.")

    mt5_loss = mt5.order_calc_profit(order_type, symbol, 1.0, float(entry), float(stop))
    loss_per_lot = abs(float(mt5_loss)) if mt5_loss is not None else 0.0
    if loss_per_lot <= 0:
        loss_per_lot = _symbol_loss_per_lot(info, entry, stop)
    if loss_per_lot <= 0:
        raise RuntimeError(
            "Unable to estimate stop loss from MT5 or symbol contract fields; shadow sizing unavailable."
        )

    risk_budget = float(account_equity) * float(risk_fraction)
    raw_volume = risk_budget / loss_per_lot
    if raw_volume < min_volume:
        return 0.0, risk_budget, float(loss_per_lot * min_volume)

    volume = min(max_volume, _floor_to_step(raw_volume, step))
    estimated_loss = float(loss_per_lot * volume)

    if estimated_loss > risk_budget * 1.001:
        raise RuntimeError(
            f"Calculated position exceeds risk budget: estimated={estimated_loss:.2f}, budget={risk_budget:.2f}"
        )

    return volume, risk_budget, estimated_loss
