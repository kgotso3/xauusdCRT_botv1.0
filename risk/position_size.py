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


def calculate_mt5_volume(
    symbol: str,
    direction: str,
    account_equity: float,
    risk_fraction: float,
    entry: float,
    stop: float,
) -> tuple[float, float, float]:
    """Size a position using MT5's own P/L calculator in account currency.

    Returns (volume, risk_budget, estimated_loss_at_stop). A zero volume means
    the broker minimum lot would exceed the configured risk budget.
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

    loss_for_one_lot = mt5.order_calc_profit(order_type, symbol, 1.0, float(entry), float(stop))
    if loss_for_one_lot is None:
        raise RuntimeError(f"order_calc_profit failed: {mt5.last_error()}")

    loss_per_lot = abs(float(loss_for_one_lot))
    if loss_per_lot <= 0:
        raise RuntimeError("MT5 returned a zero loss estimate for the stop distance.")

    risk_budget = float(account_equity) * float(risk_fraction)
    raw_volume = risk_budget / loss_per_lot
    if raw_volume < min_volume:
        min_lot_loss = abs(
            float(mt5.order_calc_profit(order_type, symbol, min_volume, float(entry), float(stop)) or 0.0)
        )
        return 0.0, risk_budget, min_lot_loss

    volume = min(max_volume, _floor_to_step(raw_volume, step))
    estimated_loss = abs(
        float(mt5.order_calc_profit(order_type, symbol, volume, float(entry), float(stop)) or 0.0)
    )

    if estimated_loss > risk_budget * 1.001:
        raise RuntimeError(
            f"Calculated position exceeds risk budget: estimated={estimated_loss:.2f}, budget={risk_budget:.2f}"
        )

    return volume, risk_budget, estimated_loss
