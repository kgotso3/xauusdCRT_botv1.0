from __future__ import annotations

import math


def calculate_volume(account_balance: float, risk_fraction: float, entry: float, stop: float, tick_size: float, tick_value: float, volume_min: float, volume_max: float, volume_step: float) -> float:
    """Calculate volume without ever increasing risk to satisfy broker minimum lot.

    Returns 0.0 when the requested risk budget cannot support the broker's
    minimum volume. This blocks the trade instead of silently over-risking.
    """
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

    steps = math.floor((raw + 1e-12) / volume_step)
    stepped = steps * volume_step
    return min(volume_max, round(stepped, 8))
