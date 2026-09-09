from __future__ import annotations


def calculate_volume(account_balance: float, risk_fraction: float, entry: float, stop: float, tick_size: float, tick_value: float, volume_min: float, volume_max: float, volume_step: float) -> float:
    distance = abs(entry - stop)
    if distance <= 0 or tick_size <= 0 or tick_value <= 0:
        raise ValueError("Invalid entry/stop/tick parameters")

    risk_money = account_balance * risk_fraction
    loss_per_lot = (distance / tick_size) * tick_value
    raw = risk_money / loss_per_lot
    stepped = int(raw / volume_step) * volume_step
    return max(volume_min, min(volume_max, round(stepped, 8)))
