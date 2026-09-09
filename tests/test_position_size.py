from risk.position_size import calculate_volume


def test_volume_does_not_exceed_risk_budget():
    volume = calculate_volume(10000, 0.0025, 2500, 2495, 0.01, 1.0, 0.01, 100.0, 0.01)
    assert volume >= 0


def test_returns_zero_if_minimum_lot_overrisks():
    volume = calculate_volume(100, 0.0025, 2500, 2400, 0.01, 1.0, 0.01, 100.0, 0.01)
    assert volume == 0.0
