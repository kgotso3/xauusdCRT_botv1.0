import pandas as pd

from backtesting.engine import _simulate_exit
from backtesting.metrics import calculate_metrics


def test_buy_target_exit():
    future = pd.DataFrame([
        {"time": pd.Timestamp("2026-01-01T10:00:00Z"), "high": 103.0, "low": 99.5, "close": 102.0},
    ])
    result = _simulate_exit(future, "BUY", 100.0, 99.0, 102.0, 24)
    assert result["exit_reason"] == "TARGET"
    assert result["r_multiple"] == 2.0


def test_same_bar_stop_is_conservative():
    future = pd.DataFrame([
        {"time": pd.Timestamp("2026-01-01T10:00:00Z"), "high": 103.0, "low": 98.5, "close": 101.0},
    ])
    result = _simulate_exit(future, "BUY", 100.0, 99.0, 102.0, 24)
    assert result["exit_reason"] == "STOP"
    assert result["r_multiple"] == -1.0


def test_metrics():
    trades = pd.DataFrame({"r_multiple": [2.0, -1.0, 2.0, -1.0]})
    trades["equity_r"] = trades["r_multiple"].cumsum()
    metrics = calculate_metrics(trades)
    assert metrics["trades"] == 4
    assert metrics["wins"] == 2
    assert metrics["losses"] == 2
    assert metrics["net_r"] == 2.0
    assert metrics["profit_factor"] == 2.0
