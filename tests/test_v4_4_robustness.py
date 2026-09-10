from __future__ import annotations

import pandas as pd

from backtesting.v4_4_robustness import RobustnessConfig, rolling_windows, robustness_summary


def _frame():
    rows = []
    start = pd.Timestamp("2025-01-01T00:00:00Z")
    for i in range(180):
        ts = start + pd.to_timedelta(int(i), unit="D")
        rows.append({"signal_close_time_utc": ts, "total_r": 1.0 if i % 3 else -1.0})
    return pd.DataFrame(rows)


def test_rolling_windows_are_time_ordered_and_nonempty():
    cfg = RobustnessConfig(window_days=60, step_days=30, min_trades_per_window=20)
    windows = rolling_windows(_frame(), cfg)
    assert not windows.empty
    assert windows["window_start"].is_monotonic_increasing
    assert (windows["trades"] >= 20).all()


def test_positive_process_can_be_marked_stable():
    cfg = RobustnessConfig(window_days=60, step_days=30, min_trades_per_window=20, min_positive_window_share=0.5, min_aggregate_profit_factor=1.01)
    frame = _frame()
    windows = rolling_windows(frame, cfg)
    result = robustness_summary(frame, windows, "TEST", cfg)
    assert result["positive_window_share"] >= 0.5
    assert result["median_window_expectancy"] > 0
    assert bool(result["stable"])


def test_drawdown_includes_zero_equity_baseline():
    frame = pd.DataFrame({
        "signal_close_time_utc": pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"], utc=True),
        "total_r": [-1.0, 2.0],
    })
    cfg = RobustnessConfig(window_days=10, step_days=10, min_trades_per_window=1)
    windows = rolling_windows(frame, cfg)
    assert float(windows.iloc[0]["max_drawdown_r"]) == -1.0
