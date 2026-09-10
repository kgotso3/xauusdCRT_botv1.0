from __future__ import annotations

import pandas as pd

from backtesting.v4_7_frozen_validation import (
    FrozenValidationConfig,
    _account_curve,
    _cost_sensitivity,
    _metrics,
    _monte_carlo,
)


def _sample():
    ts = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    return pd.DataFrame({
        "signal_close_time_utc": ts,
        "entry_model": ["OTE_0_705"] * 4 + ["OTE_0_79"] * 4,
        "direction": ["BULLISH", "BEARISH"] * 4,
        "ny_hour": [7, 14, 7, 14, 7, 14, 7, 14],
        "gross_r": [1.5, -1.0, 0.5, 1.0, 1.5, -1.0, 0.5, 1.0],
        "net_r": [1.46, -1.04, 0.46, 0.96, 1.46, -1.04, 0.46, 0.96],
    })


def test_metrics_include_zero_equity_baseline_drawdown():
    frame = _sample().iloc[[1, 2]].copy()
    m = _metrics(frame)
    assert m["max_drawdown_r"] <= -1.04


def test_cost_sensitivity_reduces_expectancy_as_cost_rises():
    out = _cost_sensitivity(_sample(), costs=(0.0, 0.10))
    for _, grp in out.groupby("entry_model"):
        grp = grp.sort_values("cost_r")
        assert grp.iloc[1]["expectancy_r"] < grp.iloc[0]["expectancy_r"]


def test_account_curve_uses_quarter_percent_risk_per_setup():
    cfg = FrozenValidationConfig(account_risk_fraction=0.0025, monte_carlo_runs=10)
    frame = _sample().iloc[[0]].copy()
    curve = _account_curve(frame, cfg, starting_equity=100000.0)
    expected = 100000.0 * (1.0 + 0.0025 * float(frame.iloc[0]["net_r"]))
    assert abs(curve.iloc[0]["equity"] - expected) < 1e-9


def test_monte_carlo_is_seed_reproducible():
    cfg = FrozenValidationConfig(monte_carlo_runs=100, monte_carlo_seed=7)
    a = _monte_carlo(_sample(), cfg)
    b = _monte_carlo(_sample(), cfg)
    assert a == b
