from __future__ import annotations

import pandas as pd

from backtesting.session_crt_split import SplitTargetConfig, simulate_split_trade


def _future(rows):
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])


def test_split_full_winner_returns_plus_1_5r():
    future = _future([
        ["2026-01-12T15:00:00Z", 100.0, 105.0, 99.0, 104.0],
        ["2026-01-12T16:00:00Z", 104.0, 110.0, 103.0, 109.0],
    ])
    out = simulate_split_trade(future, "BULLISH", entry=100.0, stop=95.0)
    assert out["tp1_hit"] is True
    assert out["tp2_hit"] is True
    assert out["leg1_r"] == 0.5
    assert out["leg2_r"] == 1.0
    assert out["total_r"] == 1.5
    assert out["outcome"] == "TP1_TP2"


def test_split_tp1_then_stop_finishes_flat():
    future = _future([
        ["2026-01-12T15:00:00Z", 100.0, 105.5, 99.0, 104.0],
        ["2026-01-12T16:00:00Z", 104.0, 104.5, 94.0, 95.0],
    ])
    out = simulate_split_trade(future, "BULLISH", entry=100.0, stop=95.0)
    assert out["tp1_hit"] is True
    assert out["tp2_hit"] is False
    assert out["leg1_r"] == 0.5
    assert out["leg2_r"] == -0.5
    assert out["total_r"] == 0.0
    assert out["outcome"] == "TP1_THEN_STOP"


def test_split_full_stop_returns_minus_1r():
    future = _future([
        ["2026-01-12T15:00:00Z", 100.0, 102.0, 94.0, 95.0],
    ])
    out = simulate_split_trade(future, "BULLISH", entry=100.0, stop=95.0)
    assert out["tp1_hit"] is False
    assert out["tp2_hit"] is False
    assert out["leg1_r"] == -0.5
    assert out["leg2_r"] == -0.5
    assert out["total_r"] == -1.0
    assert out["outcome"] == "STOP"


def test_same_bar_stop_and_target_is_conservatively_stop_first():
    future = _future([
        ["2026-01-12T15:00:00Z", 100.0, 111.0, 94.0, 100.0],
    ])
    out = simulate_split_trade(future, "BULLISH", entry=100.0, stop=95.0)
    assert out["ambiguous_intrabar"] is True
    assert out["total_r"] == -1.0
    assert out["outcome"] == "STOP"


def test_timeout_keeps_unresolved_leg_neutral():
    future = _future([
        ["2026-01-12T15:00:00Z", 100.0, 105.5, 99.0, 104.0],
    ])
    cfg = SplitTargetConfig(max_holding_bars=1)
    out = simulate_split_trade(future, "BULLISH", entry=100.0, stop=95.0, config=cfg)
    assert out["tp1_hit"] is True
    assert out["tp2_hit"] is False
    assert out["leg1_r"] == 0.5
    assert out["leg2_r"] == 0.0
    assert out["total_r"] == 0.5
    assert out["outcome"] == "TP1_TIMEOUT"
