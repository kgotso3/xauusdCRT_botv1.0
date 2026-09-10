from __future__ import annotations

import pandas as pd

from backtesting.v4_5_regime import RegimeConfig, attach_regimes, build_h1_regime_features, walk_forward_regimes


def _h1(n=500):
    rows = []
    t0 = pd.Timestamp("2025-01-01T00:00:00Z")
    price = 2000.0
    for i in range(n):
        drift = 0.4 if i < n // 2 else -0.2
        o = price
        c = price + drift
        rows.append({"time": t0 + pd.to_timedelta(i, unit="h"), "open": o, "high": max(o, c) + 1.0, "low": min(o, c) - 1.0, "close": c})
        price = c
    return pd.DataFrame(rows)


def _trades():
    rows = []
    t0 = pd.Timestamp("2025-01-20T00:00:00Z")
    for i in range(160):
        t = t0 + pd.to_timedelta(i * 24, unit="h")
        win = i % 3 == 0
        rows.append({
            "signal_close_time_utc": t,
            "direction": "BULLISH" if i % 2 == 0 else "BEARISH",
            "correct_half": True,
            "m5exec_total_r": 1.5 if win else -1.0,
            "m5exec_tp1_hit": win,
            "m5exec_tp2_hit": win,
        })
    return pd.DataFrame(rows)


def test_h1_regimes_are_timestamped_at_bar_close():
    out = build_h1_regime_features(_h1(80))
    assert out.iloc[0]["bar_close_time_utc"] == out.iloc[0]["time"] + pd.to_timedelta(1, unit="h")


def test_attach_regimes_never_uses_future_h1_bar():
    h1 = _h1(1000)
    trades = _trades().head(5)
    out = attach_regimes(trades, h1)
    assert (out["bar_close_time_utc"] <= out["signal_close_time_utc"]).all()


def test_walk_forward_keeps_train_before_test():
    h1 = _h1(5000)
    frame = attach_regimes(_trades(), h1)
    cfg = RegimeConfig(train_days=60, test_days=30, step_days=30, min_train_trades=20, min_test_trades=5)
    folds, _ = walk_forward_regimes(frame, cfg)
    assert not folds.empty
    assert (folds["train_end"] < folds["test_end"]).all()
    assert (folds["train_start"] < folds["train_end"]).all()
