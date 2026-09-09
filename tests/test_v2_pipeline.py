import numpy as np
import pandas as pd

from backtesting.coverage import coverage_table
from backtesting.deterministic_benchmark import build_deterministic_benchmark
from features.v2_features import build_causal_v2_features
from ml.v1 import MLConfig, chronological_split


def _dataset(n=240):
    times = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    one_hour = pd.offsets.Hour(1)
    rows = []
    for i, ts in enumerate(times):
        direction = "BUY" if i % 2 == 0 else "SELL"
        open_ = 2600.0 + i * 0.1
        close = open_ + (1.0 if direction == "BUY" else -1.0)
        high = max(open_, close) + 1.5
        low = min(open_, close) - 1.5
        rows.append({
            "signal_time_utc": ts,
            "decision_time_utc": ts + one_hour,
            "entry_time_utc": ts + one_hour,
            "ny_hour": i % 24,
            "day_of_week": ts.day_name(),
            "in_ny_08_13": 8 <= (i % 24) < 13,
            "direction": direction,
            "crt_direction": "BULLISH" if direction == "BUY" else "BEARISH",
            "h1_bias": "BULLISH" if i % 3 else "BEARISH",
            "m15_bias": "BULLISH" if i % 4 else "BEARISH",
            "m5_bias": "BULLISH" if i % 5 else "BEARISH",
            "alignment_count": i % 4,
            "full_alignment": i % 4 == 3,
            "open": open_, "high": high, "low": low, "close": close,
            "candle_range": high - low,
            "body_size": abs(close - open_),
            "body_pct_range": abs(close - open_) / (high - low),
            "sweep_size": 0.5 + (i % 5) * 0.2,
            "sweep_atr": 0.1 + (i % 5) * 0.1,
            "wick_size": 1.0 + (i % 3) * 0.2,
            "ema20": close - 1.0,
            "ema50": close - 2.0,
            "ema200": close - 4.0,
            "ema20_50_distance": 1.0,
            "ema50_200_distance": 2.0,
            "rsi14": 35.0 + (i % 40),
            "atr14": 3.0 + (i % 10) * 0.1,
            "atr_pct": 0.001 + i * 0.000001,
            "entry": close,
            "stop": close - 3.0 if direction == "BUY" else close + 3.0,
            "risk_distance": 3.0,
            "hit_stop": i % 3 == 0,
            "hit_1_0r": i % 3 != 0,
            "hit_1_5r": i % 5 not in {0, 1},
            "mfe_r": 0.5 + (i % 8) * 0.4,
            "mae_r": -0.3 - (i % 5) * 0.2,
        })
    return pd.DataFrame(rows)


def test_coverage_reports_common_overlap():
    h1 = pd.DataFrame({"time": pd.date_range("2025-01-01", periods=72, freq="h", tz="UTC")})
    m15 = pd.DataFrame({"time": pd.date_range("2025-01-02", periods=160, freq="15min", tz="UTC")})
    m5 = pd.DataFrame({"time": pd.date_range("2025-01-02 01:00", periods=400, freq="5min", tz="UTC")})
    out = coverage_table({"H1": h1, "M15": m15, "M5": m5})
    assert len(out) == 3
    assert out["common_start"].notna().all()
    assert out["common_end"].notna().all()


def test_causal_volatility_feature_does_not_change_past_when_future_changes():
    df = _dataset(240)
    original = build_causal_v2_features(df, volatility_window=50)
    changed = df.copy()
    changed.loc[239, "atr_pct"] = 999.0
    rebuilt = build_causal_v2_features(changed, volatility_window=50)
    assert np.isclose(
        original.loc[150, "atr_pctile_rolling"],
        rebuilt.loc[150, "atr_pctile_rolling"],
        equal_nan=True,
    )


def test_deterministic_benchmark_keeps_targets_separate():
    report = build_deterministic_benchmark(_dataset(240), min_split_samples=2)
    assert {"1R", "1.5R"}.issubset(set(report["target"]))
    assert "validation_expectancy" in report.columns


def test_ml_chronological_split_is_ordered():
    features = build_causal_v2_features(_dataset(100), volatility_window=30)
    split = chronological_split(features, MLConfig())
    assert (split.iloc[:60]["ml_split"] == "TRAIN").all()
    assert (split.iloc[60:80]["ml_split"] == "VALIDATION").all()
    assert (split.iloc[80:]["ml_split"] == "TEST").all()
