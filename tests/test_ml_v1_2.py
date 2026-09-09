import numpy as np
import pandas as pd

from features.v1_2_features import build_causal_v1_2_features
from ml.v1_2 import MLV12Config, chronological_split


def _dataset(n=180):
    times = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    rows = []
    for i, ts in enumerate(times):
        direction = "BUY" if i % 2 == 0 else "SELL"
        close = 2600 + i * 0.2
        high = close + 2.0
        low = close - 2.0
        rows.append({
            "signal_time_utc": ts,
            "decision_time_utc": ts + pd.to_timedelta(1, unit="h"),
            "entry_time_utc": ts + pd.to_timedelta(1, unit="h"),
            "ny_hour": i % 24,
            "direction": direction,
            "h1_bias": "BULLISH" if i % 3 else "BEARISH",
            "m15_bias": "BULLISH" if i % 4 else "BEARISH",
            "m5_bias": "BULLISH" if i % 5 else "BEARISH",
            "alignment_count": i % 4,
            "open": close - 0.5,
            "high": high,
            "low": low,
            "close": close,
            "candle_range": high - low,
            "body_pct_range": 0.25,
            "sweep_size": 0.5,
            "sweep_atr": 0.2,
            "wick_size": 1.0,
            "ema20_50_distance": 1.0,
            "ema50_200_distance": 2.0,
            "ema20_slope_4h": 0.4,
            "rsi14": 40 + i % 20,
            "atr14": 4.0,
            "atr_pct": 4.0 / close,
            "atr_change_4h": 0.01 * ((i % 5) - 2),
            "macd": 0.5,
            "macd_signal": 0.3,
            "macd_hist": 0.2,
            "return_1h": 0.001,
            "return_4h": 0.002,
            "return_8h": 0.003,
            "risk_distance": 3.0,
            "range_position_4h": 0.5,
            "range_position_8h": 0.5,
            "range_position_24h": 0.5,
            "distance_recent_high_4h_atr": 0.5,
            "distance_recent_low_4h_atr": 0.5,
            "distance_recent_high_8h_atr": 0.8,
            "distance_recent_low_8h_atr": 0.8,
            "distance_recent_high_24h_atr": 1.2,
            "distance_recent_low_24h_atr": 1.2,
            "prev_day_high": close + 5,
            "prev_day_low": close - 5,
            "distance_prev_day_high_atr": 1.25,
            "distance_prev_day_low_atr": 1.25,
            "hit_1_0r": i % 3 != 0,
            "hit_1_5r": i % 5 not in {0, 1},
        })
    return pd.DataFrame(rows)


def test_v12_features_include_expanded_context():
    out = build_causal_v1_2_features(_dataset())
    for col in ["directional_rsi", "macd_hist_atr", "directional_return_4h", "directional_prev_day_sweep"]:
        assert col in out.columns


def test_v12_future_change_does_not_change_earlier_rolling_feature():
    df = _dataset()
    a = build_causal_v1_2_features(df, volatility_window=50)
    changed = df.copy()
    changed.loc[len(changed) - 1, "atr_pct"] = 999.0
    b = build_causal_v1_2_features(changed, volatility_window=50)
    assert np.isclose(a.loc[100, "atr_pctile_rolling"], b.loc[100, "atr_pctile_rolling"], equal_nan=True)


def test_v12_split_remains_chronological():
    out = chronological_split(build_causal_v1_2_features(_dataset(100)), MLV12Config())
    assert (out.iloc[:60]["ml_split"] == "TRAIN").all()
    assert (out.iloc[60:80]["ml_split"] == "VALIDATION").all()
    assert (out.iloc[80:]["ml_split"] == "TEST").all()
