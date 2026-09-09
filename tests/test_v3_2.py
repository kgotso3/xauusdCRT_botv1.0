from __future__ import annotations

import numpy as np
import pandas as pd

from features.v3_2_features import build_causal_v3_2_features
from ml.v3_2_comparison import V32Config, run_v3_2_comparison


def _dataset(n: int = 1800) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    direction = np.where(np.arange(n) % 2 == 0, "BUY", "SELL")
    close = rng.normal(2000, 10, n)
    open_ = close + rng.normal(0, 1, n)
    high = np.maximum(open_, close) + rng.uniform(.2, 2, n)
    low = np.minimum(open_, close) - rng.uniform(.2, 2, n)
    candle_range = high - low
    wick_size = candle_range - np.abs(close - open_)
    atr14 = rng.uniform(1, 3, n)

    # Canonical research-layer range/liquidity features that are already present
    # in production occurrence CSVs before the V1.2/V3 feature builders run.
    range_position_4h = rng.uniform(0, 1, n)
    range_position_8h = rng.uniform(0, 1, n)
    range_position_24h = rng.uniform(0, 1, n)
    distance_recent_high_4h_atr = rng.uniform(0, 4, n)
    distance_recent_low_4h_atr = rng.uniform(0, 4, n)
    distance_recent_high_8h_atr = rng.uniform(0, 5, n)
    distance_recent_low_8h_atr = rng.uniform(0, 5, n)
    distance_recent_high_24h_atr = rng.uniform(0, 8, n)
    distance_recent_low_24h_atr = rng.uniform(0, 8, n)
    distance_prev_day_high_atr = rng.uniform(-5, 5, n)
    distance_prev_day_low_atr = rng.uniform(-5, 5, n)

    base = pd.DataFrame({
        "signal_time_utc": ts,
        "decision_time_utc": ts,
        "direction": direction,
        "alignment_count": rng.integers(0, 4, n),
        "h1_bias": np.where(rng.random(n) > .5, "BULLISH", "BEARISH"),
        "m15_bias": np.where(rng.random(n) > .5, "BULLISH", "BEARISH"),
        "m5_bias": np.where(rng.random(n) > .5, "BULLISH", "BEARISH"),
        "rsi14": rng.normal(50, 10, n),
        "atr14": atr14,
        "atr_pct": rng.uniform(.001, .01, n),
        "sweep_atr": rng.uniform(.1, 2, n),
        "sweep_size": rng.uniform(.1, 2, n),
        "body_pct_range": np.abs(close - open_) / candle_range,
        "candle_range": candle_range,
        "wick_size": wick_size,
        "ema20_50_distance": rng.normal(0, 2, n),
        "ema50_200_distance": rng.normal(0, 3, n),
        "ema20_slope_4h": rng.normal(0, .5, n),
        "macd": rng.normal(0, .5, n),
        "macd_signal": rng.normal(0, .5, n),
        "macd_hist": rng.normal(0, .2, n),
        "return_1h": rng.normal(0, .002, n),
        "return_4h": rng.normal(0, .004, n),
        "return_8h": rng.normal(0, .006, n),
        "atr_change_4h": rng.normal(0, .1, n),
        "risk_distance": rng.uniform(.5, 2, n),
        "high": high,
        "low": low,
        "open": open_,
        "close": close,
        "prev_day_high": close + rng.uniform(3, 12, n),
        "prev_day_low": close - rng.uniform(3, 12, n),
        "ny_hour": ts.tz_convert("America/New_York").hour,
        "range_position_4h": range_position_4h,
        "range_position_8h": range_position_8h,
        "range_position_24h": range_position_24h,
        "distance_recent_high_4h_atr": distance_recent_high_4h_atr,
        "distance_recent_low_4h_atr": distance_recent_low_4h_atr,
        "distance_recent_high_8h_atr": distance_recent_high_8h_atr,
        "distance_recent_low_8h_atr": distance_recent_low_8h_atr,
        "distance_recent_high_24h_atr": distance_recent_high_24h_atr,
        "distance_recent_low_24h_atr": distance_recent_low_24h_atr,
        "distance_prev_day_high_atr": distance_prev_day_high_atr,
        "distance_prev_day_low_atr": distance_prev_day_low_atr,
        "hit_1_0r": (rng.random(n) > .55).astype(int),
        "hit_1_5r": (rng.random(n) > .65).astype(int),
    })
    return base


def test_v3_2_feature_builder_does_not_need_future_labels():
    df = _dataset(250).drop(columns=["hit_1_0r", "hit_1_5r"])
    out = build_causal_v3_2_features(df)
    assert "v32_direction_x_trend_strength" in out.columns
    assert "v32_regime_transition_state" in out.columns
    assert len(out) == len(df)


def test_v3_2_excludes_forward_cohort_and_compares_fixed_rf(tmp_path):
    df = _dataset(1800)
    cutoff = df["signal_time_utc"].iloc[1700]
    cfg = V32Config(train_size=900, validation_size=200, step_size=200, development_cutoff=cutoff)
    summary, folds = run_v3_2_comparison(df, output_dir=tmp_path, config=cfg)
    assert set(summary["variant"]) == {"V3_BASE_RF", "V3_2_REFINED_RF"}
    assert pd.to_datetime(folds["validation_end"], utc=True).max() < cutoff
    assert set(folds["variant"]) == {"V3_BASE_RF", "V3_2_REFINED_RF"}
    refined_status = summary.loc[summary["variant"] == "V3_2_REFINED_RF", "promotion_status"].iloc[0]
    assert refined_status in {"RESEARCH_ONLY", "PROMOTE_TO_FINAL_FREEZE"}
