from __future__ import annotations

import numpy as np
import pandas as pd

from features.v3_2_features import build_causal_v3_2_features
from ml.v3_2_comparison import V32Config, run_v3_2_comparison


def _dataset(n: int = 1800) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    direction = np.where(np.arange(n) % 2 == 0, "BUY", "SELL")
    base = pd.DataFrame({
        "signal_time_utc": ts,
        "decision_time_utc": ts,
        "direction": direction,
        "alignment_count": rng.integers(0, 4, n),
        "h1_bias": np.where(rng.random(n) > .5, "BULLISH", "BEARISH"),
        "m15_bias": np.where(rng.random(n) > .5, "BULLISH", "BEARISH"),
        "m5_bias": np.where(rng.random(n) > .5, "BULLISH", "BEARISH"),
        "rsi14": rng.normal(50, 10, n),
        "atr14": rng.uniform(1, 3, n),
        "atr_pct": rng.uniform(.001, .01, n),
        "sweep_atr": rng.uniform(.1, 2, n),
        "sweep_size": rng.uniform(.1, 2, n),
        "body": rng.uniform(.1, 2, n),
        "range": rng.uniform(1, 3, n),
        "upper_wick": rng.uniform(0, 1, n),
        "lower_wick": rng.uniform(0, 1, n),
        "ema20": rng.normal(2000, 10, n),
        "ema50": rng.normal(2000, 10, n),
        "ema200": rng.normal(2000, 10, n),
        "ema20_slope_4h": rng.normal(0, .5, n),
        "macd": rng.normal(0, .5, n),
        "macd_signal": rng.normal(0, .5, n),
        "macd_hist": rng.normal(0, .2, n),
        "return_1h": rng.normal(0, .002, n),
        "return_4h": rng.normal(0, .004, n),
        "return_8h": rng.normal(0, .006, n),
        "atr_change_4h": rng.normal(0, .1, n),
        "risk": rng.uniform(.5, 2, n),
        "risk_atr": rng.uniform(.2, 1.5, n),
        "high": rng.normal(2001, 10, n),
        "low": rng.normal(1999, 10, n),
        "close": rng.normal(2000, 10, n),
        "prev_day_high": rng.normal(2010, 10, n),
        "prev_day_low": rng.normal(1990, 10, n),
        "killzone_name": np.where(np.arange(n) % 3 == 0, "NEW_YORK", "OUTSIDE"),
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
    assert folds["validation_end"].max() < cutoff
    assert set(folds["variant"]) == {"V3_BASE_RF", "V3_2_REFINED_RF"}
    refined_status = summary.loc[summary["variant"] == "V3_2_REFINED_RF", "promotion_status"].iloc[0]
    assert refined_status in {"RESEARCH_ONLY", "PROMOTE_TO_FINAL_FREEZE"}
