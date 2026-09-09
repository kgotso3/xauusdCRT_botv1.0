from __future__ import annotations

import numpy as np
import pandas as pd

from features.regime import build_causal_regime_features


V32_NUMERIC_FEATURES = [
    "v32_direction_x_trend_strength",
    "v32_direction_x_momentum_strength",
    "v32_direction_x_liquidity_pressure",
    "v32_direction_x_volatility",
    "v32_alignment_x_trend_strength",
    "v32_alignment_x_momentum",
    "v32_atr_momentum_interaction",
    "v32_sweep_risk_ratio",
    "v32_wick_directional_imbalance",
    "v32_prev_day_position",
    "v32_directional_prev_day_position",
    "v32_trend_transition_score",
    "v32_momentum_transition_score",
    "v32_regime_persistence_3",
]

V32_CATEGORICAL_FEATURES = [
    "v32_direction_trend_state",
    "v32_direction_momentum_state",
    "v32_direction_liquidity_state",
    "v32_regime_transition_state",
]


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def build_causal_v3_2_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Targeted V3.2 features for 1R research only.

    Every feature is derived from the current/past completed-event feature set.
    No future outcome columns are used.
    """
    df = build_causal_regime_features(dataset).copy()

    direction = _num(df, "direction_num").fillna(0.0)
    alignment = _num(df, "alignment_count").fillna(0.0) / 3.0
    trend_strength = _num(df, "regime_trend_strength").fillna(0.0)
    momentum = _num(df, "regime_momentum_strength").fillna(0.0)
    liquidity = _num(df, "regime_liquidity_pressure").fillna(0.0)
    volatility = _num(df, "regime_volatility_score").fillna(0.0)
    atr_change = _num(df, "atr_change_4h").fillna(0.0)
    sweep = _num(df, "sweep_atr")
    risk = _num(df, "risk_atr").replace(0, np.nan)
    wick_imbalance = _num(df, "wick_imbalance").fillna(0.0)

    df["v32_direction_x_trend_strength"] = direction * trend_strength
    df["v32_direction_x_momentum_strength"] = direction * momentum
    df["v32_direction_x_liquidity_pressure"] = direction * liquidity
    df["v32_direction_x_volatility"] = direction * volatility
    df["v32_alignment_x_trend_strength"] = alignment * trend_strength
    df["v32_alignment_x_momentum"] = alignment * momentum
    df["v32_atr_momentum_interaction"] = atr_change * momentum
    df["v32_sweep_risk_ratio"] = sweep / risk
    df["v32_wick_directional_imbalance"] = direction * wick_imbalance

    prev_high = _num(df, "prev_day_high")
    prev_low = _num(df, "prev_day_low")
    close = _num(df, "close")
    prev_range = (prev_high - prev_low).replace(0, np.nan)
    prev_pos = (close - prev_low) / prev_range
    df["v32_prev_day_position"] = prev_pos
    df["v32_directional_prev_day_position"] = direction * (prev_pos - 0.5)

    trend_state = df["regime_trend_state"].astype("string")
    momentum_state = df["regime_momentum_state"].astype("string")
    liquidity_state = df["regime_liquidity_state"].astype("string")

    trend_changed = trend_state.ne(trend_state.shift(1)).astype(float)
    momentum_changed = momentum_state.ne(momentum_state.shift(1)).astype(float)
    df["v32_trend_transition_score"] = trend_changed
    df["v32_momentum_transition_score"] = momentum_changed

    same_regime_1 = df["regime_composite"].astype("string").eq(df["regime_composite"].astype("string").shift(1))
    same_regime_2 = df["regime_composite"].astype("string").eq(df["regime_composite"].astype("string").shift(2))
    df["v32_regime_persistence_3"] = (same_regime_1.astype(float) + same_regime_2.astype(float)) / 2.0

    direction_label = df["direction"].astype("string")
    df["v32_direction_trend_state"] = direction_label + "|" + trend_state
    df["v32_direction_momentum_state"] = direction_label + "|" + momentum_state
    df["v32_direction_liquidity_state"] = direction_label + "|" + liquidity_state
    df["v32_regime_transition_state"] = np.select(
        [trend_changed.eq(1) & momentum_changed.eq(1), trend_changed.eq(1), momentum_changed.eq(1)],
        ["BOTH_CHANGED", "TREND_CHANGED", "MOMENTUM_CHANGED"],
        default="STABLE",
    )
    return df
