from __future__ import annotations

import numpy as np
import pandas as pd

from features.v1_2_features import build_causal_v1_2_features


REGIME_NUMERIC_FEATURES = [
    "regime_trend_strength",
    "regime_momentum_strength",
    "regime_volatility_score",
    "regime_liquidity_pressure",
    "regime_directional_alignment",
]

REGIME_CATEGORICAL_FEATURES = [
    "regime_trend_state",
    "regime_volatility_state",
    "regime_momentum_state",
    "regime_liquidity_state",
    "regime_composite",
]


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _trend_state(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    ema_fast = _num(df, "ema20_50_atr")
    ema_slow = _num(df, "ema50_200_atr")
    slope = _num(df, "ema20_slope_4h_atr")
    strength = (ema_fast.abs() + ema_slow.abs() + slope.abs()) / 3.0

    bullish = (ema_fast > 0) & (ema_slow > 0) & (slope > 0)
    bearish = (ema_fast < 0) & (ema_slow < 0) & (slope < 0)
    state = pd.Series("RANGE", index=df.index, dtype="string")
    state.loc[bullish] = "TREND_UP"
    state.loc[bearish] = "TREND_DOWN"
    state.loc[(~bullish) & (~bearish) & (strength >= strength.rolling(200, min_periods=40).median())] = "TRANSITION"
    return state, strength


def _volatility_state(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    pct = _num(df, "atr_pctile_rolling")
    state = pd.Series("UNKNOWN", index=df.index, dtype="string")
    state.loc[pct.notna() & (pct < 0.25)] = "LOW"
    state.loc[pct.notna() & (pct >= 0.25) & (pct < 0.50)] = "MID_LOW"
    state.loc[pct.notna() & (pct >= 0.50) & (pct < 0.75)] = "MID_HIGH"
    state.loc[pct.notna() & (pct >= 0.75)] = "HIGH"
    return state, pct


def _momentum_state(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    macd = _num(df, "macd_hist_atr")
    r4 = _num(df, "return_4h")
    r8 = _num(df, "return_8h")
    score = macd.fillna(0) + 0.5 * r4.fillna(0) + 0.5 * r8.fillna(0)
    state = pd.Series("FLAT", index=df.index, dtype="string")
    state.loc[score > 0] = "BULLISH"
    state.loc[score < 0] = "BEARISH"
    return state, score


def _liquidity_state(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    sweep = _num(df, "sweep_atr")
    prev_sweep = _num(df, "directional_prev_day_sweep").fillna(0)
    risk = _num(df, "risk_atr")
    pressure = sweep.fillna(0) + 0.5 * prev_sweep + 0.25 * risk.fillna(0)
    state = pd.Series("NORMAL", index=df.index, dtype="string")
    state.loc[pressure >= 1.0] = "ELEVATED"
    state.loc[pressure >= 1.5] = "EXTREME"
    return state, pressure


def build_causal_regime_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Build V3 regime labels from information available at decision time only."""
    df = build_causal_v1_2_features(dataset).copy()

    trend_state, trend_strength = _trend_state(df)
    vol_state, vol_score = _volatility_state(df)
    momentum_state, momentum_strength = _momentum_state(df)
    liquidity_state, liquidity_pressure = _liquidity_state(df)

    direction = _num(df, "direction_num").fillna(0)
    alignment = _num(df, "alignment_count").fillna(0)
    momentum_sign = np.sign(momentum_strength.fillna(0))

    df["regime_trend_state"] = trend_state
    df["regime_volatility_state"] = vol_state
    df["regime_momentum_state"] = momentum_state
    df["regime_liquidity_state"] = liquidity_state
    df["regime_trend_strength"] = trend_strength
    df["regime_momentum_strength"] = momentum_strength
    df["regime_volatility_score"] = vol_score
    df["regime_liquidity_pressure"] = liquidity_pressure
    df["regime_directional_alignment"] = direction * momentum_sign + (alignment / 3.0)
    df["regime_composite"] = (
        trend_state.astype(str)
        + "|" + vol_state.astype(str)
        + "|" + momentum_state.astype(str)
        + "|" + liquidity_state.astype(str)
    )
    return df
