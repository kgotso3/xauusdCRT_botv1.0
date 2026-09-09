from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.killzones import add_killzone_columns


BIAS_MAP = {"BEARISH": -1, "NEUTRAL": 0, "BULLISH": 1}
DIR_MAP = {"SELL": -1, "BUY": 1}


def _rolling_percentile(series: pd.Series, window: int = 200) -> pd.Series:
    def pct_rank(values: np.ndarray) -> float:
        if len(values) == 0 or np.isnan(values[-1]):
            return np.nan
        valid = values[~np.isnan(values)]
        if len(valid) < max(20, window // 5):
            return np.nan
        return float(np.mean(valid <= values[-1]))

    return series.rolling(window=window, min_periods=max(20, window // 5)).apply(pct_rank, raw=True)


def build_causal_v2_features(dataset: pd.DataFrame, volatility_window: int = 200) -> pd.DataFrame:
    """Create leakage-safe tabular features from the CRT occurrence dataset.

    The input dataset already contains only information known at decision time,
    except outcome labels. Rolling regimes are calculated using current/past rows
    in chronological order only; no full-sample quantiles are used.
    """
    df = dataset.copy()
    df["signal_time_utc"] = pd.to_datetime(df["signal_time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["signal_time_utc"]).sort_values("signal_time_utc").reset_index(drop=True)
    df = add_killzone_columns(df)

    for col in ["h1_bias", "m15_bias", "m5_bias"]:
        df[f"{col}_num"] = df[col].map(BIAS_MAP).fillna(0).astype(int)
    df["direction_num"] = df["direction"].map(DIR_MAP).fillna(0).astype(int)

    safe_atr = df["atr14"].replace(0, np.nan)
    safe_range = df["candle_range"].replace(0, np.nan)
    df["ema20_50_atr"] = df["ema20_50_distance"] / safe_atr
    df["ema50_200_atr"] = df["ema50_200_distance"] / safe_atr
    df["wick_pct_range"] = df["wick_size"] / safe_range
    df["risk_atr"] = df["risk_distance"] / safe_atr
    df["sweep_pct_range"] = df["sweep_size"] / safe_range

    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    df["upper_wick_pct"] = upper_wick / safe_range
    df["lower_wick_pct"] = lower_wick / safe_range

    df["atr_pctile_rolling"] = _rolling_percentile(df["atr_pct"].astype(float), volatility_window)
    df["volatility_regime_causal"] = pd.cut(
        df["atr_pctile_rolling"],
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["LOW", "MID_LOW", "MID_HIGH", "HIGH"],
        right=False,
    )

    df["ny_hour_sin"] = np.sin(2 * np.pi * df["ny_hour"].astype(float) / 24.0)
    df["ny_hour_cos"] = np.cos(2 * np.pi * df["ny_hour"].astype(float) / 24.0)
    dow = pd.to_datetime(df["signal_time_utc"], utc=True).dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return df


NUMERIC_FEATURES = [
    "direction_num", "alignment_count", "h1_bias_num", "m15_bias_num", "m5_bias_num",
    "rsi14", "atr_pct", "atr_pctile_rolling", "sweep_atr", "sweep_pct_range",
    "body_pct_range", "wick_pct_range", "upper_wick_pct", "lower_wick_pct",
    "ema20_50_atr", "ema50_200_atr", "risk_atr", "ny_hour_sin", "ny_hour_cos",
    "dow_sin", "dow_cos",
]

CATEGORICAL_FEATURES = ["killzone_name"]
TARGETS = {"1R": "hit_1_0r", "1.5R": "hit_1_5r"}
