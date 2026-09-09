from __future__ import annotations

import pandas as pd

from features.session_liquidity import SESSION_LIQUIDITY_NUMERIC_FEATURES
from features.v3_2_features import V32_CATEGORICAL_FEATURES, V32_NUMERIC_FEATURES, build_causal_v3_2_features

V33_NUMERIC_FEATURES = V32_NUMERIC_FEATURES + SESSION_LIQUIDITY_NUMERIC_FEATURES + [
    "v33_directional_asia_position",
    "v33_directional_london_position",
    "v33_directional_new_york_position",
    "v33_directional_asia_sweep",
    "v33_directional_london_sweep",
    "v33_directional_new_york_sweep",
]

V33_CATEGORICAL_FEATURES = V32_CATEGORICAL_FEATURES


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(float("nan"), index=df.index)


def build_causal_v3_3_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """V3.3 1R features: V3.2 plus completed-session liquidity context."""
    df = build_causal_v3_2_features(dataset).copy()
    direction = _num(df, "direction_num").fillna(0.0)

    for session in ("asia", "london", "new_york"):
        pos = _num(df, f"{session}_prev_range_position")
        high_swept = _num(df, f"{session}_prev_high_swept").fillna(0.0)
        low_swept = _num(df, f"{session}_prev_low_swept").fillna(0.0)
        directional_sweep = high_swept.where(direction < 0, low_swept.where(direction > 0, 0.0))
        df[f"v33_directional_{session}_position"] = direction * (pos - 0.5)
        df[f"v33_directional_{session}_sweep"] = directional_sweep

    return df
