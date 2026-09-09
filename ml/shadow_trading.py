from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features.regime import build_causal_regime_features
from ml.v3_walkforward import V3_CATEGORICAL_FEATURES, V3_NUMERIC_FEATURES


@dataclass(frozen=True)
class ShadowModelSpec:
    target: str
    model_path: str
    threshold: float
    target_r: float


def score_shadow_dataset(dataset: pd.DataFrame, spec: ShadowModelSpec, start: pd.Timestamp | None = None) -> pd.DataFrame:
    # Build causal/rolling features over the full history first. Only then apply
    # the shadow cohort start so rolling state is not reset at the boundary.
    df = build_causal_regime_features(dataset).sort_values("signal_time_utc").reset_index(drop=True)
    df["signal_time_utc"] = pd.to_datetime(df["signal_time_utc"], utc=True, errors="coerce")
    if start is not None:
        start = pd.Timestamp(start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        df = df.loc[df["signal_time_utc"] >= start].copy()

    feature_cols = V3_NUMERIC_FEATURES + V3_CATEGORICAL_FEATURES
    path = Path(spec.model_path)
    if not path.exists():
        raise FileNotFoundError(f"Shadow model not found: {path}")
    if df.empty:
        return pd.DataFrame()

    model = joblib.load(path)
    probability = model.predict_proba(df[feature_cols])[:, 1]
    out = pd.DataFrame({
        "signal_time_utc": df["signal_time_utc"],
        "decision_time_utc": pd.to_datetime(df["decision_time_utc"], utc=True, errors="coerce"),
        "direction": df["direction"],
        "entry": pd.to_numeric(df["entry"], errors="coerce"),
        "stop": pd.to_numeric(df["stop"], errors="coerce"),
        "probability": probability,
        "threshold": float(spec.threshold),
        "selected": probability >= spec.threshold,
        "target_r": float(spec.target_r),
        "regime_composite": df["regime_composite"],
        "regime_trend_state": df["regime_trend_state"],
        "regime_volatility_state": df["regime_volatility_state"],
        "regime_momentum_state": df["regime_momentum_state"],
        "regime_liquidity_state": df["regime_liquidity_state"],
        "killzone_name": df.get("killzone_name"),
        "hit_stop": df.get("hit_stop", np.nan),
        "hit_1_0r": df.get("hit_1_0r", np.nan),
        "hit_1_5r": df.get("hit_1_5r", np.nan),
    })
    risk = (out["entry"] - out["stop"]).abs()
    out["target"] = np.where(out["direction"].eq("BUY"), out["entry"] + spec.target_r * risk, out["entry"] - spec.target_r * risk)
    return out.reset_index(drop=True)


def upsert_shadow_journal(rows: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    key = ["signal_time_utc", "direction", "target_r"]
    new = rows.copy()
    if new.empty:
        return pd.read_csv(output) if output.exists() else new
    new["signal_time_utc"] = pd.to_datetime(new["signal_time_utc"], utc=True, errors="coerce")
    if output.exists():
        old = pd.read_csv(output)
        old["signal_time_utc"] = pd.to_datetime(old["signal_time_utc"], utc=True, errors="coerce")
        columns = list(dict.fromkeys([*old.columns, *new.columns]))
        combined = pd.concat([old.reindex(columns=columns).astype(object), new.reindex(columns=columns).astype(object)], ignore_index=True, sort=False)
    else:
        combined = new
    combined = combined.sort_values("signal_time_utc").drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    combined.to_csv(output, index=False)
    return combined
