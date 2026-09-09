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


def score_shadow_dataset(dataset: pd.DataFrame, spec: ShadowModelSpec) -> pd.DataFrame:
    df = build_causal_regime_features(dataset).sort_values("signal_time_utc").reset_index(drop=True)
    feature_cols = V3_NUMERIC_FEATURES + V3_CATEGORICAL_FEATURES
    path = Path(spec.model_path)
    if not path.exists():
        raise FileNotFoundError(f"Shadow model not found: {path}")

    model = joblib.load(path)
    probability = model.predict_proba(df[feature_cols])[:, 1]
    out = pd.DataFrame({
        "signal_time_utc": pd.to_datetime(df["signal_time_utc"], utc=True, errors="coerce"),
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

    if spec.target_r == 1.0:
        out["target"] = np.where(out["direction"].eq("BUY"), out["entry"] + (out["entry"] - out["stop"]), out["entry"] - (out["stop"] - out["entry"]))
    else:
        risk = (out["entry"] - out["stop"]).abs()
        out["target"] = np.where(out["direction"].eq("BUY"), out["entry"] + spec.target_r * risk, out["entry"] - spec.target_r * risk)
    return out


def upsert_shadow_journal(rows: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    key = ["signal_time_utc", "direction", "target_r"]
    new = rows.copy()
    new["signal_time_utc"] = pd.to_datetime(new["signal_time_utc"], utc=True, errors="coerce")
    if output.exists():
        old = pd.read_csv(output)
        old["signal_time_utc"] = pd.to_datetime(old["signal_time_utc"], utc=True, errors="coerce")
        combined = pd.concat([old.astype(object), new.astype(object)], ignore_index=True, sort=False)
    else:
        combined = new
    combined = combined.sort_values("signal_time_utc").drop_duplicates(subset=key, keep="last").reset_index(drop=True)
    combined.to_csv(output, index=False)
    return combined
