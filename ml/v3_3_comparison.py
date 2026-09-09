from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features.regime import REGIME_CATEGORICAL_FEATURES, REGIME_NUMERIC_FEATURES
from features.v1_2_features import V12_CATEGORICAL_FEATURES, V12_NUMERIC_FEATURES
from features.v3_2_features import V32_CATEGORICAL_FEATURES, V32_NUMERIC_FEATURES, build_causal_v3_2_features
from features.v3_3_features import V33_CATEGORICAL_FEATURES, V33_NUMERIC_FEATURES, build_causal_v3_3_features
from ml.v3_walkforward import DEFAULT_DEVELOPMENT_CUTOFF, V3WalkForwardConfig, iter_walkforward_slices

BASE_NUMERIC = V12_NUMERIC_FEATURES + REGIME_NUMERIC_FEATURES + V32_NUMERIC_FEATURES
BASE_CATEGORICAL = V12_CATEGORICAL_FEATURES + REGIME_CATEGORICAL_FEATURES + V32_CATEGORICAL_FEATURES
REFINED_NUMERIC = V12_NUMERIC_FEATURES + REGIME_NUMERIC_FEATURES + V33_NUMERIC_FEATURES
REFINED_CATEGORICAL = V12_CATEGORICAL_FEATURES + REGIME_CATEGORICAL_FEATURES + V33_CATEGORICAL_FEATURES


@dataclass(frozen=True)
class V33Config:
    train_size: int = 1200
    validation_size: int = 300
    step_size: int = 300
    random_state: int = 42
    development_cutoff: pd.Timestamp = DEFAULT_DEVELOPMENT_CUTOFF


def _pipeline(numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = RandomForestClassifier(
        n_estimators=800,
        max_depth=7,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", pre), ("model", model)])


def _metrics(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    yy = y.astype(int).to_numpy()
    return {
        "roc_auc": float(roc_auc_score(yy, p)) if len(np.unique(yy)) > 1 else float("nan"),
        "brier": float(brier_score_loss(yy, p)),
    }


def run_v3_3_comparison(dataset: pd.DataFrame, output_dir: str | Path = "data/research/ml_v3_3", config: V33Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or V33Config()
    base = build_causal_v3_2_features(dataset).copy()
    refined = build_causal_v3_3_features(dataset).copy()
    cutoff = pd.Timestamp(cfg.development_cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")

    for df in (base, refined):
        df["signal_time_utc"] = pd.to_datetime(df["signal_time_utc"], utc=True, errors="coerce")

    base = base.loc[base["signal_time_utc"] < cutoff].sort_values("signal_time_utc").dropna(subset=["hit_1_0r"]).reset_index(drop=True)
    refined = refined.loc[refined["signal_time_utc"] < cutoff].sort_values("signal_time_utc").dropna(subset=["hit_1_0r"]).reset_index(drop=True)
    if len(base) != len(refined):
        raise ValueError("V3.2 and V3.3 datasets are not aligned")

    required = [c for c in V33_NUMERIC_FEATURES if c.startswith(("asia_", "london_", "new_york_"))]
    missing_all = [c for c in required if c not in refined.columns]
    if missing_all:
        raise ValueError(f"Dataset must be regenerated with session-liquidity fields before V3.3: {missing_all}")

    wf_cfg = V3WalkForwardConfig(
        train_size=cfg.train_size,
        validation_size=cfg.validation_size,
        step_size=cfg.step_size,
        random_state=cfg.random_state,
        development_cutoff=cutoff,
    )
    variants = {
        "V3_2_REFINED_RF": (base, BASE_NUMERIC, BASE_CATEGORICAL),
        "V3_3_SESSION_RF": (refined, REFINED_NUMERIC, REFINED_CATEGORICAL),
    }
    rows: list[dict] = []
    target = "hit_1_0r"

    for fold, train_slice, val_slice in iter_walkforward_slices(len(base), wf_cfg):
        for variant, (df, numeric, categorical) in variants.items():
            train = df.iloc[train_slice]
            val = df.iloc[val_slice]
            pipe = _pipeline(numeric, categorical, cfg.random_state + fold)
            pipe.fit(train[numeric + categorical], train[target].astype(int))
            prob = pipe.predict_proba(val[numeric + categorical])[:, 1]
            rows.append({
                "target": "1R",
                "variant": variant,
                "fold": fold,
                "validation_start": val["signal_time_utc"].iloc[0],
                "validation_end": val["signal_time_utc"].iloc[-1],
                **_metrics(val[target], prob),
            })

    folds = pd.DataFrame(rows)
    summaries: list[dict] = []
    for variant, grp in folds.groupby("variant"):
        auc = grp["roc_auc"].astype(float)
        summaries.append({
            "variant": variant,
            "folds": len(grp),
            "mean_auc": float(auc.mean()),
            "median_auc": float(auc.median()),
            "auc_std": float(auc.std(ddof=1)) if len(auc) > 1 else float("nan"),
            "positive_auc_folds": int((auc > 0.5).sum()),
            "catastrophic_folds": int((auc < 0.45).sum()),
            "mean_brier": float(grp["brier"].mean()),
        })
    summary = pd.DataFrame(summaries).sort_values("variant").reset_index(drop=True)
    base_row = summary.loc[summary["variant"] == "V3_2_REFINED_RF"].iloc[0]
    refined_row = summary.loc[summary["variant"] == "V3_3_SESSION_RF"].iloc[0]
    summary["delta_mean_auc_vs_v32"] = summary["mean_auc"] - float(base_row["mean_auc"])
    summary["delta_median_auc_vs_v32"] = summary["median_auc"] - float(base_row["median_auc"])
    summary["delta_mean_brier_vs_v32"] = summary["mean_brier"] - float(base_row["mean_brier"])

    passed = (
        float(refined_row["mean_auc"]) >= 0.55
        and float(refined_row["median_auc"]) >= 0.55
        and int(refined_row["positive_auc_folds"]) >= 4
        and int(refined_row["catastrophic_folds"]) == 0
        and float(refined_row["auc_std"]) <= 0.08
        and float(refined_row["mean_auc"]) > float(base_row["mean_auc"])
    )
    summary["promotion_status"] = "BASELINE"
    summary.loc[summary["variant"] == "V3_3_SESSION_RF", "promotion_status"] = "PROMOTE_TO_FINAL_FREEZE" if passed else "RESEARCH_ONLY"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out / "v3_3_session_rf_folds.csv", index=False)
    summary.to_csv(out / "v3_3_session_rf_summary.csv", index=False)
    return summary, folds
