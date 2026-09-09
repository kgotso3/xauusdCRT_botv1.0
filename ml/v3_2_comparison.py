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

from features.regime import REGIME_CATEGORICAL_FEATURES, REGIME_NUMERIC_FEATURES, build_causal_regime_features
from features.v1_2_features import V12_CATEGORICAL_FEATURES, V12_NUMERIC_FEATURES
from features.v3_2_features import V32_CATEGORICAL_FEATURES, V32_NUMERIC_FEATURES, build_causal_v3_2_features
from ml.v3_walkforward import DEFAULT_DEVELOPMENT_CUTOFF, iter_walkforward_slices, V3WalkForwardConfig


BASE_NUMERIC = V12_NUMERIC_FEATURES + REGIME_NUMERIC_FEATURES
BASE_CATEGORICAL = V12_CATEGORICAL_FEATURES + REGIME_CATEGORICAL_FEATURES
REFINED_NUMERIC = BASE_NUMERIC + V32_NUMERIC_FEATURES
REFINED_CATEGORICAL = BASE_CATEGORICAL + V32_CATEGORICAL_FEATURES


@dataclass(frozen=True)
class V32Config:
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


def _metric_row(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    yy = y.astype(int).to_numpy()
    auc = float(roc_auc_score(yy, p)) if len(np.unique(yy)) > 1 else float("nan")
    return {
        "roc_auc": auc,
        "brier": float(brier_score_loss(yy, p)),
        "positive_rate": float(np.mean(yy)),
    }


def run_v3_2_comparison(dataset: pd.DataFrame, output_dir: str | Path = "data/research/ml_v3_2", config: V32Config | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or V32Config()
    base = build_causal_regime_features(dataset).copy()
    refined = build_causal_v3_2_features(dataset).copy()

    for df in (base, refined):
        df["signal_time_utc"] = pd.to_datetime(df["signal_time_utc"], utc=True, errors="coerce")

    cutoff = pd.Timestamp(cfg.development_cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    base = base.loc[base["signal_time_utc"] < cutoff].sort_values("signal_time_utc").reset_index(drop=True)
    refined = refined.loc[refined["signal_time_utc"] < cutoff].sort_values("signal_time_utc").reset_index(drop=True)

    target = "hit_1_0r"
    base = base.dropna(subset=[target]).reset_index(drop=True)
    refined = refined.dropna(subset=[target]).reset_index(drop=True)
    if len(base) != len(refined):
        raise ValueError("Baseline and refined datasets are not aligned")

    wf_cfg = V3WalkForwardConfig(train_size=cfg.train_size, validation_size=cfg.validation_size, step_size=cfg.step_size, random_state=cfg.random_state, development_cutoff=cutoff)
    rows: list[dict] = []

    variants = {
        "V3_BASE_RF": (base, BASE_NUMERIC, BASE_CATEGORICAL),
        "V3_2_REFINED_RF": (refined, REFINED_NUMERIC, REFINED_CATEGORICAL),
    }

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
                "train_start": train["signal_time_utc"].iloc[0],
                "train_end": train["signal_time_utc"].iloc[-1],
                "validation_start": val["signal_time_utc"].iloc[0],
                "validation_end": val["signal_time_utc"].iloc[-1],
                "train_samples": len(train),
                "validation_samples": len(val),
                **_metric_row(val[target], prob),
            })

    folds = pd.DataFrame(rows)
    summaries = []
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

    base_row = summary.loc[summary["variant"] == "V3_BASE_RF"].iloc[0]
    refined_row = summary.loc[summary["variant"] == "V3_2_REFINED_RF"].iloc[0]
    summary["delta_mean_auc_vs_base"] = summary["mean_auc"] - float(base_row["mean_auc"])
    summary["delta_median_auc_vs_base"] = summary["median_auc"] - float(base_row["median_auc"])
    summary["delta_mean_brier_vs_base"] = summary["mean_brier"] - float(base_row["mean_brier"])

    refined_pass = (
        float(refined_row["mean_auc"]) >= 0.55
        and float(refined_row["median_auc"]) >= 0.55
        and int(refined_row["positive_auc_folds"]) >= 4
        and int(refined_row["catastrophic_folds"]) == 0
        and float(refined_row["auc_std"]) <= 0.08
        and float(refined_row["mean_auc"]) > float(base_row["mean_auc"])
    )
    summary["promotion_status"] = "BASELINE"
    summary.loc[summary["variant"] == "V3_2_REFINED_RF", "promotion_status"] = "PROMOTE_TO_FINAL_FREEZE" if refined_pass else "RESEARCH_ONLY"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out / "v3_2_fixed_rf_folds.csv", index=False)
    summary.to_csv(out / "v3_2_fixed_rf_summary.csv", index=False)
    return summary, folds
