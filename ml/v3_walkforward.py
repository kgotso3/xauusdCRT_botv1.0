from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features.regime import REGIME_CATEGORICAL_FEATURES, REGIME_NUMERIC_FEATURES, build_causal_regime_features
from features.v1_2_features import V12_CATEGORICAL_FEATURES, V12_NUMERIC_FEATURES, V12_TARGETS


V3_NUMERIC_FEATURES = V12_NUMERIC_FEATURES + REGIME_NUMERIC_FEATURES
V3_CATEGORICAL_FEATURES = V12_CATEGORICAL_FEATURES + REGIME_CATEGORICAL_FEATURES
DEFAULT_DEVELOPMENT_CUTOFF = pd.Timestamp("2026-09-01T00:00:00Z")


@dataclass(frozen=True)
class V3WalkForwardConfig:
    train_size: int = 1200
    validation_size: int = 300
    step_size: int = 300
    select_k: int = 32
    random_state: int = 42
    development_cutoff: pd.Timestamp = DEFAULT_DEVELOPMENT_CUTOFF


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric, V3_NUMERIC_FEATURES), ("cat", categorical, V3_CATEGORICAL_FEATURES)], sparse_threshold=0.0)


def _models(random_state: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=600, max_depth=7, min_samples_leaf=10, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=600, max_depth=8, min_samples_leaf=8, class_weight="balanced", random_state=random_state, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=180, learning_rate=0.03, max_depth=2, min_samples_leaf=12, subsample=0.85, random_state=random_state),
    }


def _pipe(estimator: object, cfg: V3WalkForwardConfig) -> Pipeline:
    return Pipeline([("preprocess", _preprocessor()), ("select", SelectKBest(mutual_info_classif, k=cfg.select_k)), ("model", estimator)])


def _metrics(y_true: pd.Series, prob: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    return {
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "positive_rate": float(np.mean(y)),
    }


def iter_walkforward_slices(n: int, cfg: V3WalkForwardConfig):
    train_start = 0
    train_end = cfg.train_size
    fold = 0
    while train_end + cfg.validation_size <= n:
        val_start = train_end
        val_end = val_start + cfg.validation_size
        yield fold, slice(train_start, train_end), slice(val_start, val_end)
        fold += 1
        train_end += cfg.step_size


def run_v3_walkforward(dataset: pd.DataFrame, output_dir: str | Path = "data/research/ml_v3", model_dir: str | Path = "data/models/v3", config: V3WalkForwardConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or V3WalkForwardConfig()
    df = build_causal_regime_features(dataset).sort_values("signal_time_utc").reset_index(drop=True)
    df["signal_time_utc"] = pd.to_datetime(df["signal_time_utc"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(cfg.development_cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    df = df.loc[df["signal_time_utc"] < cutoff].reset_index(drop=True)

    feature_cols = V3_NUMERIC_FEATURES + V3_CATEGORICAL_FEATURES
    out_dir = Path(output_dir); mdl_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True); mdl_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict] = []; summary_rows: list[dict] = []

    for target_name, target_col in V12_TARGETS.items():
        target_df = df.dropna(subset=[target_col]).reset_index(drop=True)
        target_fold_rows = []
        for fold, train_slice, val_slice in iter_walkforward_slices(len(target_df), cfg):
            train = target_df.iloc[train_slice]; val = target_df.iloc[val_slice]
            candidates = []
            for model_name, estimator in _models(cfg.random_state + fold).items():
                pipe = _pipe(estimator, cfg)
                pipe.fit(train[feature_cols], train[target_col].astype(int))
                prob = pipe.predict_proba(val[feature_cols])[:, 1]
                metrics = _metrics(val[target_col], prob)
                fold_rows.append({"target":target_name,"fold":fold,"model":model_name,"train_start":train["signal_time_utc"].iloc[0],"train_end":train["signal_time_utc"].iloc[-1],"validation_start":val["signal_time_utc"].iloc[0],"validation_end":val["signal_time_utc"].iloc[-1],"train_samples":len(train),"validation_samples":len(val),**metrics})
                candidates.append((model_name, pipe, metrics))
            candidates.sort(key=lambda item: (-(item[2]["roc_auc"] if np.isfinite(item[2]["roc_auc"]) else -1.0), item[2]["brier"]))
            best_name, best_pipe, best_metrics = candidates[0]
            model_path = mdl_dir / f"v3_{target_name.replace('.', '_')}_fold{fold}_{best_name}.joblib"
            joblib.dump(best_pipe, model_path)
            target_fold_rows.append({"target":target_name,"fold":fold,"selected_model":best_name,"roc_auc":best_metrics["roc_auc"],"brier":best_metrics["brier"],"model_path":str(model_path)})
        if target_fold_rows:
            selected = pd.DataFrame(target_fold_rows)
            summary_rows.append({"target":target_name,"folds":int(len(selected)),"mean_selected_auc":float(selected["roc_auc"].mean()),"median_selected_auc":float(selected["roc_auc"].median()),"mean_selected_brier":float(selected["brier"].mean()),"positive_auc_folds":int((selected["roc_auc"] > 0.5).sum()),"best_model_mode":selected["selected_model"].mode().iloc[0]})

    folds_df = pd.DataFrame(fold_rows); summary_df = pd.DataFrame(summary_rows)
    folds_df.to_csv(out_dir / "v3_walkforward_folds.csv", index=False)
    summary_df.to_csv(out_dir / "v3_walkforward_summary.csv", index=False)
    return summary_df, folds_df
