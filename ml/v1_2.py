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
from sklearn.metrics import brier_score_loss, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features.v1_2_features import (
    V12_CATEGORICAL_FEATURES,
    V12_NUMERIC_FEATURES,
    V12_TARGETS,
    build_causal_v1_2_features,
)


@dataclass(frozen=True)
class MLV12Config:
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    random_state: int = 42
    select_k: int = 28


def chronological_split(df: pd.DataFrame, config: MLV12Config | None = None) -> pd.DataFrame:
    cfg = config or MLV12Config()
    out = df.sort_values("signal_time_utc").reset_index(drop=True).copy()
    n = len(out)
    train_end = int(n * cfg.train_fraction)
    validation_end = train_end + int(n * cfg.validation_fraction)
    out["ml_split"] = "TEST"
    out.loc[: train_end - 1, "ml_split"] = "TRAIN"
    out.loc[train_end: validation_end - 1, "ml_split"] = "VALIDATION"
    return out


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, V12_NUMERIC_FEATURES),
        ("cat", categorical, V12_CATEGORICAL_FEATURES),
    ], sparse_threshold=0.0)


def _models(random_state: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=600,
            max_depth=7,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=600,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=12,
            subsample=0.85,
            random_state=random_state,
        ),
    }


def _metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    pred = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }


def _pipeline(estimator: object, cfg: MLV12Config) -> Pipeline:
    # SelectKBest is fit on TRAIN only inside the pipeline, so feature selection
    # is target-specific without leaking VALIDATION or TEST labels.
    return Pipeline([
        ("preprocess", _preprocessor()),
        ("select", SelectKBest(mutual_info_classif, k=cfg.select_k)),
        ("model", estimator),
    ])


def _candidate_score(metrics: dict[str, float]) -> tuple[float, float]:
    auc = metrics["roc_auc"] if np.isfinite(metrics["roc_auc"]) else -1.0
    return (-auc, metrics["brier"])


def train_ml_v1_2(
    dataset: pd.DataFrame,
    output_dir: str | Path = "data/models",
    report_dir: str | Path = "data/research/ml_v1_2",
    config: MLV12Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or MLV12Config()
    df = chronological_split(build_causal_v1_2_features(dataset), cfg)
    feature_cols = V12_NUMERIC_FEATURES + V12_CATEGORICAL_FEATURES

    model_output = Path(output_dir)
    report_output = Path(report_dir)
    model_output.mkdir(parents=True, exist_ok=True)
    report_output.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict] = []
    selected_rows: list[dict] = []

    for target_name, target_col in V12_TARGETS.items():
        target_df = df.dropna(subset=[target_col]).copy()
        train = target_df[target_df["ml_split"] == "TRAIN"]
        validation = target_df[target_df["ml_split"] == "VALIDATION"]
        test = target_df[target_df["ml_split"] == "TEST"]
        if min(len(train), len(validation), len(test)) == 0:
            continue

        candidates = []
        for model_name, estimator in _models(cfg.random_state).items():
            pipe = _pipeline(estimator, cfg)
            pipe.fit(train[feature_cols], train[target_col].astype(int))
            val_prob = pipe.predict_proba(validation[feature_cols])[:, 1]
            val_metrics = _metrics(validation[target_col], val_prob)
            candidates.append((model_name, pipe, val_metrics))
            candidate_rows.append({
                "target": target_name,
                "model": model_name,
                "train_samples": int(len(train)),
                "validation_samples": int(len(validation)),
                "validation_positive_rate": float(validation[target_col].mean()),
                **{f"validation_{k}": v for k, v in val_metrics.items()},
            })

        candidates.sort(key=lambda item: _candidate_score(item[2]))
        best_name, best_pipe, best_val = candidates[0]
        test_prob = best_pipe.predict_proba(test[feature_cols])[:, 1]
        test_metrics = _metrics(test[target_col], test_prob)

        model_path = model_output / f"crt_v2_{target_name.replace('.', '_')}_v1_2_{best_name}.joblib"
        joblib.dump(best_pipe, model_path)

        selected_rows.append({
            "target": target_name,
            "selected_model": best_name,
            "train_samples": int(len(train)),
            "validation_samples": int(len(validation)),
            "test_samples": int(len(test)),
            "validation_positive_rate": float(validation[target_col].mean()),
            "test_positive_rate": float(test[target_col].mean()),
            **{f"validation_{k}": v for k, v in best_val.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "model_path": str(model_path),
        })

    candidates_df = pd.DataFrame(candidate_rows)
    selected_df = pd.DataFrame(selected_rows)
    candidates_df.to_csv(report_output / "ml_v1_2_candidates.csv", index=False)
    selected_df.to_csv(report_output / "ml_v1_2_selected.csv", index=False)
    return selected_df, candidates_df
