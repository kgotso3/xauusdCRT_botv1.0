from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features.v2_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGETS, build_causal_v2_features


@dataclass(frozen=True)
class MLConfig:
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    random_state: int = 42


def chronological_split(df: pd.DataFrame, config: MLConfig | None = None) -> pd.DataFrame:
    cfg = config or MLConfig()
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
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def _models(random_state: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def _metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    pred = (probability >= 0.5).astype(int)
    y = y_true.astype(int).to_numpy()
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }
    result["roc_auc"] = float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else float("nan")
    return result


def train_ml_v1(dataset: pd.DataFrame, output_dir: str | Path = "data/models", config: MLConfig | None = None) -> pd.DataFrame:
    cfg = config or MLConfig()
    df = chronological_split(build_causal_v2_features(dataset), cfg)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for target_name, target_col in TARGETS.items():
        target_df = df.dropna(subset=[target_col]).copy()
        train = target_df[target_df["ml_split"] == "TRAIN"]
        validation = target_df[target_df["ml_split"] == "VALIDATION"]
        test = target_df[target_df["ml_split"] == "TEST"]
        if min(len(train), len(validation), len(test)) == 0:
            continue

        candidates = []
        for model_name, estimator in _models(cfg.random_state).items():
            pipe = Pipeline([
                ("preprocess", _preprocessor()),
                ("model", estimator),
            ])
            pipe.fit(train[feature_cols], train[target_col].astype(int))
            val_prob = pipe.predict_proba(validation[feature_cols])[:, 1]
            val_metrics = _metrics(validation[target_col], val_prob)
            candidates.append((model_name, pipe, val_metrics))

        candidates.sort(key=lambda x: (-(x[2]["roc_auc"] if np.isfinite(x[2]["roc_auc"]) else -1.0), x[2]["brier"]))
        best_name, best_pipe, best_val_metrics = candidates[0]
        test_prob = best_pipe.predict_proba(test[feature_cols])[:, 1]
        test_metrics = _metrics(test[target_col], test_prob)
        model_path = output / f"crt_v2_{target_name.replace('.', '_')}_{best_name}.joblib"
        joblib.dump(best_pipe, model_path)

        row = {
            "target": target_name,
            "selected_model": best_name,
            "train_samples": int(len(train)),
            "validation_samples": int(len(validation)),
            "test_samples": int(len(test)),
            "validation_positive_rate": float(validation[target_col].mean()),
            "test_positive_rate": float(test[target_col].mean()),
            "model_path": str(model_path),
        }
        row.update({f"validation_{k}": v for k, v in best_val_metrics.items()})
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        rows.append(row)

    return pd.DataFrame(rows)
