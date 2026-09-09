from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from features.v1_2_features import V12_CATEGORICAL_FEATURES, V12_NUMERIC_FEATURES, V12_TARGETS, build_causal_v1_2_features
from features.v2_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGETS, build_causal_v2_features


@dataclass(frozen=True)
class FrozenModelSpec:
    version: str
    target: str
    model_path: str
    threshold: float = 0.50


DEFAULT_CUTOFF = pd.Timestamp("2026-09-01T00:00:00Z")
DEFAULT_MODELS = (
    FrozenModelSpec("V1", "1R", "data/models/crt_v2_1R_random_forest.joblib", 0.50),
    FrozenModelSpec("V1", "1.5R", "data/models/crt_v2_1_5R_random_forest.joblib", 0.55),
    FrozenModelSpec("V1.2", "1R", "data/models/crt_v2_1R_v1_2_extra_trees.joblib", 0.50),
    FrozenModelSpec("V1.2", "1.5R", "data/models/crt_v2_1_5R_v1_2_extra_trees.joblib", 0.50),
)


def _target_col(target: str) -> str:
    mapping = {**TARGETS, **V12_TARGETS}
    if target not in mapping:
        raise ValueError(f"Unsupported target: {target}")
    return mapping[target]


def _target_final_col(target: str) -> str:
    return "outcome_final_1_0r" if target == "1R" else "outcome_final_1_5r"


def _features_for_version(dataset: pd.DataFrame, version: str) -> tuple[pd.DataFrame, list[str]]:
    if version == "V1":
        return build_causal_v2_features(dataset), NUMERIC_FEATURES + CATEGORICAL_FEATURES
    if version == "V1.2":
        return build_causal_v1_2_features(dataset), V12_NUMERIC_FEATURES + V12_CATEGORICAL_FEATURES
    raise ValueError(f"Unsupported frozen model version: {version}")


def score_forward_occurrences(
    dataset: pd.DataFrame,
    specs: tuple[FrozenModelSpec, ...] = DEFAULT_MODELS,
    cutoff: pd.Timestamp = DEFAULT_CUTOFF,
) -> pd.DataFrame:
    raw = dataset.copy()
    raw["signal_time_utc"] = pd.to_datetime(raw["signal_time_utc"], utc=True, errors="coerce")
    raw["decision_time_utc"] = pd.to_datetime(raw["decision_time_utc"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["signal_time_utc", "decision_time_utc"]).sort_values("signal_time_utc").reset_index(drop=True)

    feature_cache: dict[str, tuple[pd.DataFrame, list[str]]] = {}
    rows: list[dict] = []

    for spec in specs:
        model_path = Path(spec.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Frozen model not found: {model_path}. Train/freeze it before forward validation.")

        if spec.version not in feature_cache:
            feature_cache[spec.version] = _features_for_version(raw, spec.version)
        features, feature_cols = feature_cache[spec.version]
        forward = features.loc[features["signal_time_utc"] >= cutoff].copy()
        if forward.empty:
            continue

        model = joblib.load(model_path)
        probabilities = model.predict_proba(forward[feature_cols])[:, 1]
        target_col = _target_col(spec.target)
        final_col = _target_final_col(spec.target)

        for (_, row), prob in zip(forward.iterrows(), probabilities):
            outcome_final = bool(row.get(final_col, False)) if pd.notna(row.get(final_col, np.nan)) else False
            label = row.get(target_col, np.nan) if outcome_final else np.nan
            hit_stop = row.get("hit_stop", np.nan) if outcome_final else np.nan
            rows.append({
                "model_version": spec.version,
                "target": spec.target,
                "model_path": str(model_path),
                "threshold_frozen": float(spec.threshold),
                "signal_time_utc": row["signal_time_utc"],
                "decision_time_utc": row["decision_time_utc"],
                "direction": row.get("direction"),
                "killzone_name": row.get("killzone_name"),
                "probability": float(prob),
                "selected_at_threshold": bool(prob >= spec.threshold),
                "outcome_final": outcome_final,
                "label": label,
                "hit_stop": hit_stop,
                "ambiguous_intrabar": row.get("ambiguous_intrabar", np.nan),
                "outcome_bars_observed": row.get("outcome_bars_observed", np.nan),
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["signal_time_utc", "model_version", "target"]).reset_index(drop=True)
    return out


def _normalize_journal_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["signal_time_utc"] = pd.to_datetime(out["signal_time_utc"], utc=True, errors="coerce")
    out["decision_time_utc"] = pd.to_datetime(out["decision_time_utc"], utc=True, errors="coerce")
    for col in ("model_version", "target", "direction"):
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
    return out


def _concat_journal_frames(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Concatenate journal frames without pandas all-NA dtype inference warnings."""
    if old.empty:
        return new.copy()
    if new.empty:
        return old.copy()

    columns = list(dict.fromkeys([*old.columns, *new.columns]))
    old = old.reindex(columns=columns)
    new = new.reindex(columns=columns)

    # Object dtype is intentional for the intermediate merge. The journal keys
    # are normalized immediately afterwards and numeric metrics are coerced by
    # the reporting layer when consumed.
    old = old.astype(object)
    new = new.astype(object)
    return pd.concat([old, new], ignore_index=True, sort=False)


def upsert_prediction_journal(new_rows: pd.DataFrame, journal_path: str | Path) -> pd.DataFrame:
    path = Path(journal_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = ["model_version", "target", "signal_time_utc", "direction"]

    normalized_new = _normalize_journal_keys(new_rows)
    if path.exists():
        old = _normalize_journal_keys(pd.read_csv(path))
        combined = _concat_journal_frames(old, normalized_new)
    else:
        combined = normalized_new.copy()

    if not combined.empty:
        combined = _normalize_journal_keys(combined)
        combined = combined.sort_values("signal_time_utc")
        combined = combined.drop_duplicates(subset=key, keep="last").reset_index(drop=True)
        combined.to_csv(path, index=False)
    return combined


def _expectancy(selected: pd.DataFrame, target_r: float) -> tuple[float, float]:
    if selected.empty:
        return float("nan"), float("nan")
    label = pd.to_numeric(selected["label"], errors="coerce")
    stop = selected["hit_stop"].astype("boolean")
    resolved = label.notna() & stop.notna()
    if not resolved.any():
        return float("nan"), float("nan")
    r = np.where(label[resolved].astype(bool), target_r, np.where(stop[resolved].astype(bool), -1.0, 0.0))
    return float(np.mean(r)), float(np.sum(r))


def forward_metrics(journal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if journal.empty:
        return pd.DataFrame()

    work = journal.copy()
    work["label"] = pd.to_numeric(work["label"], errors="coerce")

    for (version, target), grp in work.groupby(["model_version", "target"], dropna=False):
        labelled = grp.dropna(subset=["label"]).copy()
        y = labelled["label"].astype(int).to_numpy() if not labelled.empty else np.array([], dtype=int)
        p = labelled["probability"].astype(float).to_numpy() if not labelled.empty else np.array([], dtype=float)
        threshold = float(grp["threshold_frozen"].iloc[0])
        selected = labelled[labelled["probability"] >= threshold].copy()
        target_r = 1.0 if target == "1R" else 1.5
        expectancy_r, net_r = _expectancy(selected, target_r)

        auc = float(roc_auc_score(y, p)) if len(y) and len(np.unique(y)) > 1 else float("nan")
        brier = float(brier_score_loss(y, p)) if len(y) else float("nan")
        hit_rate = float(selected["label"].mean()) if len(selected) else float("nan")

        rows.append({
            "model_version": version,
            "target": target,
            "forward_predictions": int(len(grp)),
            "pending_predictions": int(grp["label"].isna().sum()),
            "labelled_predictions": int(len(labelled)),
            "positive_rate": float(labelled["label"].mean()) if len(labelled) else float("nan"),
            "roc_auc": auc,
            "brier": brier,
            "threshold_frozen": threshold,
            "selected_trades": int(len(selected)),
            "selected_coverage": float(len(selected) / len(labelled)) if len(labelled) else float("nan"),
            "selected_hit_rate": hit_rate,
            "selected_expectancy_r": expectancy_r,
            "selected_net_r": net_r,
        })

    return pd.DataFrame(rows).sort_values(["target", "model_version"]).reset_index(drop=True)
