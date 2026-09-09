from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PromotionGate:
    min_mean_auc: float = 0.55
    min_median_auc: float = 0.55
    min_positive_folds: int = 4
    max_catastrophic_folds: int = 0
    catastrophic_auc: float = 0.45
    max_auc_std: float = 0.08


def _normal_ci(values: pd.Series, z: float = 1.96) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(x.mean())
    if n == 1:
        return mean, mean
    se = float(x.std(ddof=1)) / sqrt(n)
    return mean - z * se, mean + z * se


def selected_fold_table(folds: pd.DataFrame) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()
    work = folds.copy()
    work["roc_auc"] = pd.to_numeric(work["roc_auc"], errors="coerce")
    work["brier"] = pd.to_numeric(work["brier"], errors="coerce")
    work = work.sort_values(["target", "fold", "roc_auc", "brier"], ascending=[True, True, False, True])
    return work.groupby(["target", "fold"], as_index=False, sort=True).first()


def candidate_consistency(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if folds.empty:
        return pd.DataFrame()
    for (target, model), grp in folds.groupby(["target", "model"], dropna=False):
        auc = pd.to_numeric(grp["roc_auc"], errors="coerce")
        brier = pd.to_numeric(grp["brier"], errors="coerce")
        lo, hi = _normal_ci(auc)
        rows.append({
            "target": target,
            "model": model,
            "folds": int(len(grp)),
            "mean_auc": float(auc.mean()),
            "median_auc": float(auc.median()),
            "auc_std": float(auc.std(ddof=1)) if len(auc) > 1 else 0.0,
            "auc_ci95_low": lo,
            "auc_ci95_high": hi,
            "positive_auc_folds": int((auc > 0.5).sum()),
            "catastrophic_folds": int((auc < 0.45).sum()),
            "mean_brier": float(brier.mean()),
        })
    return pd.DataFrame(rows).sort_values(["target", "mean_auc"], ascending=[True, False]).reset_index(drop=True)


def selected_stability(folds: pd.DataFrame, gate: PromotionGate | None = None) -> pd.DataFrame:
    cfg = gate or PromotionGate()
    selected = selected_fold_table(folds)
    rows: list[dict] = []
    if selected.empty:
        return pd.DataFrame()
    for target, grp in selected.groupby("target", dropna=False):
        auc = pd.to_numeric(grp["roc_auc"], errors="coerce")
        brier = pd.to_numeric(grp["brier"], errors="coerce")
        lo, hi = _normal_ci(auc)
        mean_auc = float(auc.mean())
        median_auc = float(auc.median())
        auc_std = float(auc.std(ddof=1)) if len(auc) > 1 else 0.0
        positive = int((auc > 0.5).sum())
        catastrophic = int((auc < cfg.catastrophic_auc).sum())
        pass_gate = (
            mean_auc >= cfg.min_mean_auc
            and median_auc >= cfg.min_median_auc
            and positive >= cfg.min_positive_folds
            and catastrophic <= cfg.max_catastrophic_folds
            and auc_std <= cfg.max_auc_std
        )
        status = "PROMOTE_TO_FINAL_FREEZE" if pass_gate else "RESEARCH_ONLY"
        rows.append({
            "target": target,
            "folds": int(len(grp)),
            "mean_selected_auc": mean_auc,
            "median_selected_auc": median_auc,
            "selected_auc_std": auc_std,
            "auc_ci95_low": lo,
            "auc_ci95_high": hi,
            "mean_selected_brier": float(brier.mean()),
            "positive_auc_folds": positive,
            "catastrophic_folds": catastrophic,
            "best_model_mode": grp["model"].mode().iloc[0],
            "gate_mean_auc": cfg.min_mean_auc,
            "gate_median_auc": cfg.min_median_auc,
            "gate_min_positive_folds": cfg.min_positive_folds,
            "gate_catastrophic_auc": cfg.catastrophic_auc,
            "gate_max_catastrophic_folds": cfg.max_catastrophic_folds,
            "gate_max_auc_std": cfg.max_auc_std,
            "promotion_status": status,
        })
    return pd.DataFrame(rows).sort_values("target").reset_index(drop=True)


def regime_breakdown(dataset: pd.DataFrame, development_cutoff: str | pd.Timestamp = "2026-09-01T00:00:00Z", min_samples: int = 30) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    df = dataset.copy()
    df["signal_time_utc"] = pd.to_datetime(df["signal_time_utc"], utc=True, errors="coerce")
    cutoff = pd.Timestamp(development_cutoff)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    df = df.loc[df["signal_time_utc"] < cutoff].copy()

    regime_cols = [
        "regime_trend_state",
        "regime_volatility_state",
        "regime_momentum_state",
        "regime_liquidity_state",
        "regime_composite",
    ]
    target_cols = {"1R": "hit_1_0r", "1.5R": "hit_1_5r"}
    rows: list[dict] = []
    for target, target_col in target_cols.items():
        if target_col not in df.columns:
            continue
        for regime_col in regime_cols:
            if regime_col not in df.columns:
                continue
            for regime_value, grp in df.dropna(subset=[target_col]).groupby(regime_col, dropna=False):
                n = len(grp)
                if n < min_samples:
                    continue
                hit = float(pd.to_numeric(grp[target_col], errors="coerce").mean())
                rows.append({
                    "target": target,
                    "regime_dimension": regime_col,
                    "regime_value": regime_value,
                    "samples": int(n),
                    "hit_rate": hit,
                    "naive_expectancy_r": hit * (1.0 if target == "1R" else 1.5) - (1.0 - hit),
                })
    return pd.DataFrame(rows).sort_values(["target", "regime_dimension", "samples"], ascending=[True, True, False]).reset_index(drop=True)


def write_v3_1_reports(
    folds: pd.DataFrame,
    regime_dataset: pd.DataFrame | None = None,
    output_dir: str | Path = "data/research/ml_v3_1",
    gate: PromotionGate | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = selected_fold_table(folds)
    candidates = candidate_consistency(folds)
    stability = selected_stability(folds, gate=gate)
    regimes = regime_breakdown(regime_dataset) if regime_dataset is not None else pd.DataFrame()

    selected.to_csv(out / "v3_1_selected_folds.csv", index=False)
    candidates.to_csv(out / "v3_1_candidate_consistency.csv", index=False)
    stability.to_csv(out / "v3_1_promotion_gate.csv", index=False)
    regimes.to_csv(out / "v3_1_regime_breakdown.csv", index=False)
    return stability, candidates, selected, regimes
