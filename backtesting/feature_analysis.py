from __future__ import annotations

import numpy as np
import pandas as pd

from features.v2_features import NUMERIC_FEATURES, TARGETS, build_causal_v2_features


def _expectancy(hit_rate: float, target_r: float) -> float:
    return float(hit_rate * target_r - (1.0 - hit_rate))


def chronological_development_split(df: pd.DataFrame, development_fraction: float = 0.80) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("signal_time_utc").reset_index(drop=True)
    cut = int(len(ordered) * development_fraction)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def analyze_features(dataset: pd.DataFrame, min_bin_samples: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build_causal_v2_features(dataset)
    development, _ = chronological_development_split(df)
    importance_rows: list[dict] = []
    bin_rows: list[dict] = []

    for target_name, target_col in TARGETS.items():
        target_r = 1.0 if target_name == "1R" else 1.5
        for feature in NUMERIC_FEATURES:
            if feature not in development.columns:
                continue
            pair = development[[feature, target_col]].dropna()
            if len(pair) < min_bin_samples or pair[feature].nunique() < 2:
                continue
            corr = pair[feature].corr(pair[target_col].astype(float), method="spearman")
            importance_rows.append({
                "target": target_name,
                "feature": feature,
                "samples": int(len(pair)),
                "spearman": float(corr) if pd.notna(corr) else 0.0,
                "abs_spearman": abs(float(corr)) if pd.notna(corr) else 0.0,
            })
            try:
                buckets = pd.qcut(pair[feature], q=5, duplicates="drop")
            except ValueError:
                continue
            tmp = pair.assign(bucket=buckets)
            for bucket, group in tmp.groupby("bucket", observed=True):
                if len(group) < min_bin_samples:
                    continue
                hit = float(group[target_col].mean())
                bin_rows.append({
                    "target": target_name,
                    "feature": feature,
                    "bucket": str(bucket),
                    "samples": int(len(group)),
                    "hit_rate": hit,
                    "expectancy": _expectancy(hit, target_r),
                })

    importance = pd.DataFrame(importance_rows)
    bins = pd.DataFrame(bin_rows)
    if not importance.empty:
        importance = importance.sort_values(["target", "abs_spearman"], ascending=[True, False]).reset_index(drop=True)
    if not bins.empty:
        bins = bins.sort_values(["target", "expectancy", "samples"], ascending=[True, False, False]).reset_index(drop=True)
    return importance, bins
