from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def _segment(df: pd.DataFrame, cols: list[str], min_samples: int) -> pd.DataFrame:
    grouped = df.groupby(cols, dropna=False, observed=True)
    rows: list[dict] = []
    for keys, group in grouped:
        if len(group) < min_samples:
            continue
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(cols, keys)}
        row.update({
            "samples": int(len(group)),
            "hit_1r": _safe_rate(group["hit_1_0r"]),
            "hit_1_5r": _safe_rate(group["hit_1_5r"]),
            "hit_2r": _safe_rate(group["hit_2_0r"]),
            "hit_3r": _safe_rate(group["hit_3_0r"]),
            "avg_mfe_r": float(group["mfe_r"].mean()),
            "avg_mae_r": float(group["mae_r"].mean()),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["hit_2r", "samples"], ascending=[False, False]).reset_index(drop=True)
    return out


def add_research_buckets(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    df["rsi_band"] = pd.cut(
        df["rsi14"],
        bins=[-np.inf, 30, 40, 50, 60, 70, np.inf],
        labels=["<30", "30-40", "40-50", "50-60", "60-70", ">=70"],
        right=False,
    )
    df["sweep_atr_band"] = pd.cut(
        df["sweep_atr"],
        bins=[-np.inf, 0.10, 0.20, 0.35, 0.50, np.inf],
        labels=["<0.10", "0.10-0.20", "0.20-0.35", "0.35-0.50", ">=0.50"],
        right=False,
    )
    if "atr_pct" in df.columns and df["atr_pct"].notna().sum() >= 4:
        ranked = df["atr_pct"].rank(method="first")
        df["volatility_regime"] = pd.qcut(
            ranked,
            4,
            labels=["LOW", "MID_LOW", "MID_HIGH", "HIGH"],
        )
    else:
        df["volatility_regime"] = "UNKNOWN"
    return df


def build_segmentation_tables(dataset: pd.DataFrame, min_samples: int = 20) -> dict[str, pd.DataFrame]:
    df = add_research_buckets(dataset)
    ny = df[df["in_ny_08_13"]].copy()

    tables = {
        "ny_hour": _segment(ny, ["ny_hour"], min_samples),
        "direction": _segment(df, ["direction"], min_samples),
        "weekday": _segment(df, ["day_of_week"], min_samples),
        "alignment": _segment(df, ["alignment_count"], min_samples),
        "direction_alignment": _segment(df, ["direction", "alignment_count"], min_samples),
        "rsi_band": _segment(df, ["rsi_band"], min_samples),
        "volatility": _segment(df, ["volatility_regime"], min_samples),
        "sweep_atr": _segment(df, ["sweep_atr_band"], min_samples),
        "ny_hour_direction": _segment(ny, ["ny_hour", "direction"], min_samples),
        "ny_alignment": _segment(ny, ["alignment_count"], min_samples),
        "ny_hour_direction_alignment": _segment(
            ny, ["ny_hour", "direction", "alignment_count"], min_samples
        ),
        "ny_direction_rsi": _segment(ny, ["direction", "rsi_band"], min_samples),
        "ny_direction_sweep_atr": _segment(
            ny, ["direction", "sweep_atr_band"], min_samples
        ),
        "ny_direction_volatility": _segment(
            ny, ["direction", "volatility_regime"], min_samples
        ),
        "ny_hour_alignment": _segment(ny, ["ny_hour", "alignment_count"], min_samples),
    }
    return tables
