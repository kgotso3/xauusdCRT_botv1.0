from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.segmentation import add_research_buckets


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    min_samples_per_split: int = 8

    def validate(self) -> None:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if not np.isclose(total, 1.0):
            raise ValueError("train/validation/test fractions must sum to 1.0")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("all split fractions must be positive")


def _expectancy(hit_rate: float, target_r: float) -> float:
    return float(hit_rate * target_r - (1.0 - hit_rate))


def chronological_split(df: pd.DataFrame, config: SplitConfig | None = None) -> pd.DataFrame:
    cfg = config or SplitConfig()
    cfg.validate()
    out = df.copy()
    out["signal_time_utc"] = pd.to_datetime(out["signal_time_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["signal_time_utc"]).sort_values("signal_time_utc").reset_index(drop=True)
    n = len(out)
    train_end = int(n * cfg.train_fraction)
    validation_end = train_end + int(n * cfg.validation_fraction)
    out["split"] = "TEST"
    out.loc[: train_end - 1, "split"] = "TRAIN"
    out.loc[train_end: validation_end - 1, "split"] = "VALIDATION"
    return out


def _candidate_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {"ALL": pd.Series(True, index=df.index)}

    for kz in ["ASIA", "LONDON", "NEW_YORK"]:
        masks[f"KZ={kz}"] = df["killzone_name"].eq(kz)
        for direction in ["BUY", "SELL"]:
            masks[f"KZ={kz}|DIR={direction}"] = df["killzone_name"].eq(kz) & df["direction"].eq(direction)
            for alignment in [0, 1, 2, 3]:
                masks[f"KZ={kz}|DIR={direction}|ALIGN={alignment}"] = (
                    df["killzone_name"].eq(kz)
                    & df["direction"].eq(direction)
                    & df["alignment_count"].eq(alignment)
                )

    for direction in ["BUY", "SELL"]:
        masks[f"DIR={direction}"] = df["direction"].eq(direction)
    for alignment in [0, 1, 2, 3]:
        masks[f"ALIGN={alignment}"] = df["alignment_count"].eq(alignment)

    return masks


def build_walkforward_report(
    dataset: pd.DataFrame,
    config: SplitConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate V2 candidate filters chronologically for 1R and 1.5R only.

    The candidate rules are deterministic filters. No model selection is made
    on the test split; test metrics are reported only after the same candidate
    definition has been evaluated on train and validation data.
    """
    cfg = config or SplitConfig()
    df = add_research_buckets(dataset)
    df = chronological_split(df, cfg)

    rows: list[dict] = []
    masks = _candidate_masks(df)
    for candidate, mask in masks.items():
        candidate_df = df.loc[mask].copy()
        if candidate_df.empty:
            continue
        row: dict[str, object] = {"candidate": candidate, "samples_total": int(len(candidate_df))}
        stable = True
        for split in ["TRAIN", "VALIDATION", "TEST"]:
            part = candidate_df[candidate_df["split"] == split]
            prefix = split.lower()
            n = len(part)
            row[f"{prefix}_samples"] = int(n)
            if n == 0:
                row[f"{prefix}_hit_1r"] = np.nan
                row[f"{prefix}_hit_1_5r"] = np.nan
                row[f"{prefix}_exp_1r"] = np.nan
                row[f"{prefix}_exp_1_5r"] = np.nan
                stable = False
                continue
            hit_1r = float(part["hit_1_0r"].mean())
            hit_1_5r = float(part["hit_1_5r"].mean())
            row[f"{prefix}_hit_1r"] = hit_1r
            row[f"{prefix}_hit_1_5r"] = hit_1_5r
            row[f"{prefix}_exp_1r"] = _expectancy(hit_1r, 1.0)
            row[f"{prefix}_exp_1_5r"] = _expectancy(hit_1_5r, 1.5)
            if n < cfg.min_samples_per_split:
                stable = False

        row["sample_stable"] = bool(stable)
        if stable:
            row["validation_score"] = float(
                0.65 * row["validation_exp_1_5r"] + 0.35 * row["validation_exp_1r"]
            )
            row["test_positive_both"] = bool(
                row["test_exp_1r"] > 0 and row["test_exp_1_5r"] > 0
            )
        else:
            row["validation_score"] = np.nan
            row["test_positive_both"] = False
        rows.append(row)

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(
            ["sample_stable", "validation_score", "samples_total"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)

    period_rows: list[dict] = []
    df["month"] = df["signal_time_utc"].dt.to_period("M").astype(str)
    df["quarter"] = df["signal_time_utc"].dt.to_period("Q").astype(str)
    for period_type in ["month", "quarter"]:
        for period, part in df.groupby(period_type, observed=True):
            hit_1r = float(part["hit_1_0r"].mean())
            hit_1_5r = float(part["hit_1_5r"].mean())
            period_rows.append({
                "period_type": period_type.upper(),
                "period": period,
                "samples": int(len(part)),
                "hit_1r": hit_1r,
                "hit_1_5r": hit_1_5r,
                "expectancy_1r": _expectancy(hit_1r, 1.0),
                "expectancy_1_5r": _expectancy(hit_1_5r, 1.5),
            })
    periods = pd.DataFrame(period_rows)
    return report, periods
