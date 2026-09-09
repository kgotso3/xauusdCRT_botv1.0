from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from backtesting.killzones import add_killzone_columns
from backtesting.segmentation import add_research_buckets


@dataclass(frozen=True)
class CandidateRule:
    name: str
    description: str
    apply: Callable[[pd.DataFrame], pd.Series]


def default_candidate_rules() -> list[CandidateRule]:
    def base(df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=df.index)

    return [
        CandidateRule("all_crt", "All recorded H1 CRT sweeps", base),
        CandidateRule("ny_08_13", "All CRT sweeps inside NY 08:00-13:00", lambda d: d["in_ny_08_13"].astype(bool)),
        CandidateRule("ny_09_sell", "SELL CRT sweeps at 09:00 New York", lambda d: (d["ny_hour"] == 9) & (d["direction"] == "SELL")),
        CandidateRule("ny_align_2", "NY 08:00-13:00 with alignment_count = 2", lambda d: d["in_ny_08_13"].astype(bool) & (d["alignment_count"] == 2)),
        CandidateRule("ny_09_sell_align_1", "09:00 New York SELL with alignment_count = 1", lambda d: (d["ny_hour"] == 9) & (d["direction"] == "SELL") & (d["alignment_count"] == 1)),
        CandidateRule("ny_12_align_2", "12:00 New York with alignment_count = 2", lambda d: (d["ny_hour"] == 12) & (d["alignment_count"] == 2)),
        CandidateRule("kz_08_10_sell", "Killzone 08:00-10:00 New York SELL", lambda d: d["killzone_NY_08_10"].astype(bool) & (d["direction"] == "SELL")),
        CandidateRule("kz_09_11_sell", "Killzone 09:00-11:00 New York SELL", lambda d: d["killzone_NY_09_11"].astype(bool) & (d["direction"] == "SELL")),
    ]


def _summarize(group: pd.DataFrame) -> dict:
    if group.empty:
        return {
            "samples": 0,
            "hit_1r": 0.0,
            "hit_1_5r": 0.0,
            "expectancy_1r": 0.0,
            "expectancy_1_5r": 0.0,
            "avg_mfe_r": 0.0,
            "avg_mae_r": 0.0,
        }
    hit_1r = float(group["hit_1_0r"].mean())
    hit_1_5r = float(group["hit_1_5r"].mean())
    return {
        "samples": int(len(group)),
        "hit_1r": hit_1r,
        "hit_1_5r": hit_1_5r,
        "expectancy_1r": hit_1r - (1.0 - hit_1r),
        "expectancy_1_5r": hit_1_5r * 1.5 - (1.0 - hit_1_5r),
        "avg_mfe_r": float(group["mfe_r"].mean()),
        "avg_mae_r": float(group["mae_r"].mean()),
    }


def prepare_walk_forward_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    for col in ["signal_time_utc", "decision_time_utc", "entry_time_utc"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    df = add_killzone_columns(df)
    df = add_research_buckets(df)
    if "decision_time_utc" not in df.columns:
        raise ValueError("dataset must contain decision_time_utc")
    df["month"] = df["decision_time_utc"].dt.to_period("M").astype(str)
    df["quarter"] = df["decision_time_utc"].dt.to_period("Q").astype(str)
    return df


def evaluate_candidate_rules(
    dataset: pd.DataFrame,
    min_period_samples: int = 5,
    rules: list[CandidateRule] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare_walk_forward_dataset(dataset)
    rules = rules or default_candidate_rules()
    summary_rows: list[dict] = []
    period_rows: list[dict] = []

    for rule in rules:
        mask = rule.apply(df).fillna(False)
        subset = df[mask].copy()
        overall = _summarize(subset)
        periods = []
        for period_type in ["month", "quarter"]:
            for period, group in subset.groupby(period_type, observed=True):
                stats = _summarize(group)
                stats.update({
                    "rule": rule.name,
                    "description": rule.description,
                    "period_type": period_type,
                    "period": period,
                    "usable_period": stats["samples"] >= min_period_samples,
                })
                period_rows.append(stats)
                if period_type == "month" and stats["samples"] >= min_period_samples:
                    periods.append(stats)

        usable_months = len(periods)
        positive_1_5_months = sum(1 for p in periods if p["expectancy_1_5r"] > 0)
        positive_1r_months = sum(1 for p in periods if p["expectancy_1r"] > 0)
        worst_month_1_5 = min((p["expectancy_1_5r"] for p in periods), default=0.0)

        summary_rows.append({
            "rule": rule.name,
            "description": rule.description,
            **overall,
            "usable_months": usable_months,
            "positive_1r_months": positive_1r_months,
            "positive_1_5r_months": positive_1_5_months,
            "positive_1_5r_month_pct": positive_1_5_months / usable_months if usable_months else 0.0,
            "worst_month_expectancy_1_5r": worst_month_1_5,
        })

    summary = pd.DataFrame(summary_rows)
    periods = pd.DataFrame(period_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["expectancy_1_5r", "positive_1_5r_month_pct", "samples"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return summary, periods
