from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.outcomes import outcome_summary
from features.v2_features import build_causal_v2_features


@dataclass(frozen=True)
class BenchmarkRule:
    name: str
    description: str


def _split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("signal_time_utc").reset_index(drop=True).copy()
    n = len(out)
    train_end = int(n * 0.60)
    validation_end = train_end + int(n * 0.20)
    out["split"] = "TEST"
    out.loc[: train_end - 1, "split"] = "TRAIN"
    out.loc[train_end: validation_end - 1, "split"] = "VALIDATION"
    return out


def _rule_masks(df: pd.DataFrame) -> dict[BenchmarkRule, pd.Series]:
    directional_rsi = (
        (df["direction"].eq("BUY") & df["rsi14"].ge(50))
        | (df["direction"].eq("SELL") & df["rsi14"].le(50))
    )
    return {
        BenchmarkRule("ALL_CRT", "All CRT occurrences"): pd.Series(True, index=df.index),
        BenchmarkRule("KILLZONE_ONLY", "Asia/London/New York killzones only"): df["killzone_name"].ne("OUTSIDE"),
        BenchmarkRule("ALIGN_GE_2", "At least two timeframe biases aligned"): df["alignment_count"].ge(2),
        BenchmarkRule("SWEEP_ATR_GE_020", "Sweep size at least 0.20 ATR"): df["sweep_atr"].ge(0.20),
        BenchmarkRule("DIRECTIONAL_RSI", "BUY RSI>=50 or SELL RSI<=50"): directional_rsi,
        BenchmarkRule("KZ_ALIGN_GE_2", "Killzone and alignment>=2"): df["killzone_name"].ne("OUTSIDE") & df["alignment_count"].ge(2),
        BenchmarkRule("KZ_SWEEP_RSI", "Killzone, sweep>=0.20 ATR, directional RSI"): df["killzone_name"].ne("OUTSIDE") & df["sweep_atr"].ge(0.20) & directional_rsi,
        BenchmarkRule("KZ_ALIGN_SWEEP_RSI", "Killzone, alignment>=2, sweep>=0.20 ATR, directional RSI"): df["killzone_name"].ne("OUTSIDE") & df["alignment_count"].ge(2) & df["sweep_atr"].ge(0.20) & directional_rsi,
    }


def build_deterministic_benchmark(dataset: pd.DataFrame, min_split_samples: int = 8) -> pd.DataFrame:
    """Evaluate transparent CRT rules with explicit timeout accounting.

    Expectancy uses +target R for target hits, -1R for stop hits, and 0R for
    observations where neither target nor stop was reached inside the research
    horizon. This prevents unresolved observations from being silently treated
    as full losses.
    """
    df = _split(build_causal_v2_features(dataset))
    rows: list[dict] = []

    for rule, mask in _rule_masks(df).items():
        subset = df.loc[mask.fillna(False)].copy()
        for target_name, target_r in [("1R", 1.0), ("1.5R", 1.5)]:
            row: dict[str, object] = {
                "rule": rule.name,
                "description": rule.description,
                "target": target_name,
                "samples_total": int(len(subset)),
            }
            stable = True
            for split in ["TRAIN", "VALIDATION", "TEST"]:
                part = subset[subset["split"] == split]
                prefix = split.lower()
                stats = outcome_summary(part, target_r=target_r, timeout_r=0.0) if len(part) else {
                    "samples": 0, "wins": 0, "stops": 0, "timeouts": 0,
                    "resolved_rate": float("nan"), "hit_rate_all": float("nan"),
                    "expectancy_r_timeout_neutral": float("nan"),
                }
                row[f"{prefix}_samples"] = stats["samples"]
                row[f"{prefix}_hit_rate"] = stats["hit_rate_all"]
                row[f"{prefix}_resolved_rate"] = stats["resolved_rate"]
                row[f"{prefix}_timeouts"] = stats["timeouts"]
                row[f"{prefix}_expectancy"] = stats["expectancy_r_timeout_neutral"]
                if stats["samples"] < min_split_samples:
                    stable = False
            row["sample_stable"] = stable
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["target", "sample_stable", "validation_expectancy", "samples_total"],
            ascending=[True, False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return out
