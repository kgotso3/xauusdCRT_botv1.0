from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from features.v2_features import build_causal_v2_features


@dataclass(frozen=True)
class BenchmarkRule:
    name: str
    description: str


def _expectancy(hit_rate: float, target_r: float) -> float:
    return float(hit_rate * target_r - (1.0 - hit_rate))


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
    df = _split(build_causal_v2_features(dataset))
    rows: list[dict] = []

    for rule, mask in _rule_masks(df).items():
        subset = df.loc[mask.fillna(False)].copy()
        for target_name, target_col, target_r in [
            ("1R", "hit_1_0r", 1.0),
            ("1.5R", "hit_1_5r", 1.5),
        ]:
            row: dict[str, object] = {
                "rule": rule.name,
                "description": rule.description,
                "target": target_name,
                "samples_total": int(len(subset)),
            }
            stable = True
            for split in ["TRAIN", "VALIDATION", "TEST"]:
                part = subset[subset["split"] == split]
                n = len(part)
                hit = float(part[target_col].mean()) if n else float("nan")
                prefix = split.lower()
                row[f"{prefix}_samples"] = int(n)
                row[f"{prefix}_hit_rate"] = hit
                row[f"{prefix}_expectancy"] = _expectancy(hit, target_r) if n else float("nan")
                if n < min_split_samples:
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
