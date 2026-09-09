from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.outcomes import realized_r, target_column


@dataclass(frozen=True)
class ThresholdConfig:
    thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
    min_validation_trades: int = 25
    timeout_r: float = 0.0


def _max_drawdown_r(r: pd.Series) -> float:
    if r.empty:
        return 0.0
    equity = r.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    return float(abs(drawdown.min()))


def threshold_metrics(
    df: pd.DataFrame,
    probability: np.ndarray | pd.Series,
    target_r: float,
    threshold: float,
    timeout_r: float = 0.0,
) -> dict[str, float | int]:
    prob = pd.Series(np.asarray(probability, dtype=float), index=df.index)
    selected = df.loc[prob >= threshold].copy()
    n = len(selected)
    coverage = n / len(df) if len(df) else 0.0
    if n == 0:
        return {
            "threshold": threshold,
            "trades": 0,
            "coverage": coverage,
            "hit_rate": float("nan"),
            "expectancy_r": float("nan"),
            "max_drawdown_r": float("nan"),
            "timeouts": 0,
            "resolved_rate": float("nan"),
        }

    target_col = target_column(target_r)
    r = realized_r(selected, target_r=target_r, timeout_r=timeout_r)
    resolved = selected[target_col].fillna(False).astype(bool) | selected["hit_stop"].fillna(False).astype(bool)
    return {
        "threshold": float(threshold),
        "trades": int(n),
        "coverage": float(coverage),
        "hit_rate": float(selected[target_col].fillna(False).astype(bool).mean()),
        "expectancy_r": float(r.mean()),
        "net_r": float(r.sum()),
        "max_drawdown_r": _max_drawdown_r(r),
        "timeouts": int((~resolved).sum()),
        "resolved_rate": float(resolved.mean()),
    }


def scan_validation_thresholds(
    validation_df: pd.DataFrame,
    validation_probability: np.ndarray,
    target_r: float,
    config: ThresholdConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ThresholdConfig()
    rows = [
        threshold_metrics(
            validation_df,
            validation_probability,
            target_r=target_r,
            threshold=t,
            timeout_r=cfg.timeout_r,
        )
        for t in cfg.thresholds
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["eligible"] = out["trades"] >= cfg.min_validation_trades
    out["score"] = np.where(
        out["eligible"],
        out["expectancy_r"] - 0.02 * out["max_drawdown_r"],
        np.nan,
    )
    return out.sort_values(["eligible", "score", "expectancy_r", "trades"], ascending=[False, False, False, False], na_position="last").reset_index(drop=True)


def choose_threshold(validation_scan: pd.DataFrame) -> float | None:
    eligible = validation_scan[validation_scan["eligible"]].copy()
    if eligible.empty:
        return None
    return float(eligible.iloc[0]["threshold"])
