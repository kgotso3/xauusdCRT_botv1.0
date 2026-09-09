from __future__ import annotations

import numpy as np
import pandas as pd


def target_column(target_r: float) -> str:
    if target_r == 1.0:
        return "hit_1_0r"
    if target_r == 1.5:
        return "hit_1_5r"
    raise ValueError("target_r must be 1.0 or 1.5")


def resolved_trade_mask(df: pd.DataFrame, target_r: float) -> pd.Series:
    """Rows where either the requested target or the stop was reached.

    Rows where neither target nor stop was reached inside the research horizon
    are timeouts/unresolved observations and must not silently be counted as -1R.
    """
    target_col = target_column(target_r)
    if target_col not in df.columns or "hit_stop" not in df.columns:
        raise ValueError(f"dataset requires {target_col} and hit_stop")
    return df[target_col].fillna(False).astype(bool) | df["hit_stop"].fillna(False).astype(bool)


def realized_r(df: pd.DataFrame, target_r: float, timeout_r: float = 0.0) -> pd.Series:
    """Create an explicit R outcome series.

    Target before stop => +target_r.
    Stop before target => -1R.
    Neither within the horizon => timeout_r (default 0R, conservative neutral).

    This is a research accounting convention, not a live exit rule. A later
    backtest should mark timeouts to the actual horizon-close price.
    """
    target_col = target_column(target_r)
    target_hit = df[target_col].fillna(False).astype(bool)
    stop_hit = df["hit_stop"].fillna(False).astype(bool)
    out = np.full(len(df), float(timeout_r), dtype=float)
    out[stop_hit.to_numpy()] = -1.0
    out[target_hit.to_numpy()] = float(target_r)
    return pd.Series(out, index=df.index, name=f"realized_{str(target_r).replace('.', '_')}r")


def outcome_summary(df: pd.DataFrame, target_r: float, timeout_r: float = 0.0) -> dict[str, float]:
    r = realized_r(df, target_r=target_r, timeout_r=timeout_r)
    resolved = resolved_trade_mask(df, target_r)
    target_col = target_column(target_r)
    return {
        "samples": int(len(df)),
        "wins": int(df[target_col].fillna(False).astype(bool).sum()),
        "stops": int(df["hit_stop"].fillna(False).astype(bool).sum()),
        "timeouts": int((~resolved).sum()),
        "resolved_rate": float(resolved.mean()) if len(df) else float("nan"),
        "hit_rate_all": float(df[target_col].fillna(False).astype(bool).mean()) if len(df) else float("nan"),
        "expectancy_r_timeout_neutral": float(r.mean()) if len(df) else float("nan"),
    }
