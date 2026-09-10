from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from backtesting.v4_2_causal import _variant_frame


@dataclass(frozen=True)
class RobustnessConfig:
    window_days: int = 90
    step_days: int = 30
    min_trades_per_window: int = 20
    min_positive_window_share: float = 0.60
    min_median_expectancy: float = 0.0
    min_aggregate_profit_factor: float = 1.05


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0, "total_r": 0.0, "expectancy_r": np.nan, "win_rate": np.nan, "profit_factor": np.nan, "max_drawdown_r": 0.0}
    x = df.sort_values("signal_close_time_utc")
    r = x["total_r"].astype(float)
    wins = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    eq = r.cumsum().reset_index(drop=True)
    peak = pd.concat([pd.Series([0.0]), eq]).cummax().iloc[1:].reset_index(drop=True)
    dd = eq - peak
    return {"trades": int(len(r)), "total_r": float(r.sum()), "expectancy_r": float(r.mean()), "win_rate": float((r > 0).mean()), "profit_factor": float(pf), "max_drawdown_r": float(dd.min())}


def rolling_windows(frame: pd.DataFrame, cfg: RobustnessConfig) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    x = frame.copy()
    x["signal_close_time_utc"] = pd.to_datetime(x["signal_close_time_utc"], utc=True, errors="coerce")
    x = x.dropna(subset=["signal_close_time_utc"]).sort_values("signal_close_time_utc")
    start = x["signal_close_time_utc"].min().floor("D")
    end = x["signal_close_time_utc"].max().ceil("D")
    rows = []
    cursor = start
    while cursor < end:
        stop = cursor + pd.to_timedelta(int(cfg.window_days), unit="D")
        sample = x.loc[(x["signal_close_time_utc"] >= cursor) & (x["signal_close_time_utc"] < stop)]
        if len(sample) >= cfg.min_trades_per_window:
            rows.append({"window_start": cursor, "window_end": stop, **_metrics(sample)})
        cursor = cursor + pd.to_timedelta(int(cfg.step_days), unit="D")
    return pd.DataFrame(rows)


def robustness_summary(frame: pd.DataFrame, windows: pd.DataFrame, variant: str, cfg: RobustnessConfig) -> dict:
    agg = _metrics(frame)
    if windows.empty:
        return {"variant": variant, **agg, "windows": 0, "positive_window_share": np.nan, "median_window_expectancy": np.nan, "worst_window_expectancy": np.nan, "stable": False}
    positive_share = float((windows["expectancy_r"] > 0).mean())
    median_exp = float(windows["expectancy_r"].median())
    worst_exp = float(windows["expectancy_r"].min())
    stable = bool(positive_share >= cfg.min_positive_window_share and median_exp > cfg.min_median_expectancy and agg["profit_factor"] >= cfg.min_aggregate_profit_factor)
    return {"variant": variant, **agg, "windows": int(len(windows)), "positive_window_share": positive_share, "median_window_expectancy": median_exp, "worst_window_expectancy": worst_exp, "stable": stable}


def run_v4_4_robustness(trades: pd.DataFrame, output_dir: str | Path = "data/research/v4_4_robustness", config: RobustnessConfig | None = None):
    cfg = config or RobustnessConfig()
    variants = ["RAW_CAUSAL", "MSS_CAUSAL", "MSS_FVG_CAUSAL", "FULL_MODEL_CAUSAL", "FULL_M5_CAUSAL"]
    summaries = []
    window_rows = []
    attribution_rows = []

    previous = None
    for variant in variants:
        frame = _variant_frame(trades, variant)
        windows = rolling_windows(frame, cfg)
        summaries.append(robustness_summary(frame, windows, variant, cfg))
        if not windows.empty:
            w = windows.copy(); w["variant"] = variant; window_rows.append(w)
        if previous is not None:
            attribution_rows.append({
                "from_variant": previous["variant"], "to_variant": variant,
                "trade_change": int(len(frame) - previous["trades"]),
                "expectancy_change": float(_metrics(frame)["expectancy_r"] - previous["expectancy"]),
                "pf_change": float(_metrics(frame)["profit_factor"] - previous["pf"]),
            })
        m = _metrics(frame)
        previous = {"variant": variant, "trades": len(frame), "expectancy": m["expectancy_r"], "pf": m["profit_factor"]}

    full = _variant_frame(trades, "FULL_M5_CAUSAL")
    segment_rows = []
    for dim in ["direction", "ny_hour", "correct_half", "year"]:
        if dim not in full.columns or full.empty:
            continue
        for value, grp in full.groupby(dim, dropna=False):
            segment_rows.append({"dimension": dim, "value": value, **_metrics(grp)})

    summary = pd.DataFrame(summaries)
    windows_all = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()
    attribution = pd.DataFrame(attribution_rows)
    segments = pd.DataFrame(segment_rows)

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "v4_4_robustness_summary.csv", index=False)
    windows_all.to_csv(out / "v4_4_rolling_windows.csv", index=False)
    attribution.to_csv(out / "v4_4_stage_attribution.csv", index=False)
    segments.to_csv(out / "v4_4_full_m5_segments.csv", index=False)
    return summary, windows_all, attribution, segments
