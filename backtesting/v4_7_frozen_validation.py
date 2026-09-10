from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.session_crt_split import SplitTargetConfig
from backtesting.v4_6_entry_retest import EntryRetestConfig, build_entry_retest_trades


FROZEN_MODELS = ("OTE_0_705", "OTE_0_79")


@dataclass(frozen=True)
class FrozenValidationConfig:
    retest_window_bars: int = 24
    max_holding_hours: int = 24
    all_in_cost_r: float = 0.04
    account_risk_fraction: float = 0.0025
    validation_window_days: int = 90
    validation_step_days: int = 60
    min_trades_per_window: int = 8
    monte_carlo_runs: int = 5000
    monte_carlo_seed: int = 42

    def __post_init__(self) -> None:
        if self.retest_window_bars <= 0:
            raise ValueError("retest_window_bars must be positive")
        if self.max_holding_hours <= 0:
            raise ValueError("max_holding_hours must be positive")
        if self.all_in_cost_r < 0:
            raise ValueError("all_in_cost_r must be non-negative")
        if not 0 < self.account_risk_fraction < 1:
            raise ValueError("account_risk_fraction must be between 0 and 1")
        if self.validation_window_days <= 0 or self.validation_step_days <= 0:
            raise ValueError("validation windows must be positive")
        if self.min_trades_per_window <= 0:
            raise ValueError("min_trades_per_window must be positive")
        if self.monte_carlo_runs <= 0:
            raise ValueError("monte_carlo_runs must be positive")


def _metrics(frame: pd.DataFrame, r_col: str = "net_r") -> dict:
    if frame.empty:
        return {"trades": 0, "total_r": 0.0, "expectancy_r": np.nan, "win_rate": np.nan, "profit_factor": np.nan, "max_drawdown_r": 0.0}
    x = frame.sort_values("signal_close_time_utc")
    r = pd.to_numeric(x[r_col], errors="coerce").dropna().reset_index(drop=True)
    if r.empty:
        return {"trades": 0, "total_r": 0.0, "expectancy_r": np.nan, "win_rate": np.nan, "profit_factor": np.nan, "max_drawdown_r": 0.0}
    wins = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    eq = r.cumsum()
    peak = pd.concat([pd.Series([0.0]), eq]).cummax().iloc[1:].reset_index(drop=True)
    dd = eq - peak
    return {
        "trades": int(len(r)),
        "total_r": float(r.sum()),
        "expectancy_r": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(pf),
        "max_drawdown_r": float(dd.min()) if len(dd) else 0.0,
    }


def _validation_windows(frame: pd.DataFrame, cfg: FrozenValidationConfig) -> pd.DataFrame:
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
        stop = cursor + pd.to_timedelta(cfg.validation_window_days, unit="D")
        sample = x.loc[(x["signal_close_time_utc"] >= cursor) & (x["signal_close_time_utc"] < stop)]
        if len(sample) >= cfg.min_trades_per_window:
            rows.append({"window_start": cursor, "window_end": stop, **_metrics(sample)})
        cursor += pd.to_timedelta(cfg.validation_step_days, unit="D")
    return pd.DataFrame(rows)


def _account_curve(frame: pd.DataFrame, cfg: FrozenValidationConfig, starting_equity: float = 100000.0) -> pd.DataFrame:
    x = frame.sort_values("signal_close_time_utc").copy()
    equity = float(starting_equity)
    peak = equity
    rows = []
    for _, row in x.iterrows():
        net_r = float(row["net_r"])
        equity *= 1.0 + cfg.account_risk_fraction * net_r
        peak = max(peak, equity)
        rows.append({
            "signal_close_time_utc": row["signal_close_time_utc"],
            "entry_model": row["entry_model"],
            "net_r": net_r,
            "equity": equity,
            "drawdown_pct": (equity / peak - 1.0) * 100.0,
        })
    return pd.DataFrame(rows)


def _monte_carlo(frame: pd.DataFrame, cfg: FrozenValidationConfig) -> dict:
    r = pd.to_numeric(frame["net_r"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(r) == 0:
        return {"mc_runs": 0, "mc_median_max_dd_r": np.nan, "mc_p95_max_dd_r": np.nan, "mc_p99_max_dd_r": np.nan, "mc_loss_probability": np.nan}
    rng = np.random.default_rng(cfg.monte_carlo_seed)
    max_dd = np.empty(cfg.monte_carlo_runs, dtype=float)
    terminal = np.empty(cfg.monte_carlo_runs, dtype=float)
    for i in range(cfg.monte_carlo_runs):
        sample = rng.choice(r, size=len(r), replace=True)
        eq = np.cumsum(sample)
        peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
        dd = eq - peak
        max_dd[i] = -float(dd.min()) if len(dd) else 0.0
        terminal[i] = float(eq[-1])
    return {
        "mc_runs": int(cfg.monte_carlo_runs),
        "mc_median_max_dd_r": float(np.quantile(max_dd, 0.50)),
        "mc_p95_max_dd_r": float(np.quantile(max_dd, 0.95)),
        "mc_p99_max_dd_r": float(np.quantile(max_dd, 0.99)),
        "mc_loss_probability": float((terminal <= 0).mean()),
    }


def _segment_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if frame.empty:
        return pd.DataFrame()
    x = frame.copy()
    x["signal_close_time_utc"] = pd.to_datetime(x["signal_close_time_utc"], utc=True, errors="coerce")
    x["year"] = x["signal_close_time_utc"].dt.year
    x["quarter"] = x["signal_close_time_utc"].dt.to_period("Q").astype(str)
    for dim in ["entry_model", "direction", "ny_hour", "year", "quarter"]:
        if dim not in x.columns:
            continue
        for value, grp in x.groupby(dim, dropna=False):
            rows.append({"dimension": dim, "value": value, **_metrics(grp)})
    return pd.DataFrame(rows)


def _cost_sensitivity(frame: pd.DataFrame, costs: tuple[float, ...] = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)) -> pd.DataFrame:
    rows = []
    for model, grp in frame.groupby("entry_model", sort=False):
        for cost in costs:
            sample = grp.copy()
            sample["net_r"] = pd.to_numeric(sample["gross_r"], errors="coerce") - float(cost)
            rows.append({"entry_model": model, "cost_r": cost, **_metrics(sample)})
    return pd.DataFrame(rows)


def run_v4_7_frozen_validation(
    v4_2_trades: pd.DataFrame,
    m5: pd.DataFrame,
    output_dir: str | Path = "data/research/v4_7_frozen_validation",
    config: FrozenValidationConfig | None = None,
):
    cfg = config or FrozenValidationConfig()
    retest_cfg = EntryRetestConfig(retest_window_bars=cfg.retest_window_bars, ote_levels=(0.705, 0.79))
    m5_holding_bars = int(cfg.max_holding_hours * 60 / 5)
    split_cfg = SplitTargetConfig(max_holding_bars=m5_holding_bars)
    rebuilt = build_entry_retest_trades(v4_2_trades, m5, retest_cfg, split_cfg)
    filled = rebuilt.loc[
        rebuilt["entry_model"].isin(FROZEN_MODELS)
        & rebuilt["fill_status"].eq("FILLED")
        & rebuilt.get("total_r", pd.Series(index=rebuilt.index, dtype=float)).notna()
    ].copy()
    filled["signal_close_time_utc"] = pd.to_datetime(filled["signal_close_time_utc"], utc=True, errors="coerce")
    filled["gross_r"] = pd.to_numeric(filled["total_r"], errors="coerce")
    filled["cost_r"] = float(cfg.all_in_cost_r)
    filled["net_r"] = filled["gross_r"] - filled["cost_r"]

    summary_rows = []
    window_rows = []
    mc_rows = []
    curves = []
    for model in FROZEN_MODELS:
        grp = filled.loc[filled["entry_model"].eq(model)].copy()
        windows = _validation_windows(grp, cfg)
        positive_share = float((windows["expectancy_r"] > 0).mean()) if not windows.empty else np.nan
        median_exp = float(windows["expectancy_r"].median()) if not windows.empty else np.nan
        m = _metrics(grp)
        frozen_pass = bool(
            m["trades"] >= 50
            and m["expectancy_r"] > 0
            and m["profit_factor"] >= 1.15
            and not np.isnan(positive_share)
            and positive_share >= 0.60
            and median_exp > 0
        )
        summary_rows.append({
            "entry_model": model,
            **m,
            "validation_windows": int(len(windows)),
            "positive_window_share": positive_share,
            "median_window_expectancy": median_exp,
            "frozen_validation_pass": frozen_pass,
        })
        if not windows.empty:
            w = windows.copy(); w["entry_model"] = model; window_rows.append(w)
        mc_rows.append({"entry_model": model, **_monte_carlo(grp, cfg)})
        curve = _account_curve(grp, cfg)
        if not curve.empty:
            curves.append(curve)

    summary = pd.DataFrame(summary_rows)
    windows_all = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()
    monte_carlo = pd.DataFrame(mc_rows)
    segments = _segment_report(filled)
    sensitivity = _cost_sensitivity(filled)
    account = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(out / "v4_7_all_retest_setups.csv", index=False)
    filled.to_csv(out / "v4_7_frozen_trades.csv", index=False)
    summary.to_csv(out / "v4_7_summary.csv", index=False)
    windows_all.to_csv(out / "v4_7_validation_windows.csv", index=False)
    monte_carlo.to_csv(out / "v4_7_monte_carlo.csv", index=False)
    segments.to_csv(out / "v4_7_segments.csv", index=False)
    sensitivity.to_csv(out / "v4_7_cost_sensitivity.csv", index=False)
    account.to_csv(out / "v4_7_account_curve.csv", index=False)
    return summary, windows_all, monte_carlo, segments, sensitivity, filled
