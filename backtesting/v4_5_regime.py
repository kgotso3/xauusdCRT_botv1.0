from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting.v4_2_causal import _variant_frame
from backtesting.v4_4_robustness import _metrics


@dataclass(frozen=True)
class RegimeConfig:
    train_days: int = 180
    test_days: int = 90
    step_days: int = 60
    min_train_trades: int = 25
    min_test_trades: int = 8
    min_train_expectancy: float = 0.0
    min_train_profit_factor: float = 1.05


def _normalize_h1(h1: pd.DataFrame) -> pd.DataFrame:
    x = h1.copy()
    x["time"] = pd.to_datetime(x["time"], utc=True, errors="coerce")
    x = x.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return x


def _rolling_percentile_last(values: np.ndarray) -> float:
    if len(values) == 0 or np.isnan(values[-1]):
        return np.nan
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid <= values[-1]))


def build_h1_regime_features(h1: pd.DataFrame) -> pd.DataFrame:
    """Build regime features using only information available by each H1 close."""
    x = _normalize_h1(h1)
    prev_close = x["close"].shift(1)
    tr = pd.concat([
        (x["high"] - x["low"]).abs(),
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    ema20 = x["close"].ewm(span=20, adjust=False).mean()
    ema50 = x["close"].ewm(span=50, adjust=False).mean()

    x["bar_close_time_utc"] = x["time"] + pd.to_timedelta(1, unit="h")
    x["atr14"] = atr14
    x["atr_pctile_240"] = atr14.rolling(240, min_periods=60).apply(_rolling_percentile_last, raw=True)
    x["trend_gap_atr"] = (ema20 - ema50) / atr14.replace(0, np.nan)
    x["momentum_4h"] = x["close"].pct_change(4)
    x["momentum_8h"] = x["close"].pct_change(8)
    x["range_atr"] = (x["high"] - x["low"]).abs() / atr14.replace(0, np.nan)

    x["trend_state"] = "RANGE"
    x.loc[(x["trend_gap_atr"] > 0.25) & (x["momentum_4h"] > 0), "trend_state"] = "TREND_UP"
    x.loc[(x["trend_gap_atr"] < -0.25) & (x["momentum_4h"] < 0), "trend_state"] = "TREND_DOWN"

    x["volatility_state"] = "UNKNOWN"
    p = x["atr_pctile_240"]
    x.loc[p.notna() & (p < 0.25), "volatility_state"] = "LOW"
    x.loc[p.notna() & (p >= 0.25) & (p < 0.50), "volatility_state"] = "MID_LOW"
    x.loc[p.notna() & (p >= 0.50) & (p < 0.75), "volatility_state"] = "MID_HIGH"
    x.loc[p.notna() & (p >= 0.75), "volatility_state"] = "HIGH"

    x["range_state"] = "NORMAL"
    x.loc[x["range_atr"] < 0.75, "range_state"] = "COMPRESSED"
    x.loc[x["range_atr"] >= 1.50, "range_state"] = "EXPANDED"

    cols = [
        "time", "bar_close_time_utc", "atr14", "atr_pctile_240", "trend_gap_atr",
        "momentum_4h", "momentum_8h", "range_atr", "trend_state",
        "volatility_state", "range_state",
    ]
    return x[cols].copy()


def attach_regimes(trades: pd.DataFrame, h1: pd.DataFrame, variant: str = "FULL_M5_CAUSAL") -> pd.DataFrame:
    frame = _variant_frame(trades, variant).copy()
    if frame.empty:
        return frame
    frame["signal_close_time_utc"] = pd.to_datetime(frame["signal_close_time_utc"], utc=True, errors="coerce")
    regimes = build_h1_regime_features(h1).sort_values("bar_close_time_utc")
    frame = frame.sort_values("signal_close_time_utc")
    out = pd.merge_asof(
        frame,
        regimes.drop(columns=["time"]),
        left_on="signal_close_time_utc",
        right_on="bar_close_time_utc",
        direction="backward",
        allow_exact_matches=True,
    )
    out["trend_alignment"] = "NEUTRAL"
    out.loc[(out["direction"] == "BULLISH") & (out["trend_state"] == "TREND_UP"), "trend_alignment"] = "ALIGNED"
    out.loc[(out["direction"] == "BEARISH") & (out["trend_state"] == "TREND_DOWN"), "trend_alignment"] = "ALIGNED"
    out.loc[(out["direction"] == "BULLISH") & (out["trend_state"] == "TREND_DOWN"), "trend_alignment"] = "COUNTER"
    out.loc[(out["direction"] == "BEARISH") & (out["trend_state"] == "TREND_UP"), "trend_alignment"] = "COUNTER"
    out["regime_composite"] = out["trend_state"].astype(str) + "|" + out["volatility_state"].astype(str) + "|" + out["range_state"].astype(str)
    return out


def _pf(frame: pd.DataFrame) -> float:
    r = frame["total_r"].astype(float)
    wins = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses <= 0:
        return np.inf if wins > 0 else np.nan
    return wins / losses


def _candidate_rows(train: pd.DataFrame, cfg: RegimeConfig) -> pd.DataFrame:
    rows: list[dict] = []
    dimensions = ["trend_state", "volatility_state", "range_state", "trend_alignment", "regime_composite"]
    for dim in dimensions:
        for value, grp in train.groupby(dim, dropna=False):
            m = _metrics(grp)
            selected = bool(
                m["trades"] >= cfg.min_train_trades
                and m["expectancy_r"] > cfg.min_train_expectancy
                and m["profit_factor"] >= cfg.min_train_profit_factor
            )
            rows.append({"dimension": dim, "value": value, "selected": selected, **m})
    return pd.DataFrame(rows)


def walk_forward_regimes(frame: pd.DataFrame, cfg: RegimeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    x = frame.dropna(subset=["signal_close_time_utc"]).sort_values("signal_close_time_utc").copy()
    start = x["signal_close_time_utc"].min().floor("D")
    end = x["signal_close_time_utc"].max().ceil("D")
    fold_rows: list[dict] = []
    selection_rows: list[dict] = []
    cursor = start
    fold = 0
    while True:
        train_end = cursor + pd.to_timedelta(cfg.train_days, unit="D")
        test_end = train_end + pd.to_timedelta(cfg.test_days, unit="D")
        if train_end >= end:
            break
        train = x.loc[(x["signal_close_time_utc"] >= cursor) & (x["signal_close_time_utc"] < train_end)]
        test = x.loc[(x["signal_close_time_utc"] >= train_end) & (x["signal_close_time_utc"] < test_end)]
        if len(train) >= cfg.min_train_trades and len(test) >= cfg.min_test_trades:
            candidates = _candidate_rows(train, cfg)
            selected = candidates.loc[candidates["selected"]].copy()
            mask = pd.Series(False, index=test.index)
            for _, rule in selected.iterrows():
                mask |= test[rule["dimension"]].eq(rule["value"])
                selection_rows.append({
                    "fold": fold,
                    "train_start": cursor,
                    "train_end": train_end,
                    "test_end": test_end,
                    "dimension": rule["dimension"],
                    "value": rule["value"],
                    "train_trades": int(rule["trades"]),
                    "train_expectancy_r": float(rule["expectancy_r"]),
                    "train_profit_factor": float(rule["profit_factor"]),
                })
            selected_test = test.loc[mask]
            baseline = _metrics(test)
            chosen = _metrics(selected_test)
            fold_rows.append({
                "fold": fold,
                "train_start": cursor,
                "train_end": train_end,
                "test_end": test_end,
                "selected_rules": int(len(selected)),
                **{f"baseline_{k}": v for k, v in baseline.items()},
                **{f"selected_{k}": v for k, v in chosen.items()},
            })
            fold += 1
        cursor = cursor + pd.to_timedelta(cfg.step_days, unit="D")
        if cursor >= end:
            break
    return pd.DataFrame(fold_rows), pd.DataFrame(selection_rows)


def run_v4_5_regime_study(
    trades: pd.DataFrame,
    h1: pd.DataFrame,
    output_dir: str | Path = "data/research/v4_5_regime",
    variant: str = "FULL_M5_CAUSAL",
    config: RegimeConfig | None = None,
):
    cfg = config or RegimeConfig()
    frame = attach_regimes(trades, h1, variant)
    folds, selections = walk_forward_regimes(frame, cfg)

    segment_rows = []
    for dim in ["trend_state", "volatility_state", "range_state", "trend_alignment", "regime_composite"]:
        if dim not in frame.columns:
            continue
        for value, grp in frame.groupby(dim, dropna=False):
            segment_rows.append({"dimension": dim, "value": value, **_metrics(grp)})
    segments = pd.DataFrame(segment_rows)

    if folds.empty:
        summary = pd.DataFrame([{"folds": 0, "positive_selected_fold_share": np.nan, "median_selected_expectancy": np.nan, "aggregate_selected_r": 0.0, "stable": False}])
    else:
        eligible = folds.loc[folds["selected_trades"] >= cfg.min_test_trades]
        positive_share = float((eligible["selected_expectancy_r"] > 0).mean()) if len(eligible) else np.nan
        median_exp = float(eligible["selected_expectancy_r"].median()) if len(eligible) else np.nan
        total_r = float(eligible["selected_total_r"].sum()) if len(eligible) else 0.0
        stable = bool(len(eligible) >= 3 and positive_share >= 0.60 and median_exp > 0)
        summary = pd.DataFrame([{
            "folds": int(len(folds)),
            "eligible_selected_folds": int(len(eligible)),
            "positive_selected_fold_share": positive_share,
            "median_selected_expectancy": median_exp,
            "aggregate_selected_r": total_r,
            "stable": stable,
        }])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "v4_5_regime_trades.csv", index=False)
    segments.to_csv(out / "v4_5_regime_segments.csv", index=False)
    folds.to_csv(out / "v4_5_walk_forward_folds.csv", index=False)
    selections.to_csv(out / "v4_5_selected_rules.csv", index=False)
    summary.to_csv(out / "v4_5_summary.csv", index=False)
    return summary, folds, selections, segments, frame
