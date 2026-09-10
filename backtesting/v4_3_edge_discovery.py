from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from backtesting.v4_2_causal import _variant_frame


@dataclass(frozen=True)
class EdgeDiscoveryConfig:
    holdout_start: str = "2026-07-01"
    min_dev_trades: int = 50
    min_dev_expectancy: float = 0.0
    min_dev_profit_factor: float = 1.10
    min_positive_years: int = 2
    quantile_bins: int = 4


def _pf(x: pd.Series) -> float:
    wins = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    if losses <= 0:
        return np.inf if wins > 0 else np.nan
    return wins / losses


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0, "total_r": 0.0, "expectancy_r": np.nan, "win_rate": np.nan, "profit_factor": np.nan, "max_drawdown_r": 0.0}
    r = df["total_r"].astype(float)
    eq = r.cumsum()
    dd = eq - eq.cummax()
    return {
        "trades": int(len(df)),
        "total_r": float(r.sum()),
        "expectancy_r": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(_pf(r)),
        "max_drawdown_r": float(dd.min()),
    }


def enrich_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["signal_close_time_utc"] = pd.to_datetime(out["signal_close_time_utc"], utc=True, errors="coerce")
    out["range_size"] = (out["range_high"] - out["range_low"]).abs()
    out["stop_distance"] = (out["h1_entry"] - out["stop"]).abs()
    out["sweep_depth"] = np.where(out["direction"].eq("BULLISH"), out["range_low"] - out["stop"], out["stop"] - out["range_high"])
    for prefix in ["m15_mss", "m15_fvg", "m5_fvg"]:
        tcol = prefix + "_time"
        if tcol in out.columns:
            ts = pd.to_datetime(out[tcol], utc=True, errors="coerce")
            out[prefix + "_delay_min"] = (ts - out["signal_close_time_utc"]).dt.total_seconds() / 60.0
    if "m15_fvg_entry" in out.columns:
        out["m15_entry_displacement"] = (out["m15_fvg_entry"] - out["h1_entry"]).abs()
    if "m5_fvg_entry" in out.columns:
        out["m5_entry_displacement"] = (out["m5_fvg_entry"] - out["h1_entry"]).abs()
    return out


def _quantile_edges(dev: pd.Series, q: int) -> list[float]:
    s = pd.to_numeric(dev, errors="coerce").dropna()
    if s.nunique() < 2:
        return []
    qs = np.linspace(0, 1, q + 1)
    vals = sorted(set(float(v) for v in s.quantile(qs).tolist()))
    if len(vals) < 3:
        return []
    vals[0], vals[-1] = -np.inf, np.inf
    return vals


def _positive_years(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    years = df.assign(_year=df["signal_close_time_utc"].dt.year).groupby("_year")["total_r"].mean()
    return int((years > 0).sum())


def discover_edges(trades: pd.DataFrame, variant: str = "FULL_M5_CAUSAL", config: EdgeDiscoveryConfig | None = None):
    cfg = config or EdgeDiscoveryConfig()
    frame = enrich_features(_variant_frame(trades, variant))
    cutoff = pd.Timestamp(cfg.holdout_start, tz="UTC")
    dev = frame.loc[frame["signal_close_time_utc"] < cutoff].copy()
    holdout = frame.loc[frame["signal_close_time_utc"] >= cutoff].copy()

    candidates: list[dict] = []

    def add_rule(name: str, mask: pd.Series, rule_type: str, feature: str, value: str):
        sample = dev.loc[mask.fillna(False)].copy()
        m = _metrics(sample)
        pos_years = _positive_years(sample)
        selected = bool(m["trades"] >= cfg.min_dev_trades and m["expectancy_r"] > cfg.min_dev_expectancy and m["profit_factor"] >= cfg.min_dev_profit_factor and pos_years >= cfg.min_positive_years)
        candidates.append({"rule_name": name, "rule_type": rule_type, "feature": feature, "value": value, "positive_dev_years": pos_years, "selected": selected, **m})

    for col in ["ny_hour", "direction", "correct_half"]:
        if col in dev.columns:
            for value in sorted(dev[col].dropna().unique(), key=str):
                add_rule(f"{col}={value}", dev[col].eq(value), "categorical", col, str(value))

    numeric = ["range_size", "stop_distance", "sweep_depth", "m15_mss_delay_min", "m15_fvg_delay_min", "m5_fvg_delay_min", "m15_entry_displacement", "m5_entry_displacement"]
    bin_defs: dict[str, list[float]] = {}
    for col in numeric:
        if col not in dev.columns:
            continue
        edges = _quantile_edges(dev[col], cfg.quantile_bins)
        if not edges:
            continue
        bin_defs[col] = edges
        bins = pd.cut(pd.to_numeric(dev[col], errors="coerce"), bins=edges, include_lowest=True, duplicates="drop")
        for interval in bins.dropna().unique():
            add_rule(f"{col}:{interval}", bins.eq(interval), "quantile", col, str(interval))

    for direction in ["BULLISH", "BEARISH"]:
        if "direction" not in dev.columns:
            continue
        for hour in sorted(dev["ny_hour"].dropna().unique()):
            add_rule(f"direction={direction}&ny_hour={hour}", dev["direction"].eq(direction) & dev["ny_hour"].eq(hour), "combo", "direction+ny_hour", f"{direction}|{hour}")

    cand = pd.DataFrame(candidates).sort_values(["selected", "expectancy_r", "profit_factor", "trades"], ascending=[False, False, False, False]).reset_index(drop=True)

    hold_rows: list[dict] = []
    for _, rule in cand.loc[cand["selected"]].iterrows():
        if rule["rule_type"] == "categorical":
            raw = rule["value"]
            col = rule["feature"]
            if col == "ny_hour": value = int(float(raw))
            elif col == "correct_half": value = raw.lower() == "true"
            else: value = raw
            mask = holdout[col].eq(value)
        elif rule["rule_type"] == "combo":
            direction, hour = str(rule["value"]).split("|")
            mask = holdout["direction"].eq(direction) & holdout["ny_hour"].eq(int(float(hour)))
        else:
            col = rule["feature"]
            edges = bin_defs.get(col, [])
            hb = pd.cut(pd.to_numeric(holdout[col], errors="coerce"), bins=edges, include_lowest=True, duplicates="drop")
            mask = hb.astype(str).eq(str(rule["value"]))
        sample = holdout.loc[mask.fillna(False)].copy()
        hm = _metrics(sample)
        hold_rows.append({"rule_name": rule["rule_name"], "dev_expectancy_r": rule["expectancy_r"], "dev_profit_factor": rule["profit_factor"], "dev_trades": rule["trades"], **{f"holdout_{k}": v for k, v in hm.items()}})

    hold_report = pd.DataFrame(hold_rows)
    overall = pd.DataFrame([
        {"period": "DEVELOPMENT", **_metrics(dev)},
        {"period": "HOLDOUT", **_metrics(holdout)},
    ])
    return cand, hold_report, overall, dev, holdout


def run_v4_3_edge_discovery(trades: pd.DataFrame, output_dir: str | Path = "data/research/v4_3_edge", variant: str = "FULL_M5_CAUSAL", config: EdgeDiscoveryConfig | None = None):
    candidates, holdout, overall, dev, hold = discover_edges(trades, variant, config)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out / "v4_3_candidates.csv", index=False)
    holdout.to_csv(out / "v4_3_holdout_validation.csv", index=False)
    overall.to_csv(out / "v4_3_period_summary.csv", index=False)
    dev.to_csv(out / "v4_3_development_trades.csv", index=False)
    hold.to_csv(out / "v4_3_holdout_trades.csv", index=False)
    return candidates, holdout, overall
