from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtesting.session_crt_split import SplitTargetConfig, simulate_split_trade

NY = ZoneInfo("America/New_York")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    return out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def _m15_confirmation(m15: pd.DataFrame, signal_time: pd.Timestamp, direction: str, entry: float, stop: float) -> tuple[bool, bool]:
    """Causal post-sweep M15 MSS and FVG proxy confirmation.

    Only completed M15 bars after the H1 manipulation close are inspected.
    MSS requires displacement through the prior M15 candle extreme. FVG uses a
    standard three-candle imbalance in the signal direction. This is a research
    proxy to quantify confirmation value before any execution rule is frozen.
    """
    window = m15.loc[(m15["time"] > signal_time) & (m15["time"] <= signal_time + pd.Timedelta(hours=2))].copy()
    if len(window) < 3:
        return False, False
    mss = False
    fvg = False
    rows = window.reset_index(drop=True)
    for i in range(1, len(rows)):
        prev = rows.iloc[i - 1]
        cur = rows.iloc[i]
        if direction == "BULLISH":
            displacement = float(cur["close"]) > float(prev["high"]) and float(cur["close"]) > float(cur["open"])
        else:
            displacement = float(cur["close"]) < float(prev["low"]) and float(cur["close"]) < float(cur["open"])
        mss = mss or displacement
        if i >= 2:
            first = rows.iloc[i - 2]
            if direction == "BULLISH":
                imbalance = float(cur["low"]) > float(first["high"])
            else:
                imbalance = float(cur["high"]) < float(first["low"])
            fvg = fvg or imbalance
    return bool(mss), bool(mss and fvg)


def build_unrestricted_h1_setups(h1: pd.DataFrame, m15: pd.DataFrame, split_config: SplitTargetConfig | None = None) -> pd.DataFrame:
    """Scan every consecutive completed H1 pair, with no killzone/time barrier."""
    cfg = split_config or SplitTargetConfig()
    bars = _normalize(h1)
    lower = _normalize(m15)
    rows: list[dict] = []
    for i in range(1, len(bars)):
        prev = bars.iloc[i - 1]
        cur = bars.iloc[i]
        rh, rl = float(prev["high"]), float(prev["low"])
        high, low, close = float(cur["high"]), float(cur["low"]), float(cur["close"])
        swept_high = high > rh
        swept_low = low < rl
        if swept_high == swept_low:  # neither side or ambiguous double sweep
            continue
        inside = rl < close < rh
        if not inside:
            continue
        if swept_low:
            direction, stop = "BULLISH", low
        else:
            direction, stop = "BEARISH", high
        if abs(close - stop) <= 0:
            continue
        signal_time = pd.Timestamp(cur["time"])
        future = bars.loc[bars["time"] > signal_time]
        result = simulate_split_trade(future, direction, close, stop, cfg)
        eq = (rh + rl) / 2.0
        correct_half = close <= eq if direction == "BULLISH" else close >= eq
        mss, mss_fvg = _m15_confirmation(lower, signal_time, direction, close, stop)
        ny_time = signal_time.tz_convert(NY)
        rows.append({
            "range_time_utc": prev["time"], "signal_time_utc": signal_time,
            "ny_hour": int(ny_time.hour), "year": int(ny_time.year),
            "direction": direction, "entry": close, "stop": stop,
            "range_high": rh, "range_low": rl, "equilibrium": eq,
            "correct_half": bool(correct_half), "mss": mss, "mss_fvg": mss_fvg,
            **result,
        })
    return pd.DataFrame(rows)


def _metrics(scope: str, variant: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"scope": scope, "variant": variant, "trades": 0, "total_r": 0.0, "expectancy_r": np.nan, "win_rate": np.nan, "profit_factor": np.nan, "max_drawdown_r": 0.0, "tp1_rate": np.nan, "tp2_rate": np.nan}
    wins = df.loc[df.total_r > 0, "total_r"].sum()
    losses = -df.loc[df.total_r < 0, "total_r"].sum()
    eq = df.sort_values("signal_time_utc").total_r.cumsum()
    dd = eq - eq.cummax()
    return {"scope": scope, "variant": variant, "trades": len(df), "total_r": float(df.total_r.sum()), "expectancy_r": float(df.total_r.mean()), "win_rate": float((df.total_r > 0).mean()), "profit_factor": float(wins / losses) if losses > 0 else np.inf, "max_drawdown_r": float(dd.min()), "tp1_rate": float(df.tp1_hit.mean()), "tp2_rate": float(df.tp2_hit.mean())}


def run_v4_1_confirmation_study(h1: pd.DataFrame, m15: pd.DataFrame, output_dir="data/research/v4_1_unrestricted", split_config=None):
    trades = build_unrestricted_h1_setups(h1, m15, split_config)
    variants = {
        "RAW": pd.Series(True, index=trades.index),
        "MSS": trades["mss"] if not trades.empty else pd.Series(dtype=bool),
        "MSS_FVG": trades["mss_fvg"] if not trades.empty else pd.Series(dtype=bool),
        "FULL_MODEL": (trades["mss_fvg"] & trades["correct_half"]) if not trades.empty else pd.Series(dtype=bool),
    }
    summary = pd.DataFrame([_metrics("ALL", name, trades.loc[mask]) for name, mask in variants.items()])
    breakdown_rows = []
    for name, mask in variants.items():
        sample = trades.loc[mask].copy()
        for dimension in ["ny_hour", "direction", "correct_half", "year"]:
            for value, grp in sample.groupby(dimension, dropna=False):
                row = _metrics(f"{dimension}={value}", name, grp)
                row["dimension"] = dimension
                row["value"] = value
                breakdown_rows.append(row)
    breakdown = pd.DataFrame(breakdown_rows)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "v4_1_unrestricted_trades.csv", index=False)
    summary.to_csv(out / "v4_1_unrestricted_summary.csv", index=False)
    breakdown.to_csv(out / "v4_1_unrestricted_breakdown.csv", index=False)
    return summary, breakdown, trades
