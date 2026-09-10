from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtesting.session_crt_split import SplitTargetConfig, simulate_split_trade

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class CausalConfirmationConfig:
    m15_window_hours: int = 3
    m5_window_hours: int = 2
    structure_lookback: int = 3


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    return out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def _bar_close_time(open_time: pd.Timestamp, minutes: int) -> pd.Timestamp:
    return pd.Timestamp(open_time) + pd.to_timedelta(int(minutes), unit="min")


def _valid_entry(direction: str, entry: float, stop: float) -> bool:
    return entry > stop if direction == "BULLISH" else entry < stop


def _empty_confirmation() -> dict:
    return {
        "mss": False,
        "mss_time": pd.NaT,
        "mss_entry": np.nan,
        "fvg": False,
        "fvg_time": pd.NaT,
        "fvg_entry": np.nan,
    }


def _find_mss_fvg(
    bars: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    direction: str,
    timeframe_minutes: int,
    lookback: int,
) -> dict:
    """Find a causal MSS and then a direction-aligned FVG.

    MT5 timestamps represent bar OPEN times.  A confirmation bar is eligible only
    after its close, and that close must fall inside [start_time, end_time].
    Crucially, the MSS structure lookback may use bars that closed BEFORE
    start_time.  Those bars are already-known context, not future information.
    This fixes the old implementation which sliced them away.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")

    data = _normalize(bars)
    if data.empty:
        return _empty_confirmation()

    data = data.loc[data["time"] < end_time].copy().reset_index(drop=True)
    if len(data) < max(lookback + 1, 3):
        return _empty_confirmation()

    close_times = data["time"] + pd.to_timedelta(int(timeframe_minutes), unit="min")
    mss_idx = None
    mss_time = pd.NaT
    mss_entry = np.nan

    for i in range(lookback, len(data)):
        cur_close_time = pd.Timestamp(close_times.iloc[i])
        if cur_close_time < start_time:
            continue
        if cur_close_time > end_time:
            break

        cur = data.iloc[i]
        prior = data.iloc[i - lookback:i]
        close = float(cur["close"])
        open_ = float(cur["open"])
        if direction == "BULLISH":
            shifted = close > float(prior["high"].max()) and close > open_
        else:
            shifted = close < float(prior["low"].min()) and close < open_
        if shifted:
            mss_idx = i
            mss_time = cur_close_time
            mss_entry = close
            break

    if mss_idx is None:
        return _empty_confirmation()

    fvg_time = pd.NaT
    fvg_entry = np.nan
    for i in range(max(2, mss_idx), len(data)):
        cur_close_time = pd.Timestamp(close_times.iloc[i])
        if cur_close_time < mss_time:
            continue
        if cur_close_time > end_time:
            break

        first = data.iloc[i - 2]
        cur = data.iloc[i]
        if direction == "BULLISH":
            imbalance = float(cur["low"]) > float(first["high"])
        else:
            imbalance = float(cur["high"]) < float(first["low"])
        if imbalance:
            fvg_time = cur_close_time
            fvg_entry = float(cur["close"])
            break

    return {
        "mss": True,
        "mss_time": mss_time,
        "mss_entry": float(mss_entry),
        "fvg": bool(pd.notna(fvg_time)),
        "fvg_time": fvg_time,
        "fvg_entry": float(fvg_entry) if pd.notna(fvg_time) else np.nan,
    }


def _simulate_from_timestamp(
    bars: pd.DataFrame,
    entry_time: pd.Timestamp,
    direction: str,
    entry: float,
    stop: float,
    split_cfg: SplitTargetConfig,
) -> dict | None:
    if not _valid_entry(direction, entry, stop):
        return None
    future = bars.loc[bars["time"] >= entry_time].copy()
    if future.empty:
        return None
    return simulate_split_trade(future, direction, entry, stop, split_cfg)


def build_causal_setups(
    h1: pd.DataFrame,
    m15: pd.DataFrame,
    m5: pd.DataFrame | None = None,
    split_config: SplitTargetConfig | None = None,
    confirmation_config: CausalConfirmationConfig | None = None,
) -> pd.DataFrame:
    """Build unrestricted 24-hour H1 CRT candidates with causal confirmations."""
    split_cfg = split_config or SplitTargetConfig()
    conf_cfg = confirmation_config or CausalConfirmationConfig()
    h1b = _normalize(h1)
    m15b = _normalize(m15)
    m5b = _normalize(m5) if m5 is not None and not m5.empty else None

    rows: list[dict] = []
    for i in range(1, len(h1b)):
        prev = h1b.iloc[i - 1]
        cur = h1b.iloc[i]
        rh, rl = float(prev["high"]), float(prev["low"])
        high, low, reclaim_close = float(cur["high"]), float(cur["low"]), float(cur["close"])
        swept_high, swept_low = high > rh, low < rl
        if swept_high == swept_low or not (rl < reclaim_close < rh):
            continue

        if swept_low:
            direction, stop = "BULLISH", low
        else:
            direction, stop = "BEARISH", high
        if not _valid_entry(direction, reclaim_close, stop):
            continue

        signal_open = pd.Timestamp(cur["time"])
        signal_close = _bar_close_time(signal_open, 60)
        eq = (rh + rl) / 2.0
        correct_half = reclaim_close <= eq if direction == "BULLISH" else reclaim_close >= eq
        ny = signal_close.tz_convert(NY)

        base = {
            "range_time_utc": pd.Timestamp(prev["time"]),
            "signal_open_time_utc": signal_open,
            "signal_close_time_utc": signal_close,
            "ny_hour": int(ny.hour),
            "year": int(ny.year),
            "direction": direction,
            "h1_entry": reclaim_close,
            "stop": stop,
            "range_high": rh,
            "range_low": rl,
            "equilibrium": eq,
            "correct_half": bool(correct_half),
        }

        raw_result = _simulate_from_timestamp(h1b, signal_close, direction, reclaim_close, stop, split_cfg)
        if raw_result is not None:
            for k, v in raw_result.items():
                base[f"raw_{k}"] = v

        m15_conf = _find_mss_fvg(
            m15b,
            signal_close,
            signal_close + pd.to_timedelta(int(conf_cfg.m15_window_hours), unit="h"),
            direction,
            15,
            conf_cfg.structure_lookback,
        )
        base.update({f"m15_{k}": v for k, v in m15_conf.items()})

        if m15_conf["mss"] and _valid_entry(direction, float(m15_conf["mss_entry"]), stop):
            mss_result = _simulate_from_timestamp(m15b, pd.Timestamp(m15_conf["mss_time"]), direction, float(m15_conf["mss_entry"]), stop, split_cfg)
            if mss_result is not None:
                for k, v in mss_result.items():
                    base[f"mss_{k}"] = v

        if m15_conf["fvg"] and _valid_entry(direction, float(m15_conf["fvg_entry"]), stop):
            fvg_result = _simulate_from_timestamp(m15b, pd.Timestamp(m15_conf["fvg_time"]), direction, float(m15_conf["fvg_entry"]), stop, split_cfg)
            if fvg_result is not None:
                for k, v in fvg_result.items():
                    base[f"fvg_{k}"] = v

            if m5b is not None:
                m5_conf = _find_mss_fvg(
                    m5b,
                    pd.Timestamp(m15_conf["fvg_time"]),
                    pd.Timestamp(m15_conf["fvg_time"]) + pd.to_timedelta(int(conf_cfg.m5_window_hours), unit="h"),
                    direction,
                    5,
                    conf_cfg.structure_lookback,
                )
                base.update({f"m5_{k}": v for k, v in m5_conf.items()})
                if m5_conf["fvg"] and _valid_entry(direction, float(m5_conf["fvg_entry"]), stop):
                    m5_result = _simulate_from_timestamp(m5b, pd.Timestamp(m5_conf["fvg_time"]), direction, float(m5_conf["fvg_entry"]), stop, split_cfg)
                    if m5_result is not None:
                        for k, v in m5_result.items():
                            base[f"m5exec_{k}"] = v

        rows.append(base)

    return pd.DataFrame(rows)


def _variant_frame(trades: pd.DataFrame, variant: str) -> pd.DataFrame:
    mapping = {
        "RAW_CAUSAL": "raw_",
        "MSS_CAUSAL": "mss_",
        "MSS_FVG_CAUSAL": "fvg_",
        "FULL_MODEL_CAUSAL": "fvg_",
        "FULL_M5_CAUSAL": "m5exec_",
    }
    prefix = mapping[variant]
    col = prefix + "total_r"
    if col not in trades.columns:
        return trades.iloc[0:0].copy()
    out = trades.loc[trades[col].notna()].copy()
    if variant in {"FULL_MODEL_CAUSAL", "FULL_M5_CAUSAL"}:
        out = out.loc[out["correct_half"]]
    out = out.rename(columns={
        prefix + "total_r": "total_r",
        prefix + "tp1_hit": "tp1_hit",
        prefix + "tp2_hit": "tp2_hit",
    })
    return out


def _metrics(scope: str, variant: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"scope": scope, "variant": variant, "trades": 0, "total_r": 0.0, "expectancy_r": np.nan, "win_rate": np.nan, "profit_factor": np.nan, "max_drawdown_r": 0.0, "tp1_rate": np.nan, "tp2_rate": np.nan}
    wins = float(df.loc[df["total_r"] > 0, "total_r"].sum())
    losses = float(-df.loc[df["total_r"] < 0, "total_r"].sum())
    eq = df.sort_values("signal_close_time_utc")["total_r"].cumsum()
    peak = pd.concat([pd.Series([0.0]), eq.reset_index(drop=True)]).cummax().iloc[1:].reset_index(drop=True)
    dd = eq.reset_index(drop=True) - peak
    return {
        "scope": scope,
        "variant": variant,
        "trades": int(len(df)),
        "total_r": float(df["total_r"].sum()),
        "expectancy_r": float(df["total_r"].mean()),
        "win_rate": float((df["total_r"] > 0).mean()),
        "profit_factor": wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan),
        "max_drawdown_r": float(dd.min()) if len(dd) else 0.0,
        "tp1_rate": float(df["tp1_hit"].mean()),
        "tp2_rate": float(df["tp2_hit"].mean()),
    }


def run_v4_2_causal_study(
    h1: pd.DataFrame,
    m15: pd.DataFrame,
    m5: pd.DataFrame | None = None,
    output_dir: str | Path = "data/research/v4_2_causal",
    split_config: SplitTargetConfig | None = None,
    confirmation_config: CausalConfirmationConfig | None = None,
):
    trades = build_causal_setups(h1, m15, m5, split_config, confirmation_config)
    variants = ["RAW_CAUSAL", "MSS_CAUSAL", "MSS_FVG_CAUSAL", "FULL_MODEL_CAUSAL"]
    if m5 is not None:
        variants.append("FULL_M5_CAUSAL")

    summary_rows, breakdown_rows = [], []
    for variant in variants:
        frame = _variant_frame(trades, variant)
        summary_rows.append(_metrics("ALL", variant, frame))
        for dimension in ["ny_hour", "direction", "correct_half", "year"]:
            if frame.empty:
                continue
            for value, grp in frame.groupby(dimension, dropna=False):
                row = _metrics(f"{dimension}={value}", variant, grp)
                row["dimension"] = dimension
                row["value"] = value
                breakdown_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    breakdown = pd.DataFrame(breakdown_rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "v4_2_causal_trades.csv", index=False)
    summary.to_csv(out / "v4_2_causal_summary.csv", index=False)
    breakdown.to_csv(out / "v4_2_causal_breakdown.csv", index=False)
    return summary, breakdown, trades
