from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from features.indicators import add_indicators, trend_bias
from strategy.crt import detect_crt

NY = ZoneInfo("America/New_York")
H1_DELTA = pd.to_timedelta(1, unit="h")


@dataclass(frozen=True)
class ResearchConfig:
    warmup_h1: int = 220
    max_holding_bars: int = 24
    target_rs: tuple[float, ...] = (1.0, 1.5)


def _slice_completed(df: pd.DataFrame, decision_time: pd.Timestamp, timeframe_minutes: int) -> pd.DataFrame:
    duration = pd.to_timedelta(timeframe_minutes, unit="min")
    return df.loc[(df["time"] + duration) <= decision_time].copy()


def _recent_range_features(h1_ind: pd.DataFrame, close: float, atr: float | None) -> dict[str, float | None]:
    """Features from completed H1 history only; current signal bar is included."""
    out: dict[str, float | None] = {}
    safe_atr = atr if atr is not None and np.isfinite(atr) and atr > 0 else None
    for window in (4, 8, 24):
        recent = h1_ind.tail(window)
        hi = float(recent["high"].max())
        lo = float(recent["low"].min())
        width = hi - lo
        out[f"recent_high_{window}h"] = hi
        out[f"recent_low_{window}h"] = lo
        out[f"distance_recent_high_{window}h_atr"] = (hi - close) / safe_atr if safe_atr else None
        out[f"distance_recent_low_{window}h_atr"] = (close - lo) / safe_atr if safe_atr else None
        out[f"range_position_{window}h"] = (close - lo) / width if width > 0 else 0.5
    return out


def _previous_ny_day_range(h1_hist: pd.DataFrame, decision_time: pd.Timestamp) -> dict[str, float | None]:
    hist = h1_hist.copy()
    times = pd.to_datetime(hist["time"], utc=True, errors="coerce")
    ny_dates = times.dt.tz_convert(NY).dt.date
    current_ny_date = decision_time.tz_convert(NY).date()
    prior_dates = sorted({d for d in ny_dates.dropna().tolist() if d < current_ny_date})
    if not prior_dates:
        return {"prev_day_high": None, "prev_day_low": None, "distance_prev_day_high_atr": None, "distance_prev_day_low_atr": None}
    prev_date = prior_dates[-1]
    prev = hist.loc[ny_dates == prev_date]
    if prev.empty:
        return {"prev_day_high": None, "prev_day_low": None, "distance_prev_day_high_atr": None, "distance_prev_day_low_atr": None}
    return {"prev_day_high": float(prev["high"].max()), "prev_day_low": float(prev["low"].min())}


def _target_outcomes(
    future_h1: pd.DataFrame,
    direction: str,
    entry: float,
    stop: float,
    target_rs: tuple[float, ...],
    max_holding_bars: int,
) -> dict:
    risk = abs(entry - stop)
    if risk <= 0:
        raise ValueError("risk distance must be positive")

    hit = {r: False for r in target_rs}
    stop_hit = False
    bars_to_stop = None
    bars_to_target: dict[float, int | None] = {r: None for r in target_rs}
    mfe_r = 0.0
    mae_r = 0.0
    ambiguous_intrabar = False
    observed_bars = 0

    for held, (_, bar) in enumerate(future_h1.head(max_holding_bars).iterrows(), start=1):
        observed_bars = held
        high = float(bar["high"])
        low = float(bar["low"])

        if direction == "BUY":
            mfe_r = max(mfe_r, (high - entry) / risk)
            mae_r = min(mae_r, (low - entry) / risk)
            bar_stop = low <= stop
            target_touched = {r: high >= entry + r * risk for r in target_rs}
        else:
            mfe_r = max(mfe_r, (entry - low) / risk)
            mae_r = min(mae_r, (entry - high) / risk)
            bar_stop = high >= stop
            target_touched = {r: low <= entry - r * risk for r in target_rs}

        if bar_stop and any(target_touched.values()):
            ambiguous_intrabar = True

        if bar_stop:
            stop_hit = True
            bars_to_stop = held
            break

        for r, touched in target_touched.items():
            if touched and not hit[r]:
                hit[r] = True
                bars_to_target[r] = held

    horizon_complete = observed_bars >= max_holding_bars
    out = {
        "hit_stop": stop_hit,
        "bars_to_stop": bars_to_stop,
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
        "ambiguous_intrabar": bool(ambiguous_intrabar),
        "outcome_bars_observed": int(observed_bars),
        "outcome_horizon_complete": bool(horizon_complete),
    }
    for r in target_rs:
        key = str(r).replace(".", "_")
        target_final = bool(hit[r] or stop_hit or horizon_complete)
        out[f"hit_{key}r"] = bool(hit[r])
        out[f"bars_to_{key}r"] = bars_to_target[r]
        out[f"outcome_final_{key}r"] = target_final
    return out


def build_crt_occurrence_dataset(
    h1: pd.DataFrame,
    m15: pd.DataFrame,
    m5: pd.DataFrame,
    config: ResearchConfig | None = None,
) -> pd.DataFrame:
    """Create one causal research row for every H1 CRT sweep occurrence."""
    cfg = config or ResearchConfig()
    rows: list[dict] = []

    if len(h1) <= cfg.warmup_h1 + 1:
        return pd.DataFrame()

    h1_all_ind = add_indicators(h1)

    for i in range(cfg.warmup_h1, len(h1) - 1):
        signal_bar = h1.iloc[i]
        decision_time = pd.Timestamp(signal_bar["time"]) + H1_DELTA
        h1_hist = h1.iloc[: i + 1].copy()

        crt = detect_crt(h1_hist)
        if not crt["swept"]:
            continue

        m15_hist = _slice_completed(m15, decision_time, 15)
        m5_hist = _slice_completed(m5, decision_time, 5)
        if len(m15_hist) < 200 or len(m5_hist) < 200:
            continue

        h1_ind = h1_all_ind.iloc[: i + 1].copy()
        m15_ind = add_indicators(m15_hist)
        m5_ind = add_indicators(m5_hist)
        h1_row = h1_ind.iloc[-1]
        prev = h1_ind.iloc[-2]

        h1_bias = trend_bias(h1_ind)
        m15_bias = trend_bias(m15_ind)
        m5_bias = trend_bias(m5_ind)
        aligned_count = sum(bias == crt["direction"] for bias in (h1_bias, m15_bias, m5_bias))

        entry_bar = h1.iloc[i + 1]
        entry = float(entry_bar["open"])
        if crt["direction"] == "BULLISH":
            direction = "BUY"
            stop = float(signal_bar["low"])
            sweep_size = max(0.0, float(prev["low"] - signal_bar["low"]))
            wick_size = float(signal_bar["close"] - signal_bar["low"])
        else:
            direction = "SELL"
            stop = float(signal_bar["high"])
            sweep_size = max(0.0, float(signal_bar["high"] - prev["high"]))
            wick_size = float(signal_bar["high"] - signal_bar["close"])

        risk = abs(entry - stop)
        if risk <= 0:
            continue

        future = h1.iloc[i + 1 :].copy()
        outcomes = _target_outcomes(future, direction, entry, stop, cfg.target_rs, cfg.max_holding_bars)

        ny_time = decision_time.tz_convert(NY)
        candle_range = float(signal_bar["high"] - signal_bar["low"])
        body_size = abs(float(signal_bar["close"] - signal_bar["open"]))
        atr = float(h1_row["atr14"]) if pd.notna(h1_row["atr14"]) else None
        close = float(signal_bar["close"])
        recent = _recent_range_features(h1_ind, close, atr)
        prev_day = _previous_ny_day_range(h1_hist, decision_time)
        safe_atr = atr if atr and atr > 0 else None
        prev_day_high = prev_day.get("prev_day_high")
        prev_day_low = prev_day.get("prev_day_low")
        prev_day["distance_prev_day_high_atr"] = (prev_day_high - close) / safe_atr if safe_atr and prev_day_high is not None else None
        prev_day["distance_prev_day_low_atr"] = (close - prev_day_low) / safe_atr if safe_atr and prev_day_low is not None else None

        rows.append({
            "signal_time_utc": signal_bar["time"],
            "decision_time_utc": decision_time,
            "entry_time_utc": entry_bar["time"],
            "ny_date": ny_time.date().isoformat(),
            "ny_hour": int(ny_time.hour),
            "day_of_week": ny_time.day_name(),
            "in_ny_08_13": bool(8 <= ny_time.hour < 13),
            "direction": direction,
            "crt_direction": crt["direction"],
            "h1_bias": h1_bias,
            "m15_bias": m15_bias,
            "m5_bias": m5_bias,
            "alignment_count": int(aligned_count),
            "full_alignment": bool(aligned_count == 3),
            "open": float(signal_bar["open"]),
            "high": float(signal_bar["high"]),
            "low": float(signal_bar["low"]),
            "close": close,
            "candle_range": candle_range,
            "body_size": body_size,
            "body_pct_range": body_size / candle_range if candle_range > 0 else 0.0,
            "sweep_size": sweep_size,
            "sweep_atr": sweep_size / atr if atr and atr > 0 else None,
            "wick_size": wick_size,
            "ema20": float(h1_row["ema20"]),
            "ema50": float(h1_row["ema50"]),
            "ema200": float(h1_row["ema200"]),
            "ema20_50_distance": float(h1_row["ema20"] - h1_row["ema50"]),
            "ema50_200_distance": float(h1_row["ema50"] - h1_row["ema200"]),
            "ema20_slope_4h": float(h1_row["ema20_slope_4h"]) if pd.notna(h1_row["ema20_slope_4h"]) else None,
            "macd": float(h1_row["macd"]) if pd.notna(h1_row["macd"]) else None,
            "macd_signal": float(h1_row["macd_signal"]) if pd.notna(h1_row["macd_signal"]) else None,
            "macd_hist": float(h1_row["macd_hist"]) if pd.notna(h1_row["macd_hist"]) else None,
            "return_1h": float(h1_row["return_1h"]) if pd.notna(h1_row["return_1h"]) else None,
            "return_4h": float(h1_row["return_4h"]) if pd.notna(h1_row["return_4h"]) else None,
            "return_8h": float(h1_row["return_8h"]) if pd.notna(h1_row["return_8h"]) else None,
            "atr_change_4h": float(h1_row["atr_change_4h"]) if pd.notna(h1_row["atr_change_4h"]) else None,
            "rsi14": float(h1_row["rsi14"]) if pd.notna(h1_row["rsi14"]) else None,
            "atr14": atr,
            "atr_pct": atr / close if atr and close != 0 else None,
            "entry": entry,
            "stop": stop,
            "risk_distance": risk,
            **recent,
            **prev_day,
            **outcomes,
        })

    return pd.DataFrame(rows)
