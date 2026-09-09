from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

from features.indicators import add_indicators, trend_bias
from strategy.crt import detect_crt

NY = ZoneInfo("America/New_York")
H1_DELTA = pd.to_timedelta(1, unit="h")


@dataclass(frozen=True)
class ResearchConfig:
    warmup_h1: int = 220
    max_holding_bars: int = 24
    target_rs: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)


def _slice_completed(df: pd.DataFrame, decision_time: pd.Timestamp, timeframe_minutes: int) -> pd.DataFrame:
    duration = pd.to_timedelta(timeframe_minutes, unit="min")
    return df.loc[(df["time"] + duration) <= decision_time].copy()


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

    for held, (_, bar) in enumerate(future_h1.head(max_holding_bars).iterrows(), start=1):
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

        # Conservative OHLC assumption: if stop and target are both touched
        # in the same H1 bar, count the stop first because intrabar path is unknown.
        if bar_stop:
            stop_hit = True
            bars_to_stop = held
            break

        for r, touched in target_touched.items():
            if touched and not hit[r]:
                hit[r] = True
                bars_to_target[r] = held

    out = {
        "hit_stop": stop_hit,
        "bars_to_stop": bars_to_stop,
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
    }
    for r in target_rs:
        key = str(r).replace(".", "_")
        out[f"hit_{key}r"] = bool(hit[r])
        out[f"bars_to_{key}r"] = bars_to_target[r]
    return out


def build_crt_occurrence_dataset(
    h1: pd.DataFrame,
    m15: pd.DataFrame,
    m5: pd.DataFrame,
    config: ResearchConfig | None = None,
) -> pd.DataFrame:
    """Create one causal research row for every H1 CRT sweep occurrence.

    Features use information available by the H1 decision time only. Outcome
    labels are computed separately from future H1 bars, making the result safe
    to split chronologically for later ML experiments.
    """
    cfg = config or ResearchConfig()
    rows: list[dict] = []

    if len(h1) <= cfg.warmup_h1 + 1:
        return pd.DataFrame()

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

        h1_ind = add_indicators(h1_hist)
        m15_ind = add_indicators(m15_hist)
        m5_ind = add_indicators(m5_hist)
        h1_row = h1_ind.iloc[-1]
        prev = h1_ind.iloc[-2]

        h1_bias = trend_bias(h1_ind)
        m15_bias = trend_bias(m15_ind)
        m5_bias = trend_bias(m5_ind)
        aligned_count = sum(
            bias == crt["direction"]
            for bias in (h1_bias, m15_bias, m5_bias)
        )

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
        outcomes = _target_outcomes(
            future_h1=future,
            direction=direction,
            entry=entry,
            stop=stop,
            target_rs=cfg.target_rs,
            max_holding_bars=cfg.max_holding_bars,
        )

        ny_time = decision_time.tz_convert(NY)
        candle_range = float(signal_bar["high"] - signal_bar["low"])
        body_size = abs(float(signal_bar["close"] - signal_bar["open"]))
        atr = float(h1_row["atr14"]) if pd.notna(h1_row["atr14"]) else None
        close = float(signal_bar["close"])

        rows.append(
            {
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
                "rsi14": float(h1_row["rsi14"]) if pd.notna(h1_row["rsi14"]) else None,
                "atr14": atr,
                "atr_pct": atr / close if atr and close != 0 else None,
                "entry": entry,
                "stop": stop,
                "risk_distance": risk,
                **outcomes,
            }
        )

    return pd.DataFrame(rows)
