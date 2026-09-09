from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.metrics import calculate_metrics
from strategy.signal_engine import build_signal


@dataclass(frozen=True)
class BacktestConfig:
    minimum_score: int = 70
    reward_risk: float = 2.0
    warmup_h1: int = 220
    max_holding_bars: int = 24


def _slice_completed(df: pd.DataFrame, decision_time: pd.Timestamp, timeframe_minutes: int) -> pd.DataFrame:
    duration = pd.Timedelta(minutes=timeframe_minutes)
    return df.loc[(df["time"] + duration) <= decision_time].copy()


def _simulate_exit(
    future_h1: pd.DataFrame,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    max_holding_bars: int,
) -> dict:
    """Simulate bar-based exits conservatively.

    If both stop and target are touched within the same H1 candle, the stop is
    assumed to have been hit first because OHLC data cannot reveal intrabar path.
    """
    mfe_r = 0.0
    mae_r = 0.0
    risk = abs(entry - stop)
    if risk <= 0:
        raise ValueError("risk distance must be positive")

    sample = future_h1.head(max_holding_bars)
    for held, (_, bar) in enumerate(sample.iterrows(), start=1):
        high = float(bar["high"])
        low = float(bar["low"])

        if direction == "BUY":
            mfe_r = max(mfe_r, (high - entry) / risk)
            mae_r = min(mae_r, (low - entry) / risk)
            stop_hit = low <= stop
            target_hit = high >= target
        else:
            mfe_r = max(mfe_r, (entry - low) / risk)
            mae_r = min(mae_r, (entry - high) / risk)
            stop_hit = high >= stop
            target_hit = low <= target

        if stop_hit:
            return {
                "exit_time": bar["time"],
                "exit_price": stop,
                "exit_reason": "STOP",
                "r_multiple": -1.0,
                "bars_held": held,
                "mfe_r": float(mfe_r),
                "mae_r": float(mae_r),
            }
        if target_hit:
            rr = abs(target - entry) / risk
            return {
                "exit_time": bar["time"],
                "exit_price": target,
                "exit_reason": "TARGET",
                "r_multiple": float(rr),
                "bars_held": held,
                "mfe_r": float(mfe_r),
                "mae_r": float(mae_r),
            }

    if sample.empty:
        return {}

    final = sample.iloc[-1]
    exit_price = float(final["close"])
    r_multiple = (
        (exit_price - entry) / risk if direction == "BUY" else (entry - exit_price) / risk
    )
    return {
        "exit_time": final["time"],
        "exit_price": exit_price,
        "exit_reason": "TIME",
        "r_multiple": float(r_multiple),
        "bars_held": int(len(sample)),
        "mfe_r": float(mfe_r),
        "mae_r": float(mae_r),
    }


def run_backtest(
    h1: pd.DataFrame,
    m15: pd.DataFrame,
    m5: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run a no-lookahead CRT backtest using next-H1-open entries.

    The engine mirrors the live V1 one-position rule: a new historical signal
    is ignored while a previously simulated position is still open.
    """
    cfg = config or BacktestConfig()
    records: list[dict] = []
    position_open_until: pd.Timestamp | None = None

    if len(h1) <= cfg.warmup_h1 + 1:
        trades = pd.DataFrame()
        return trades, calculate_metrics(trades)

    for i in range(cfg.warmup_h1, len(h1) - 1):
        signal_bar = h1.iloc[i]
        decision_time = pd.Timestamp(signal_bar["time"]) + pd.Timedelta(hours=1)

        if position_open_until is not None and decision_time <= position_open_until:
            continue

        h1_hist = h1.iloc[: i + 1].copy()
        m15_hist = _slice_completed(m15, decision_time, 15)
        m5_hist = _slice_completed(m5, decision_time, 5)
        if len(m15_hist) < 200 or len(m5_hist) < 200:
            continue

        signal = build_signal(
            h1_hist,
            m15_hist,
            m5_hist,
            now=decision_time.to_pydatetime(),
            minimum_score=cfg.minimum_score,
        )
        if not signal["approved"]:
            continue

        direction = signal["direction"]
        entry_bar = h1.iloc[i + 1]
        entry = float(entry_bar["open"])
        if direction == "BUY":
            stop = float(signal_bar["low"])
            risk = entry - stop
            target = entry + cfg.reward_risk * risk
        else:
            stop = float(signal_bar["high"])
            risk = stop - entry
            target = entry - cfg.reward_risk * risk

        if risk <= 0:
            continue

        future = h1.iloc[i + 1 :].copy()
        outcome = _simulate_exit(
            future_h1=future,
            direction=direction,
            entry=entry,
            stop=stop,
            target=target,
            max_holding_bars=cfg.max_holding_bars,
        )
        if not outcome:
            continue

        position_open_until = pd.Timestamp(outcome["exit_time"])
        records.append(
            {
                "signal_time": signal_bar["time"],
                "decision_time": decision_time,
                "entry_time": entry_bar["time"],
                "direction": direction,
                "score": signal["score"],
                "h1_bias": signal["h1_bias"],
                "m15_bias": signal["m15_bias"],
                "m5_bias": signal["m5_bias"],
                "crt_direction": signal["crt"]["direction"],
                "crt_swept": signal["crt"]["swept"],
                "entry": entry,
                "stop": stop,
                "target": target,
                **outcome,
            }
        )

    trades = pd.DataFrame(records)
    if not trades.empty:
        trades["equity_r"] = trades["r_multiple"].cumsum()
    metrics = calculate_metrics(trades)
    return trades, metrics
