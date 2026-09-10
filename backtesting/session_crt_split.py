from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtesting.killzones import DEFAULT_KILLZONES, Killzone
from strategy.session_crt import SessionCRTConfig, detect_session_crt

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SplitTargetConfig:
    """Risk and execution assumptions for the V4 split-target experiment.

    Total setup risk is 1.0R split equally between two independent legs:
    - leg 1: 0.5R risk, target at +1R price distance -> +0.5R payoff
    - leg 2: 0.5R risk, target at +2R price distance -> +1.0R payoff

    The runner leg keeps the original stop after TP1 in this first experiment.
    This avoids adding an untested break-even rule.
    """

    leg1_weight: float = 0.5
    leg2_weight: float = 0.5
    tp1_r: float = 1.0
    tp2_r: float = 2.0
    max_holding_bars: int = 24
    starting_equity_r: float = 0.0
    intrabar_policy: str = "STOP_FIRST"

    def __post_init__(self) -> None:
        if not np.isclose(self.leg1_weight + self.leg2_weight, 1.0):
            raise ValueError("leg weights must sum to 1.0")
        if self.leg1_weight <= 0 or self.leg2_weight <= 0:
            raise ValueError("leg weights must be positive")
        if self.tp1_r <= 0 or self.tp2_r <= self.tp1_r:
            raise ValueError("require 0 < tp1_r < tp2_r")
        if self.max_holding_bars <= 0:
            raise ValueError("max_holding_bars must be positive")
        if self.intrabar_policy != "STOP_FIRST":
            raise ValueError("only conservative STOP_FIRST intrabar handling is supported")


def _session_date_for_timestamp(ts_ny: pd.Timestamp, kz: Killzone):
    date = ts_ny.date()
    if kz.start_hour > kz.end_hour and ts_ny.hour < kz.end_hour:
        return (ts_ny - pd.DateOffset(days=1)).date()
    return date


def _session_dates(h1: pd.DataFrame, kz: Killzone) -> list:
    times = pd.to_datetime(h1["time"], utc=True, errors="coerce")
    ny = times.dt.tz_convert(NY)
    mask = ny.dt.hour.map(kz.contains_hour)
    dates = [_session_date_for_timestamp(t, kz) for t in ny.loc[mask].dropna()]
    return sorted(set(dates))


def _targets(direction: str, entry: float, stop: float, cfg: SplitTargetConfig) -> tuple[float, float]:
    risk = abs(entry - stop)
    if risk <= 0:
        raise ValueError("entry-stop risk distance must be positive")
    if direction == "BULLISH":
        return entry + cfg.tp1_r * risk, entry + cfg.tp2_r * risk
    if direction == "BEARISH":
        return entry - cfg.tp1_r * risk, entry - cfg.tp2_r * risk
    raise ValueError(f"unsupported direction: {direction}")


def simulate_split_trade(
    future_h1: pd.DataFrame,
    direction: str,
    entry: float,
    stop: float,
    config: SplitTargetConfig | None = None,
) -> dict:
    """Simulate two equal-risk legs using completed future H1 bars only.

    Conservative ambiguity rule: if a bar touches the stop and any still-open
    target in the same H1 candle, the stop is assumed to occur first.

    Timeout handling is neutral for any still-open leg rather than treating an
    unresolved trade as a loss.
    """
    cfg = config or SplitTargetConfig()
    tp1, tp2 = _targets(direction, entry, stop, cfg)

    leg1_open = True
    leg2_open = True
    leg1_r = 0.0
    leg2_r = 0.0
    tp1_hit = False
    tp2_hit = False
    stopped = False
    ambiguous_intrabar = False
    bars_held = 0

    ordered = future_h1.copy()
    ordered["time"] = pd.to_datetime(ordered["time"], utc=True, errors="coerce")
    ordered = ordered.dropna(subset=["time"]).sort_values("time")

    for held, (_, bar) in enumerate(ordered.head(cfg.max_holding_bars).iterrows(), start=1):
        bars_held = held
        high = float(bar["high"])
        low = float(bar["low"])

        if direction == "BULLISH":
            stop_touched = low <= stop
            tp1_touched = high >= tp1
            tp2_touched = high >= tp2
        else:
            stop_touched = high >= stop
            tp1_touched = low <= tp1
            tp2_touched = low <= tp2

        any_open_target_touched = (leg1_open and tp1_touched) or (leg2_open and tp2_touched)
        if stop_touched and any_open_target_touched:
            ambiguous_intrabar = True

        if stop_touched:
            if leg1_open:
                leg1_r = -cfg.leg1_weight
                leg1_open = False
            if leg2_open:
                leg2_r = -cfg.leg2_weight
                leg2_open = False
            stopped = True
            break

        if leg1_open and tp1_touched:
            leg1_r = cfg.leg1_weight * cfg.tp1_r
            leg1_open = False
            tp1_hit = True

        if leg2_open and tp2_touched:
            leg2_r = cfg.leg2_weight * cfg.tp2_r
            leg2_open = False
            tp2_hit = True

        if not leg1_open and not leg2_open:
            break

    total_r = float(leg1_r + leg2_r)
    if tp2_hit:
        outcome = "TP1_TP2"
    elif tp1_hit and stopped:
        outcome = "TP1_THEN_STOP"
    elif stopped:
        outcome = "STOP"
    elif tp1_hit:
        outcome = "TP1_TIMEOUT"
    else:
        outcome = "TIMEOUT"

    return {
        "tp1_price": float(tp1),
        "tp2_price": float(tp2),
        "tp1_hit": bool(tp1_hit),
        "tp2_hit": bool(tp2_hit),
        "stopped": bool(stopped),
        "leg1_r": float(leg1_r),
        "leg2_r": float(leg2_r),
        "total_r": total_r,
        "outcome": outcome,
        "bars_held": int(bars_held),
        "ambiguous_intrabar": bool(ambiguous_intrabar),
        "leg1_open_at_timeout": bool(leg1_open),
        "leg2_open_at_timeout": bool(leg2_open),
    }


def run_split_target_backtest(
    h1: pd.DataFrame,
    output_dir: str | Path = "data/research/v4_split",
    split_config: SplitTargetConfig | None = None,
    detector_config: SessionCRTConfig | None = None,
    killzones: tuple[Killzone, ...] = DEFAULT_KILLZONES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backtest the H1 session CRT detector with 50/50 1R + 2R exits."""
    cfg = split_config or SplitTargetConfig()
    # For this experiment, a sweep/reclaim is allowed to enter even when the
    # opposite C1 extreme itself offers <2R. Exit targets are fixed at 1R/2R.
    detector_cfg = detector_config or SessionCRTConfig(min_rr=0.0)

    bars = h1.copy()
    bars["time"] = pd.to_datetime(bars["time"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    rows: list[dict] = []
    for kz in killzones:
        for session_date in _session_dates(bars, kz):
            signal = detect_session_crt(bars, kz.name, session_date, detector_cfg, killzones)
            if not signal.get("valid", False):
                continue
            manipulation_time = pd.Timestamp(signal["manipulation_time_utc"])
            future = bars.loc[bars["time"] > manipulation_time].copy()
            result = simulate_split_trade(
                future,
                signal["direction"],
                float(signal["entry"]),
                float(signal["stop"]),
                cfg,
            )
            rows.append({
                "session": kz.name,
                "session_date": str(session_date),
                "range_time_utc": signal["range_time_utc"],
                "signal_time_utc": manipulation_time,
                "direction": signal["direction"],
                "entry": float(signal["entry"]),
                "stop": float(signal["stop"]),
                "range_high": float(signal["range_high"]),
                "range_low": float(signal["range_low"]),
                "equilibrium": float(signal["equilibrium"]),
                "correct_half": signal.get("correct_half"),
                **result,
            })

    trades = pd.DataFrame(rows)
    if trades.empty:
        summary = pd.DataFrame([{
            "scope": "ALL",
            "trades": 0,
            "total_r": 0.0,
            "expectancy_r": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "max_drawdown_r": 0.0,
            "tp1_rate": np.nan,
            "tp2_rate": np.nan,
            "positive_growth": False,
        }])
        equity = pd.DataFrame(columns=["signal_time_utc", "trade_r", "equity_r", "peak_r", "drawdown_r"])
    else:
        trades = trades.sort_values("signal_time_utc").reset_index(drop=True)
        trades["equity_r"] = cfg.starting_equity_r + trades["total_r"].cumsum()
        trades["peak_r"] = trades["equity_r"].cummax()
        trades["drawdown_r"] = trades["equity_r"] - trades["peak_r"]
        equity = trades[["signal_time_utc", "total_r", "equity_r", "peak_r", "drawdown_r"]].rename(columns={"total_r": "trade_r"})

        def summarize(scope: str, grp: pd.DataFrame) -> dict:
            wins = grp.loc[grp["total_r"] > 0, "total_r"]
            losses = grp.loc[grp["total_r"] < 0, "total_r"]
            gross_profit = float(wins.sum())
            gross_loss = float(-losses.sum())
            eq = grp["total_r"].cumsum()
            peak = eq.cummax()
            dd = eq - peak
            pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else np.nan)
            return {
                "scope": scope,
                "trades": int(len(grp)),
                "total_r": float(grp["total_r"].sum()),
                "expectancy_r": float(grp["total_r"].mean()),
                "win_rate": float((grp["total_r"] > 0).mean()),
                "profit_factor": float(pf) if np.isfinite(pf) else pf,
                "max_drawdown_r": float(dd.min()) if len(dd) else 0.0,
                "tp1_rate": float(grp["tp1_hit"].mean()),
                "tp2_rate": float(grp["tp2_hit"].mean()),
                "positive_growth": bool(grp["total_r"].sum() > 0),
            }

        summary_rows = [summarize("ALL", trades)]
        for session, grp in trades.groupby("session", sort=False):
            summary_rows.append(summarize(session, grp))
        summary = pd.DataFrame(summary_rows)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "v4_split_trades.csv", index=False)
    equity.to_csv(out / "v4_split_equity.csv", index=False)
    summary.to_csv(out / "v4_split_summary.csv", index=False)
    return summary, trades, equity
