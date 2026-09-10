from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtesting.v4_2_causal import build_causal_setups
from backtesting.v4_6_entry_retest import _fvg_zone_at_confirmation, _ote_entry


FROZEN_ENTRY_LEVEL = 0.79
FROZEN_BENCHMARK = {
    "entry_model": "OTE_0_79",
    "fill_rate": 0.194320,
    "expectancy_r": 0.113846,
    "profit_factor": 1.242147,
    "max_drawdown_r": -7.36,
    "cost_r": 0.04,
}


@dataclass(frozen=True)
class ShadowConfig:
    target_closed_fills: int = 40
    retest_window_minutes: int = 120
    holding_window_hours: int = 24
    risk_fraction: float = 0.01
    history_h1: int = 400
    history_m15: int = 1200
    history_m5: int = 3000

    def __post_init__(self) -> None:
        if not 30 <= self.target_closed_fills <= 50:
            raise ValueError("target_closed_fills must be between 30 and 50")
        if self.retest_window_minutes <= 0 or self.holding_window_hours <= 0:
            raise ValueError("shadow windows must be positive")
        if not 0 < self.risk_fraction < 1:
            raise ValueError("risk_fraction must be between 0 and 1")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts(value: Any) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def load_state(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"started_at_utc": _utc_now(), "last_completed_m5": None, "signals": {}}
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("started_at_utc", _utc_now())
    data.setdefault("last_completed_m5", None)
    data.setdefault("signals", {})
    return data


def save_state(path: str | Path, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True, default=str)
    tmp.replace(p)


def signal_key(symbol: str, confirmation_time: Any, direction: str) -> str:
    return f"{symbol}|V5|{_ts(confirmation_time).isoformat()}|{direction.upper()}|OTE079"


def _candidate_from_latest_causal(h1: pd.DataFrame, m15: pd.DataFrame, m5: pd.DataFrame, started_at: Any) -> dict | None:
    setups = build_causal_setups(h1, m15, m5)
    if setups.empty or "m5_fvg" not in setups.columns:
        return None
    correct_half = setups["correct_half"].astype("boolean").fillna(False)
    m5_fvg = setups["m5_fvg"].astype("boolean").fillna(False)
    x = setups.loc[correct_half & m5_fvg].copy()
    if x.empty:
        return None
    x["m5_fvg_time"] = pd.to_datetime(x["m5_fvg_time"], utc=True, errors="coerce")
    x = x.dropna(subset=["m5_fvg_time"]).sort_values("m5_fvg_time")
    if x.empty:
        return None
    row = x.iloc[-1]
    conf = _ts(row["m5_fvg_time"])
    if conf < _ts(started_at):
        return None
    zone = _fvg_zone_at_confirmation(m5.reset_index(drop=True), conf, str(row["direction"]))
    if zone is None:
        return None
    stop = float(row["stop"])
    entry = float(_ote_entry(str(row["direction"]), stop, float(zone["impulse_extreme"]), FROZEN_ENTRY_LEVEL))
    valid = entry > stop if str(row["direction"]) == "BULLISH" else entry < stop
    if not valid:
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    tp1 = entry + risk if str(row["direction"]) == "BULLISH" else entry - risk
    tp2 = entry + 2 * risk if str(row["direction"]) == "BULLISH" else entry - 2 * risk
    return {
        "confirmation_time_utc": conf.isoformat(),
        "signal_close_time_utc": _ts(row["signal_close_time_utc"]).isoformat(),
        "direction": str(row["direction"]),
        "entry": entry,
        "stop": stop,
        "tp1": float(tp1),
        "tp2": float(tp2),
        "risk_distance": float(risk),
        "fvg_low": float(zone["fvg_low"]),
        "fvg_high": float(zone["fvg_high"]),
        "impulse_extreme": float(zone["impulse_extreme"]),
        "ny_hour": int(row["ny_hour"]),
    }


def add_new_candidate(state: dict, symbol: str, candidate: dict | None, observed_at: Any) -> str | None:
    if candidate is None:
        return None
    key = signal_key(symbol, candidate["confirmation_time_utc"], candidate["direction"])
    if key in state["signals"]:
        return None
    conf = _ts(candidate["confirmation_time_utc"])
    if conf < _ts(state["started_at_utc"]):
        return None
    state["signals"][key] = {
        **candidate,
        "signal_id": key,
        "symbol": symbol,
        "entry_model": "OTE_0_79",
        "status": "PENDING_RETEST",
        "first_seen_utc": _ts(observed_at).isoformat(),
        "retest_expiry_utc": (conf + pd.Timedelta(minutes=120)).isoformat(),
        "filled_at_utc": None,
        "closed_at_utc": None,
        "tp1_hit": False,
        "tp2_hit": False,
        "gross_r": None,
        "spread_at_signal": None,
        "spread_at_fill": None,
        "spread_r_at_fill": None,
        "shadow_slippage_r": None,
        "volume": None,
        "risk_budget": None,
        "estimated_stop_loss": None,
        "sizing_error": None,
    }
    return key


def _close(sig: dict, when: Any, status: str, gross_r: float) -> None:
    sig["status"] = status
    sig["closed_at_utc"] = _ts(when).isoformat()
    sig["gross_r"] = float(gross_r)


def update_signal_with_tick(
    sig: dict,
    tick: dict,
    account_equity: float | None = None,
    volume_info: tuple | None = None,
    risk_fraction: float = 0.01,
) -> None:
    if sig["status"] in {"CLOSED_STOP", "CLOSED_TP2", "CLOSED_TIMEOUT", "CANCELLED_STOP_BEFORE_FILL", "CANCELLED_NO_RETEST"}:
        return
    now = _ts(tick["time"])
    direction = sig["direction"]
    entry = float(sig["entry"]); stop = float(sig["stop"]); tp1 = float(sig["tp1"]); tp2 = float(sig["tp2"])
    bid = float(tick["bid"]); ask = float(tick["ask"])

    if sig["status"] == "PENDING_RETEST":
        if now > _ts(sig["retest_expiry_utc"]):
            sig["status"] = "CANCELLED_NO_RETEST"
            sig["closed_at_utc"] = now.isoformat()
            return
        stop_touched = bid <= stop if direction == "BULLISH" else ask >= stop
        fill_touched = ask <= entry if direction == "BULLISH" else bid >= entry
        if stop_touched:
            sig["status"] = "CANCELLED_STOP_BEFORE_FILL"
            sig["closed_at_utc"] = now.isoformat()
            return
        if fill_touched:
            fill_price = ask if direction == "BULLISH" else bid
            sig["status"] = "FILLED"
            sig["filled_at_utc"] = now.isoformat()
            sig["fill_price"] = float(fill_price)
            sig["spread_at_fill"] = float(tick.get("spread", ask - bid))
            sig["spread_points_at_fill"] = float(tick.get("spread_points", np.nan))
            sig["spread_r_at_fill"] = float(sig["spread_at_fill"] / sig["risk_distance"])
            if direction == "BULLISH":
                sig["shadow_slippage_r"] = float((fill_price - entry) / sig["risk_distance"])
            else:
                sig["shadow_slippage_r"] = float((entry - fill_price) / sig["risk_distance"])
            if volume_info is not None:
                sig["volume"], sig["risk_budget"], sig["estimated_stop_loss"] = map(float, volume_info)
            elif account_equity is not None:
                sig["risk_budget"] = float(account_equity) * float(risk_fraction)
        return

    if sig["status"] in {"FILLED", "TP1_HIT"}:
        filled_at = _ts(sig["filled_at_utc"])
        if now > filled_at + pd.Timedelta(hours=24):
            _close(sig, now, "CLOSED_TIMEOUT", 0.5 if sig["tp1_hit"] else 0.0)
            return
        exit_price = bid if direction == "BULLISH" else ask
        stop_touched = exit_price <= stop if direction == "BULLISH" else exit_price >= stop
        tp1_touched = exit_price >= tp1 if direction == "BULLISH" else exit_price <= tp1
        tp2_touched = exit_price >= tp2 if direction == "BULLISH" else exit_price <= tp2
        if stop_touched:
            _close(sig, now, "CLOSED_STOP", 0.0 if sig["tp1_hit"] else -1.0)
            return
        if not sig["tp1_hit"] and tp1_touched:
            sig["tp1_hit"] = True
            sig["status"] = "TP1_HIT"
            sig["tp1_hit_at_utc"] = now.isoformat()
        if tp2_touched:
            sig["tp1_hit"] = True
            sig["tp2_hit"] = True
            _close(sig, now, "CLOSED_TP2", 1.5)


def shadow_metrics(state: dict) -> dict:
    signals = list(state.get("signals", {}).values())
    setups = len(signals)
    fills = [s for s in signals if s.get("filled_at_utc")]
    closed = [s for s in fills if s.get("gross_r") is not None]
    r = np.array([float(s["gross_r"]) for s in closed], dtype=float)
    if len(r):
        wins = float(r[r > 0].sum()); losses = float(-r[r < 0].sum())
        pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
        eq = np.cumsum(r); peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
        dd = float((eq - peak).min())
        expectancy = float(r.mean())
    else:
        pf = np.nan; dd = 0.0; expectancy = np.nan
    spreads = [float(s["spread_r_at_fill"]) for s in fills if s.get("spread_r_at_fill") is not None]
    slips = [float(s["shadow_slippage_r"]) for s in fills if s.get("shadow_slippage_r") is not None]
    return {
        "prospective_setups": setups,
        "prospective_fills": len(fills),
        "closed_fills": len(closed),
        "fill_rate": float(len(fills) / setups) if setups else np.nan,
        "expectancy_r": expectancy,
        "profit_factor": float(pf),
        "max_drawdown_r": dd,
        "avg_spread_r": float(np.mean(spreads)) if spreads else np.nan,
        "median_spread_r": float(np.median(spreads)) if spreads else np.nan,
        "avg_shadow_slippage_r": float(np.mean(slips)) if slips else np.nan,
    }


def comparison_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for metric in ["fill_rate", "expectancy_r", "profit_factor", "max_drawdown_r"]:
        rows.append({"metric": metric, "v4_7_benchmark": FROZEN_BENCHMARK[metric], "v5_shadow": metrics.get(metric), "difference": (metrics.get(metric) - FROZEN_BENCHMARK[metric]) if pd.notna(metrics.get(metric)) else np.nan})
    return pd.DataFrame(rows)


def export_reports(state: dict, output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = list(state.get("signals", {}).values())
    journal = pd.DataFrame(rows)
    metrics = shadow_metrics(state)
    summary = pd.DataFrame([metrics])
    journal.to_csv(out / "v5_shadow_journal.csv", index=False)
    summary.to_csv(out / "v5_shadow_summary.csv", index=False)
    comparison_frame(metrics).to_csv(out / "v5_vs_v4_7_benchmark.csv", index=False)
    return journal, summary
