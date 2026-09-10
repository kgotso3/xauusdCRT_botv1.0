from __future__ import annotations

import pandas as pd

from shadow.v5_engine import (
    ShadowConfig,
    add_new_candidate,
    shadow_metrics,
    update_signal_with_tick,
)


def _candidate(t="2026-09-10T10:00:00Z", direction="BULLISH"):
    return {
        "confirmation_time_utc": t,
        "signal_close_time_utc": "2026-09-10T09:00:00Z",
        "direction": direction,
        "entry": 2500.0,
        "stop": 2490.0 if direction == "BULLISH" else 2510.0,
        "tp1": 2510.0 if direction == "BULLISH" else 2490.0,
        "tp2": 2520.0 if direction == "BULLISH" else 2480.0,
        "risk_distance": 10.0,
        "fvg_low": 2502.0,
        "fvg_high": 2504.0,
        "impulse_extreme": 2537.619,
        "ny_hour": 6,
    }


def test_shadow_target_is_limited_to_30_50_and_default_risk_is_one_percent():
    cfg = ShadowConfig(target_closed_fills=40)
    assert cfg.target_closed_fills == 40
    assert cfg.risk_fraction == 0.01
    try:
        ShadowConfig(target_closed_fills=51)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_old_confirmation_cannot_enter_prospective_cohort():
    state = {"started_at_utc": "2026-09-10T12:00:00Z", "signals": {}, "last_completed_m5": None}
    added = add_new_candidate(state, "GOLD", _candidate("2026-09-10T10:00:00Z"), "2026-09-10T12:01:00Z")
    assert added is None
    assert state["signals"] == {}


def test_buy_shadow_fill_uses_ask_then_tracks_tp2_with_bid():
    state = {"started_at_utc": "2026-09-10T09:00:00Z", "signals": {}, "last_completed_m5": None}
    key = add_new_candidate(state, "GOLD", _candidate(), "2026-09-10T10:00:01Z")
    sig = state["signals"][key]
    update_signal_with_tick(sig, {"time": pd.Timestamp("2026-09-10T10:10:00Z"), "bid": 2499.8, "ask": 2500.0, "spread": 0.2, "spread_points": 20})
    assert sig["status"] == "FILLED"
    assert sig["spread_r_at_fill"] == 0.02
    update_signal_with_tick(sig, {"time": pd.Timestamp("2026-09-10T11:00:00Z"), "bid": 2510.0, "ask": 2510.2, "spread": 0.2, "spread_points": 20})
    assert sig["status"] == "TP1_HIT"
    update_signal_with_tick(sig, {"time": pd.Timestamp("2026-09-10T12:00:00Z"), "bid": 2520.0, "ask": 2520.2, "spread": 0.2, "spread_points": 20})
    assert sig["status"] == "CLOSED_TP2"
    assert sig["gross_r"] == 1.5


def test_tp1_then_stop_is_zero_r_and_metrics_count_closed_fill():
    state = {"started_at_utc": "2026-09-10T09:00:00Z", "signals": {}, "last_completed_m5": None}
    key = add_new_candidate(state, "GOLD", _candidate(), "2026-09-10T10:00:01Z")
    sig = state["signals"][key]
    update_signal_with_tick(sig, {"time": pd.Timestamp("2026-09-10T10:10:00Z"), "bid": 2499.8, "ask": 2500.0, "spread": 0.2, "spread_points": 20})
    update_signal_with_tick(sig, {"time": pd.Timestamp("2026-09-10T11:00:00Z"), "bid": 2510.0, "ask": 2510.2, "spread": 0.2, "spread_points": 20})
    update_signal_with_tick(sig, {"time": pd.Timestamp("2026-09-10T12:00:00Z"), "bid": 2489.9, "ask": 2490.1, "spread": 0.2, "spread_points": 20})
    assert sig["gross_r"] == 0.0
    m = shadow_metrics(state)
    assert m["prospective_fills"] == 1
    assert m["closed_fills"] == 1


def test_shadow_fallback_risk_budget_uses_configured_one_percent():
    state = {"started_at_utc": "2026-09-10T09:00:00Z", "signals": {}, "last_completed_m5": None}
    key = add_new_candidate(state, "GOLD", _candidate(), "2026-09-10T10:00:01Z")
    sig = state["signals"][key]
    update_signal_with_tick(
        sig,
        {"time": pd.Timestamp("2026-09-10T10:10:00Z"), "bid": 2499.8, "ask": 2500.0, "spread": 0.2, "spread_points": 20},
        account_equity=100000.0,
        volume_info=None,
        risk_fraction=0.01,
    )
    assert sig["status"] == "FILLED"
    assert sig["risk_budget"] == 1000.0
