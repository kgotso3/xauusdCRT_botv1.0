from __future__ import annotations

import pandas as pd

from backtesting.killzones import DEFAULT_KILLZONES
from strategy.session_crt import SessionCRTConfig, detect_session_crt


def _bars(rows):
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])


def test_new_york_killzone_runs_to_14_ny_exclusive():
    ny = next(k for k in DEFAULT_KILLZONES if k.name == "NEW_YORK")
    assert ny.start_hour == 8
    assert ny.end_hour == 14
    assert ny.contains_hour(8)
    assert ny.contains_hour(13)
    assert not ny.contains_hour(14)


def test_bullish_session_crt_range_sweep_reclaim_and_expansion_ready():
    # January is EST: 13:00 UTC == 08:00 NY.
    df = _bars([
        ["2026-01-12T13:00:00Z", 100.0, 110.0, 100.0, 106.0],  # 08:00 range
        ["2026-01-12T14:00:00Z", 106.0, 107.0, 97.0, 102.0],   # 09:00 sweep low + reclaim
        ["2026-01-12T15:00:00Z", 102.0, 109.0, 101.0, 108.0],
    ])
    signal = detect_session_crt(df, "NEW_YORK", "2026-01-12", SessionCRTConfig(min_rr=2.0))
    assert signal["direction"] == "BULLISH"
    assert signal["phase"] == "EXPANSION_READY"
    assert signal["swept_low"] is True
    assert signal["reclaimed"] is True
    assert signal["target"] == 110.0
    assert signal["stop"] == 97.0
    assert signal["rr_to_opposite_range"] >= 2.0
    assert signal["valid"] is True


def test_bearish_session_crt_range_sweep_reclaim_and_expansion_ready():
    df = _bars([
        ["2026-01-12T13:00:00Z", 100.0, 110.0, 100.0, 104.0],
        ["2026-01-12T14:00:00Z", 104.0, 113.0, 103.0, 108.0],
    ])
    signal = detect_session_crt(df, "NEW_YORK", "2026-01-12", SessionCRTConfig(min_rr=2.0))
    assert signal["direction"] == "BEARISH"
    assert signal["phase"] == "EXPANSION_READY"
    assert signal["swept_high"] is True
    assert signal["target"] == 100.0
    assert signal["stop"] == 113.0
    assert signal["valid"] is True


def test_double_sweep_is_rejected_as_ambiguous():
    df = _bars([
        ["2026-01-12T13:00:00Z", 100.0, 110.0, 100.0, 105.0],
        ["2026-01-12T14:00:00Z", 105.0, 112.0, 98.0, 105.0],
    ])
    signal = detect_session_crt(df, "NEW_YORK", "2026-01-12")
    assert signal["phase"] == "AMBIGUOUS"
    assert signal["valid"] is False


def test_sweep_without_close_back_inside_is_not_confirmed():
    df = _bars([
        ["2026-01-12T13:00:00Z", 100.0, 110.0, 100.0, 105.0],
        ["2026-01-12T14:00:00Z", 105.0, 108.0, 96.0, 98.0],
    ])
    signal = detect_session_crt(df, "NEW_YORK", "2026-01-12")
    assert signal["direction"] == "BULLISH"
    assert signal["phase"] == "MANIPULATION_UNCONFIRMED"
    assert signal["reclaimed"] is False
    assert signal["valid"] is False
