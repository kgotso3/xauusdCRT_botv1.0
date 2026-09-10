from __future__ import annotations

import pandas as pd

from backtesting.v4_6_entry_retest import (
    EntryRetestConfig,
    _fvg_zone_at_confirmation,
    _limit_fill,
    _ote_entry,
)


def _m5_fixture():
    t = pd.date_range("2026-01-01T10:00:00Z", periods=7, freq="5min")
    return pd.DataFrame({
        "time": t,
        "open": [100.0, 100.2, 100.4, 101.0, 101.4, 101.1, 101.3],
        "high": [100.4, 100.6, 101.2, 101.6, 101.5, 101.4, 101.8],
        "low":  [99.8, 100.0, 100.8, 101.1, 101.0, 100.9, 101.0],
        "close":[100.2, 100.4, 101.0, 101.4, 101.1, 101.3, 101.6],
    })


def test_fvg_zone_only_exists_when_third_candle_has_closed():
    bars = _m5_fixture()
    # Third candle opens 10:10 and closes 10:15. first.high=100.4, third.low=100.8.
    zone = _fvg_zone_at_confirmation(bars, pd.Timestamp("2026-01-01T10:15:00Z"), "BULLISH")
    assert zone is not None
    assert zone["fvg_low"] == 100.4
    assert zone["fvg_high"] == 100.8


def test_limit_fill_cancels_if_stop_is_touched_before_retest():
    bars = _m5_fixture().copy()
    bars.loc[3, "low"] = 98.5
    fill_time, _, status = _limit_fill(
        bars,
        pd.Timestamp("2026-01-01T10:15:00Z"),
        "BULLISH",
        100.6,
        99.0,
        4,
    )
    assert pd.isna(fill_time)
    assert status == "STOP_BEFORE_FILL"


def test_ote_entry_moves_deeper_as_retracement_increases():
    shallow = _ote_entry("BULLISH", 99.0, 103.0, 0.62)
    deep = _ote_entry("BULLISH", 99.0, 103.0, 0.79)
    assert deep < shallow
    assert 99.0 < deep < 103.0


def test_entry_retest_config_rejects_invalid_retest_window():
    try:
        EntryRetestConfig(retest_window_bars=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
