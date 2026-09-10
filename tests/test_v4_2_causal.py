import pandas as pd

from backtesting.session_crt_split import SplitTargetConfig
from backtesting.v4_2_causal import _find_mss_fvg, build_causal_setups


def _df(rows):
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])


def test_mss_timestamp_is_bar_close_not_open():
    m15 = _df([
        ["2026-01-01T10:00:00Z", 100, 101, 99, 100],
        ["2026-01-01T10:15:00Z", 100, 102, 99.5, 101],
        ["2026-01-01T10:30:00Z", 101, 103, 100, 102],
        ["2026-01-01T10:45:00Z", 102, 105, 101.5, 104.5],
    ])
    m15["time"] = pd.to_datetime(m15["time"], utc=True)
    result = _find_mss_fvg(m15, pd.Timestamp("2026-01-01T10:00:00Z"), pd.Timestamp("2026-01-01T12:00:00Z"), "BULLISH", 15, 3)
    assert result["mss"] is True
    assert result["mss_time"] == pd.Timestamp("2026-01-01T11:00:00Z")


def test_fvg_must_be_at_or_after_mss():
    m15 = _df([
        ["2026-01-01T10:00:00Z", 100, 101, 99, 100],
        ["2026-01-01T10:15:00Z", 100, 101.5, 100.5, 101],
        ["2026-01-01T10:30:00Z", 102, 103, 102, 102.5],
        ["2026-01-01T10:45:00Z", 102.5, 104, 102, 103.5],
        ["2026-01-01T11:00:00Z", 103.5, 106, 103, 105.5],
        ["2026-01-01T11:15:00Z", 105.5, 108, 105, 107],
    ])
    m15["time"] = pd.to_datetime(m15["time"], utc=True)
    result = _find_mss_fvg(m15, pd.Timestamp("2026-01-01T10:00:00Z"), pd.Timestamp("2026-01-01T12:00:00Z"), "BULLISH", 15, 3)
    assert result["mss"] is True
    if result["fvg"]:
        assert result["fvg_time"] >= result["mss_time"]


def test_h1_reclaim_is_known_one_hour_after_h1_open():
    h1 = _df([
        ["2026-01-01T08:00:00Z", 100, 110, 100, 106],
        ["2026-01-01T09:00:00Z", 106, 108, 98, 104],
        ["2026-01-01T10:00:00Z", 104, 112, 103, 111],
    ])
    m15 = _df([
        ["2026-01-01T10:00:00Z", 104, 105, 103, 104.5],
        ["2026-01-01T10:15:00Z", 104.5, 106, 104, 105.5],
        ["2026-01-01T10:30:00Z", 105.5, 107, 105, 106.5],
        ["2026-01-01T10:45:00Z", 106.5, 109, 106, 108.5],
    ])
    trades = build_causal_setups(h1, m15, split_config=SplitTargetConfig(max_holding_bars=4))
    first = trades.iloc[0]
    assert first["signal_open_time_utc"] == pd.Timestamp("2026-01-01T09:00:00Z")
    assert first["signal_close_time_utc"] == pd.Timestamp("2026-01-01T10:00:00Z")


def test_double_sweep_is_not_candidate():
    h1 = _df([
        ["2026-01-01T08:00:00Z", 100, 110, 100, 106],
        ["2026-01-01T09:00:00Z", 106, 112, 98, 105],
    ])
    m15 = _df([
        ["2026-01-01T10:00:00Z", 105, 106, 104, 105],
        ["2026-01-01T10:15:00Z", 105, 106, 104, 105],
        ["2026-01-01T10:30:00Z", 105, 106, 104, 105],
        ["2026-01-01T10:45:00Z", 105, 106, 104, 105],
    ])
    assert build_causal_setups(h1, m15).empty
