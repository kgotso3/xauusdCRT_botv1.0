import pandas as pd

from backtesting.v4_2_causal import _find_mss_fvg


def _bars(rows):
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])


def test_mss_can_use_completed_pre_start_bars_as_context():
    bars = _bars([
        ("2026-09-10T09:00:00Z", 100, 101, 99, 100),
        ("2026-09-10T09:15:00Z", 100, 102, 99, 101),
        ("2026-09-10T09:30:00Z", 101, 103, 100, 102),
        ("2026-09-10T09:45:00Z", 102, 105, 101, 104),
        ("2026-09-10T10:00:00Z", 104, 106, 104, 105),
        ("2026-09-10T10:15:00Z", 105, 108, 105, 107),
    ])
    out = _find_mss_fvg(
        bars,
        pd.Timestamp("2026-09-10T10:00:00Z"),
        pd.Timestamp("2026-09-10T11:00:00Z"),
        "BULLISH",
        15,
        3,
    )
    assert out["mss"] is True
    assert out["mss_time"] == pd.Timestamp("2026-09-10T10:00:00Z")


def test_confirmation_bar_must_be_closed_before_window_end():
    bars = _bars([
        ("2026-09-10T09:00:00Z", 100, 101, 99, 100),
        ("2026-09-10T09:15:00Z", 100, 102, 99, 101),
        ("2026-09-10T09:30:00Z", 101, 103, 100, 102),
        ("2026-09-10T10:00:00Z", 102, 110, 102, 109),
    ])
    out = _find_mss_fvg(
        bars,
        pd.Timestamp("2026-09-10T10:00:00Z"),
        pd.Timestamp("2026-09-10T10:10:00Z"),
        "BULLISH",
        15,
        3,
    )
    assert out["mss"] is False
