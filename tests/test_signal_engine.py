from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from strategy.signal_engine import build_signal


def _bars():
    return pd.DataFrame([
        {"high": 100 + i * 0.1, "low": 99 + i * 0.1, "close": 99.5 + i * 0.1}
        for i in range(220)
    ])


def test_bullish_signal_maps_to_buy():
    df = _bars()
    with patch("strategy.signal_engine.trend_bias", side_effect=["BULLISH", "BULLISH", "BULLISH"]), \
         patch("strategy.signal_engine.detect_crt", return_value={"direction": "BULLISH", "swept": True, "score": 40}), \
         patch("strategy.signal_engine.in_ny_window", return_value=True):
        signal = build_signal(df, df, df, now=datetime.now(timezone.utc))
    assert signal["direction"] == "BUY"
    assert signal["approved"] is True


def test_bearish_signal_maps_to_sell():
    df = _bars()
    with patch("strategy.signal_engine.trend_bias", side_effect=["BEARISH", "BEARISH", "BEARISH"]), \
         patch("strategy.signal_engine.detect_crt", return_value={"direction": "BEARISH", "swept": True, "score": 40}), \
         patch("strategy.signal_engine.in_ny_window", return_value=True):
        signal = build_signal(df, df, df, now=datetime.now(timezone.utc))
    assert signal["direction"] == "SELL"
    assert signal["approved"] is True
