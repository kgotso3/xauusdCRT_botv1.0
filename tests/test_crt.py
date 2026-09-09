import pandas as pd

from strategy.crt import detect_crt


def test_bullish_sweep():
    df = pd.DataFrame([
        {"high": 100, "low": 95, "close": 98},
        {"high": 99, "low": 94, "close": 97},
    ])
    result = detect_crt(df)
    assert result["direction"] == "BULLISH"
    assert result["swept"] is True


def test_bearish_sweep():
    df = pd.DataFrame([
        {"high": 100, "low": 95, "close": 98},
        {"high": 101, "low": 96, "close": 99},
    ])
    result = detect_crt(df)
    assert result["direction"] == "BEARISH"
    assert result["swept"] is True


def test_no_sweep():
    df = pd.DataFrame([
        {"high": 100, "low": 95, "close": 98},
        {"high": 99, "low": 96, "close": 98},
    ])
    result = detect_crt(df)
    assert result["direction"] == "NONE"
    assert result["swept"] is False


def test_requires_two_candles():
    df = pd.DataFrame([
        {"high": 100, "low": 95, "close": 98},
    ])
    result = detect_crt(df)
    assert result["direction"] == "NONE"
    assert result["swept"] is False
