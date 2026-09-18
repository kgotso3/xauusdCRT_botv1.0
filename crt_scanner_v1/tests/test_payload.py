import pytest
from pydantic import ValidationError

from schemas import TradingViewPayload


SYMBOLS = [
    ("XAUUSD", "OANDA:XAUUSD"),
    ("NAS100", "OANDA:NAS100USD"),
    ("US500", "OANDA:SPX500USD"),
    ("US30", "OANDA:US30USD"),
    ("EURUSD", "OANDA:EURUSD"),
    ("GBPUSD", "OANDA:GBPUSD"),
    ("USDJPY", "OANDA:USDJPY"),
    ("USDCAD", "OANDA:USDCAD"),
    ("AUDUSD", "OANDA:AUDUSD"),
    ("BTCUSD", "COINBASE:BTCUSD"),
]


def make_symbol(label: str, ticker: str) -> dict:
    return {
        "label": label,
        "ticker": ticker,
        "bias": "BULLISH",
        "trend_score": 5,
        "sweep": "LOW",
        "close_inside": True,
        "valid_crt": True,
        "direction": "BULLISH",
        "aligned": True,
        "c1_high": 100.0,
        "c1_low": 90.0,
        "c1_mid": 95.0,
        "c2_close": 92.0,
    }


def test_ten_symbol_payload_validates():
    payload = {
        "secret": "12345678",
        "version": "1.0.4",
        "scan_type": "AM",
        "time_ny": "2026-09-18 09:00 NY",
        "symbols": [make_symbol(label, ticker) for label, ticker in SYMBOLS],
    }
    parsed = TradingViewPayload.model_validate(payload)
    assert len(parsed.symbols) == 10
    assert parsed.symbols[0].label == "XAUUSD"
    assert parsed.symbols[-1].label == "BTCUSD"


def test_more_than_ten_symbols_is_rejected():
    payload = {
        "secret": "12345678",
        "version": "1.0.4",
        "scan_type": "AM",
        "time_ny": "2026-09-18 09:00 NY",
        "symbols": [make_symbol(label, ticker) for label, ticker in SYMBOLS]
        + [make_symbol("EXTRA", "TEST:EXTRA")],
    }
    with pytest.raises(ValidationError):
        TradingViewPayload.model_validate(payload)
