from schemas import TradingViewPayload


def test_payload_validates():
    payload = {
        "secret": "12345678",
        "version": "1.0.1",
        "scan_type": "AM",
        "time_ny": "2026-09-18 09:00 NY",
        "symbols": [
            {
                "label": "XAUUSD",
                "ticker": "OANDA:XAUUSD",
                "bias": "BULLISH",
                "trend_score": 5,
                "sweep": "LOW",
                "close_inside": True,
                "valid_crt": True,
                "direction": "BULLISH",
                "aligned": True,
                "c1_high": 3680.0,
                "c1_low": 3650.0,
                "c1_mid": 3665.0,
                "c2_close": 3658.0,
            }
        ],
    }
    parsed = TradingViewPayload.model_validate(payload)
    assert parsed.symbols[0].valid_crt is True
