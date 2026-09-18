from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

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


def symbol_row(label: str, ticker: str, index: int) -> dict:
    valid = index in {0, 1, 5}
    direction = "BULLISH" if index != 1 else "BEARISH"
    bias = direction if index != 5 else "BEARISH"
    return {
        "label": label,
        "ticker": ticker,
        "bias": bias,
        "trend_score": 5 if bias == "BULLISH" else -5,
        "sweep": "LOW" if direction == "BULLISH" else "HIGH",
        "close_inside": valid,
        "valid_crt": valid,
        "direction": direction if valid else "-",
        "aligned": valid and bias == direction,
        "c1_high": 100.0 + index,
        "c1_low": 90.0 + index,
        "c1_mid": 95.0 + index,
        "c2_close": 92.0 + index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a 10-market CRT Scanner V1 test payload.")
    parser.add_argument("--url", required=True, help="Webhook URL ending in /tradingview")
    parser.add_argument("--secret", required=True, help="Value matching TV_WEBHOOK_SECRET")
    parser.add_argument("--scan", choices=["AM", "PM"], default="AM")
    args = parser.parse_args()

    now_ny = datetime.now(ZoneInfo("America/New_York"))
    payload = {
        "secret": args.secret,
        "version": "1.0.4-test",
        "scan_type": args.scan,
        "time_ny": now_ny.strftime("%Y-%m-%d %H:%M NY"),
        "symbols": [symbol_row(label, ticker, i) for i, (label, ticker) in enumerate(SYMBOLS)],
    }

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            print(response.status, response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.code, exc.read().decode("utf-8"))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
