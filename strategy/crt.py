from __future__ import annotations

import pandas as pd


def detect_crt(df: pd.DataFrame) -> dict:
    """Detect a conservative two-candle CRT-style sweep.

    The previous completed candle defines the reference range. The latest
    completed candle must sweep one side and close back inside that range.
    """
    if len(df) < 3:
        return {"direction": "NONE", "swept": False, "score": 0}

    prev = df.iloc[-2]
    cur = df.iloc[-1]
    direction = "NONE"
    swept = False

    bullish = cur["low"] < prev["low"] and cur["close"] > prev["low"] and cur["close"] <= prev["high"]
    bearish = cur["high"] > prev["high"] and cur["close"] < prev["high"] and cur["close"] >= prev["low"]

    if bullish:
        direction, swept = "BULLISH", True
    elif bearish:
        direction, swept = "BEARISH", True

    return {"direction": direction, "swept": swept, "score": 40 if swept else 0}
