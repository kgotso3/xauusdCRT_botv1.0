from __future__ import annotations

import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["ema200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    out["rsi14"] = 100 - (100 / (1 + rs))

    high_low = out["high"] - out["low"]
    high_close = (out["high"] - close.shift()).abs()
    low_close = (out["low"] - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()
    return out


def trend_bias(df: pd.DataFrame) -> str:
    row = df.iloc[-1]
    if row["ema20"] > row["ema50"] > row["ema200"]:
        return "BULLISH"
    if row["ema20"] < row["ema50"] < row["ema200"]:
        return "BEARISH"
    return "NEUTRAL"
