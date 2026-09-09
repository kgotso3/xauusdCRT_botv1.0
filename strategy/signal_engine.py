from __future__ import annotations

from features.indicators import add_indicators, trend_bias
from strategy.crt import detect_crt


def build_signal(h1, m15, m5, minimum_score: int = 70) -> dict:
    h1 = add_indicators(h1)
    m15 = add_indicators(m15)
    m5 = add_indicators(m5)

    bias = trend_bias(h1)
    crt = detect_crt(h1)
    m15_bias = trend_bias(m15)
    m5_bias = trend_bias(m5)

    score = 0
    if bias in {"BULLISH", "BEARISH"}:
        score += 20
    if crt["swept"] and crt["direction"] == bias:
        score += 40
    if m15_bias == bias:
        score += 20
    if m5_bias == bias:
        score += 20

    direction = bias if score >= minimum_score and bias != "NEUTRAL" else "NONE"
    return {
        "direction": direction,
        "score": score,
        "h1_bias": bias,
        "m15_bias": m15_bias,
        "m5_bias": m5_bias,
        "crt": crt,
        "approved": direction != "NONE",
    }
