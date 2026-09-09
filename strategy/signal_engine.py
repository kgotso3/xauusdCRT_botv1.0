from __future__ import annotations

from datetime import datetime, timezone

from features.indicators import add_indicators, trend_bias
from strategy.crt import detect_crt
from strategy.session import in_ny_window


def build_signal(h1, m15, m5, now: datetime | None = None, minimum_score: int = 70) -> dict:
    now = now or datetime.now(timezone.utc)
    session_ok = in_ny_window(now)

    h1 = add_indicators(h1)
    m15 = add_indicators(m15)
    m5 = add_indicators(m5)

    bias = trend_bias(h1)
    crt = detect_crt(h1)
    m15_bias = trend_bias(m15)
    m5_bias = trend_bias(m5)

    score = 0
    reasons: list[str] = []

    if bias in {"BULLISH", "BEARISH"}:
        score += 20
        reasons.append(f"H1 bias {bias}")
    if crt["swept"] and crt["direction"] == bias:
        score += 40
        reasons.append("H1 CRT sweep agrees with bias")
    if m15_bias == bias and bias != "NEUTRAL":
        score += 20
        reasons.append("M15 confirms H1")
    if m5_bias == bias and bias != "NEUTRAL":
        score += 20
        reasons.append("M5 confirms H1")

    direction = bias if session_ok and score >= minimum_score and bias != "NEUTRAL" else "NONE"
    return {
        "direction": direction,
        "score": score,
        "h1_bias": bias,
        "m15_bias": m15_bias,
        "m5_bias": m5_bias,
        "crt": crt,
        "ny_session": session_ok,
        "reasons": reasons,
        "approved": direction != "NONE",
    }
