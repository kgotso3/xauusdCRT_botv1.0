from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from backtesting.killzones import DEFAULT_KILLZONES
from features.indicators import add_indicators, trend_bias
from strategy.crt import detect_crt

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class V2Rule:
    name: str
    killzone: str
    direction: str
    alignment_count: int | None = None
    target_r: float = 1.5
    enabled: bool = False

    def validate(self) -> None:
        if self.killzone not in {k.name for k in DEFAULT_KILLZONES}:
            raise ValueError(f"unsupported killzone: {self.killzone}")
        if self.direction not in {"BUY", "SELL"}:
            raise ValueError("direction must be BUY or SELL")
        if self.alignment_count is not None and self.alignment_count not in {0, 1, 2, 3}:
            raise ValueError("alignment_count must be 0..3 or None")
        if self.target_r not in {1.0, 1.5}:
            raise ValueError("V2 target_r must be 1.0 or 1.5")


# Intentionally empty until walk-forward candidates pass the strict gate.
APPROVED_RULES: tuple[V2Rule, ...] = ()


def killzone_for_time(now: datetime) -> str:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    hour = now.astimezone(NY).hour
    for kz in DEFAULT_KILLZONES:
        if kz.contains_hour(hour):
            return kz.name
    return "OUTSIDE"


def _alignment_count(direction: str, h1_bias: str, m15_bias: str, m5_bias: str) -> int:
    expected = "BULLISH" if direction == "BUY" else "BEARISH"
    return sum(bias == expected for bias in (h1_bias, m15_bias, m5_bias))


def build_v2_signal(h1, m15, m5, now: datetime, rules: tuple[V2Rule, ...] | None = None) -> dict:
    """Build a deterministic V2 signal from CRT + killzone + validated rule filters.

    No rule is executable by default. A rule must be explicitly enabled after
    research validation. This function never selects/tunes rules from live data.
    """
    active_rules = rules if rules is not None else APPROVED_RULES
    for rule in active_rules:
        rule.validate()

    h1_i = add_indicators(h1)
    m15_i = add_indicators(m15)
    m5_i = add_indicators(m5)

    crt = detect_crt(h1_i)
    h1_bias = trend_bias(h1_i)
    m15_bias = trend_bias(m15_i)
    m5_bias = trend_bias(m5_i)
    kz = killzone_for_time(now)

    if not crt["swept"]:
        return {
            "approved": False, "direction": "NONE", "target_r": None,
            "killzone": kz, "alignment_count": None, "rule": None,
            "h1_bias": h1_bias, "m15_bias": m15_bias, "m5_bias": m5_bias,
            "crt": crt, "reason": "No completed H1 CRT sweep",
        }

    direction = "BUY" if crt["direction"] == "BULLISH" else "SELL"
    alignment = _alignment_count(direction, h1_bias, m15_bias, m5_bias)

    matched = []
    for rule in active_rules:
        if not rule.enabled:
            continue
        if rule.killzone != kz or rule.direction != direction:
            continue
        if rule.alignment_count is not None and rule.alignment_count != alignment:
            continue
        matched.append(rule)

    if not matched:
        return {
            "approved": False, "direction": direction, "target_r": None,
            "killzone": kz, "alignment_count": alignment, "rule": None,
            "h1_bias": h1_bias, "m15_bias": m15_bias, "m5_bias": m5_bias,
            "crt": crt, "reason": "CRT detected but no enabled V2 rule matched",
        }

    # If more than one validated rule matches, prefer the smaller target for the
    # project's win-rate-first policy. Rules remain deterministic/config-driven.
    chosen = sorted(matched, key=lambda r: (r.target_r, r.name))[0]
    return {
        "approved": True, "direction": direction, "target_r": chosen.target_r,
        "killzone": kz, "alignment_count": alignment, "rule": chosen.name,
        "h1_bias": h1_bias, "m15_bias": m15_bias, "m5_bias": m5_bias,
        "crt": crt, "reason": f"Matched enabled V2 rule: {chosen.name}",
    }
