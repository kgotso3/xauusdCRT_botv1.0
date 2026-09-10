from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtesting.killzones import DEFAULT_KILLZONES, Killzone

NY = ZoneInfo("America/New_York")
H1 = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class SessionCRTConfig:
    """Configuration for the H1 session-anchored CRT model."""

    min_rr: float = 2.0
    require_reclaim_close: bool = True
    reject_double_sweep: bool = True
    require_correct_half: bool = False


@dataclass(frozen=True)
class SessionCRTSignal:
    session: str
    session_date: str
    direction: str
    phase: str
    range_time_utc: pd.Timestamp
    manipulation_time_utc: pd.Timestamp | None
    range_high: float
    range_low: float
    equilibrium: float
    entry: float | None
    stop: float | None
    target: float | None
    rr_to_opposite_range: float | None
    swept_high: bool
    swept_low: bool
    reclaimed: bool
    correct_half: bool | None
    valid: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "session": self.session,
            "session_date": self.session_date,
            "direction": self.direction,
            "phase": self.phase,
            "range_time_utc": self.range_time_utc,
            "manipulation_time_utc": self.manipulation_time_utc,
            "range_high": self.range_high,
            "range_low": self.range_low,
            "equilibrium": self.equilibrium,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "rr_to_opposite_range": self.rr_to_opposite_range,
            "swept_high": self.swept_high,
            "swept_low": self.swept_low,
            "reclaimed": self.reclaimed,
            "correct_half": self.correct_half,
            "valid": self.valid,
            "reason": self.reason,
        }


def _normalize_h1(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        raise ValueError("H1 dataframe must contain a 'time' column")
    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"H1 dataframe missing required OHLC columns: {missing}")

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    out["ny_time"] = out["time"].dt.tz_convert(NY)
    out["ny_hour"] = out["ny_time"].dt.hour.astype(int)
    out["ny_date"] = out["ny_time"].dt.date
    return out


def _session_date_for_bar(ny_time: pd.Timestamp, kz: Killzone):
    date = ny_time.date()
    if kz.start_hour > kz.end_hour and ny_time.hour < kz.end_hour:
        return (ny_time - pd.Timedelta(days=1)).date()
    return date


def _session_rows(h1: pd.DataFrame, kz: Killzone) -> pd.DataFrame:
    mask = h1["ny_hour"].map(kz.contains_hour)
    out = h1.loc[mask].copy()
    if out.empty:
        return out
    out["session_date"] = out["ny_time"].map(lambda t: _session_date_for_bar(t, kz))
    return out


def _rr(entry: float, stop: float, target: float) -> float | None:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if not np.isfinite(risk) or risk <= 0:
        return None
    return float(reward / risk)


def detect_session_crt(
    h1: pd.DataFrame,
    session: str,
    session_date=None,
    config: SessionCRTConfig | None = None,
    killzones: tuple[Killzone, ...] = DEFAULT_KILLZONES,
) -> dict:
    """Detect H1 Range -> Manipulation -> Expansion logic inside one killzone.

    Rules:
    1. The first H1 candle opening inside the requested killzone defines C1, the
       CRT range.
    2. Later H1 candles in the same killzone may manipulate one side of C1.
    3. A bullish setup sweeps C1 low and closes back inside C1. A bearish setup
       sweeps C1 high and closes back inside C1.
    4. A candle that sweeps both sides is ambiguous and is rejected by default.
    5. The theoretical target is the opposite C1 extreme. The manipulation wick
       is the theoretical stop. Minimum 1:2 R:R is a configurable quality gate.

    This is an H1-native detector. M15/M5 MSS/FVG confirmation can be layered on
    after this function identifies a valid session CRT manipulation.
    """
    cfg = config or SessionCRTConfig()
    normalized = _normalize_h1(h1)

    kz = next((k for k in killzones if k.name.upper() == session.upper()), None)
    if kz is None:
        raise ValueError(f"Unknown session {session!r}")

    rows = _session_rows(normalized, kz)
    if rows.empty:
        return {"valid": False, "phase": "NO_SESSION_DATA", "session": kz.name, "reason": "no H1 bars in session"}

    if session_date is None:
        session_date = rows["session_date"].iloc[-1]
    else:
        session_date = pd.Timestamp(session_date).date()

    rows = rows.loc[rows["session_date"] == session_date].copy().reset_index(drop=True)
    if rows.empty:
        return {"valid": False, "phase": "NO_SESSION_DATA", "session": kz.name, "reason": "no H1 bars for requested session date"}

    c1 = rows.iloc[0]
    range_high = float(c1["high"])
    range_low = float(c1["low"])
    eq = (range_high + range_low) / 2.0

    if len(rows) == 1:
        return SessionCRTSignal(
            session=kz.name,
            session_date=str(session_date),
            direction="NONE",
            phase="RANGE",
            range_time_utc=pd.Timestamp(c1["time"]),
            manipulation_time_utc=None,
            range_high=range_high,
            range_low=range_low,
            equilibrium=eq,
            entry=None,
            stop=None,
            target=None,
            rr_to_opposite_range=None,
            swept_high=False,
            swept_low=False,
            reclaimed=False,
            correct_half=None,
            valid=False,
            reason="range established; waiting for manipulation",
        ).as_dict()

    for _, cur in rows.iloc[1:].iterrows():
        high = float(cur["high"])
        low = float(cur["low"])
        close = float(cur["close"])
        swept_high = high > range_high
        swept_low = low < range_low

        if not swept_high and not swept_low:
            continue

        if swept_high and swept_low and cfg.reject_double_sweep:
            return SessionCRTSignal(
                session=kz.name,
                session_date=str(session_date),
                direction="NONE",
                phase="AMBIGUOUS",
                range_time_utc=pd.Timestamp(c1["time"]),
                manipulation_time_utc=pd.Timestamp(cur["time"]),
                range_high=range_high,
                range_low=range_low,
                equilibrium=eq,
                entry=None,
                stop=None,
                target=None,
                rr_to_opposite_range=None,
                swept_high=True,
                swept_low=True,
                reclaimed=False,
                correct_half=None,
                valid=False,
                reason="manipulation candle swept both sides of the range",
            ).as_dict()

        inside_close = range_low < close < range_high

        if swept_low and not swept_high:
            direction = "BULLISH"
            reclaimed = close > range_low if cfg.require_reclaim_close else True
            entry = close
            stop = low
            target = range_high
            correct_half = close <= eq
        else:
            direction = "BEARISH"
            reclaimed = close < range_high if cfg.require_reclaim_close else True
            entry = close
            stop = high
            target = range_low
            correct_half = close >= eq

        reclaimed = bool(reclaimed and inside_close)
        rr = _rr(entry, stop, target) if reclaimed else None
        rr_ok = bool(rr is not None and rr >= cfg.min_rr)
        half_ok = bool(correct_half or not cfg.require_correct_half)
        valid = bool(reclaimed and rr_ok and half_ok)

        if not reclaimed:
            reason = "liquidity swept but H1 candle did not close back inside the range"
            phase = "MANIPULATION_UNCONFIRMED"
        elif not half_ok:
            reason = "reclaim confirmed but close is not in the required premium/discount half"
            phase = "MANIPULATION_CONFIRMED"
        elif not rr_ok:
            reason = f"reclaim confirmed but opposite-range target offers less than {cfg.min_rr:.1f}R"
            phase = "MANIPULATION_CONFIRMED"
        else:
            reason = "range swept and reclaimed; expansion setup confirmed"
            phase = "EXPANSION_READY"

        return SessionCRTSignal(
            session=kz.name,
            session_date=str(session_date),
            direction=direction,
            phase=phase,
            range_time_utc=pd.Timestamp(c1["time"]),
            manipulation_time_utc=pd.Timestamp(cur["time"]),
            range_high=range_high,
            range_low=range_low,
            equilibrium=eq,
            entry=entry,
            stop=stop,
            target=target,
            rr_to_opposite_range=rr,
            swept_high=swept_high,
            swept_low=swept_low,
            reclaimed=reclaimed,
            correct_half=bool(correct_half),
            valid=valid,
            reason=reason,
        ).as_dict()

    return SessionCRTSignal(
        session=kz.name,
        session_date=str(session_date),
        direction="NONE",
        phase="RANGE",
        range_time_utc=pd.Timestamp(c1["time"]),
        manipulation_time_utc=None,
        range_high=range_high,
        range_low=range_low,
        equilibrium=eq,
        entry=None,
        stop=None,
        target=None,
        rr_to_opposite_range=None,
        swept_high=False,
        swept_low=False,
        reclaimed=False,
        correct_half=None,
        valid=False,
        reason="range established; no qualifying manipulation yet",
    ).as_dict()


def scan_latest_session_crt(
    h1: pd.DataFrame,
    config: SessionCRTConfig | None = None,
    killzones: tuple[Killzone, ...] = DEFAULT_KILLZONES,
) -> list[dict]:
    """Return the latest H1 CRT state for Asia, London and New York."""
    normalized = _normalize_h1(h1)
    states: list[dict] = []
    for kz in killzones:
        session_rows = _session_rows(normalized, kz)
        if session_rows.empty:
            states.append({"valid": False, "phase": "NO_SESSION_DATA", "session": kz.name, "reason": "no H1 bars in session"})
            continue
        latest_date = session_rows["session_date"].iloc[-1]
        states.append(detect_session_crt(normalized.drop(columns=["ny_time", "ny_hour", "ny_date"], errors="ignore"), kz.name, latest_date, config, killzones))
    return states
