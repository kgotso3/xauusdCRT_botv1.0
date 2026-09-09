from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtesting.killzones import DEFAULT_KILLZONES, Killzone

NY = ZoneInfo("America/New_York")
DAY = timedelta(days=1)

SESSION_LIQUIDITY_NUMERIC_FEATURES = [
    "asia_prev_high_dist_atr",
    "asia_prev_low_dist_atr",
    "asia_prev_range_atr",
    "asia_prev_range_position",
    "asia_prev_high_swept",
    "asia_prev_low_swept",
    "london_prev_high_dist_atr",
    "london_prev_low_dist_atr",
    "london_prev_range_atr",
    "london_prev_range_position",
    "london_prev_high_swept",
    "london_prev_low_swept",
    "new_york_prev_high_dist_atr",
    "new_york_prev_low_dist_atr",
    "new_york_prev_range_atr",
    "new_york_prev_range_position",
    "new_york_prev_high_swept",
    "new_york_prev_low_swept",
]


@dataclass(frozen=True)
class CompletedSessionRange:
    high: float | None
    low: float | None
    start_utc: pd.Timestamp | None
    end_utc: pd.Timestamp | None


def _session_local_dates(times_ny: pd.Series, kz: Killzone) -> pd.Series:
    dates = times_ny.dt.date.astype("object")
    if kz.start_hour > kz.end_hour:
        # Wrapped session, e.g. Asia 20:00-00:00. Bars after midnight belong
        # to the prior session date.
        after_midnight = times_ny.dt.hour < kz.end_hour
        previous_dates = (times_ny - DAY).dt.date
        dates = dates.where(~after_midnight, previous_dates)
    return dates


def _completed_session_range(
    h1_hist: pd.DataFrame,
    decision_time: pd.Timestamp,
    kz: Killzone,
) -> CompletedSessionRange:
    if h1_hist.empty:
        return CompletedSessionRange(None, None, None, None)

    hist = h1_hist.copy()
    times_utc = pd.to_datetime(hist["time"], utc=True, errors="coerce")
    times_ny = times_utc.dt.tz_convert(NY)
    in_session = times_ny.dt.hour.apply(kz.contains_hour)
    if not in_session.any():
        return CompletedSessionRange(None, None, None, None)

    session_dates = _session_local_dates(times_ny, kz)
    hist = hist.assign(_time_utc=times_utc, _time_ny=times_ny, _session_date=session_dates, _in_session=in_session)
    hist = hist.loc[hist["_in_session"]].copy()
    if hist.empty:
        return CompletedSessionRange(None, None, None, None)

    decision_ny = pd.Timestamp(decision_time).tz_convert(NY)
    candidate_dates = sorted(set(hist["_session_date"].dropna().tolist()))
    for session_date in reversed(candidate_dates):
        rows = hist.loc[hist["_session_date"] == session_date].copy()
        if rows.empty:
            continue

        local_start = pd.Timestamp(session_date, tz=NY) + timedelta(hours=int(kz.start_hour))
        end_date = session_date if kz.start_hour < kz.end_hour else session_date + DAY
        local_end = pd.Timestamp(end_date, tz=NY) + timedelta(hours=int(kz.end_hour))

        # Only sessions fully completed before the decision timestamp are valid.
        if local_end > decision_ny:
            continue

        return CompletedSessionRange(
            high=float(rows["high"].max()),
            low=float(rows["low"].min()),
            start_utc=local_start.tz_convert("UTC"),
            end_utc=local_end.tz_convert("UTC"),
        )

    return CompletedSessionRange(None, None, None, None)


def build_completed_session_liquidity_features(
    h1_hist: pd.DataFrame,
    decision_time: pd.Timestamp,
    close: float,
    signal_high: float,
    signal_low: float,
    atr: float | None,
    killzones: tuple[Killzone, ...] = DEFAULT_KILLZONES,
) -> dict[str, float | None]:
    """Return causal features from the most recent fully completed session ranges.

    The current/incomplete session is never used. This makes the features safe at
    the H1 CRT decision timestamp.
    """
    safe_atr = atr if atr is not None and np.isfinite(atr) and atr > 0 else None
    out: dict[str, float | None] = {}

    for kz in killzones:
        prefix = kz.name.lower()
        session = _completed_session_range(h1_hist, decision_time, kz)
        hi, lo = session.high, session.low
        width = (hi - lo) if hi is not None and lo is not None else None

        out[f"{prefix}_prev_high"] = hi
        out[f"{prefix}_prev_low"] = lo
        out[f"{prefix}_prev_high_dist_atr"] = (hi - close) / safe_atr if safe_atr and hi is not None else None
        out[f"{prefix}_prev_low_dist_atr"] = (close - lo) / safe_atr if safe_atr and lo is not None else None
        out[f"{prefix}_prev_range_atr"] = width / safe_atr if safe_atr and width is not None else None
        out[f"{prefix}_prev_range_position"] = (close - lo) / width if width is not None and width > 0 else None
        out[f"{prefix}_prev_high_swept"] = float(signal_high > hi) if hi is not None else None
        out[f"{prefix}_prev_low_swept"] = float(signal_low < lo) if lo is not None else None

    return out
