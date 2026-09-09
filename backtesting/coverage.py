from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CoverageSummary:
    timeframe: str
    rows: int
    first_time: pd.Timestamp | None
    last_time: pd.Timestamp | None
    span_days: float
    median_gap_minutes: float | None
    max_gap_minutes: float | None


def summarize_history(df: pd.DataFrame, timeframe: str) -> CoverageSummary:
    if df.empty or "time" not in df.columns:
        return CoverageSummary(timeframe, 0, None, None, 0.0, None, None)

    times = pd.to_datetime(df["time"], utc=True, errors="coerce").dropna().sort_values()
    if times.empty:
        return CoverageSummary(timeframe, 0, None, None, 0.0, None, None)

    gaps = times.diff().dropna().dt.total_seconds().div(60.0)
    first = times.iloc[0]
    last = times.iloc[-1]
    return CoverageSummary(
        timeframe=timeframe,
        rows=int(len(times)),
        first_time=first,
        last_time=last,
        span_days=float((last - first).total_seconds() / 86400.0),
        median_gap_minutes=float(gaps.median()) if len(gaps) else None,
        max_gap_minutes=float(gaps.max()) if len(gaps) else None,
    )


def load_history_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df


def common_coverage(summaries: list[CoverageSummary]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    valid = [s for s in summaries if s.first_time is not None and s.last_time is not None]
    if not valid:
        return None, None
    start = max(s.first_time for s in valid if s.first_time is not None)
    end = min(s.last_time for s in valid if s.last_time is not None)
    if start > end:
        return None, None
    return start, end


def coverage_table(histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summaries = [summarize_history(df, tf) for tf, df in histories.items()]
    common_start, common_end = common_coverage(summaries)
    rows = []
    for s in summaries:
        rows.append({
            "timeframe": s.timeframe,
            "rows": s.rows,
            "first_time": s.first_time,
            "last_time": s.last_time,
            "span_days": s.span_days,
            "median_gap_minutes": s.median_gap_minutes,
            "max_gap_minutes": s.max_gap_minutes,
            "common_start": common_start,
            "common_end": common_end,
        })
    return pd.DataFrame(rows)
