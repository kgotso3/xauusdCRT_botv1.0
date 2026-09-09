from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")


def to_new_york(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(NEW_YORK)


def in_ny_window(dt: datetime, start_hour: int = 8, end_hour: int = 13) -> bool:
    """True during the configured New York trading window.

    Uses America/New_York so DST is handled automatically. The end is exclusive:
    08:00 <= NY time < 13:00 by default.
    """
    ny = to_new_york(dt)
    if ny.weekday() >= 5:
        return False
    current = ny.time().replace(tzinfo=None)
    return time(start_hour, 0) <= current < time(end_hour, 0)
