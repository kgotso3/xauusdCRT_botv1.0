from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Killzone:
    """Trading killzone expressed in America/New_York local clock hours.

    Hours are [start_hour, end_hour). If start_hour > end_hour the window wraps
    over midnight, which is useful for the Asia session.
    """

    name: str
    start_hour: int
    end_hour: int

    def contains_hour(self, hour: int) -> bool:
        hour = int(hour)
        if self.start_hour < self.end_hour:
            return self.start_hour <= hour < self.end_hour
        if self.start_hour > self.end_hour:
            return hour >= self.start_hour or hour < self.end_hour
        return True


# Session-anchored CRT research defaults. All times are expressed in
# America/New_York local clock time, so UTC conversion stays DST-aware.
#
# The New York window is intentionally extended through 14:00 NY time for the
# H1 CRT model. With [start, end) semantics, NEW_YORK includes H1 candles that
# open at 08:00, 09:00, 10:00, 11:00, 12:00, and 13:00 NY time.
DEFAULT_KILLZONES: tuple[Killzone, ...] = (
    Killzone("ASIA", 20, 0),       # 20:00-00:00 New York time
    Killzone("LONDON", 2, 5),     # 02:00-05:00 New York time
    Killzone("NEW_YORK", 8, 14),  # 08:00-14:00 New York time
)


def add_killzone_columns(
    df: pd.DataFrame,
    killzones: tuple[Killzone, ...] = DEFAULT_KILLZONES,
) -> pd.DataFrame:
    """Add one boolean column per killzone plus a primary killzone label."""
    out = df.copy()
    if "ny_hour" not in out.columns:
        raise ValueError("dataset must contain ny_hour")

    out["killzone_name"] = "OUTSIDE"
    for kz in killzones:
        col = f"killzone_{kz.name}"
        out[col] = out["ny_hour"].apply(kz.contains_hour)
        mask = out[col] & (out["killzone_name"] == "OUTSIDE")
        out.loc[mask, "killzone_name"] = kz.name

    return out
