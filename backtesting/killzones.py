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


# V2 research defaults. These are deliberately configuration constants rather
# than hard-coded strategy rules so they can be changed and revalidated later.
# All times are America/New_York local time and therefore DST-aware when the
# research dataset's ny_hour is generated from timezone-aware UTC timestamps.
DEFAULT_KILLZONES: tuple[Killzone, ...] = (
    Killzone("ASIA", 20, 0),       # 20:00-00:00 New York time
    Killzone("LONDON", 2, 5),     # 02:00-05:00 New York time
    Killzone("NEW_YORK", 8, 11),  # 08:00-11:00 New York time
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
