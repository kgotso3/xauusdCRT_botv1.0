from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Killzone:
    name: str
    start_hour: int
    end_hour: int

    def contains_hour(self, hour: int) -> bool:
        return self.start_hour <= int(hour) < self.end_hour


DEFAULT_KILLZONES: tuple[Killzone, ...] = (
    Killzone("NY_08_10", 8, 10),
    Killzone("NY_09_11", 9, 11),
    Killzone("NY_10_12", 10, 12),
    Killzone("NY_11_13", 11, 13),
    Killzone("NY_08_13", 8, 13),
)


def add_killzone_columns(
    df: pd.DataFrame,
    killzones: tuple[Killzone, ...] = DEFAULT_KILLZONES,
) -> pd.DataFrame:
    out = df.copy()
    if "ny_hour" not in out.columns:
        raise ValueError("dataset must contain ny_hour")
    for kz in killzones:
        out[f"killzone_{kz.name}"] = out["ny_hour"].apply(kz.contains_hour)
    out["killzone_name"] = "OUTSIDE"
    for kz in killzones:
        mask = out[f"killzone_{kz.name}"] & (out["killzone_name"] == "OUTSIDE")
        out.loc[mask, "killzone_name"] = kz.name
    return out
