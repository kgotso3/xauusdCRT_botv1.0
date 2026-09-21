from __future__ import annotations

import sys

import pandas as pd

import mt5_crt_m5_statistical_robustness as base


# Preserve the original statistical test, but ensure Monte Carlo's observed path
# uses the same chronological ordering as performance_stats()/max_drawdown_r().
_original_monte_carlo_sequence = base.monte_carlo_sequence


def chronological_monte_carlo_sequence(df, reps, rng):
    ordered = df.copy()
    ordered["_entry_dt_sort"] = pd.to_datetime(
        ordered["entry_time_ny"], errors="coerce", utc=True
    )
    ordered = ordered.sort_values(
        ["_entry_dt_sort", "logical_symbol"], kind="stable"
    ).drop(columns=["_entry_dt_sort"]).reset_index(drop=True)

    summary, paths = _original_monte_carlo_sequence(ordered, reps, rng)

    # Force the displayed observed drawdown to use the exact same canonical
    # calculation as the headline performance table. This is a consistency
    # check only; simulated permutation/bootstrap paths are unchanged.
    observed_dd = base.max_drawdown_r(ordered)
    if not summary.empty:
        summary["Observed Max DD R"] = observed_dd

    return summary, paths


base.monte_carlo_sequence = chronological_monte_carlo_sequence


if __name__ == "__main__":
    sys.exit(base.main())
