import pandas as pd

from backtesting.killzones import add_killzone_columns
from backtesting.walk_forward import evaluate_candidate_rules


def _dataset():
    rows = []
    base = pd.Timestamp("2026-01-01 14:00:00", tz="UTC")
    # 14:00 UTC in January is 09:00 New York.
    for i in range(12):
        rows.append({
            "signal_time_utc": base + pd.Timedelta(days=i),
            "decision_time_utc": base + pd.Timedelta(days=i, hours=1),
            "entry_time_utc": base + pd.Timedelta(days=i, hours=1),
            "ny_hour": 9,
            "in_ny_08_13": True,
            "direction": "SELL",
            "alignment_count": 1,
            "day_of_week": "Monday",
            "rsi14": 55,
            "sweep_atr": 0.15,
            "atr_pct": 0.002,
            "hit_1_0r": i < 8,
            "hit_1_5r": i < 7,
            "hit_2_0r": i < 5,
            "mfe_r": 1.8,
            "mae_r": -0.8,
        })
    return pd.DataFrame(rows)


def test_killzone_columns_mark_ny_09():
    df = add_killzone_columns(pd.DataFrame({"ny_hour": [9, 12, 14]}))
    assert bool(df.loc[0, "killzone_NY_08_10"])
    assert bool(df.loc[0, "killzone_NY_09_11"])
    assert bool(df.loc[1, "killzone_NY_11_13"])
    assert df.loc[2, "killzone_name"] == "OUTSIDE"


def test_walk_forward_candidate_summary_focuses_1r_1_5r():
    summary, periods = evaluate_candidate_rules(_dataset(), min_period_samples=2)
    assert not summary.empty
    assert "expectancy_1r" in summary.columns
    assert "expectancy_1_5r" in summary.columns
    assert "ny_09_sell" in set(summary["rule"])
    row = summary.loc[summary["rule"] == "ny_09_sell"].iloc[0]
    assert row["samples"] == 12
    assert row["hit_1_5r"] > 0.5
    assert not periods.empty
