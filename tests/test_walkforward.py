import pandas as pd

from backtesting.killzones import DEFAULT_KILLZONES, add_killzone_columns
from backtesting.walkforward import SplitConfig, build_walkforward_report, chronological_split


def _dataset(n=120):
    times = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    rows = []
    for i, ts in enumerate(times):
        rows.append({
            "signal_time_utc": ts,
            "ny_hour": i % 24,
            "direction": "BUY" if i % 2 == 0 else "SELL",
            "alignment_count": i % 4,
            "rsi14": 45 + (i % 20),
            "sweep_atr": 0.15 + (i % 5) * 0.05,
            "atr_pct": 0.001 + i * 0.000001,
            "hit_1_0r": i % 3 != 0,
            "hit_1_5r": i % 4 != 0,
            "mfe_r": 1.0 + (i % 6) * 0.4,
            "mae_r": -0.5 - (i % 4) * 0.2,
        })
    return pd.DataFrame(rows)


def test_default_killzones_cover_asia_london_new_york():
    assert [k.name for k in DEFAULT_KILLZONES] == ["ASIA", "LONDON", "NEW_YORK"]
    df = pd.DataFrame({"ny_hour": [21, 3, 9, 15]})
    out = add_killzone_columns(df)
    assert list(out["killzone_name"]) == ["ASIA", "LONDON", "NEW_YORK", "OUTSIDE"]


def test_chronological_split_preserves_order():
    df = _dataset(100)
    out = chronological_split(df, SplitConfig())
    assert (out.iloc[:60]["split"] == "TRAIN").all()
    assert (out.iloc[60:80]["split"] == "VALIDATION").all()
    assert (out.iloc[80:]["split"] == "TEST").all()


def test_walkforward_reports_only_1r_and_1_5r_metrics():
    report, periods = build_walkforward_report(_dataset(240), SplitConfig(min_samples_per_split=2))
    assert not report.empty
    assert not periods.empty
    assert "validation_hit_1r" in report.columns
    assert "validation_hit_1_5r" in report.columns
    assert "test_exp_1r" in report.columns
    assert "test_exp_1_5r" in report.columns
    assert "validation_hit_2r" not in report.columns
