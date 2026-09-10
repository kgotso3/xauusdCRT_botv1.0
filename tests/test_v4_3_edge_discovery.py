from __future__ import annotations

import pandas as pd

from backtesting.v4_3_edge_discovery import EdgeDiscoveryConfig, discover_edges, enrich_features


def _sample():
    rows = []
    for year in [2024, 2025, 2026]:
        for hour in [8, 9]:
            for i in range(60):
                ts = pd.Timestamp(f"{year}-03-01T00:00:00Z") + pd.Timedelta(hours=i)
                if year == 2026 and i >= 30:
                    ts = pd.Timestamp("2026-08-01T00:00:00Z") + pd.Timedelta(hours=i)
                win = hour == 8
                r = 1.5 if win else -1.0
                rows.append({
                    "signal_close_time_utc": ts,
                    "ny_hour": hour,
                    "direction": "BULLISH" if hour == 8 else "BEARISH",
                    "correct_half": True,
                    "h1_entry": 100.0,
                    "stop": 99.0,
                    "range_high": 102.0,
                    "range_low": 98.0,
                    "m5exec_total_r": r,
                    "m5exec_tp1_hit": win,
                    "m5exec_tp2_hit": win,
                    "m15_mss_time": ts + pd.Timedelta(minutes=15),
                    "m15_fvg_time": ts + pd.Timedelta(minutes=30),
                    "m5_fvg_time": ts + pd.Timedelta(minutes=40),
                    "m15_fvg_entry": 100.5,
                    "m5_fvg_entry": 100.25,
                })
    return pd.DataFrame(rows)


def test_enrich_features_adds_causal_numeric_fields():
    out = enrich_features(_sample().head(1))
    assert out.iloc[0]["range_size"] == 4.0
    assert out.iloc[0]["stop_distance"] == 1.0
    assert out.iloc[0]["m15_mss_delay_min"] == 15.0


def test_discovery_uses_development_before_holdout_only():
    cfg = EdgeDiscoveryConfig(holdout_start="2026-07-01", min_dev_trades=50, min_dev_profit_factor=1.1, min_positive_years=2)
    candidates, holdout, overall, dev, test = discover_edges(_sample(), config=cfg)
    cutoff = pd.Timestamp("2026-07-01", tz="UTC")
    assert (dev["signal_close_time_utc"] < cutoff).all()
    assert (test["signal_close_time_utc"] >= cutoff).all()
    assert set(overall["period"]) == {"DEVELOPMENT", "HOLDOUT"}
    assert not candidates.empty


def test_profitable_hour_can_be_selected_on_development():
    cfg = EdgeDiscoveryConfig(holdout_start="2026-07-01", min_dev_trades=50, min_dev_profit_factor=1.1, min_positive_years=2)
    candidates, holdout, *_ = discover_edges(_sample(), config=cfg)
    rule = candidates.loc[candidates["rule_name"].eq("ny_hour=8")].iloc[0]
    assert bool(rule["selected"])
    assert rule["expectancy_r"] > 0
    assert not holdout.empty
