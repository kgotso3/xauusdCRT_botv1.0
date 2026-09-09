import pandas as pd

from backtesting.research import ResearchConfig, build_crt_occurrence_dataset


def _bars(times, base=100.0):
    rows = []
    for i, ts in enumerate(times):
        price = base + i * 0.1
        rows.append({
            "time": ts,
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price + 0.2,
            "tick_volume": 100,
            "spread": 10,
            "real_volume": 0,
        })
    return pd.DataFrame(rows)


def test_occurrence_dataset_records_crt_sweep_without_strategy_approval():
    h1_times = pd.date_range("2026-01-01", periods=230, freq="h", tz="UTC")
    h1 = _bars(h1_times)

    # Force a bullish CRT sweep on bar 220: sweep prior low and close back inside.
    prev_low = float(h1.loc[219, "low"])
    prev_high = float(h1.loc[219, "high"])
    h1.loc[220, "low"] = prev_low - 0.5
    h1.loc[220, "high"] = prev_high - 0.1
    h1.loc[220, "close"] = prev_low + 0.2
    h1.loc[221, "open"] = prev_low + 0.4

    m15 = _bars(pd.date_range("2025-12-20", periods=2000, freq="15min", tz="UTC"))
    m5 = _bars(pd.date_range("2025-12-20", periods=6000, freq="5min", tz="UTC"))

    dataset = build_crt_occurrence_dataset(
        h1,
        m15,
        m5,
        ResearchConfig(warmup_h1=220, max_holding_bars=5),
    )

    assert len(dataset) >= 1
    row = dataset.iloc[0]
    assert row["direction"] == "BUY"
    assert row["crt_direction"] == "BULLISH"
    assert "hit_2_0r" in dataset.columns
    assert "mfe_r" in dataset.columns
    assert "alignment_count" in dataset.columns


def test_research_requires_history_after_warmup():
    h1 = _bars(pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"))
    m15 = _bars(pd.date_range("2026-01-01", periods=40, freq="15min", tz="UTC"))
    m5 = _bars(pd.date_range("2026-01-01", periods=120, freq="5min", tz="UTC"))
    dataset = build_crt_occurrence_dataset(h1, m15, m5, ResearchConfig(warmup_h1=20))
    assert dataset.empty
