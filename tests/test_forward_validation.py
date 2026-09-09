import numpy as np
import pandas as pd

from ml.forward_validation import forward_metrics, upsert_prediction_journal


def test_forward_metrics_uses_frozen_threshold_and_timeout_zero_r(tmp_path):
    journal = pd.DataFrame([
        {"model_version":"V1","target":"1R","threshold_frozen":0.50,"signal_time_utc":"2026-09-02T10:00:00Z","decision_time_utc":"2026-09-02T11:00:00Z","direction":"BUY","probability":0.70,"label":1,"hit_stop":False},
        {"model_version":"V1","target":"1R","threshold_frozen":0.50,"signal_time_utc":"2026-09-03T10:00:00Z","decision_time_utc":"2026-09-03T11:00:00Z","direction":"SELL","probability":0.60,"label":0,"hit_stop":True},
        {"model_version":"V1","target":"1R","threshold_frozen":0.50,"signal_time_utc":"2026-09-04T10:00:00Z","decision_time_utc":"2026-09-04T11:00:00Z","direction":"BUY","probability":0.55,"label":0,"hit_stop":False},
        {"model_version":"V1","target":"1R","threshold_frozen":0.50,"signal_time_utc":"2026-09-05T10:00:00Z","decision_time_utc":"2026-09-05T11:00:00Z","direction":"BUY","probability":0.40,"label":1,"hit_stop":False},
    ])
    out = forward_metrics(journal)
    row = out.iloc[0]
    assert row["selected_trades"] == 3
    assert np.isclose(row["selected_hit_rate"], 1/3)
    assert np.isclose(row["selected_expectancy_r"], 0.0)
    assert np.isclose(row["selected_net_r"], 0.0)


def test_forward_journal_upserts_same_signal(tmp_path):
    path = tmp_path / "forward.csv"
    first = pd.DataFrame([{
        "model_version":"V1.2","target":"1R","signal_time_utc":"2026-09-02T10:00:00Z",
        "decision_time_utc":"2026-09-02T11:00:00Z","direction":"BUY","probability":0.60,
        "threshold_frozen":0.50,"label":np.nan,"hit_stop":pd.NA,
    }]).astype({"hit_stop":"boolean"})
    second = first.copy()
    second.loc[0, "probability"] = 0.60
    second.loc[0, "label"] = 1
    second.loc[0, "hit_stop"] = False

    upsert_prediction_journal(first, path)
    out = upsert_prediction_journal(second, path)
    assert len(out) == 1
    assert float(out.iloc[0]["label"]) == 1.0
