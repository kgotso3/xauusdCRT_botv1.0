import numpy as np
import pandas as pd

from backtesting.outcomes import outcome_summary, realized_r
from ml.thresholds import ThresholdConfig, choose_threshold, scan_validation_thresholds


def _rows():
    return pd.DataFrame({
        "hit_stop": [False, True, False, False],
        "hit_1_0r": [True, False, False, True],
        "hit_1_5r": [False, False, False, True],
    })


def test_unresolved_timeout_is_not_full_loss():
    df = _rows()
    r = realized_r(df, target_r=1.0, timeout_r=0.0)
    assert list(r) == [1.0, -1.0, 0.0, 1.0]
    stats = outcome_summary(df, target_r=1.0, timeout_r=0.0)
    assert stats["timeouts"] == 1
    assert np.isclose(stats["expectancy_r_timeout_neutral"], 0.25)


def test_threshold_selection_uses_eligible_validation_rows():
    n = 40
    df = pd.DataFrame({
        "hit_stop": [False if i < 25 else True for i in range(n)],
        "hit_1_0r": [True if i < 25 else False for i in range(n)],
        "hit_1_5r": [True if i < 20 else False for i in range(n)],
    })
    prob = np.linspace(0.95, 0.10, n)
    scan = scan_validation_thresholds(
        df,
        prob,
        target_r=1.0,
        config=ThresholdConfig(thresholds=(0.5, 0.7), min_validation_trades=5),
    )
    chosen = choose_threshold(scan)
    assert chosen in {0.5, 0.7}
    assert scan.loc[0, "eligible"]
    assert scan.loc[0, "expectancy_r"] >= scan.loc[1, "expectancy_r"]
