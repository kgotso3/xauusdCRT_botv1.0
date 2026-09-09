import pandas as pd

from backtesting.candidate_gate import apply_candidate_gate


def test_candidate_gate_requires_positive_both_across_all_splits():
    df = pd.DataFrame([
        {
            "candidate": "pass",
            "sample_stable": True,
            "train_exp_1r": 0.10,
            "train_exp_1_5r": 0.20,
            "validation_exp_1r": 0.05,
            "validation_exp_1_5r": 0.10,
            "test_exp_1r": 0.02,
            "test_exp_1_5r": 0.03,
        },
        {
            "candidate": "research",
            "sample_stable": True,
            "train_exp_1r": 0.10,
            "train_exp_1_5r": 0.20,
            "validation_exp_1r": -0.01,
            "validation_exp_1_5r": 0.10,
            "test_exp_1r": 0.02,
            "test_exp_1_5r": 0.03,
        },
    ])
    out = apply_candidate_gate(df)
    assert bool(out.loc[out["candidate"] == "pass", "strict_pass"].iloc[0])
    assert out.loc[out["candidate"] == "pass", "research_status"].iloc[0] == "STRICT_PASS"
    assert not bool(out.loc[out["candidate"] == "research", "strict_pass"].iloc[0])
    assert out.loc[out["candidate"] == "research", "research_status"].iloc[0] == "RESEARCH_ONLY"
