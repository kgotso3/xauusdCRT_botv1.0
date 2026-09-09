from __future__ import annotations

import pandas as pd


def apply_candidate_gate(report: pd.DataFrame) -> pd.DataFrame:
    """Classify walk-forward candidates without silently promoting weak rules.

    STRICT_PASS requires:
    - enough samples in every chronological split (sample_stable)
    - positive 1R and 1.5R expectancy on TRAIN
    - positive 1R and 1.5R expectancy on VALIDATION
    - positive 1R and 1.5R expectancy on TEST

    TEST is an audit-only holdout. Once TEST has been inspected, it must not be
    used to invent/tune new rules; new rules require a future holdout period.
    """
    out = report.copy()
    if out.empty:
        return out

    required = [
        "train_exp_1r", "train_exp_1_5r",
        "validation_exp_1r", "validation_exp_1_5r",
        "test_exp_1r", "test_exp_1_5r",
        "sample_stable",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"candidate report missing columns: {missing}")

    out["train_positive_both"] = (out["train_exp_1r"] > 0) & (out["train_exp_1_5r"] > 0)
    out["validation_positive_both"] = (
        (out["validation_exp_1r"] > 0) & (out["validation_exp_1_5r"] > 0)
    )
    out["test_positive_both"] = (out["test_exp_1r"] > 0) & (out["test_exp_1_5r"] > 0)
    out["strict_pass"] = (
        out["sample_stable"].astype(bool)
        & out["train_positive_both"]
        & out["validation_positive_both"]
        & out["test_positive_both"]
    )

    out["research_status"] = "REJECT"
    out.loc[out["sample_stable"].astype(bool), "research_status"] = "RESEARCH_ONLY"
    out.loc[out["strict_pass"], "research_status"] = "STRICT_PASS"
    return out
