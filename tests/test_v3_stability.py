import pandas as pd

from ml.v3_stability import PromotionGate, candidate_consistency, selected_stability


def _fold_rows(aucs):
    rows = []
    for fold, auc in enumerate(aucs):
        rows.append({"target":"1R","fold":fold,"model":"random_forest","roc_auc":auc,"brier":0.24})
        rows.append({"target":"1R","fold":fold,"model":"extra_trees","roc_auc":auc-0.01,"brier":0.25})
    return pd.DataFrame(rows)


def test_promotion_gate_passes_stable_candidate():
    folds = _fold_rows([0.56, 0.57, 0.55, 0.58, 0.56])
    out = selected_stability(folds)
    row = out.iloc[0]
    assert row["promotion_status"] == "PROMOTE_TO_FINAL_FREEZE"
    assert row["positive_auc_folds"] == 5
    assert row["catastrophic_folds"] == 0


def test_promotion_gate_rejects_low_mean_auc():
    folds = _fold_rows([0.54, 0.55, 0.56, 0.54, 0.55])
    out = selected_stability(folds)
    assert out.iloc[0]["promotion_status"] == "RESEARCH_ONLY"


def test_promotion_gate_rejects_catastrophic_fold():
    folds = _fold_rows([0.60, 0.61, 0.62, 0.60, 0.40])
    out = selected_stability(folds, PromotionGate(max_auc_std=0.20))
    row = out.iloc[0]
    assert row["catastrophic_folds"] == 1
    assert row["promotion_status"] == "RESEARCH_ONLY"


def test_candidate_consistency_returns_each_model():
    folds = _fold_rows([0.56, 0.57, 0.55, 0.58, 0.56])
    out = candidate_consistency(folds)
    assert set(out["model"]) == {"random_forest", "extra_trees"}
    assert set(out["folds"]) == {5}
