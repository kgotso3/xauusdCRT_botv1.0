from __future__ import annotations

from typing import Any

PRIMARY_COHORT = {"XAUUSD", "US500", "US30"}


def classify_model_eligibility(symbol_row: dict[str, Any]) -> dict[str, bool]:
    label = str(symbol_row.get("label", "")).upper()
    valid = bool(symbol_row.get("valid_crt", False))
    aligned = bool(symbol_row.get("aligned", False))
    in_primary_cohort = label in PRIMARY_COHORT

    v1_eligible = in_primary_cohort and valid
    v2_eligible = v1_eligible and aligned

    return {
        "in_primary_cohort": in_primary_cohort,
        "v1_eligible": v1_eligible,
        "v2_eligible": v2_eligible,
    }
