from v2_policy import classify_model_eligibility


def test_primary_aligned_valid_is_v2_candidate():
    row = {"label": "XAUUSD", "valid_crt": True, "aligned": True}
    result = classify_model_eligibility(row)
    assert result["in_primary_cohort"] is True
    assert result["v1_eligible"] is True
    assert result["v2_eligible"] is True


def test_primary_misaligned_valid_stays_v1_control_only():
    row = {"label": "US500", "valid_crt": True, "aligned": False}
    result = classify_model_eligibility(row)
    assert result["v1_eligible"] is True
    assert result["v2_eligible"] is False


def test_non_primary_market_is_not_in_v1_or_v2_candidate_cohort():
    row = {"label": "EURUSD", "valid_crt": True, "aligned": True}
    result = classify_model_eligibility(row)
    assert result["in_primary_cohort"] is False
    assert result["v1_eligible"] is False
    assert result["v2_eligible"] is False


def test_invalid_primary_setup_is_not_eligible():
    row = {"label": "US30", "valid_crt": False, "aligned": False}
    result = classify_model_eligibility(row)
    assert result["v1_eligible"] is False
    assert result["v2_eligible"] is False
