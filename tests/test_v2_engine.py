from datetime import datetime, timezone

import pytest

from strategy.v2_engine import APPROVED_RULES, V2Rule, killzone_for_time


def test_v2_has_no_approved_rules_by_default():
    assert APPROVED_RULES == ()


def test_v2_rule_allows_only_1r_or_1_5r():
    V2Rule("ok", "NEW_YORK", "SELL", alignment_count=0, target_r=1.5).validate()
    with pytest.raises(ValueError):
        V2Rule("bad", "NEW_YORK", "SELL", alignment_count=0, target_r=2.0).validate()


def test_killzone_for_time_is_dst_aware():
    # 13:00 UTC on 1 July 2026 is 09:00 EDT -> NEW_YORK killzone.
    now = datetime(2026, 7, 1, 13, 0, tzinfo=timezone.utc)
    assert killzone_for_time(now) == "NEW_YORK"
