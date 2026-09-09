from datetime import datetime, timezone

from strategy.session import in_ny_window


def test_ny_session_open_during_edt():
    # September: New York is EDT (UTC-4). 12:00 UTC == 08:00 NY.
    assert in_ny_window(datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc))


def test_ny_session_closed_at_13():
    assert not in_ny_window(datetime(2026, 9, 9, 17, 0, tzinfo=timezone.utc))


def test_weekend_is_closed():
    assert not in_ny_window(datetime(2026, 9, 12, 14, 0, tzinfo=timezone.utc))
