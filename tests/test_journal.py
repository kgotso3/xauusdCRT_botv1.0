from pathlib import Path

from database.journal import execution_status, make_signal_id, mark_execution, record_signal, signal_seen


def test_journal_records_signal_once_and_tracks_execution(tmp_path: Path):
    db_path = tmp_path / "journal.sqlite3"
    signal = {"direction": "BUY", "score": 80, "approved": True}
    signal_id = make_signal_id("GOLD", "2026-09-09T12:00:00+00:00", "BUY")

    assert signal_seen(signal_id, db_path) is False
    assert record_signal(signal_id, "GOLD", "2026-09-09T12:00:00+00:00", signal, db_path) is True
    assert signal_seen(signal_id, db_path) is True
    assert record_signal(signal_id, "GOLD", "2026-09-09T12:00:00+00:00", signal, db_path) is False
    assert execution_status(signal_id, db_path) == "NOT_SENT"

    mark_execution(signal_id, "PENDING", db_path=db_path)
    assert execution_status(signal_id, db_path) == "PENDING"

    mark_execution(signal_id, "SENT", order_ticket=123, deal_ticket=456, db_path=db_path)
    assert execution_status(signal_id, db_path) == "SENT"
