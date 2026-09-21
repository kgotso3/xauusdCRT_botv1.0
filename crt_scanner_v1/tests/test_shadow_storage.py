import importlib
from pathlib import Path


def test_shadow_setup_created_and_resolved(tmp_path: Path, monkeypatch):
    db = tmp_path / "shadow.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db.as_posix()}")

    import storage
    importlib.reload(storage)

    payload = {
        "secret": "12345678",
        "version": "2.0.0",
        "scan_type": "AM",
        "time_ny": "2026-09-21 09:00 NY",
        "symbols": [
            {
                "label": "XAUUSD",
                "ticker": "OANDA:XAUUSD",
                "bias": "BULLISH",
                "trend_score": 5,
                "sweep": "LOW",
                "close_inside": True,
                "valid_crt": True,
                "direction": "BULLISH",
                "aligned": True,
                "c1_high": 100.0,
                "c1_low": 90.0,
                "c1_mid": 95.0,
                "c2_close": 92.0,
            }
        ],
    }

    scan_id = storage.save_scan(payload)
    rows = storage.recent_shadow_setups()

    assert scan_id == 1
    assert len(rows) == 1
    assert rows[0]["label"] == "XAUUSD"
    assert rows[0]["v1_eligible"] is True
    assert rows[0]["v2_eligible"] is True
    assert rows[0]["outcome"] == "PENDING"

    assert storage.update_shadow_outcome(rows[0]["id"], "TP2") is True
    resolved = storage.recent_shadow_setups()[0]
    assert resolved["outcome"] == "TP2"
    assert resolved["model_r"] == 2.0
    assert resolved["resolved_at"] is not None
