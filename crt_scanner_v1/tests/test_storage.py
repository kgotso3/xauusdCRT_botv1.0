import importlib
from pathlib import Path


def test_save_and_load(tmp_path: Path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db.as_posix()}")
    import storage
    importlib.reload(storage)

    payload = {
        "secret": "12345678",
        "version": "1.0.1",
        "scan_type": "PM",
        "time_ny": "2026-09-18 21:00 NY",
        "symbols": [
            {
                "label": "EURUSD",
                "ticker": "OANDA:EURUSD",
                "bias": "BEARISH",
                "trend_score": -5,
                "sweep": "HIGH",
                "close_inside": True,
                "valid_crt": True,
                "direction": "BEARISH",
                "aligned": True,
                "c1_high": 1.18,
                "c1_low": 1.17,
                "c1_mid": 1.175,
                "c2_close": 1.179,
            }
        ],
    }
    scan_id = storage.save_scan(payload)
    latest = storage.latest_scan()
    assert scan_id == 1
    assert latest is not None
    assert latest["symbols"][0]["label"] == "EURUSD"
