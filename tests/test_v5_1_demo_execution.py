from types import SimpleNamespace

import pytest

import mt5.v5_1_demo_execution as ex


def test_floor_volume_respects_broker_step():
    assert ex._floor_volume(0.037, 0.01) == 0.03


def test_split_volume_requires_two_valid_legs():
    info = SimpleNamespace(volume_step=0.01, volume_min=0.01)
    assert ex._split_volume(0.04, info) == (0.02, 0.02)
    with pytest.raises(RuntimeError):
        ex._split_volume(0.01, info)


def test_non_demo_account_is_hard_blocked(monkeypatch):
    monkeypatch.setattr(ex.mt5, "ACCOUNT_TRADE_MODE_DEMO", 0, raising=False)
    monkeypatch.setattr(ex.mt5, "account_info", lambda: SimpleNamespace(trade_mode=2))
    with pytest.raises(RuntimeError, match="not DEMO"):
        ex.account_is_demo_only()


def test_demo_account_passes_hard_lock(monkeypatch):
    monkeypatch.setattr(ex.mt5, "ACCOUNT_TRADE_MODE_DEMO", 0, raising=False)
    monkeypatch.setattr(ex.mt5, "account_info", lambda: SimpleNamespace(trade_mode=0))
    assert ex.account_is_demo_only() is True
