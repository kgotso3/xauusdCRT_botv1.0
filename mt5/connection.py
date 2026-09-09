from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import MetaTrader5 as mt5
from dotenv import load_dotenv


@dataclass(frozen=True)
class MT5Settings:
    login: int
    password: str
    server: str
    path: str
    mode: str
    symbol: str
    risk_per_trade: float
    max_daily_loss: float
    max_open_positions: int
    max_spread_points: float


def load_settings() -> MT5Settings:
    load_dotenv()
    return MT5Settings(
        login=int(os.environ["MT5_LOGIN"]),
        password=os.environ["MT5_PASSWORD"],
        server=os.environ["MT5_SERVER"],
        path=os.environ["MT5_PATH"],
        mode=os.getenv("TRADING_MODE", "DEMO").upper(),
        symbol=os.getenv("SYMBOL", "XAUUSD"),
        risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.0025")),
        max_daily_loss=float(os.getenv("MAX_DAILY_LOSS", "0.02")),
        max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "1")),
        max_spread_points=float(os.getenv("MAX_SPREAD_POINTS", "80")),
    )


def connect() -> MT5Settings:
    settings = load_settings()
    if settings.mode != "DEMO":
        raise RuntimeError("V1 safety lock: TRADING_MODE must be DEMO.")
    if settings.max_spread_points <= 0:
        raise RuntimeError("MAX_SPREAD_POINTS must be greater than zero.")

    if not Path(settings.path).exists():
        raise RuntimeError(f"MT5 terminal not found at: {settings.path}")

    if not mt5.initialize(
        path=settings.path,
        login=settings.login,
        password=settings.password,
        server=settings.server,
    ):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    account = mt5.account_info()
    if account is None:
        mt5.shutdown()
        raise RuntimeError(f"Unable to read MT5 account: {mt5.last_error()}")

    if account.login != settings.login:
        mt5.shutdown()
        raise RuntimeError("Connected MT5 login does not match MT5_LOGIN. Trading blocked.")

    demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)
    if getattr(account, "trade_mode", None) != demo_mode:
        mt5.shutdown()
        raise RuntimeError("V1 safety lock: connected account is not an MT5 DEMO account.")

    if not getattr(account, "trade_allowed", False):
        mt5.shutdown()
        raise RuntimeError("MT5 account does not currently allow trading.")

    return settings


def disconnect() -> None:
    mt5.shutdown()
