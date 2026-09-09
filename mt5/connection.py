from __future__ import annotations

import os
from dataclasses import dataclass

import MetaTrader5 as mt5
from dotenv import load_dotenv


@dataclass(frozen=True)
class MT5Settings:
    login: int
    password: str
    server: str
    path: str
    mode: str


def load_settings() -> MT5Settings:
    load_dotenv()
    return MT5Settings(
        login=int(os.environ["MT5_LOGIN"]),
        password=os.environ["MT5_PASSWORD"],
        server=os.environ["MT5_SERVER"],
        path=os.environ["MT5_PATH"],
        mode=os.getenv("TRADING_MODE", "DEMO").upper(),
    )


def connect() -> MT5Settings:
    settings = load_settings()
    if settings.mode != "DEMO":
        raise RuntimeError("V1 safety lock: TRADING_MODE must be DEMO.")

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

    return settings


def disconnect() -> None:
    mt5.shutdown()
