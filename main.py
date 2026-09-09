from __future__ import annotations

import argparse

import MetaTrader5 as mt5

from mt5.connection import connect, disconnect
from mt5.market_data import get_rates, latest_tick
from mt5.execution import market_order
from strategy.signal_engine import build_signal

SYMBOL = "XAUUSD"


def run(mode: str) -> None:
    connect()
    try:
        account = mt5.account_info()
        print(f"Connected: login={account.login} server={account.server} balance={account.balance:.2f} currency={account.currency}")
        print(f"Trade mode: {account.trade_mode} (V1 requires demo)")

        tick = latest_tick(SYMBOL)
        print(f"{SYMBOL} bid={tick['bid']} ask={tick['ask']} spread={tick['spread']}")

        h1 = get_rates(SYMBOL, "H1", 300)
        m15 = get_rates(SYMBOL, "M15", 300)
        m5 = get_rates(SYMBOL, "M5", 300)
        signal = build_signal(h1, m15, m5)
        print("SIGNAL:", signal)

        if mode == "demo" and signal["approved"]:
            # V1 intentionally uses a conservative structural stop and 2R target.
            row = h1.iloc[-1]
            entry = tick["ask"] if signal["direction"] == "BUY" else tick["bid"]
            if signal["direction"] == "BUY":
                sl = float(row["low"])
                risk = entry - sl
                tp = entry + 2 * risk
            else:
                sl = float(row["high"])
                risk = sl - entry
                tp = entry - 2 * risk

            if risk <= 0:
                raise RuntimeError("Invalid structural risk distance; trade blocked.")

            info = mt5.symbol_info(SYMBOL)
            account = mt5.account_info()
            tick_size = info.trade_tick_size or info.point
            tick_value = info.trade_tick_value
            volume = max(info.volume_min, info.volume_min)
            if tick_value > 0:
                risk_money = account.balance * 0.0025
                loss_per_lot = (risk / tick_size) * tick_value
                raw = risk_money / loss_per_lot
                volume = min(info.volume_max, max(info.volume_min, int(raw / info.volume_step) * info.volume_step))

            result = market_order(SYMBOL, signal["direction"], volume, sl, tp)
            print("DEMO ORDER SENT:", result)
    finally:
        disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "signal", "demo"], default="test")
    args = parser.parse_args()
    run(args.mode)
