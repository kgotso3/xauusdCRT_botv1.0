from __future__ import annotations

import argparse

import MetaTrader5 as mt5

from mt5.connection import connect, disconnect
from mt5.market_data import get_rates, latest_tick
from mt5.execution import market_order
from risk.position_size import calculate_volume
from strategy.signal_engine import build_signal


def run(mode: str) -> None:
    settings = connect()
    try:
        symbol = settings.symbol
        account = mt5.account_info()
        print(f"Connected: login={account.login} server={account.server} balance={account.balance:.2f} currency={account.currency}")
        print("Safety: MT5 DEMO account verified")

        tick = latest_tick(symbol)
        print(f"{symbol} bid={tick['bid']} ask={tick['ask']} spread={tick['spread']}")

        h1 = get_rates(symbol, "H1", 300)
        m15 = get_rates(symbol, "M15", 300)
        m5 = get_rates(symbol, "M5", 300)
        signal = build_signal(h1, m15, m5, now=tick["time"])
        print("SIGNAL:", signal)

        if mode != "demo":
            return
        if not signal["approved"]:
            print("NO TRADE: setup/session requirements not met.")
            return

        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            raise RuntimeError(f"Unable to read open positions: {mt5.last_error()}")
        if len(positions) >= settings.max_open_positions:
            print("NO TRADE: maximum open positions reached.")
            return

        row = h1.iloc[-1]
        entry = tick["ask"] if signal["direction"] == "BUY" else tick["bid"]
        if signal["direction"] == "BUY":
            sl = float(row["low"])
            distance = entry - sl
            tp = entry + 2.0 * distance
        else:
            sl = float(row["high"])
            distance = sl - entry
            tp = entry - 2.0 * distance

        if distance <= 0:
            raise RuntimeError("Invalid structural stop distance; trade blocked.")

        info = mt5.symbol_info(symbol)
        volume = calculate_volume(
            account_balance=float(account.balance),
            risk_fraction=settings.risk_per_trade,
            entry=float(entry),
            stop=float(sl),
            tick_size=float(info.trade_tick_size or info.point),
            tick_value=float(info.trade_tick_value),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
        )
        if volume <= 0:
            print("NO TRADE: broker minimum lot would exceed configured risk.")
            return

        print(f"DEMO ORDER CANDIDATE: {signal['direction']} {volume} {symbol} entry={entry} sl={sl} tp={tp}")
        result = market_order(symbol, signal["direction"], volume, sl, tp)
        print(f"DEMO ORDER SENT: order={getattr(result, 'order', None)} deal={getattr(result, 'deal', None)}")
    finally:
        disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAUUSD CRT Trader V1 - XM demo only")
    parser.add_argument("--mode", choices=["test", "signal", "demo"], default="test")
    args = parser.parse_args()
    run(args.mode)
