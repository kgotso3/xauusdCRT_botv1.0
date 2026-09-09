from __future__ import annotations

import argparse

import MetaTrader5 as mt5

from database.journal import execution_status, make_signal_id, mark_execution, record_signal
from mt5.connection import connect, disconnect
from mt5.execution import market_order, validate_trade_levels
from mt5.market_data import get_rates, latest_tick, resolve_symbol
from risk.position_size import calculate_mt5_volume
from strategy.signal_engine import build_signal


def _print_symbol_diagnostics(symbol: str, max_spread_points: float) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Unable to read symbol diagnostics for {symbol}.")

    print(
        "CONTRACT: "
        f"digits={info.digits} point={info.point} "
        f"tick_size={info.trade_tick_size} tick_value={info.trade_tick_value} "
        f"volume_min={info.volume_min} volume_max={info.volume_max} volume_step={info.volume_step}"
    )
    print(
        "EXECUTION: "
        f"trade_mode={info.trade_mode} filling_mode={info.filling_mode} "
        f"stops_level={info.trade_stops_level} freeze_level={info.trade_freeze_level} "
        f"max_spread_points={max_spread_points}"
    )


def run(mode: str) -> None:
    settings = connect()
    try:
        requested_symbol = settings.symbol
        symbol = resolve_symbol(requested_symbol)
        account = mt5.account_info()
        print(
            f"Connected: login={account.login} server={account.server} "
            f"balance={account.balance:.2f} equity={account.equity:.2f} currency={account.currency}"
        )
        print("Safety: MT5 DEMO account verified")
        if symbol != requested_symbol:
            print(f"Symbol resolved: {requested_symbol} -> {symbol}")

        _print_symbol_diagnostics(symbol, settings.max_spread_points)

        tick = latest_tick(symbol)
        print(f"MT5 latest tick UTC: {tick['time'].isoformat()}")
        print(
            f"{symbol} bid={tick['bid']} ask={tick['ask']} "
            f"spread={tick['spread']} ({tick['spread_points']:.1f} points)"
        )

        # Position 0 is still forming; asof=tick time adds a second safety check
        # so a future/incomplete bar can never reach the signal engine.
        h1 = get_rates(symbol, "H1", 300, completed_only=True, asof=tick["time"])
        m15 = get_rates(symbol, "M15", 300, completed_only=True, asof=tick["time"])
        m5 = get_rates(symbol, "M5", 300, completed_only=True, asof=tick["time"])
        signal = build_signal(h1, m15, m5, now=tick["time"])

        h1_bar_time = h1.iloc[-1]["time"]
        signal_id = make_signal_id(symbol, h1_bar_time, signal["direction"])
        is_new = record_signal(signal_id, symbol, h1_bar_time, signal)

        print(f"Last completed H1 bar: {h1_bar_time}")
        print("SIGNAL:", signal)
        print(f"Signal ID: {signal_id}")
        if not is_new:
            print("Journal: signal already recorded.")

        if mode != "demo":
            return
        if not signal["approved"]:
            print("NO TRADE: setup/session requirements not met.")
            return

        if tick["spread_points"] > settings.max_spread_points:
            print(
                "NO TRADE: spread safety limit exceeded: "
                f"current={tick['spread_points']:.1f} points, "
                f"maximum={settings.max_spread_points:.1f} points."
            )
            return

        prior_status = execution_status(signal_id)
        if prior_status in {"PENDING", "SENT"}:
            print(f"NO TRADE: duplicate execution blocked; journal status={prior_status}.")
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

        entry, sl, tp = validate_trade_levels(symbol, signal["direction"], entry, sl, tp)
        volume, risk_budget, estimated_loss = calculate_mt5_volume(
            symbol=symbol,
            direction=signal["direction"],
            account_equity=float(account.equity),
            risk_fraction=settings.risk_per_trade,
            entry=float(entry),
            stop=float(sl),
        )
        if volume <= 0:
            print(
                "NO TRADE: broker minimum lot would exceed configured risk. "
                f"Risk budget={risk_budget:.2f} {account.currency}; "
                f"minimum-lot estimated loss={estimated_loss:.2f} {account.currency}."
            )
            return

        print(
            f"DEMO ORDER CANDIDATE: {signal['direction']} {volume} {symbol} "
            f"entry={entry} sl={sl} tp={tp} risk_budget={risk_budget:.2f} "
            f"estimated_stop_loss={estimated_loss:.2f} {account.currency}"
        )

        mark_execution(signal_id, "PENDING")
        try:
            result = market_order(symbol, signal["direction"], volume, sl, tp)
        except Exception as exc:
            mark_execution(signal_id, "ERROR", error_text=str(exc))
            raise

        mark_execution(
            signal_id,
            "SENT",
            order_ticket=getattr(result, "order", None),
            deal_ticket=getattr(result, "deal", None),
        )
        print(
            f"DEMO ORDER SENT: order={getattr(result, 'order', None)} "
            f"deal={getattr(result, 'deal', None)}"
        )
    finally:
        disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAUUSD CRT Trader V1 - XM demo only")
    parser.add_argument("--mode", choices=["test", "signal", "demo"], default="test")
    args = parser.parse_args()
    run(args.mode)
