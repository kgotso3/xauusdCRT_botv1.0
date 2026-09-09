from __future__ import annotations

import MetaTrader5 as mt5


def account_is_demo() -> bool:
    account = mt5.account_info()
    if account is None:
        raise RuntimeError(f"Unable to read account: {mt5.last_error()}")
    # MT5 trade mode values: demo is 0, contest is 1, real is 2.
    return getattr(account, "trade_mode", None) in (mt5.ACCOUNT_TRADE_MODE_DEMO, mt5.ACCOUNT_TRADE_MODE_CONTEST)


def market_order(symbol: str, direction: str, volume: float, sl: float, tp: float, deviation: int = 50):
    if not account_is_demo():
        raise RuntimeError("V1 safety lock: live/real MT5 accounts are blocked.")

    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        raise RuntimeError(f"Unable to read {symbol} market information.")

    if direction == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    elif direction == "SELL":
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        raise ValueError("direction must be BUY or SELL")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": float(price),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": deviation,
        "magic": 26090901,
        "comment": "XAUUSD_CRT_V1_DEMO",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    check = mt5.order_check(request)
    if check is None:
        raise RuntimeError(f"order_check failed: {mt5.last_error()}")
    if getattr(check, "retcode", 0) != 0:
        raise RuntimeError(f"order_check rejected request: {check}")

    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"order_send failed: {mt5.last_error()}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"order_send rejected: {result}")
    return result
