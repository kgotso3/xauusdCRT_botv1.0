from __future__ import annotations

import MetaTrader5 as mt5

MAGIC = 26090901


def account_is_demo() -> bool:
    account = mt5.account_info()
    if account is None:
        raise RuntimeError(f"Unable to read account: {mt5.last_error()}")
    return getattr(account, "trade_mode", None) == getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)


def _filling_mode(info) -> int:
    mode = getattr(info, "filling_mode", None)
    supported = {
        getattr(mt5, "ORDER_FILLING_FOK", -101),
        getattr(mt5, "ORDER_FILLING_IOC", -102),
        getattr(mt5, "ORDER_FILLING_RETURN", -103),
    }
    if mode in supported:
        return mode
    return mt5.ORDER_FILLING_IOC


def market_order(symbol: str, direction: str, volume: float, sl: float, tp: float, deviation: int = 50):
    if not account_is_demo():
        raise RuntimeError("V1 safety lock: live/real MT5 accounts are blocked.")
    if volume <= 0:
        raise RuntimeError("Trade blocked: calculated volume is zero.")

    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        raise RuntimeError(f"Unable to read {symbol} market information.")
    if not getattr(info, "trade_mode", 0):
        raise RuntimeError(f"Trading is disabled for {symbol}.")

    direction = direction.upper()
    if direction == "BUY":
        order_type, price = mt5.ORDER_TYPE_BUY, tick.ask
        if not sl < price < tp:
            raise RuntimeError("Invalid BUY levels: require SL < entry < TP.")
    elif direction == "SELL":
        order_type, price = mt5.ORDER_TYPE_SELL, tick.bid
        if not tp < price < sl:
            raise RuntimeError("Invalid SELL levels: require TP < entry < SL.")
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
        "deviation": int(deviation),
        "magic": MAGIC,
        "comment": "XAUUSD_CRT_V1_DEMO",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _filling_mode(info),
    }

    check = mt5.order_check(request)
    if check is None:
        raise RuntimeError(f"order_check failed: {mt5.last_error()}")
    if getattr(check, "retcode", -1) != 0:
        raise RuntimeError(f"order_check rejected request: {check}")

    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"order_send failed: {mt5.last_error()}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"order_send rejected: {result}")
    return result
