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


def validate_trade_levels(symbol: str, direction: str, entry: float, sl: float, tp: float) -> tuple[float, float, float]:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Unable to read symbol information for {symbol}.")

    point = float(info.point or 0.0)
    digits = int(info.digits)
    if point <= 0:
        raise RuntimeError(f"Invalid point size for {symbol}.")

    entry = round(float(entry), digits)
    sl = round(float(sl), digits)
    tp = round(float(tp), digits)

    direction = direction.upper()
    if direction == "BUY":
        if not sl < entry < tp:
            raise RuntimeError("Invalid BUY levels: require SL < entry < TP.")
        sl_distance = entry - sl
        tp_distance = tp - entry
    elif direction == "SELL":
        if not tp < entry < sl:
            raise RuntimeError("Invalid SELL levels: require TP < entry < SL.")
        sl_distance = sl - entry
        tp_distance = entry - tp
    else:
        raise ValueError("direction must be BUY or SELL")

    stops_level = int(getattr(info, "trade_stops_level", 0) or 0)
    freeze_level = int(getattr(info, "trade_freeze_level", 0) or 0)
    required_points = max(stops_level, freeze_level)
    required_distance = required_points * point

    if required_distance > 0:
        tolerance = point * 0.1
        if sl_distance + tolerance < required_distance:
            raise RuntimeError(
                f"Stop loss too close for {symbol}: distance={sl_distance:.{digits}f}, "
                f"broker minimum={required_distance:.{digits}f} ({required_points} points)."
            )
        if tp_distance + tolerance < required_distance:
            raise RuntimeError(
                f"Take profit too close for {symbol}: distance={tp_distance:.{digits}f}, "
                f"broker minimum={required_distance:.{digits}f} ({required_points} points)."
            )

    return entry, sl, tp


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
        order_type, price = mt5.ORDER_TYPE_BUY, float(tick.ask)
    elif direction == "SELL":
        order_type, price = mt5.ORDER_TYPE_SELL, float(tick.bid)
    else:
        raise ValueError("direction must be BUY or SELL")

    price, sl, tp = validate_trade_levels(symbol, direction, price, sl, tp)

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
