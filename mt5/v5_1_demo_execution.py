from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math

import MetaTrader5 as mt5

from risk.position_size import calculate_mt5_volume

MAGIC_V51 = 26091051
COMMENT_PREFIX = "CRT_V51_DEMO"


@dataclass(frozen=True)
class DemoRiskConfig:
    risk_fraction: float = 0.01
    max_open_setups: int = 1
    daily_loss_limit_r: float = 2.0
    max_consecutive_losses: int = 3


def account_is_demo_only() -> bool:
    account = mt5.account_info()
    if account is None:
        raise RuntimeError(f"Unable to read account: {mt5.last_error()}")
    demo_mode = getattr(mt5, "ACCOUNT_TRADE_MODE_DEMO", 0)
    if getattr(account, "trade_mode", None) != demo_mode:
        raise RuntimeError("V5.1 HARD LOCK: connected MT5 account is not DEMO. No order may be sent.")
    return True


def _floor_volume(value: float, step: float) -> float:
    return round(math.floor((value + 1e-12) / step) * step, 8)


def _symbol_info(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Unable to read symbol information for {symbol}.")
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"Unable to select symbol {symbol}: {mt5.last_error()}")
    if int(getattr(info, "trade_mode", 0) or 0) == 0:
        raise RuntimeError(f"Trading is disabled for {symbol}.")
    return info


def _existing_v51_exposure(symbol: str) -> tuple[int, int]:
    positions = mt5.positions_get(symbol=symbol) or ()
    orders = mt5.orders_get(symbol=symbol) or ()
    pos = [p for p in positions if int(getattr(p, "magic", 0)) == MAGIC_V51]
    pend = [o for o in orders if int(getattr(o, "magic", 0)) == MAGIC_V51]
    return len(pos), len(pend)


def _utc_day_start() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _closed_setup_pnl_today() -> list[float]:
    deals = mt5.history_deals_get(_utc_day_start(), datetime.now(timezone.utc)) or ()
    ours = [d for d in deals if int(getattr(d, "magic", 0)) == MAGIC_V51 and int(getattr(d, "entry", -1)) == getattr(mt5, "DEAL_ENTRY_OUT", 1)]
    by_position: dict[int, float] = {}
    for d in ours:
        pid = int(getattr(d, "position_id", 0))
        pnl = float(getattr(d, "profit", 0.0)) + float(getattr(d, "commission", 0.0)) + float(getattr(d, "swap", 0.0)) + float(getattr(d, "fee", 0.0))
        by_position[pid] = by_position.get(pid, 0.0) + pnl
    return [v for _, v in sorted(by_position.items())]


def hard_risk_gate(symbol: str, journal_daily_r: float, cfg: DemoRiskConfig) -> None:
    account_is_demo_only()
    pos, pend = _existing_v51_exposure(symbol)
    # V5.1 uses two broker legs per setup.  With max_open_setups=1, any
    # existing V5.1 position/order means a second setup is forbidden.
    if cfg.max_open_setups == 1 and (pos > 0 or pend > 0):
        raise RuntimeError("V5.1 duplicate/risk gate: an existing V5.1 GOLD position or pending order already exists.")
    if cfg.max_open_setups > 1 and pos + pend >= cfg.max_open_setups * 2:
        raise RuntimeError("V5.1 risk gate: maximum open/pending setup exposure reached.")
    if journal_daily_r <= -abs(cfg.daily_loss_limit_r):
        raise RuntimeError("V5.1 risk gate: daily loss limit reached.")

    pnl = _closed_setup_pnl_today()
    streak = 0
    for x in reversed(pnl):
        if x < 0:
            streak += 1
        else:
            break
    if streak >= cfg.max_consecutive_losses:
        raise RuntimeError("V5.1 risk gate: consecutive-loss limit reached.")


def _filling_mode(info) -> int:
    mode = getattr(info, "filling_mode", None)
    supported = {
        getattr(mt5, "ORDER_FILLING_FOK", -101),
        getattr(mt5, "ORDER_FILLING_IOC", -102),
        getattr(mt5, "ORDER_FILLING_RETURN", -103),
    }
    if mode in supported:
        return int(mode)
    return int(getattr(mt5, "ORDER_FILLING_RETURN", mt5.ORDER_FILLING_IOC))


def _split_volume(total_volume: float, info) -> tuple[float, float]:
    step = float(info.volume_step)
    minimum = float(info.volume_min)
    half = _floor_volume(total_volume / 2.0, step)
    if half < minimum:
        raise RuntimeError(
            f"Calculated total volume {total_volume} cannot be split into two broker-valid legs; minimum leg is {minimum}."
        )
    return half, half


def build_limit_requests(symbol: str, signal: dict, cfg: DemoRiskConfig) -> tuple[list[dict], dict]:
    account_is_demo_only()
    info = _symbol_info(symbol)
    account = mt5.account_info()
    if account is None:
        raise RuntimeError(f"Unable to read account: {mt5.last_error()}")

    direction = "BUY" if signal["direction"] == "BULLISH" else "SELL"
    entry = float(signal["entry"])
    stop = float(signal["stop"])
    tp1 = float(signal["tp1"])
    tp2 = float(signal["tp2"])
    digits = int(info.digits)

    total_volume, risk_budget, estimated_loss = calculate_mt5_volume(
        symbol=symbol,
        direction=direction,
        account_equity=float(account.equity),
        risk_fraction=cfg.risk_fraction,
        entry=entry,
        stop=stop,
    )
    if total_volume <= 0:
        raise RuntimeError("V5.1 risk gate: broker minimum volume exceeds the configured setup risk budget.")

    v1, v2 = _split_volume(total_volume, info)
    order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
    margin_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
    required_margin = 0.0
    for volume in (v1, v2):
        margin = mt5.order_calc_margin(margin_type, symbol, volume, entry)
        if margin is None:
            raise RuntimeError(f"order_calc_margin failed: {mt5.last_error()}")
        required_margin += float(margin)
    if required_margin > float(account.margin_free):
        raise RuntimeError(
            f"V5.1 margin gate: required margin {required_margin:.2f} exceeds free margin {float(account.margin_free):.2f}."
        )

    expiration = int(datetime.fromisoformat(signal["retest_expiry_utc"].replace("Z", "+00:00")).timestamp())
    base = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "type": order_type,
        "price": round(entry, digits),
        "sl": round(stop, digits),
        "deviation": 50,
        "magic": MAGIC_V51,
        "type_time": mt5.ORDER_TIME_SPECIFIED,
        "expiration": expiration,
        "type_filling": _filling_mode(info),
    }
    req1 = {**base, "volume": v1, "tp": round(tp1, digits), "comment": f"{COMMENT_PREFIX}_TP1"}
    req2 = {**base, "volume": v2, "tp": round(tp2, digits), "comment": f"{COMMENT_PREFIX}_TP2"}
    meta = {
        "total_volume": total_volume,
        "leg_volumes": [v1, v2],
        "risk_budget": risk_budget,
        "estimated_loss_at_stop": estimated_loss,
        "required_margin": required_margin,
        "free_margin": float(account.margin_free),
    }
    return [req1, req2], meta


def validate_requests(requests: list[dict]) -> list:
    checks = []
    for req in requests:
        check = mt5.order_check(req)
        if check is None:
            raise RuntimeError(f"order_check failed: {mt5.last_error()}")
        if int(getattr(check, "retcode", -1)) != 0:
            raise RuntimeError(f"order_check rejected request: {check}")
        checks.append(check)
    return checks


def submit_demo_requests(requests: list[dict], dry_run: bool = True) -> list:
    account_is_demo_only()
    validate_requests(requests)
    if dry_run:
        return []

    results = []
    for req in requests:
        result = mt5.order_send(req)
        if result is None:
            raise RuntimeError(f"order_send failed: {mt5.last_error()}")
        if int(result.retcode) not in {
            int(getattr(mt5, "TRADE_RETCODE_DONE", 10009)),
            int(getattr(mt5, "TRADE_RETCODE_PLACED", 10008)),
        }:
            raise RuntimeError(f"order_send rejected: {result}")
        results.append(result)
    return results
