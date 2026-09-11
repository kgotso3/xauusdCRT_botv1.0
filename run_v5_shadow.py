from __future__ import annotations

import argparse
import time
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from mt5.connection import connect, disconnect
from mt5.market_data import get_rates, latest_tick, resolve_symbol
from risk.position_size import calculate_mt5_volume
from shadow.v5_engine import (
    ShadowConfig,
    _candidate_from_latest_causal,
    add_new_candidate,
    export_reports,
    load_state,
    save_state,
    shadow_metrics,
    update_signal_with_tick,
)


def _crosses_limit(sig: dict, tick: dict) -> bool:
    if sig.get("status") != "PENDING_RETEST":
        return False
    if sig["direction"] == "BULLISH":
        return float(tick["ask"]) <= float(sig["entry"])
    return float(tick["bid"]) >= float(sig["entry"])


def run(args) -> None:
    cfg = ShadowConfig(
        target_closed_fills=args.target_fills,
        retest_window_minutes=120,
        holding_window_hours=24,
        risk_fraction=args.risk,
    )
    state_path = Path(args.state)
    state = load_state(state_path)
    save_state(state_path, state)

    settings = connect()
    try:
        preferred_symbol = getattr(args, "symbol", None) or settings.symbol
        symbol = resolve_symbol(preferred_symbol)
        account = mt5.account_info()
        if account is None:
            raise RuntimeError(f"Unable to read account: {mt5.last_error()}")

        print("CRT V5.1 — PROSPECTIVE SHADOW MODE")
        print(f"Symbol: {symbol} | frozen entry: OTE 0.79 | risk model: {cfg.risk_fraction:.3%}")
        print(f"Target: {cfg.target_closed_fills} completed shadow fills")
        print("SAFETY LOCK: market data + simulated orders only; mt5.order_send() is not called.\n")
        print(f"Prospective cohort started: {state['started_at_utc']}")

        while True:
            tick = latest_tick(symbol)
            account = mt5.account_info()
            if account is None:
                raise RuntimeError(f"Unable to refresh account: {mt5.last_error()}")

            m5_tail = get_rates(symbol, "M5", 10, completed_only=True, asof=tick["time"])
            latest_completed_m5 = pd.Timestamp(m5_tail.iloc[-1]["time"]).isoformat()

            if state.get("last_completed_m5") != latest_completed_m5:
                h1 = get_rates(symbol, "H1", cfg.history_h1, completed_only=True, asof=tick["time"])
                m15 = get_rates(symbol, "M15", cfg.history_m15, completed_only=True, asof=tick["time"])
                m5 = get_rates(symbol, "M5", cfg.history_m5, completed_only=True, asof=tick["time"])
                candidate = _candidate_from_latest_causal(h1, m15, m5, state["started_at_utc"])
                new_id = add_new_candidate(state, symbol, candidate, tick["time"])
                if new_id:
                    sig = state["signals"][new_id]
                    sig["spread_at_signal"] = float(tick["spread"])
                    sig["spread_points_at_signal"] = float(tick["spread_points"])
                    print(
                        f"NEW SHADOW SETUP {sig['direction']} confirmation={sig['confirmation_time_utc']} "
                        f"entry={sig['entry']:.3f} stop={sig['stop']:.3f} "
                        f"tp1={sig['tp1']:.3f} tp2={sig['tp2']:.3f}"
                    )
                    save_state(state_path, state)
                state["last_completed_m5"] = latest_completed_m5

            for sig in state["signals"].values():
                old_status = sig.get("status")
                volume_info = None
                if _crosses_limit(sig, tick):
                    direction = "BUY" if sig["direction"] == "BULLISH" else "SELL"
                    try:
                        volume_info = calculate_mt5_volume(
                            symbol=symbol,
                            direction=direction,
                            account_equity=float(account.equity),
                            risk_fraction=cfg.risk_fraction,
                            entry=float(sig["entry"]),
                            stop=float(sig["stop"]),
                        )
                        sig["sizing_error"] = None
                    except Exception as exc:
                        sig["sizing_error"] = str(exc)
                        print(f"SHADOW SIZING WARNING {sig['signal_id']}: {exc}")

                update_signal_with_tick(
                    sig,
                    tick,
                    float(account.equity),
                    volume_info,
                    risk_fraction=cfg.risk_fraction,
                )
                if sig.get("status") != old_status:
                    print(f"{sig['signal_id']} : {old_status} -> {sig['status']}")

            save_state(state_path, state)
            export_reports(state, args.output_dir)
            m = shadow_metrics(state)
            print(
                f"STATUS setups={m['prospective_setups']} fills={m['prospective_fills']} "
                f"closed={m['closed_fills']}/{cfg.target_closed_fills} "
                f"exp={m['expectancy_r'] if pd.notna(m['expectancy_r']) else float('nan'):.4f}R "
                f"PF={m['profit_factor'] if pd.notna(m['profit_factor']) else float('nan'):.3f}"
            )

            if m["closed_fills"] >= cfg.target_closed_fills:
                print("\nTARGET COHORT COMPLETE. Shadow engine stopped; run the comparison report before any demo execution decision.")
                break
            if args.once:
                break
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        print("\nShadow mode stopped by user. State and reports were preserved.")
        save_state(state_path, state)
        export_reports(state, args.output_dir)
    finally:
        disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="V5.1 frozen OTE 0.79 prospective MT5 shadow trader")
    p.add_argument("--target-fills", type=int, default=40, choices=range(30, 51))
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--poll-seconds", type=int, default=5)
    p.add_argument("--symbol", default=None)
    p.add_argument("--state", default="data/shadow/v5_shadow_state.json")
    p.add_argument("--output-dir", default="data/shadow/v5")
    p.add_argument("--once", action="store_true", help="Run one polling cycle for diagnostics")
    run(p.parse_args())
