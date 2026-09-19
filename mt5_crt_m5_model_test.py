from __future__ import annotations

import argparse
import bisect
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from mt5_crt_frequency_test import (
    NY,
    UTC,
    SYMBOL_ALIASES,
    analyze_symbol,
    load_m15,
    parse_date,
    resolve_symbols,
    utc_bounds,
)


def load_m5(symbol: str, start_day: date, end_day: date) -> pd.DataFrame:
    """Load enough M5 history for causal swing detection and PM C3 windows crossing midnight."""
    if not mt5.symbol_select(symbol, True):
        return pd.DataFrame()

    context_start = start_day - timedelta(days=3)
    context_end = end_day + timedelta(days=2)
    start_utc, end_utc = utc_bounds(context_start, context_end)

    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_utc, end_utc)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_ny"] = df["time_utc"].dt.tz_convert("America/New_York")
    return df.sort_values("time_ny").reset_index(drop=True)


def broker_pip_size(mt5_symbol: str) -> float:
    """Broker-aware pip heuristic.

    5/3-digit instruments: 1 pip = 10 points.
    Other digit formats: 1 pip = 1 broker point.
    The actual pip size is written to every trade record for auditability.
    """
    info = mt5.symbol_info(mt5_symbol)
    if info is None:
        raise RuntimeError(f"symbol_info unavailable for {mt5_symbol}")
    point = float(info.point)
    digits = int(info.digits)
    return point * 10.0 if digits in (3, 5) else point


def confirmed_pivots(df: pd.DataFrame, left: int = 2, right: int = 2) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Return causal pivot lows/highs as (confirm_idx, pivot_idx, price)."""
    lows = df["low"].to_numpy()
    highs = df["high"].to_numpy()
    pivot_lows: list[tuple[int, int, float]] = []
    pivot_highs: list[tuple[int, int, float]] = []

    for i in range(left, len(df) - right):
        low = float(lows[i])
        high = float(highs[i])

        if all(low < float(lows[j]) for j in range(i - left, i)) and all(low < float(lows[j]) for j in range(i + 1, i + right + 1)):
            pivot_lows.append((i + right, i, low))

        if all(high > float(highs[j]) for j in range(i - left, i)) and all(high > float(highs[j]) for j in range(i + 1, i + right + 1)):
            pivot_highs.append((i + right, i, high))

    return pivot_lows, pivot_highs


def latest_confirmed(pivots: list[tuple[int, int, float]], confirm_before_idx: int) -> tuple[int, int, float] | None:
    if not pivots or confirm_before_idx < 0:
        return None
    confirm_indices = [x[0] for x in pivots]
    pos = bisect.bisect_right(confirm_indices, confirm_before_idx) - 1
    return pivots[pos] if pos >= 0 else None


def c3_bounds(session_day: date, session: str) -> tuple[datetime, datetime]:
    if session == "AM":
        return (
            datetime.combine(session_day, time(9, 0), tzinfo=NY),
            datetime.combine(session_day, time(13, 0), tzinfo=NY),
        )
    return (
        datetime.combine(session_day, time(21, 0), tzinfo=NY),
        datetime.combine(session_day + timedelta(days=1), time(1, 0), tzinfo=NY),
    )


def find_index_range(df: pd.DataFrame, start: datetime, end: datetime) -> tuple[int, int] | None:
    mask = (df["time_ny"] >= start) & (df["time_ny"] < end)
    idx = df.index[mask].tolist()
    if not idx:
        return None
    return idx[0], idx[-1]


def is_bullish_fvg(df: pd.DataFrame, i: int) -> bool:
    if i < 2:
        return False
    candle1_high = float(df.at[i - 2, "high"])
    candle3_low = float(df.at[i, "low"])
    middle_bull = float(df.at[i - 1, "close"]) > float(df.at[i - 1, "open"])
    return middle_bull and candle3_low > candle1_high


def is_bearish_fvg(df: pd.DataFrame, i: int) -> bool:
    if i < 2:
        return False
    candle1_low = float(df.at[i - 2, "low"])
    candle3_high = float(df.at[i, "high"])
    middle_bear = float(df.at[i - 1, "close"]) < float(df.at[i - 1, "open"])
    return middle_bear and candle3_high < candle1_low


def level_hit(row: pd.Series, level: float) -> bool:
    return float(row["low"]) <= level <= float(row["high"])


def manage_trade_m5(
    df: pd.DataFrame,
    entry_idx: int,
    end_idx: int,
    direction: str,
    entry: float,
    stop: float,
    c1_target: float,
) -> dict:
    risk = entry - stop if direction == "BULLISH" else stop - entry
    if risk <= 0:
        return {"trade_status": "INVALID_RISK"}

    one_r = entry + risk if direction == "BULLISH" else entry - risk
    two_r = entry + 2.0 * risk if direction == "BULLISH" else entry - 2.0 * risk

    one_r_hit = False
    two_r_hit = False
    c1_hit = False
    be_hit = False
    ambiguous = False
    status = "OPEN_C3_END"
    one_r_idx = None
    exit_idx = None

    erow = df.loc[entry_idx]
    entry_stop = float(erow["low"]) <= stop if direction == "BULLISH" else float(erow["high"]) >= stop
    entry_one = float(erow["high"]) >= one_r if direction == "BULLISH" else float(erow["low"]) <= one_r
    if entry_stop or entry_one:
        return {
            "trade_status": "ENTRY_BAR_AMBIGUOUS",
            "one_r_hit": False,
            "two_r_hit": False,
            "c1_hit": False,
            "be_hit": False,
            "ambiguous": True,
            "one_r": one_r,
            "two_r": two_r,
            "risk": risk,
            "one_r_time_ny": None,
            "exit_time_ny": None,
        }

    be_armed = False

    for i in range(entry_idx + 1, end_idx + 1):
        row = df.loc[i]

        if not one_r_hit:
            stop_hit = float(row["low"]) <= stop if direction == "BULLISH" else float(row["high"]) >= stop
            hit1 = float(row["high"]) >= one_r if direction == "BULLISH" else float(row["low"]) <= one_r

            if stop_hit and hit1:
                ambiguous = True
                status = "PRE_1R_AMBIGUOUS"
                exit_idx = i
                break
            if stop_hit:
                status = "STOP"
                exit_idx = i
                break
            if hit1:
                one_r_hit = True
                one_r_idx = i

                hit2_same = float(row["high"]) >= two_r if direction == "BULLISH" else float(row["low"]) <= two_r
                c1_same = float(row["high"]) >= c1_target if direction == "BULLISH" else float(row["low"]) <= c1_target
                c1_hit = c1_hit or c1_same

                if hit2_same:
                    two_r_hit = True
                    status = "TP2"
                    exit_idx = i
                    break

                be_armed = True
                continue

        if be_armed:
            hit_be = float(row["low"]) <= entry if direction == "BULLISH" else float(row["high"]) >= entry
            hit2 = float(row["high"]) >= two_r if direction == "BULLISH" else float(row["low"]) <= two_r
            hit_c1 = float(row["high"]) >= c1_target if direction == "BULLISH" else float(row["low"]) <= c1_target

            if hit_be and (hit2 or hit_c1):
                ambiguous = True
                c1_hit = c1_hit or hit_c1
                status = "POST_1R_AMBIGUOUS"
                exit_idx = i
                break

            if hit2:
                two_r_hit = True
                c1_hit = c1_hit or hit_c1
                status = "TP2"
                exit_idx = i
                break

            if hit_c1:
                c1_hit = True

            if hit_be:
                be_hit = True
                status = "BE"
                exit_idx = i
                break

    if status == "OPEN_C3_END" and one_r_hit:
        status = "ONE_R_OPEN_C3_END"

    return {
        "trade_status": status,
        "one_r_hit": one_r_hit,
        "two_r_hit": two_r_hit,
        "c1_hit": c1_hit,
        "be_hit": be_hit,
        "ambiguous": ambiguous,
        "one_r": one_r,
        "two_r": two_r,
        "risk": risk,
        "one_r_time_ny": df.at[one_r_idx, "time_ny"].isoformat() if one_r_idx is not None else None,
        "exit_time_ny": df.at[exit_idx, "time_ny"].isoformat() if exit_idx is not None else None,
    }


def evaluate_m5_model(
    crt: pd.Series,
    m5: pd.DataFrame,
    pivot_lows: list[tuple[int, int, float]],
    pivot_highs: list[tuple[int, int, float]],
    pip_size: float,
    sl_buffer_pips: float,
    min_c3_bars: int,
) -> dict:
    session_day = date.fromisoformat(str(crt["date_ny"]))
    session = str(crt["session"])
    direction = str(crt["direction"])
    c3_start, c3_end = c3_bounds(session_day, session)
    idx_range = find_index_range(m5, c3_start, c3_end)

    base = {
        "logical_symbol": crt["logical_symbol"],
        "mt5_symbol": crt["mt5_symbol"],
        "date_ny": crt["date_ny"],
        "session": session,
        "direction": direction,
        "c1_high": float(crt["c1_high"]),
        "c1_low": float(crt["c1_low"]),
        "c2_high": float(crt["c2_high"]),
        "c2_low": float(crt["c2_low"]),
        "c2_close": float(crt["c2_close"]),
        "pip_size": pip_size,
        "sl_buffer_pips": sl_buffer_pips,
        "sl_buffer_price": pip_size * sl_buffer_pips,
        "c3_bars": 0,
        "raid": False,
        "raid_level": None,
        "raid_time_ny": None,
        "raid_reclaimed": False,
        "mss": False,
        "mss_level": None,
        "mss_time_ny": None,
        "fvg": False,
        "fvg_low": None,
        "fvg_high": None,
        "fvg_time_ny": None,
        "retest_entry": False,
        "entry": None,
        "entry_time_ny": None,
        "stop": None,
        "stop_swing": None,
        "trade_status": "NO_C3_DATA",
        "one_r_hit": False,
        "two_r_hit": False,
        "c1_hit": False,
        "be_hit": False,
        "ambiguous": False,
    }

    if idx_range is None:
        return base

    start_idx, end_idx = idx_range
    c3_count = end_idx - start_idx + 1
    base["c3_bars"] = c3_count
    if c3_count < min_c3_bars:
        base["trade_status"] = "INSUFFICIENT_C3"
        return base

    raid_idx = None
    raid_pivot = None
    mss_pivot = None

    for i in range(start_idx, end_idx + 1):
        if direction == "BULLISH":
            liq = latest_confirmed(pivot_lows, i - 1)
            struct = latest_confirmed(pivot_highs, i - 1)
            if liq and struct and float(m5.at[i, "low"]) < liq[2]:
                raid_idx, raid_pivot, mss_pivot = i, liq, struct
                base["raid_reclaimed"] = bool(float(m5.at[i, "close"]) > liq[2])
                break
        else:
            liq = latest_confirmed(pivot_highs, i - 1)
            struct = latest_confirmed(pivot_lows, i - 1)
            if liq and struct and float(m5.at[i, "high"]) > liq[2]:
                raid_idx, raid_pivot, mss_pivot = i, liq, struct
                base["raid_reclaimed"] = bool(float(m5.at[i, "close"]) < liq[2])
                break

    if raid_idx is None or raid_pivot is None or mss_pivot is None:
        base["trade_status"] = "NO_RAID"
        return base

    base["raid"] = True
    base["raid_level"] = float(raid_pivot[2])
    base["raid_time_ny"] = m5.at[raid_idx, "time_ny"].isoformat()
    base["mss_level"] = float(mss_pivot[2])

    mss_idx = None
    for i in range(raid_idx, end_idx + 1):
        close = float(m5.at[i, "close"])
        if direction == "BULLISH" and close > float(mss_pivot[2]):
            mss_idx = i
            break
        if direction == "BEARISH" and close < float(mss_pivot[2]):
            mss_idx = i
            break

    if mss_idx is None:
        base["trade_status"] = "NO_MSS"
        return base

    base["mss"] = True
    base["mss_time_ny"] = m5.at[mss_idx, "time_ny"].isoformat()

    fvg_idx = None
    zone_low = zone_high = entry = None

    for i in range(max(mss_idx + 1, start_idx + 2), end_idx + 1):
        if i - 2 < start_idx:
            continue

        if direction == "BULLISH" and is_bullish_fvg(m5, i):
            zone_low = float(m5.at[i - 2, "high"])
            zone_high = float(m5.at[i, "low"])
            entry = zone_high
            fvg_idx = i
            break

        if direction == "BEARISH" and is_bearish_fvg(m5, i):
            zone_low = float(m5.at[i, "high"])
            zone_high = float(m5.at[i - 2, "low"])
            entry = zone_low
            fvg_idx = i
            break

    if fvg_idx is None or entry is None:
        base["trade_status"] = "NO_FVG"
        return base

    base["fvg"] = True
    base["fvg_low"] = zone_low
    base["fvg_high"] = zone_high
    base["fvg_time_ny"] = m5.at[fvg_idx, "time_ny"].isoformat()

    entry_idx = None
    for i in range(fvg_idx + 1, end_idx + 1):
        if level_hit(m5.loc[i], entry):
            entry_idx = i
            break

    if entry_idx is None:
        base["trade_status"] = "NO_RETEST"
        return base

    base["retest_entry"] = True
    base["entry"] = entry
    base["entry_time_ny"] = m5.at[entry_idx, "time_ny"].isoformat()

    buffer_price = pip_size * sl_buffer_pips
    if direction == "BULLISH":
        stop_pivot = latest_confirmed(pivot_lows, entry_idx - 1)
        if stop_pivot is None:
            base["trade_status"] = "NO_STOP_SWING"
            return base
        stop = float(stop_pivot[2]) - buffer_price
        c1_target = float(crt["c1_high"])
    else:
        stop_pivot = latest_confirmed(pivot_highs, entry_idx - 1)
        if stop_pivot is None:
            base["trade_status"] = "NO_STOP_SWING"
            return base
        stop = float(stop_pivot[2]) + buffer_price
        c1_target = float(crt["c1_low"])

    base["stop_swing"] = float(stop_pivot[2])
    base["stop"] = stop

    if (direction == "BULLISH" and stop >= entry) or (direction == "BEARISH" and stop <= entry):
        base["trade_status"] = "INVALID_STOP"
        return base

    management = manage_trade_m5(m5, entry_idx, end_idx, direction, entry, stop, c1_target)
    base.update(management)
    return base


def build_ranking(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for logical in SYMBOL_ALIASES:
        part = results[results["logical_symbol"] == logical] if not results.empty else pd.DataFrame()
        valid_crt = len(part)
        if valid_crt == 0:
            rows.append({
                "Symbol": logical,
                "Valid CRT": 0,
                "Raid %": 0.0,
                "MSS %": 0.0,
                "FVG %": 0.0,
                "Entry %": 0.0,
                "Trades": 0,
                "1R %": 0.0,
                "2R %": 0.0,
                "BE %": 0.0,
                "Stop %": 0.0,
                "C1 Target %": 0.0,
                "Ambiguous": 0,
            })
            continue

        raids = int(part["raid"].sum())
        mss = int(part["mss"].sum())
        fvgs = int(part["fvg"].sum())
        entries = int(part["retest_entry"].sum())
        trade_mask = part["trade_status"].isin([
            "STOP", "BE", "TP2", "OPEN_C3_END", "ONE_R_OPEN_C3_END",
            "ENTRY_BAR_AMBIGUOUS", "PRE_1R_AMBIGUOUS", "POST_1R_AMBIGUOUS",
        ])
        trades = part[trade_mask]
        resolved = trades[~trades["ambiguous"]]
        denom = len(resolved)

        rows.append({
            "Symbol": logical,
            "Valid CRT": valid_crt,
            "Raid %": round(raids / valid_crt * 100, 2),
            "MSS %": round(mss / valid_crt * 100, 2),
            "FVG %": round(fvgs / valid_crt * 100, 2),
            "Entry %": round(entries / valid_crt * 100, 2),
            "Trades": len(trades),
            "1R %": round(float(resolved["one_r_hit"].mean() * 100), 2) if denom else 0.0,
            "2R %": round(float(resolved["two_r_hit"].mean() * 100), 2) if denom else 0.0,
            "BE %": round(float((resolved["trade_status"] == "BE").mean() * 100), 2) if denom else 0.0,
            "Stop %": round(float((resolved["trade_status"] == "STOP").mean() * 100), 2) if denom else 0.0,
            "C1 Target %": round(float(resolved["c1_hit"].mean() * 100), 2) if denom else 0.0,
            "Ambiguous": int(trades["ambiguous"].sum()) if not trades.empty else 0,
        })

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(["2R %", "1R %", "Entry %", "Valid CRT"], ascending=[False, False, False, False]).reset_index(drop=True)
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    return ranking


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest the CRT Scanner V1 M5 confirmation/FVG entry model on MT5 history.")
    parser.add_argument("--start", type=parse_date, required=True)
    parser.add_argument("--end", type=parse_date, required=True)
    parser.add_argument("--min-bars", type=int, default=12, help="Minimum M15 bars in each C1/C2 window")
    parser.add_argument("--c3-min-bars", type=int, default=36, help="Minimum M5 bars required in C3 (default 36 of 48)")
    parser.add_argument("--sl-buffer-pips", type=float, default=10.0)
    parser.add_argument("--sessions", choices=["both", "am", "pm"], default="both")
    parser.add_argument("--output", default="results/crt_m5_model")
    parser.add_argument("--terminal", default=None)
    parser.add_argument("--max-candidates", type=int, default=12)
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must be on or after --start")
    if not 1 <= args.min_bars <= 16:
        parser.error("--min-bars must be between 1 and 16")
    if not 1 <= args.c3_min_bars <= 48:
        parser.error("--c3-min-bars must be between 1 and 48")
    if args.sl_buffer_pips <= 0:
        parser.error("--sl-buffer-pips must be positive")

    print("CRT Scanner V1 - M5 confirmation / FVG backtest")
    print(f"Period: {args.start} to {args.end}")
    print(f"Sessions: {args.sessions.upper()} | C1/C2 minimum: {args.min_bars}/16 | C3 minimum: {args.c3_min_bars}/48 M5 bars")
    print("Rules: valid 4H CRT -> M5 liquidity raid -> MSS -> displacement/FVG -> first-edge retest.")
    print(f"SL: latest confirmed M5 swing +/- {args.sl_buffer_pips:g} broker-aware pips.")
    print("Management: +1R arms break-even; track 2R, opposite C1 boundary, BE, stop.")
    print("M5 pivot definition: 2-left / 2-right, used only after confirmation.\n")

    initialized = mt5.initialize(args.terminal) if args.terminal else mt5.initialize()
    if not initialized:
        print(f"ERROR: Could not initialize MT5. last_error={mt5.last_error()}")
        return 2

    try:
        mapping, missing, resolver_diag = resolve_symbols(args.start, args.end, max_candidates=args.max_candidates)
        if missing:
            print("WARNING: missing usable history for:", ", ".join(missing))

        all_results: list[pd.DataFrame] = []

        print("ANALYZING 10-MARKET MODEL")
        print("-" * 100)

        for logical in SYMBOL_ALIASES:
            mt5_symbol = mapping.get(logical)
            if not mt5_symbol:
                print(f"{logical:<10} skipped - no usable MT5 history")
                continue

            m15 = load_m15(mt5_symbol, args.start, args.end)
            if m15.empty:
                print(f"{logical:<10} {mt5_symbol:<18} no M15 history")
                continue

            scans = analyze_symbol(logical, mt5_symbol, m15, args.sessions, args.min_bars)
            valid = scans[scans["valid_crt"]].copy() if not scans.empty else pd.DataFrame()
            if valid.empty:
                print(f"{logical:<10} {mt5_symbol:<18} 0 valid CRTs")
                continue

            m5 = load_m5(mt5_symbol, args.start, args.end)
            if m5.empty:
                print(f"{logical:<10} {mt5_symbol:<18} no M5 history")
                continue

            pivot_lows, pivot_highs = confirmed_pivots(m5)
            pip_size = broker_pip_size(mt5_symbol)

            rows: list[dict] = []
            for _, crt in valid.iterrows():
                rows.append(
                    evaluate_m5_model(
                        crt,
                        m5,
                        pivot_lows,
                        pivot_highs,
                        pip_size,
                        args.sl_buffer_pips,
                        args.c3_min_bars,
                    )
                )

            result = pd.DataFrame(rows)
            all_results.append(result)

            entries = int(result["retest_entry"].sum()) if not result.empty else 0
            hit1 = int(result["one_r_hit"].sum()) if not result.empty else 0
            hit2 = int(result["two_r_hit"].sum()) if not result.empty else 0
            print(
                f"{logical:<10} {mt5_symbol:<18} valid CRT={len(valid):>3} | "
                f"raid={int(result['raid'].sum()):>3} | MSS={int(result['mss'].sum()):>3} | "
                f"FVG={int(result['fvg'].sum()):>3} | entries={entries:>3} | 1R={hit1:>3} | 2R={hit2:>3}"
            )

        results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        ranking = build_ranking(results)

        print("\nM5 MODEL EFFECTIVENESS RANKING")
        print("=" * 132)
        cols = ["Rank", "Symbol", "Valid CRT", "Raid %", "MSS %", "FVG %", "Entry %", "Trades", "1R %", "2R %", "BE %", "Stop %", "C1 Target %", "Ambiguous"]
        print(ranking[cols].to_string(index=False))

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        ranking_csv = out_dir / "crt_m5_model_ranking.csv"
        results_csv = out_dir / "crt_m5_model_setups.csv"
        resolver_csv = out_dir / "symbol_resolver_diagnostics.csv"
        ranking.to_csv(ranking_csv, index=False)
        results.to_csv(results_csv, index=False)
        resolver_diag.to_csv(resolver_csv, index=False)

        xlsx_path = out_dir / "crt_m5_model_results.xlsx"
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                ranking.to_excel(writer, sheet_name="Ranking", index=False)
                results.to_excel(writer, sheet_name="All Valid CRT", index=False)
                results[results["retest_entry"]].to_excel(writer, sheet_name="Trades", index=False)
                results[results["session"] == "AM"].to_excel(writer, sheet_name="AM", index=False)
                results[results["session"] == "PM"].to_excel(writer, sheet_name="PM", index=False)
                resolver_diag.to_excel(writer, sheet_name="Resolver", index=False)
            excel_msg = str(xlsx_path)
        except (ImportError, ModuleNotFoundError):
            excel_msg = "not written (install openpyxl)"

        print("\nSaved:")
        print(f"  {ranking_csv}")
        print(f"  {results_csv}")
        print(f"  {resolver_csv}")
        print(f"  Excel: {excel_msg}")
        print("\nNOTE: M5 OHLC cannot resolve all intrabar ordering. Ambiguous trades are flagged and excluded from 1R/2R rate denominators.")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
