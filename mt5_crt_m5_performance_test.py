from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from mt5_crt_frequency_test import (
    SYMBOL_ALIASES,
    analyze_symbol,
    load_m15,
    parse_date,
    resolve_symbols,
)
from mt5_crt_m5_model_test import (
    broker_pip_size,
    c3_bounds,
    confirmed_pivots,
    evaluate_m5_model,
    load_m5,
)


AMBIGUOUS_STATUSES = {"ENTRY_BAR_AMBIGUOUS", "PRE_1R_AMBIGUOUS", "POST_1R_AMBIGUOUS"}


def c3_last_close(m5: pd.DataFrame, session_day: date, session: str) -> float | None:
    start, end = c3_bounds(session_day, session)
    part = m5[(m5["time_ny"] >= start) & (m5["time_ny"] < end)]
    if part.empty:
        return None
    return float(part.iloc[-1]["close"])


def add_performance_fields(result: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return result.copy()

    out = result.copy()
    outcomes: list[str] = []
    model_rs: list[float] = []
    c3_closes: list[float] = []
    timeout_rs: list[float] = []

    for _, row in out.iterrows():
        if not bool(row.get("retest_entry", False)):
            outcomes.append("")
            model_rs.append(float("nan"))
            c3_closes.append(float("nan"))
            timeout_rs.append(float("nan"))
            continue

        status = str(row.get("trade_status", ""))
        if bool(row.get("ambiguous", False)) or status in AMBIGUOUS_STATUSES:
            outcomes.append("AMBIGUOUS")
            model_rs.append(float("nan"))
            c3_closes.append(float("nan"))
            timeout_rs.append(float("nan"))
            continue

        if status == "STOP":
            outcomes.append("STOP")
            model_rs.append(-1.0)
            c3_closes.append(float("nan"))
            timeout_rs.append(float("nan"))
            continue

        if status == "BE":
            outcomes.append("BE")
            model_rs.append(0.0)
            c3_closes.append(float("nan"))
            timeout_rs.append(float("nan"))
            continue

        if status == "TP2":
            outcomes.append("TP2")
            model_rs.append(2.0)
            c3_closes.append(float("nan"))
            timeout_rs.append(float("nan"))
            continue

        if status in {"OPEN_C3_END", "ONE_R_OPEN_C3_END"}:
            session_day = date.fromisoformat(str(row["date_ny"]))
            close = c3_last_close(m5, session_day, str(row["session"]))
            entry = float(row["entry"])
            risk = float(row.get("risk", float("nan")))
            direction = str(row["direction"])

            if close is None or not math.isfinite(risk) or risk <= 0:
                outcomes.append("TIMEOUT_NO_CLOSE")
                model_rs.append(float("nan"))
                c3_closes.append(float("nan") if close is None else close)
                timeout_rs.append(float("nan"))
                continue

            timeout_r = (close - entry) / risk if direction == "BULLISH" else (entry - close) / risk
            outcomes.append("TIMEOUT")
            model_rs.append(float(timeout_r))
            c3_closes.append(close)
            timeout_rs.append(float(timeout_r))
            continue

        outcomes.append(status or "UNCLASSIFIED")
        model_rs.append(float("nan"))
        c3_closes.append(float("nan"))
        timeout_rs.append(float("nan"))

    out["final_outcome"] = outcomes
    out["model_r"] = model_rs
    out["c3_close"] = c3_closes
    out["timeout_close_r"] = timeout_rs
    return out


def max_drawdown_r(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    ordered = trades.copy()
    ordered["entry_dt"] = pd.to_datetime(ordered["entry_time_ny"], errors="coerce", utc=True)
    ordered = ordered.sort_values(["entry_dt", "logical_symbol"], kind="stable")
    equity = ordered["model_r"].fillna(0.0).cumsum()
    curve = pd.concat([pd.Series([0.0]), equity.reset_index(drop=True)], ignore_index=True)
    peaks = curve.cummax()
    drawdowns = curve - peaks
    return float(drawdowns.min())


def performance_stats(part: pd.DataFrame) -> dict:
    empty = {
        "Trades": 0,
        "TP2": 0,
        "BE": 0,
        "Stop": 0,
        "Timeout": 0,
        "TP2 %": 0.0,
        "BE %": 0.0,
        "Stop %": 0.0,
        "Timeout %": 0.0,
        "Positive R %": 0.0,
        "Expectancy R": 0.0,
        "Total R": 0.0,
        "Profit Factor": 0.0,
        "Max DD R": 0.0,
    }
    if part.empty:
        return empty

    clean = part[part["model_r"].notna()].copy()
    n = len(clean)
    if n == 0:
        return empty

    tp2 = int((clean["final_outcome"] == "TP2").sum())
    be = int((clean["final_outcome"] == "BE").sum())
    stop = int((clean["final_outcome"] == "STOP").sum())
    timeout = int((clean["final_outcome"] == "TIMEOUT").sum())
    gains = float(clean.loc[clean["model_r"] > 0, "model_r"].sum())
    losses = float(-clean.loc[clean["model_r"] < 0, "model_r"].sum())

    if losses > 0:
        pf = gains / losses
    elif gains > 0:
        pf = float("inf")
    else:
        pf = 0.0

    return {
        "Trades": n,
        "TP2": tp2,
        "BE": be,
        "Stop": stop,
        "Timeout": timeout,
        "TP2 %": round(tp2 / n * 100.0, 2),
        "BE %": round(be / n * 100.0, 2),
        "Stop %": round(stop / n * 100.0, 2),
        "Timeout %": round(timeout / n * 100.0, 2),
        "Positive R %": round(float((clean["model_r"] > 0).mean() * 100.0), 2),
        "Expectancy R": round(float(clean["model_r"].mean()), 3),
        "Total R": round(float(clean["model_r"].sum()), 2),
        "Profit Factor": round(float(pf), 3) if math.isfinite(pf) else float("inf"),
        "Max DD R": round(max_drawdown_r(clean), 2),
    }


def build_performance_tables(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    sessions: list[dict] = []

    for logical in SYMBOL_ALIASES:
        part = results[results["logical_symbol"] == logical] if not results.empty else pd.DataFrame()
        entries = part[part["retest_entry"]] if not part.empty else pd.DataFrame()
        ambiguous = int((entries["final_outcome"] == "AMBIGUOUS").sum()) if not entries.empty else 0

        stats = performance_stats(entries)
        rows.append({
            "Symbol": logical,
            "Valid CRT": len(part),
            "Entries": len(entries),
            "Ambiguous": ambiguous,
            **stats,
        })

        for session in ("AM", "PM"):
            s_part = entries[entries["session"] == session] if not entries.empty else pd.DataFrame()
            s_ambiguous = int((s_part["final_outcome"] == "AMBIGUOUS").sum()) if not s_part.empty else 0
            sessions.append({
                "Symbol": logical,
                "Session": session,
                "Entries": len(s_part),
                "Ambiguous": s_ambiguous,
                **performance_stats(s_part),
            })

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        ["Expectancy R", "Profit Factor", "Total R", "Trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    return ranking, pd.DataFrame(sessions)


def portfolio_summary(results: pd.DataFrame) -> pd.DataFrame:
    entries = results[results["retest_entry"]] if not results.empty else pd.DataFrame()
    ambiguous = int((entries["final_outcome"] == "AMBIGUOUS").sum()) if not entries.empty else 0
    return pd.DataFrame([{
        "Scope": "All 10 symbols",
        "Entries": len(entries),
        "Ambiguous": ambiguous,
        **performance_stats(entries),
    }])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate realized R, expectancy, profit factor, drawdown and AM/PM performance for the frozen CRT M5 model."
    )
    parser.add_argument("--start", type=parse_date, required=True)
    parser.add_argument("--end", type=parse_date, required=True)
    parser.add_argument("--min-bars", type=int, default=12)
    parser.add_argument("--c3-min-bars", type=int, default=36)
    parser.add_argument("--sl-buffer-pips", type=float, default=10.0)
    parser.add_argument("--sessions", choices=["both", "am", "pm"], default="both")
    parser.add_argument("--output", default="results/crt_m5_performance")
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

    print("CRT Scanner V1 - Frozen M5 model performance test")
    print(f"Period: {args.start} to {args.end}")
    print(f"Sessions: {args.sessions.upper()} | C1/C2 minimum: {args.min_bars}/16 | C3 minimum: {args.c3_min_bars}/48")
    print("Entry rules are unchanged from mt5_crt_m5_model_test.py.")
    print("Outcome policy: STOP=-1R, BE=0R, TP2=+2R.")
    print("If still open at C3 end, the trade is classified TIMEOUT and marked to the final available C3 M5 close.")
    print("Ambiguous M5 bars are excluded from expectancy, profit factor and drawdown.\n")

    initialized = mt5.initialize(args.terminal) if args.terminal else mt5.initialize()
    if not initialized:
        print(f"ERROR: Could not initialize MT5. last_error={mt5.last_error()}")
        return 2

    try:
        mapping, missing, resolver_diag = resolve_symbols(args.start, args.end, max_candidates=args.max_candidates)
        if missing:
            print("WARNING: missing usable history for:", ", ".join(missing))

        all_results: list[pd.DataFrame] = []
        print("REPLAYING FROZEN MODEL")
        print("-" * 108)

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

            result = add_performance_fields(pd.DataFrame(rows), m5)
            all_results.append(result)

            entries = result[result["retest_entry"]]
            clean = entries[entries["model_r"].notna()]
            stats = performance_stats(entries)
            ambiguous = int((entries["final_outcome"] == "AMBIGUOUS").sum())

            print(
                f"{logical:<10} {mt5_symbol:<18} CRT={len(valid):>3} | entries={len(entries):>3} | "
                f"clean={len(clean):>3} | amb={ambiguous:>2} | "
                f"Exp={stats['Expectancy R']:>6.3f}R | Total={stats['Total R']:>7.2f}R | "
                f"PF={stats['Profit Factor']:>6}"
            )

        results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        ranking, session_breakdown = build_performance_tables(results)
        portfolio = portfolio_summary(results)

        print("\nFROZEN MODEL PERFORMANCE")
        print("=" * 156)
        display_cols = [
            "Rank", "Symbol", "Valid CRT", "Entries", "Trades", "Ambiguous",
            "TP2 %", "BE %", "Stop %", "Timeout %",
            "Expectancy R", "Total R", "Profit Factor", "Max DD R",
        ]
        print(ranking[display_cols].to_string(index=False))

        print("\nAM / PM BREAKDOWN")
        print("=" * 138)
        session_cols = [
            "Symbol", "Session", "Entries", "Trades", "Ambiguous",
            "TP2 %", "BE %", "Stop %", "Timeout %",
            "Expectancy R", "Total R", "Profit Factor", "Max DD R",
        ]
        print(session_breakdown[session_cols].to_string(index=False))

        print("\n10-SYMBOL PORTFOLIO (1R risk per trade, sequential R aggregation)")
        print("=" * 108)
        print(portfolio.to_string(index=False))

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        ranking_csv = out_dir / "crt_m5_performance_ranking.csv"
        trades_csv = out_dir / "crt_m5_performance_trades.csv"
        session_csv = out_dir / "crt_m5_session_breakdown.csv"
        portfolio_csv = out_dir / "crt_m5_portfolio_summary.csv"
        resolver_csv = out_dir / "symbol_resolver_diagnostics.csv"

        ranking.to_csv(ranking_csv, index=False)
        results.to_csv(trades_csv, index=False)
        session_breakdown.to_csv(session_csv, index=False)
        portfolio.to_csv(portfolio_csv, index=False)
        resolver_diag.to_csv(resolver_csv, index=False)

        xlsx_path = out_dir / "crt_m5_performance_results.xlsx"
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                ranking.to_excel(writer, sheet_name="Performance Ranking", index=False)
                session_breakdown.to_excel(writer, sheet_name="AM PM Breakdown", index=False)
                portfolio.to_excel(writer, sheet_name="Portfolio", index=False)
                results.to_excel(writer, sheet_name="All Setups", index=False)
                results[results["retest_entry"]].to_excel(writer, sheet_name="Entries", index=False)
                results[results["model_r"].notna()].to_excel(writer, sheet_name="Clean Trades", index=False)
                resolver_diag.to_excel(writer, sheet_name="Resolver", index=False)
            excel_msg = str(xlsx_path)
        except (ImportError, ModuleNotFoundError):
            excel_msg = "not written (install openpyxl)"

        print("\nSaved:")
        print(f"  {ranking_csv}")
        print(f"  {trades_csv}")
        print(f"  {session_csv}")
        print(f"  {portfolio_csv}")
        print(f"  {resolver_csv}")
        print(f"  Excel: {excel_msg}")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
