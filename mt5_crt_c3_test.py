from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from mt5_crt_frequency_test import NY, UTC, SYMBOL_ALIASES, parse_date, resolve_symbols, print_mapping


def load_m15_extended(symbol: str, start_day: date, end_day: date) -> pd.DataFrame:
    if not mt5.symbol_select(symbol, True):
        return pd.DataFrame()

    start_local = datetime.combine(start_day, time(0, 0), tzinfo=NY)
    # Include the final PM C3: end_day 21:00 -> next day 01:00 NY.
    end_local = datetime.combine(end_day + timedelta(days=1), time(1, 0), tzinfo=NY)

    rates = mt5.copy_rates_range(
        symbol,
        mt5.TIMEFRAME_M15,
        start_local.astimezone(UTC),
        end_local.astimezone(UTC),
    )
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_ny"] = df["time_utc"].dt.tz_convert("America/New_York")
    return df.sort_values("time_ny").copy()


def ny_window(df: pd.DataFrame, day: date, start_hour: int, end_hour: int, end_day_offset: int = 0) -> pd.DataFrame:
    start_dt = datetime.combine(day, time(start_hour, 0), tzinfo=NY)
    end_dt = datetime.combine(day + timedelta(days=end_day_offset), time(end_hour, 0), tzinfo=NY)
    return df[(df["time_ny"] >= start_dt) & (df["time_ny"] < end_dt)].sort_values("time_ny")


def first_touch(c3: pd.DataFrame, direction: str, stop: float, target: float) -> str:
    for _, bar in c3.iterrows():
        high = float(bar["high"])
        low = float(bar["low"])

        if direction == "BULLISH":
            hit_stop = low <= stop
            hit_target = high >= target
        else:
            hit_stop = high >= stop
            hit_target = low <= target

        if hit_stop and hit_target:
            return "AMBIGUOUS"
        if hit_target:
            return "WIN"
        if hit_stop:
            return "LOSS"

    return "NO_HIT"


def evaluate_setup(
    logical: str,
    broker_symbol: str,
    day: date,
    session: str,
    c1: pd.DataFrame,
    c2: pd.DataFrame,
    c3: pd.DataFrame,
    min_bars: int,
    c3_min_bars: int,
) -> dict | None:
    if len(c1) < min_bars or len(c2) < min_bars:
        return None

    c1_high = float(c1["high"].max())
    c1_low = float(c1["low"].min())
    c2_high = float(c2["high"].max())
    c2_low = float(c2["low"].min())
    c2_close = float(c2.iloc[-1]["close"])

    swept_high = c2_high > c1_high
    swept_low = c2_low < c1_low
    exactly_one = swept_high != swept_low
    close_inside = c1_low < c2_close < c1_high
    valid = exactly_one and close_inside

    sweep = "BOTH" if swept_high and swept_low else "HIGH" if swept_high else "LOW" if swept_low else "NONE"
    direction = "BULLISH" if valid and swept_low else "BEARISH" if valid and swept_high else "-"

    row = {
        "symbol": logical,
        "mt5_symbol": broker_symbol,
        "date_ny": day.isoformat(),
        "session": session,
        "c1_bars": len(c1),
        "c2_bars": len(c2),
        "c3_bars": len(c3),
        "c1_high": c1_high,
        "c1_low": c1_low,
        "c2_high": c2_high,
        "c2_low": c2_low,
        "c2_close": c2_close,
        "sweep": sweep,
        "close_inside": close_inside,
        "valid_crt": valid,
        "direction": direction,
        "c3_tested": False,
        "entry": None,
        "stop": None,
        "risk": None,
        "target_1r": None,
        "target_2r": None,
        "opposite_c1": None,
        "c3_high": None,
        "c3_low": None,
        "mfe_r": None,
        "mae_r": None,
        "outcome_1r": None,
        "outcome_2r": None,
        "outcome_opposite_c1": None,
    }

    if not valid or len(c3) < c3_min_bars:
        return row

    # Entry at the first tradable C3 M15 open. This handles session gaps more honestly
    # than assuming the C2 close is always executable as the C3 entry.
    entry = float(c3.iloc[0]["open"])
    stop = c2_low if direction == "BULLISH" else c2_high
    risk = abs(entry - stop)
    if risk <= 0:
        return row

    if direction == "BULLISH":
        target_1r = entry + risk
        target_2r = entry + 2.0 * risk
        opposite_c1 = c1_high
        mfe = max(0.0, float(c3["high"].max()) - entry)
        mae = max(0.0, entry - float(c3["low"].min()))
    else:
        target_1r = entry - risk
        target_2r = entry - 2.0 * risk
        opposite_c1 = c1_low
        mfe = max(0.0, entry - float(c3["low"].min()))
        mae = max(0.0, float(c3["high"].max()) - entry)

    row.update({
        "c3_tested": True,
        "entry": entry,
        "stop": stop,
        "risk": risk,
        "target_1r": target_1r,
        "target_2r": target_2r,
        "opposite_c1": opposite_c1,
        "c3_high": float(c3["high"].max()),
        "c3_low": float(c3["low"].min()),
        "mfe_r": round(mfe / risk, 4),
        "mae_r": round(mae / risk, 4),
        "outcome_1r": first_touch(c3, direction, stop, target_1r),
        "outcome_2r": first_touch(c3, direction, stop, target_2r),
        "outcome_opposite_c1": first_touch(c3, direction, stop, opposite_c1),
    })
    return row


def analyze_symbol(
    logical: str,
    broker_symbol: str,
    df: pd.DataFrame,
    start_day: date,
    end_day: date,
    sessions: str,
    min_bars: int,
    c3_min_bars: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    day = start_day

    while day <= end_day:
        if sessions in {"both", "am"}:
            rec = evaluate_setup(
                logical,
                broker_symbol,
                day,
                "AM",
                ny_window(df, day, 1, 5),
                ny_window(df, day, 5, 9),
                ny_window(df, day, 9, 13),
                min_bars,
                c3_min_bars,
            )
            if rec is not None:
                rows.append(rec)

        if sessions in {"both", "pm"}:
            rec = evaluate_setup(
                logical,
                broker_symbol,
                day,
                "PM",
                ny_window(df, day, 13, 17),
                ny_window(df, day, 17, 21),
                ny_window(df, day, 21, 1, end_day_offset=1),
                min_bars,
                c3_min_bars,
            )
            if rec is not None:
                rows.append(rec)

        day += timedelta(days=1)

    return pd.DataFrame(rows)


def outcome_counts(df: pd.DataFrame, column: str) -> dict:
    tested = df[df["c3_tested"] == True]
    wins = int((tested[column] == "WIN").sum())
    losses = int((tested[column] == "LOSS").sum())
    ambiguous = int((tested[column] == "AMBIGUOUS").sum())
    no_hit = int((tested[column] == "NO_HIT").sum())
    decisive = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "ambiguous": ambiguous,
        "no_hit": no_hit,
        "win_pct": round(wins / decisive * 100.0, 2) if decisive else 0.0,
    }


def build_summary(details: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    rows: list[dict] = []

    for logical in SYMBOL_ALIASES:
        part = details[details["symbol"] == logical] if not details.empty else pd.DataFrame()
        eligible = len(part)
        valid = part[part["valid_crt"] == True] if eligible else pd.DataFrame()
        tested = valid[valid["c3_tested"] == True] if not valid.empty else pd.DataFrame()

        one_r = outcome_counts(valid, "outcome_1r") if not valid.empty else {"wins": 0, "losses": 0, "ambiguous": 0, "no_hit": 0, "win_pct": 0.0}
        two_r = outcome_counts(valid, "outcome_2r") if not valid.empty else {"wins": 0, "losses": 0, "ambiguous": 0, "no_hit": 0, "win_pct": 0.0}
        opp = outcome_counts(valid, "outcome_opposite_c1") if not valid.empty else {"wins": 0, "losses": 0, "ambiguous": 0, "no_hit": 0, "win_pct": 0.0}

        rows.append({
            "Symbol": logical,
            "MT5 Symbol": mapping.get(logical, "NOT FOUND"),
            "Eligible Scans": eligible,
            "Valid CRT": int(len(valid)),
            "Valid %": round(len(valid) / eligible * 100.0, 2) if eligible else 0.0,
            "C3 Tested": int(len(tested)),
            "1R Wins": one_r["wins"],
            "1R Losses": one_r["losses"],
            "1R Ambiguous": one_r["ambiguous"],
            "1R No Hit": one_r["no_hit"],
            "1R Win %": one_r["win_pct"],
            "2R Wins": two_r["wins"],
            "2R Losses": two_r["losses"],
            "2R Ambiguous": two_r["ambiguous"],
            "2R No Hit": two_r["no_hit"],
            "2R Win %": two_r["win_pct"],
            "Opp C1 Wins": opp["wins"],
            "Opp C1 Losses": opp["losses"],
            "Opp C1 Ambiguous": opp["ambiguous"],
            "Opp C1 No Hit": opp["no_hit"],
            "Opp C1 Win %": opp["win_pct"],
            "Avg MFE R": round(float(tested["mfe_r"].mean()), 3) if not tested.empty else 0.0,
            "Avg MAE R": round(float(tested["mae_r"].mean()), 3) if not tested.empty else 0.0,
            "AM Valid": int(((valid["session"] == "AM")).sum()) if not valid.empty else 0,
            "PM Valid": int(((valid["session"] == "PM")).sum()) if not valid.empty else 0,
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["2R Win %", "1R Win %", "C3 Tested", "Avg MFE R"], ascending=[False, False, False, False]).reset_index(drop=True)
    summary.insert(0, "C3 Rank", range(1, len(summary) + 1))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure C3 outcomes after valid CRT setups using the same history-aware MT5 symbol resolver.")
    parser.add_argument("--start", type=parse_date, required=True)
    parser.add_argument("--end", type=parse_date, required=True)
    parser.add_argument("--sessions", choices=["both", "am", "pm"], default="both")
    parser.add_argument("--min-bars", type=int, default=12, help="Minimum M15 bars in each C1/C2 window")
    parser.add_argument("--c3-min-bars", type=int, default=12, help="Minimum M15 bars in C3 to score the outcome")
    parser.add_argument("--output", default="results/crt_c3")
    parser.add_argument("--terminal", default=None)
    parser.add_argument("--max-candidates", type=int, default=12)
    args = parser.parse_args()

    if args.end < args.start:
        parser.error("--end must be on or after --start")
    if not 1 <= args.min_bars <= 16:
        parser.error("--min-bars must be between 1 and 16")
    if not 1 <= args.c3_min_bars <= 16:
        parser.error("--c3-min-bars must be between 1 and 16")

    print("CRT Scanner V1 - MT5 C3 outcome test")
    print(f"Period: {args.start} to {args.end}")
    print(f"Sessions: {args.sessions.upper()} | C1/C2 minimum: {args.min_bars}/16 | C3 minimum: {args.c3_min_bars}/16")
    print("CRT: exactly one C1 side swept + C2 closes back inside C1.")
    print("C3: AM 09:00-13:00 NY; PM 21:00-01:00 NY.")
    print("Entry = first C3 M15 open; stop = swept C2 extreme; targets = 1R, 2R, opposite C1.")
    print("If stop and target are touched in the same M15 bar, outcome = AMBIGUOUS.\n")

    initialized = mt5.initialize(args.terminal) if args.terminal else mt5.initialize()
    if not initialized:
        print(f"ERROR: Could not initialize MT5. last_error={mt5.last_error()}")
        return 2

    try:
        mapping, missing, resolver_diag = resolve_symbols(args.start, args.end, max_candidates=args.max_candidates)
        print_mapping(mapping, missing, resolver_diag)

        frames: list[pd.DataFrame] = []
        print("\nDOWNLOADING / ANALYZING C3")
        print("-" * 84)

        for logical in SYMBOL_ALIASES:
            broker_symbol = mapping.get(logical)
            if not broker_symbol:
                print(f"{logical:<10} skipped - no usable history")
                continue

            df = load_m15_extended(broker_symbol, args.start, args.end)
            result = analyze_symbol(logical, broker_symbol, df, args.start, args.end, args.sessions, args.min_bars, args.c3_min_bars)
            if not result.empty:
                frames.append(result)

            valid = int(result["valid_crt"].sum()) if not result.empty else 0
            tested = int(((result["valid_crt"] == True) & (result["c3_tested"] == True)).sum()) if not result.empty else 0
            print(f"{logical:<10} {broker_symbol:<18} eligible {len(result):>4} | valid {valid:>3} | C3 tested {tested:>3}")

        details = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        summary = build_summary(details, mapping)

        print("\nC3 OUTCOME RANKING")
        print("=" * 124)
        cols = ["C3 Rank", "Symbol", "Valid CRT", "C3 Tested", "1R Win %", "2R Win %", "Opp C1 Win %", "Avg MFE R", "Avg MAE R", "1R Ambiguous", "2R Ambiguous"]
        print(summary[cols].to_string(index=False))

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        ranking_csv = out_dir / "crt_c3_ranking.csv"
        details_csv = out_dir / "crt_c3_all_scans.csv"
        valid_csv = out_dir / "crt_c3_valid_setups.csv"
        resolver_csv = out_dir / "symbol_resolver_diagnostics.csv"

        summary.to_csv(ranking_csv, index=False)
        details.to_csv(details_csv, index=False)
        valid_details = details[details["valid_crt"] == True] if not details.empty else pd.DataFrame()
        valid_details.to_csv(valid_csv, index=False)
        resolver_diag.to_csv(resolver_csv, index=False)

        xlsx_path = out_dir / "crt_c3_results.xlsx"
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                summary.to_excel(writer, sheet_name="C3 Ranking", index=False)
                valid_details.to_excel(writer, sheet_name="Valid CRT C3", index=False)
                details.to_excel(writer, sheet_name="All Scans", index=False)
                resolver_diag.to_excel(writer, sheet_name="Symbol Resolver", index=False)
                if not valid_details.empty:
                    valid_details[valid_details["session"] == "AM"].to_excel(writer, sheet_name="AM Valid", index=False)
                    valid_details[valid_details["session"] == "PM"].to_excel(writer, sheet_name="PM Valid", index=False)
            excel_msg = str(xlsx_path)
        except (ImportError, ModuleNotFoundError):
            excel_msg = "not written (install openpyxl)"

        print("\nSaved:")
        print(f"  {ranking_csv}")
        print(f"  {details_csv}")
        print(f"  {valid_csv}")
        print(f"  {resolver_csv}")
        print(f"  Excel: {excel_msg}")

        incomplete = summary[(summary["Valid CRT"] > 0) & (summary["C3 Tested"] < summary["Valid CRT"])]
        if not incomplete.empty:
            print("\nC3 COVERAGE NOTE:")
            for _, row in incomplete.iterrows():
                print(f"  {row['Symbol']}: {int(row['C3 Tested'])}/{int(row['Valid CRT'])} valid CRTs had enough C3 bars.")

        print("\nWin % uses decisive first-touch outcomes only (WIN vs LOSS). AMBIGUOUS and NO_HIT are reported separately.")
        print("This is a historical analysis, not a guarantee of future trading performance.")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
