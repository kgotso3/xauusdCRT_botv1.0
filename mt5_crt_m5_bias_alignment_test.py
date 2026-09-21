from __future__ import annotations

import argparse
import math
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from mt5_crt_m5_performance_test import performance_stats


NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
PRIMARY_COHORT = ["US30", "US500", "XAUUSD"]


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def chunk_dates(start_day: date, end_day: date, days: int = 90):
    cur = start_day
    while cur <= end_day:
        chunk_end = min(cur + timedelta(days=days - 1), end_day)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def utc_bounds(start_day: date, end_day: date) -> tuple[datetime, datetime]:
    start_dt = datetime.combine(start_day, time.min, tzinfo=UTC)
    end_dt = datetime.combine(end_day + timedelta(days=1), time.min, tzinfo=UTC)
    return start_dt, end_dt


def load_rates_chunked(symbol: str, timeframe: int, start_day: date, end_day: date) -> pd.DataFrame:
    if not mt5.symbol_select(symbol, True):
        return pd.DataFrame()

    mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
    frames: list[pd.DataFrame] = []

    for chunk_start, chunk_end in chunk_dates(start_day, end_day):
        start_utc, end_utc = utc_bounds(chunk_start, chunk_end)
        rates = mt5.copy_rates_range(symbol, timeframe, start_utc, end_utc)
        if rates is None or len(rates) == 0:
            continue
        frames.append(pd.DataFrame(rates))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["time"], keep="first").sort_values("time").reset_index(drop=True)
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["next_open_utc"] = df["time_utc"].shift(-1)
    df["ema20"] = df["close"].astype(float).ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].astype(float).ewm(span=50, adjust=False).mean()
    return df


def scan_timestamp_ny(row: pd.Series) -> datetime:
    session_day = date.fromisoformat(str(row["date_ny"]))
    session = str(row["session"]).upper()
    scan_hour = 9 if session == "AM" else 21
    return datetime.combine(session_day, time(scan_hour, 0), tzinfo=NY)


def last_completed_index(df: pd.DataFrame, scan_utc: pd.Timestamp) -> int | None:
    if df.empty or "next_open_utc" not in df.columns:
        return None
    mask = df["next_open_utc"].notna() & (df["next_open_utc"] <= scan_utc)
    idx = df.index[mask]
    if len(idx) == 0:
        return None
    return int(idx[-1])


def classify_bias(d1: pd.DataFrame, h4: pd.DataFrame, scan_ny: datetime) -> dict:
    scan_utc = pd.Timestamp(scan_ny.astimezone(UTC))
    d_idx = last_completed_index(d1, scan_utc)
    h_idx = last_completed_index(h4, scan_utc)

    empty = {
        "bias_available": False,
        "bias": "MISSING",
        "trend_score": np.nan,
        "d_close": np.nan,
        "d_ema20": np.nan,
        "d_ema50": np.nan,
        "d_ema50_old": np.nan,
        "h4_close": np.nan,
        "h4_ema20": np.nan,
        "h4_ema50": np.nan,
        "d_bar_time_utc": "",
        "h4_bar_time_utc": "",
    }

    # Pine uses D1 close/EMA values at [1] and EMA50 at [6]. Once the last
    # completed daily bar is located, [6] is five completed D1 bars behind [1].
    if d_idx is None or h_idx is None or d_idx < 5:
        return empty

    d = d1.loc[d_idx]
    d_old = d1.loc[d_idx - 5]
    h = h4.loc[h_idx]

    values = [d["close"], d["ema20"], d["ema50"], d_old["ema50"], h["close"], h["ema20"], h["ema50"]]
    if not all(math.isfinite(float(v)) for v in values):
        return empty

    score = 0
    score += 2 if float(d["close"]) > float(d["ema50"]) else -2
    score += 1 if float(d["ema20"]) > float(d["ema50"]) else -1
    score += 1 if float(d["ema50"]) > float(d_old["ema50"]) else -1
    score += 1 if float(h["close"]) > float(h["ema50"]) else -1
    score += 1 if float(h["ema20"]) > float(h["ema50"]) else -1
    score += 1 if float(h["close"]) > float(h["ema20"]) else -1

    bias = "BULLISH" if score >= 4 else "BEARISH" if score <= -4 else "NEUTRAL"
    return {
        "bias_available": True,
        "bias": bias,
        "trend_score": int(score),
        "d_close": float(d["close"]),
        "d_ema20": float(d["ema20"]),
        "d_ema50": float(d["ema50"]),
        "d_ema50_old": float(d_old["ema50"]),
        "h4_close": float(h["close"]),
        "h4_ema20": float(h["ema20"]),
        "h4_ema50": float(h["ema50"]),
        "d_bar_time_utc": str(d["time_utc"]),
        "h4_bar_time_utc": str(h["time_utc"]),
    }


def relation(direction: str, bias: str) -> str:
    direction = str(direction).upper()
    bias = str(bias).upper()
    if bias == "MISSING":
        return "MISSING"
    if bias == "NEUTRAL":
        return "NEUTRAL"
    if bias == direction:
        return "ALIGNED"
    return "MISALIGNED"


def clean_trade_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {"logical_symbol", "mt5_symbol", "model_r", "entry_time_ny", "date_ny", "session", "direction", "Clean Month"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input CSV missing columns: {', '.join(sorted(missing))}")

    out = df.copy()
    out["model_r"] = pd.to_numeric(out["model_r"], errors="coerce")
    out = out[out["model_r"].notna()].copy()
    out = out[out["logical_symbol"].isin(PRIMARY_COHORT)].copy()
    out["Clean Month"] = out["Clean Month"].astype(str)
    return out.reset_index(drop=True)


def stat_row(label: str, part: pd.DataFrame) -> dict:
    stats = performance_stats(part) if not part.empty else performance_stats(pd.DataFrame())
    return {
        "Group": label,
        "Trades": int(stats["Trades"]),
        "Expectancy R": float(stats["Expectancy R"]),
        "Total R": float(stats["Total R"]),
        "Profit Factor": float(stats["Profit Factor"]),
        "Max DD R": float(stats["Max DD R"]),
        "Positive R %": float(stats["Positive R %"]),
        "TP2 %": float(stats["TP2 %"]),
        "Timeout %": float(stats["Timeout %"]),
    }


def bootstrap_mean(part: pd.DataFrame, reps: int, rng: np.random.Generator) -> dict:
    if part.empty:
        return {"Bootstrap 95% Low": np.nan, "Bootstrap 95% High": np.nan, "Bootstrap Mean > 0 %": np.nan}
    values = part["model_r"].to_numpy(dtype=float)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        means[i] = float(rng.choice(values, size=len(values), replace=True).mean())
    return {
        "Bootstrap 95% Low": float(np.quantile(means, 0.025)),
        "Bootstrap 95% High": float(np.quantile(means, 0.975)),
        "Bootstrap Mean > 0 %": float((means > 0).mean() * 100.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Test the frozen CRT M5 baseline against the pre-defined D1/H4 trend-bias alignment rule using MT5-native higher-timeframe bars.")
    parser.add_argument("--trades", default="results/crt_m5_clean_months/clean_month_primary_trades.csv")
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=5601)
    parser.add_argument("--warmup-days", type=int, default=450)
    parser.add_argument("--output", default="results/crt_m5_bias_alignment")
    parser.add_argument("--terminal", default=None)
    args = parser.parse_args()

    if args.bootstrap_reps < 1000:
        parser.error("--bootstrap-reps must be >= 1000")
    if args.warmup_days < 100:
        parser.error("--warmup-days must be >= 100")

    trades_path = Path(args.trades)
    if not trades_path.exists():
        print(f"ERROR: Trades CSV not found: {trades_path}")
        return 2

    try:
        trades = clean_trade_rows(pd.read_csv(trades_path))
    except Exception as exc:
        print(f"ERROR: Could not prepare clean trades: {exc}")
        return 2

    if trades.empty:
        print("ERROR: No clean primary-cohort trades found.")
        return 1

    initialized = mt5.initialize(args.terminal) if args.terminal else mt5.initialize()
    if not initialized:
        print(f"ERROR: Could not initialize MT5. last_error={mt5.last_error()}")
        return 2

    try:
        scan_times = trades.apply(scan_timestamp_ny, axis=1)
        start_day = min(x.date() for x in scan_times) - timedelta(days=args.warmup_days)
        end_day = max(x.date() for x in scan_times) + timedelta(days=7)

        histories: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        print("CRT Scanner V1 - Historical Daily Bias alignment experiment")
        print("Primary cohort: US30 + US500 + XAUUSD")
        print(f"Clean baseline trades: {len(trades)} | Months: {trades['Clean Month'].nunique()}")
        print("Bias scoring is frozen to the existing Pine rule.")
        print("Higher-timeframe source: XM/MT5-native D1 and H4 bars (provider/time-boundary proxy for TradingView OANDA bars).")
        print("Bias is evaluated causally at 09:00 NY for AM CRTs and 21:00 NY for PM CRTs.\n")

        for symbol in trades["mt5_symbol"].dropna().astype(str).unique():
            d1 = load_rates_chunked(symbol, mt5.TIMEFRAME_D1, start_day, end_day)
            h4 = load_rates_chunked(symbol, mt5.TIMEFRAME_H4, start_day, end_day)
            histories[symbol] = (d1, h4)
            print(f"{symbol:<18} D1 bars={len(d1):>5} | H4 bars={len(h4):>6}")

        enriched_rows: list[dict] = []
        for _, row in trades.iterrows():
            symbol = str(row["mt5_symbol"])
            d1, h4 = histories.get(symbol, (pd.DataFrame(), pd.DataFrame()))
            scan_ny = scan_timestamp_ny(row)
            bias_info = classify_bias(d1, h4, scan_ny)
            item = row.to_dict()
            item["scan_time_ny"] = scan_ny.isoformat()
            item.update(bias_info)
            item["bias_relation"] = relation(str(row["direction"]), str(bias_info["bias"]))
            enriched_rows.append(item)

        enriched = pd.DataFrame(enriched_rows)
        available = enriched[enriched["bias_available"] == True].copy()  # noqa: E712
        coverage_pct = len(available) / len(enriched) * 100.0 if len(enriched) else 0.0

        print("\nBIAS COVERAGE")
        print("=" * 92)
        print(f"Bias available: {len(available)}/{len(enriched)} ({coverage_pct:.1f}%)")
        if len(available) < len(enriched):
            print(f"Missing bias:   {len(enriched) - len(available)}")

        relation_rows = []
        rng = np.random.default_rng(args.seed)
        for group in ["ALL AVAILABLE", "ALIGNED", "NEUTRAL", "MISALIGNED"]:
            if group == "ALL AVAILABLE":
                part = available.copy()
            else:
                part = available[available["bias_relation"] == group].copy()
            row = stat_row(group, part)
            row["Retention %"] = round(len(part) / len(available) * 100.0, 2) if len(available) else 0.0
            row.update(bootstrap_mean(part, args.bootstrap_reps, rng))
            relation_rows.append(row)
        relation_summary = pd.DataFrame(relation_rows)

        symbol_rows = []
        for symbol in PRIMARY_COHORT:
            for group in ["ALIGNED", "NEUTRAL", "MISALIGNED"]:
                part = available[(available["logical_symbol"] == symbol) & (available["bias_relation"] == group)].copy()
                row = stat_row(group, part)
                row["Symbol"] = symbol
                symbol_rows.append(row)
        symbol_summary = pd.DataFrame(symbol_rows)

        month_rows = []
        for month in sorted(available["Clean Month"].unique().tolist()):
            for group in ["ALIGNED", "NEUTRAL", "MISALIGNED"]:
                part = available[(available["Clean Month"] == month) & (available["bias_relation"] == group)].copy()
                row = stat_row(group, part)
                row["Clean Month"] = month
                month_rows.append(row)
        monthly_summary = pd.DataFrame(month_rows)

        session_rows = []
        for session in ["AM", "PM"]:
            for group in ["ALIGNED", "NEUTRAL", "MISALIGNED"]:
                part = available[(available["session"] == session) & (available["bias_relation"] == group)].copy()
                row = stat_row(group, part)
                row["Session"] = session
                session_rows.append(row)
        session_summary = pd.DataFrame(session_rows)

        print("\nBIAS RELATION PERFORMANCE")
        print("=" * 142)
        cols = ["Group", "Trades", "Retention %", "Expectancy R", "Total R", "Profit Factor", "Max DD R", "Positive R %", "Bootstrap 95% Low", "Bootstrap 95% High", "Bootstrap Mean > 0 %"]
        print(relation_summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        print("\nBY SYMBOL AND BIAS RELATION")
        print("=" * 118)
        print(symbol_summary[["Symbol", "Group", "Trades", "Expectancy R", "Total R", "Profit Factor", "Max DD R", "Positive R %"]].to_string(index=False))

        print("\nBY SESSION AND BIAS RELATION")
        print("=" * 118)
        print(session_summary[["Session", "Group", "Trades", "Expectancy R", "Total R", "Profit Factor", "Max DD R", "Positive R %"]].to_string(index=False))

        print("\nALIGNED MONTHLY STABILITY")
        print("=" * 112)
        aligned_months = monthly_summary[monthly_summary["Group"] == "ALIGNED"].copy()
        print(aligned_months[["Clean Month", "Trades", "Expectancy R", "Total R", "Profit Factor", "Max DD R", "Positive R %"]].to_string(index=False))

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        enriched_csv = out_dir / "bias_enriched_clean_trades.csv"
        relation_csv = out_dir / "bias_relation_performance.csv"
        symbol_csv = out_dir / "bias_by_symbol.csv"
        month_csv = out_dir / "bias_by_month.csv"
        session_csv = out_dir / "bias_by_session.csv"

        enriched.to_csv(enriched_csv, index=False)
        relation_summary.to_csv(relation_csv, index=False)
        symbol_summary.to_csv(symbol_csv, index=False)
        monthly_summary.to_csv(month_csv, index=False)
        session_summary.to_csv(session_csv, index=False)

        xlsx_path = out_dir / "crt_m5_bias_alignment_results.xlsx"
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                relation_summary.to_excel(writer, sheet_name="Relation Performance", index=False)
                symbol_summary.to_excel(writer, sheet_name="By Symbol", index=False)
                monthly_summary.to_excel(writer, sheet_name="By Month", index=False)
                session_summary.to_excel(writer, sheet_name="By Session", index=False)
                enriched.to_excel(writer, sheet_name="Enriched Trades", index=False)
            excel_msg = str(xlsx_path)
        except (ImportError, ModuleNotFoundError):
            excel_msg = "not written (install openpyxl)"

        print("\nSaved:")
        for path in [enriched_csv, relation_csv, symbol_csv, month_csv, session_csv]:
            print(f"  {path}")
        print(f"  Excel: {excel_msg}")
        print("\nNOTE: This test preserves the frozen scoring rule but uses XM/MT5-native D1/H4 candles; exact TradingView/OANDA bias labels can differ near provider candle boundaries.")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
