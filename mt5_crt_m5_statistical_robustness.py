from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mt5_crt_m5_performance_test import max_drawdown_r, performance_stats


PRIMARY_COHORT = ["US30", "US500", "XAUUSD"]
DEFAULT_COSTS_R = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]


def parse_costs(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0:
            raise argparse.ArgumentTypeError("cost values must be >= 0")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one cost value is required")
    return values


def clean_trade_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = {"logical_symbol", "model_r", "entry_time_ny", "Clean Month"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input CSV missing columns: {', '.join(sorted(missing))}")

    out = df.copy()
    out["model_r"] = pd.to_numeric(out["model_r"], errors="coerce")
    out = out[out["model_r"].notna()].copy()
    out = out[out["logical_symbol"].isin(PRIMARY_COHORT)].copy()
    out["Clean Month"] = out["Clean Month"].astype(str)
    return out.reset_index(drop=True)


def basic_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "Trades": 0,
            "Expectancy R": 0.0,
            "Total R": 0.0,
            "Profit Factor": 0.0,
            "Max DD R": 0.0,
            "Positive R %": 0.0,
        }
    stats = performance_stats(df)
    return {
        "Trades": int(stats["Trades"]),
        "Expectancy R": float(stats["Expectancy R"]),
        "Total R": float(stats["Total R"]),
        "Profit Factor": float(stats["Profit Factor"]),
        "Max DD R": float(stats["Max DD R"]),
        "Positive R %": float(stats["Positive R %"]),
    }


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(np.quantile(values, alpha / 2.0))
    hi = float(np.quantile(values, 1.0 - alpha / 2.0))
    return lo, hi


def trade_bootstrap(df: pd.DataFrame, reps: int, rng: np.random.Generator) -> dict:
    r = df["model_r"].to_numpy(dtype=float)
    n = len(r)
    if n == 0:
        return {}

    means = np.empty(reps, dtype=float)
    totals = np.empty(reps, dtype=float)
    for i in range(reps):
        sample = rng.choice(r, size=n, replace=True)
        means[i] = float(sample.mean())
        totals[i] = float(sample.sum())

    mean_lo, mean_hi = percentile_ci(means)
    total_lo, total_hi = percentile_ci(totals)
    return {
        "Method": "Trade bootstrap",
        "Replicates": reps,
        "Observed Expectancy R": float(r.mean()),
        "Mean R 95% Low": mean_lo,
        "Mean R 95% High": mean_hi,
        "Observed Total R": float(r.sum()),
        "Total R 95% Low": total_lo,
        "Total R 95% High": total_hi,
        "Bootstrap Mean > 0 %": float((means > 0).mean() * 100.0),
    }


def month_block_bootstrap(df: pd.DataFrame, reps: int, rng: np.random.Generator) -> dict:
    months = sorted(df["Clean Month"].dropna().astype(str).unique().tolist())
    if not months:
        return {}

    month_arrays = {
        month: df.loc[df["Clean Month"] == month, "model_r"].to_numpy(dtype=float)
        for month in months
    }

    means = np.empty(reps, dtype=float)
    totals = np.empty(reps, dtype=float)
    for i in range(reps):
        chosen = rng.choice(months, size=len(months), replace=True)
        sample = np.concatenate([month_arrays[m] for m in chosen])
        means[i] = float(sample.mean()) if len(sample) else 0.0
        totals[i] = float(sample.sum())

    mean_lo, mean_hi = percentile_ci(means)
    total_lo, total_hi = percentile_ci(totals)
    return {
        "Method": "Month-block bootstrap",
        "Replicates": reps,
        "Observed Expectancy R": float(df["model_r"].mean()),
        "Mean R 95% Low": mean_lo,
        "Mean R 95% High": mean_hi,
        "Observed Total R": float(df["model_r"].sum()),
        "Total R 95% Low": total_lo,
        "Total R 95% High": total_hi,
        "Bootstrap Mean > 0 %": float((means > 0).mean() * 100.0),
    }


def leave_one_month_out(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    months = sorted(df["Clean Month"].dropna().astype(str).unique().tolist())
    for month in months:
        part = df[df["Clean Month"] != month].copy()
        row = {"Excluded Month": month, **basic_stats(part)}
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_symbol_out(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol in PRIMARY_COHORT:
        part = df[df["logical_symbol"] != symbol].copy()
        row = {"Excluded Symbol": symbol, **basic_stats(part)}
        rows.append(row)
    return pd.DataFrame(rows)


def by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol in PRIMARY_COHORT:
        part = df[df["logical_symbol"] == symbol].copy()
        rows.append({"Symbol": symbol, **basic_stats(part)})
    return pd.DataFrame(rows)


def cost_sensitivity(df: pd.DataFrame, costs: list[float]) -> pd.DataFrame:
    rows: list[dict] = []
    for cost in costs:
        part = df.copy()
        part["model_r"] = part["model_r"].astype(float) - float(cost)
        stats = basic_stats(part)
        gains = float(part.loc[part["model_r"] > 0, "model_r"].sum())
        losses = float(-part.loc[part["model_r"] < 0, "model_r"].sum())
        pf = gains / losses if losses > 0 else (float("inf") if gains > 0 else 0.0)
        rows.append({
            "Cost R / Trade": float(cost),
            "Trades": len(part),
            "Expectancy R": round(float(part["model_r"].mean()), 4),
            "Total R": round(float(part["model_r"].sum()), 2),
            "Profit Factor": round(float(pf), 3) if math.isfinite(pf) else float("inf"),
            "Positive R %": round(float((part["model_r"] > 0).mean() * 100.0), 2),
        })
    return pd.DataFrame(rows)


def drawdown_from_sequence(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    curve = np.concatenate([[0.0], np.cumsum(values)])
    peaks = np.maximum.accumulate(curve)
    drawdowns = curve - peaks
    return float(drawdowns.min())


def monte_carlo_sequence(df: pd.DataFrame, reps: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = df["model_r"].to_numpy(dtype=float)
    if len(values) == 0:
        return pd.DataFrame(), pd.DataFrame()

    perm_dd = np.empty(reps, dtype=float)
    boot_dd = np.empty(reps, dtype=float)
    boot_total = np.empty(reps, dtype=float)

    for i in range(reps):
        perm = rng.permutation(values)
        perm_dd[i] = drawdown_from_sequence(perm)

        boot = rng.choice(values, size=len(values), replace=True)
        boot_dd[i] = drawdown_from_sequence(boot)
        boot_total[i] = float(boot.sum())

    observed_dd = drawdown_from_sequence(values)

    summary = pd.DataFrame([
        {
            "Simulation": "Permutation drawdown",
            "Replicates": reps,
            "Observed Max DD R": observed_dd,
            "Median Max DD R": float(np.median(perm_dd)),
            "5% Max DD R": float(np.quantile(perm_dd, 0.05)),
            "1% Max DD R": float(np.quantile(perm_dd, 0.01)),
            "Worst Sim Max DD R": float(perm_dd.min()),
            "Median Total R": float(values.sum()),
            "5% Total R": float(values.sum()),
        },
        {
            "Simulation": "Bootstrap path",
            "Replicates": reps,
            "Observed Max DD R": observed_dd,
            "Median Max DD R": float(np.median(boot_dd)),
            "5% Max DD R": float(np.quantile(boot_dd, 0.05)),
            "1% Max DD R": float(np.quantile(boot_dd, 0.01)),
            "Worst Sim Max DD R": float(boot_dd.min()),
            "Median Total R": float(np.median(boot_total)),
            "5% Total R": float(np.quantile(boot_total, 0.05)),
        },
    ])

    paths = pd.DataFrame({
        "Permutation Max DD R": perm_dd,
        "Bootstrap Max DD R": boot_dd,
        "Bootstrap Total R": boot_total,
    })
    return summary, paths


def monthly_sign_summary(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.groupby("Clean Month", as_index=False).agg(
        Trades=("model_r", "size"),
        Expectancy_R=("model_r", "mean"),
        Total_R=("model_r", "sum"),
    )
    monthly["Positive Month"] = monthly["Expectancy_R"] > 0
    positive = int(monthly["Positive Month"].sum())
    n = len(monthly)

    # Exact one-sided binomial tail under p=0.5. Interpret cautiously because months are not guaranteed independent.
    sign_tail = sum(math.comb(n, k) for k in range(positive, n + 1)) / (2 ** n) if n else float("nan")
    monthly["Exact sign-test tail p (same for all rows)"] = sign_tail
    return monthly


def main() -> int:
    parser = argparse.ArgumentParser(description="Statistical robustness checks for the coverage-safe frozen CRT M5 cohort.")
    parser.add_argument("--trades", default="results/crt_m5_clean_months/clean_month_primary_trades.csv")
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--mc-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=5601)
    parser.add_argument("--costs-r", type=parse_costs, default=DEFAULT_COSTS_R)
    parser.add_argument("--output", default="results/crt_m5_statistical_robustness")
    args = parser.parse_args()

    if args.bootstrap_reps < 1000:
        parser.error("--bootstrap-reps must be >= 1000")
    if args.mc_reps < 1000:
        parser.error("--mc-reps must be >= 1000")

    trades_path = Path(args.trades)
    if not trades_path.exists():
        print(f"ERROR: Trades CSV not found: {trades_path}")
        return 2

    try:
        raw = pd.read_csv(trades_path)
        trades = clean_trade_rows(raw)
    except Exception as exc:
        print(f"ERROR: Could not prepare clean trades: {exc}")
        return 2

    if trades.empty:
        print("ERROR: No clean primary-cohort trades found.")
        return 1

    rng = np.random.default_rng(args.seed)

    observed = pd.DataFrame([{"Scope": "US30+US500+XAUUSD", **basic_stats(trades)}])
    symbols = by_symbol(trades)
    trade_bs = trade_bootstrap(trades, args.bootstrap_reps, rng)
    month_bs = month_block_bootstrap(trades, args.bootstrap_reps, rng)
    bootstrap_summary = pd.DataFrame([trade_bs, month_bs])
    lomo = leave_one_month_out(trades)
    loso = leave_one_symbol_out(trades)
    costs = cost_sensitivity(trades, args.costs_r)
    mc_summary, mc_paths = monte_carlo_sequence(trades, args.mc_reps, rng)
    monthly = monthly_sign_summary(trades)

    observed_mean = float(trades["model_r"].mean())
    breakeven_cost = observed_mean

    print("CRT Scanner V1 - Statistical robustness of clean primary cohort")
    print("Primary cohort: US30 + US500 + XAUUSD")
    print(f"Clean trades: {len(trades)} | Months: {trades['Clean Month'].nunique()} | Seed: {args.seed}")
    print("No strategy rules are changed. This script analyzes the already-completed clean-trade sample.\n")

    print("OBSERVED PERFORMANCE")
    print("=" * 100)
    print(observed.to_string(index=False))
    print(f"\nSimple fixed-cost break-even: {breakeven_cost:.4f}R per trade")

    print("\nBOOTSTRAP EXPECTANCY")
    print("=" * 128)
    print(bootstrap_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nLEAVE-ONE-MONTH-OUT")
    print("=" * 110)
    print(lomo.to_string(index=False))

    print("\nLEAVE-ONE-SYMBOL-OUT")
    print("=" * 110)
    print(loso.to_string(index=False))

    print("\nFIXED COST SENSITIVITY")
    print("=" * 96)
    print(costs.to_string(index=False))

    print("\nMONTE CARLO DRAWDOWN / PATH STRESS")
    print("=" * 120)
    print(mc_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nMONTHLY SIGN CHECK")
    print("=" * 110)
    print(monthly.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    observed_csv = out_dir / "observed_performance.csv"
    symbols_csv = out_dir / "symbol_performance.csv"
    bootstrap_csv = out_dir / "bootstrap_summary.csv"
    lomo_csv = out_dir / "leave_one_month_out.csv"
    loso_csv = out_dir / "leave_one_symbol_out.csv"
    costs_csv = out_dir / "cost_sensitivity.csv"
    mc_csv = out_dir / "monte_carlo_summary.csv"
    mc_paths_csv = out_dir / "monte_carlo_paths.csv"
    monthly_csv = out_dir / "monthly_sign_check.csv"

    observed.to_csv(observed_csv, index=False)
    symbols.to_csv(symbols_csv, index=False)
    bootstrap_summary.to_csv(bootstrap_csv, index=False)
    lomo.to_csv(lomo_csv, index=False)
    loso.to_csv(loso_csv, index=False)
    costs.to_csv(costs_csv, index=False)
    mc_summary.to_csv(mc_csv, index=False)
    mc_paths.to_csv(mc_paths_csv, index=False)
    monthly.to_csv(monthly_csv, index=False)

    xlsx_path = out_dir / "crt_m5_statistical_robustness.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            observed.to_excel(writer, sheet_name="Observed", index=False)
            symbols.to_excel(writer, sheet_name="By Symbol", index=False)
            bootstrap_summary.to_excel(writer, sheet_name="Bootstrap", index=False)
            lomo.to_excel(writer, sheet_name="Leave One Month", index=False)
            loso.to_excel(writer, sheet_name="Leave One Symbol", index=False)
            costs.to_excel(writer, sheet_name="Cost Sensitivity", index=False)
            mc_summary.to_excel(writer, sheet_name="Monte Carlo", index=False)
            monthly.to_excel(writer, sheet_name="Monthly Sign", index=False)
        excel_msg = str(xlsx_path)
    except (ImportError, ModuleNotFoundError):
        excel_msg = "not written (install openpyxl)"

    print("\nSaved:")
    for path in [observed_csv, symbols_csv, bootstrap_csv, lomo_csv, loso_csv, costs_csv, mc_csv, mc_paths_csv, monthly_csv]:
        print(f"  {path}")
    print(f"  Excel: {excel_msg}")
    print("\nNOTE: Bootstrap and Monte Carlo outputs quantify sample uncertainty; they do not prove future profitability.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
