from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mt5_crt_m5_performance_test import performance_stats


DEFAULT_COSTS_R = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]
POLICIES = [
    ("BASELINE_ALL", "benchmark"),
    ("ALIGNED_ONLY", "confirmatory"),
    ("NON_NEUTRAL", "exploratory"),
    ("MISALIGNED_ONLY", "diagnostic"),
    ("NEUTRAL_ONLY", "diagnostic"),
]


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


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "logical_symbol",
        "model_r",
        "entry_time_ny",
        "Clean Month",
        "bias_relation",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input CSV missing columns: {', '.join(sorted(missing))}")

    out = df.copy()
    out["model_r"] = pd.to_numeric(out["model_r"], errors="coerce")
    out = out[out["model_r"].notna()].copy()
    out["Clean Month"] = out["Clean Month"].astype(str)
    out["bias_relation"] = out["bias_relation"].astype(str).str.upper()
    out = out[out["bias_relation"].isin({"ALIGNED", "NEUTRAL", "MISALIGNED"})].copy()
    return out.reset_index(drop=True)


def policy_part(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy == "BASELINE_ALL":
        return df.copy()
    if policy == "ALIGNED_ONLY":
        return df[df["bias_relation"] == "ALIGNED"].copy()
    if policy == "NON_NEUTRAL":
        return df[df["bias_relation"].isin({"ALIGNED", "MISALIGNED"})].copy()
    if policy == "MISALIGNED_ONLY":
        return df[df["bias_relation"] == "MISALIGNED"].copy()
    if policy == "NEUTRAL_ONLY":
        return df[df["bias_relation"] == "NEUTRAL"].copy()
    raise ValueError(f"Unknown policy: {policy}")


def stat_row(df: pd.DataFrame) -> dict:
    stats = performance_stats(df) if not df.empty else performance_stats(pd.DataFrame())
    return {
        "Trades": int(stats["Trades"]),
        "Expectancy R": float(stats["Expectancy R"]),
        "Total R": float(stats["Total R"]),
        "Profit Factor": float(stats["Profit Factor"]),
        "Max DD R": float(stats["Max DD R"]),
        "Positive R %": float(stats["Positive R %"]),
        "TP2 %": float(stats["TP2 %"]),
        "Timeout %": float(stats["Timeout %"]),
    }


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q))


def trade_bootstrap(df: pd.DataFrame, reps: int, rng: np.random.Generator) -> dict:
    values = df["model_r"].to_numpy(dtype=float)
    if len(values) == 0:
        return {
            "Trade BS 95% Low": np.nan,
            "Trade BS 95% High": np.nan,
            "Trade BS Mean > 0 %": np.nan,
        }

    means = np.empty(reps, dtype=float)
    for i in range(reps):
        means[i] = float(rng.choice(values, size=len(values), replace=True).mean())

    return {
        "Trade BS 95% Low": percentile(means, 0.025),
        "Trade BS 95% High": percentile(means, 0.975),
        "Trade BS Mean > 0 %": float((means > 0).mean() * 100.0),
    }


def month_block_bootstrap(df: pd.DataFrame, policy: str, reps: int, rng: np.random.Generator) -> dict:
    months = sorted(df["Clean Month"].dropna().unique().tolist())
    if not months:
        return {
            "Month BS 95% Low": np.nan,
            "Month BS 95% High": np.nan,
            "Month BS Mean > 0 %": np.nan,
        }

    blocks = {month: df[df["Clean Month"] == month].copy() for month in months}
    means = np.empty(reps, dtype=float)

    for i in range(reps):
        chosen = rng.choice(months, size=len(months), replace=True)
        sample = pd.concat([blocks[m] for m in chosen], ignore_index=True)
        part = policy_part(sample, policy)
        means[i] = float(part["model_r"].mean()) if len(part) else np.nan

    means = means[np.isfinite(means)]
    if len(means) == 0:
        return {
            "Month BS 95% Low": np.nan,
            "Month BS 95% High": np.nan,
            "Month BS Mean > 0 %": np.nan,
        }

    return {
        "Month BS 95% Low": percentile(means, 0.025),
        "Month BS 95% High": percentile(means, 0.975),
        "Month BS Mean > 0 %": float((means > 0).mean() * 100.0),
    }


def policy_summary(df: pd.DataFrame, reps: int, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict] = []
    baseline_n = len(df)
    for policy, status in POLICIES:
        part = policy_part(df, policy)
        row = {
            "Policy": policy,
            "Hypothesis Status": status,
            "Retention %": round(len(part) / baseline_n * 100.0, 2) if baseline_n else 0.0,
            **stat_row(part),
        }
        row.update(trade_bootstrap(part, reps, rng))
        row.update(month_block_bootstrap(df, policy, reps, rng))
        rows.append(row)
    return pd.DataFrame(rows)


def month_block_policy_differences(df: pd.DataFrame, reps: int, rng: np.random.Generator) -> pd.DataFrame:
    months = sorted(df["Clean Month"].dropna().unique().tolist())
    blocks = {month: df[df["Clean Month"] == month].copy() for month in months}
    comparisons = [
        ("ALIGNED_ONLY", "BASELINE_ALL"),
        ("NON_NEUTRAL", "BASELINE_ALL"),
        ("ALIGNED_ONLY", "NON_NEUTRAL"),
        ("ALIGNED_ONLY", "MISALIGNED_ONLY"),
        ("ALIGNED_ONLY", "NEUTRAL_ONLY"),
    ]
    sims = {pair: [] for pair in comparisons}

    for _ in range(reps):
        chosen = rng.choice(months, size=len(months), replace=True)
        sample = pd.concat([blocks[m] for m in chosen], ignore_index=True)
        means: dict[str, float] = {}
        for policy, _status in POLICIES:
            part = policy_part(sample, policy)
            means[policy] = float(part["model_r"].mean()) if len(part) else np.nan
        for a, b in comparisons:
            if math.isfinite(means[a]) and math.isfinite(means[b]):
                sims[(a, b)].append(means[a] - means[b])

    rows: list[dict] = []
    for a, b in comparisons:
        vals = np.asarray(sims[(a, b)], dtype=float)
        if len(vals):
            rows.append({
                "Policy A": a,
                "Policy B": b,
                "A - B Mean R": float(vals.mean()),
                "Difference 95% Low": percentile(vals, 0.025),
                "Difference 95% High": percentile(vals, 0.975),
                "P(A > B) %": float((vals > 0).mean() * 100.0),
                "Replicates": len(vals),
            })
        else:
            rows.append({
                "Policy A": a,
                "Policy B": b,
                "A - B Mean R": np.nan,
                "Difference 95% Low": np.nan,
                "Difference 95% High": np.nan,
                "P(A > B) %": np.nan,
                "Replicates": 0,
            })
    return pd.DataFrame(rows)


def leave_one_month_out(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    months = sorted(df["Clean Month"].dropna().unique().tolist())
    for month in months:
        remaining = df[df["Clean Month"] != month].copy()
        for policy in ["ALIGNED_ONLY", "NON_NEUTRAL"]:
            part = policy_part(remaining, policy)
            rows.append({"Excluded Month": month, "Policy": policy, **stat_row(part)})
    return pd.DataFrame(rows)


def leave_one_symbol_out(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    symbols = sorted(df["logical_symbol"].dropna().unique().tolist())
    for symbol in symbols:
        remaining = df[df["logical_symbol"] != symbol].copy()
        for policy in ["ALIGNED_ONLY", "NON_NEUTRAL"]:
            part = policy_part(remaining, policy)
            rows.append({"Excluded Symbol": symbol, "Policy": policy, **stat_row(part)})
    return pd.DataFrame(rows)


def temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    temp = df.copy()
    temp["Year"] = temp["Clean Month"].str.slice(0, 4)
    for year in sorted(temp["Year"].unique().tolist()):
        year_df = temp[temp["Year"] == year].copy()
        for policy in ["BASELINE_ALL", "ALIGNED_ONLY", "NON_NEUTRAL", "MISALIGNED_ONLY", "NEUTRAL_ONLY"]:
            part = policy_part(year_df, policy)
            rows.append({"Year": year, "Policy": policy, **stat_row(part)})
    return pd.DataFrame(rows)


def cost_sensitivity(df: pd.DataFrame, costs: list[float]) -> pd.DataFrame:
    rows: list[dict] = []
    for policy in ["BASELINE_ALL", "ALIGNED_ONLY", "NON_NEUTRAL"]:
        original = policy_part(df, policy)
        for cost in costs:
            part = original.copy()
            part["model_r"] = part["model_r"].astype(float) - float(cost)
            stats = stat_row(part)
            rows.append({
                "Policy": policy,
                "Cost R / Trade": float(cost),
                **stats,
            })
    return pd.DataFrame(rows)


def monthly_policy(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for month in sorted(df["Clean Month"].dropna().unique().tolist()):
        month_df = df[df["Clean Month"] == month].copy()
        for policy in ["ALIGNED_ONLY", "NON_NEUTRAL"]:
            part = policy_part(month_df, policy)
            rows.append({"Clean Month": month, "Policy": policy, **stat_row(part)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Robustness comparison of baseline, pre-defined aligned-only bias filter, and exploratory neutral-exclusion policy."
    )
    parser.add_argument("--trades", default="results/crt_m5_bias_alignment/bias_enriched_clean_trades.csv")
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=5601)
    parser.add_argument("--costs-r", type=parse_costs, default=DEFAULT_COSTS_R)
    parser.add_argument("--output", default="results/crt_m5_bias_policy_robustness")
    args = parser.parse_args()

    if args.bootstrap_reps < 1000:
        parser.error("--bootstrap-reps must be >= 1000")

    path = Path(args.trades)
    if not path.exists():
        print(f"ERROR: Bias-enriched trades CSV not found: {path}")
        return 2

    try:
        df = prepare(pd.read_csv(path))
    except Exception as exc:
        print(f"ERROR: Could not prepare bias-enriched trades: {exc}")
        return 2

    if df.empty:
        print("ERROR: No usable bias-enriched trades found.")
        return 1

    rng = np.random.default_rng(args.seed)

    summary = policy_summary(df, args.bootstrap_reps, rng)
    differences = month_block_policy_differences(df, args.bootstrap_reps, rng)
    lomo = leave_one_month_out(df)
    loso = leave_one_symbol_out(df)
    temporal = temporal_split(df)
    costs = cost_sensitivity(df, args.costs_r)
    monthly = monthly_policy(df)

    print("CRT Scanner V1 - Bias policy robustness comparison")
    print(f"Trades: {len(df)} | Months: {df['Clean Month'].nunique()} | Seed: {args.seed}")
    print("ALIGNED_ONLY is confirmatory: the alignment rule existed before this result.")
    print("NON_NEUTRAL is exploratory: neutral exclusion was motivated by the observed bias breakdown.\n")

    print("POLICY SUMMARY")
    print("=" * 170)
    cols = [
        "Policy", "Hypothesis Status", "Trades", "Retention %", "Expectancy R", "Total R",
        "Profit Factor", "Max DD R", "Positive R %",
        "Trade BS 95% Low", "Trade BS 95% High", "Trade BS Mean > 0 %",
        "Month BS 95% Low", "Month BS 95% High", "Month BS Mean > 0 %",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nMONTH-BLOCK POLICY DIFFERENCES")
    print("=" * 132)
    print(differences.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nLEAVE-ONE-MONTH-OUT: ALIGNED VS NON-NEUTRAL")
    print("=" * 118)
    print(lomo[["Excluded Month", "Policy", "Trades", "Expectancy R", "Total R", "Profit Factor", "Max DD R"]].to_string(index=False))

    print("\nLEAVE-ONE-SYMBOL-OUT: ALIGNED VS NON-NEUTRAL")
    print("=" * 118)
    print(loso[["Excluded Symbol", "Policy", "Trades", "Expectancy R", "Total R", "Profit Factor", "Max DD R"]].to_string(index=False))

    print("\nTEMPORAL SPLIT")
    print("=" * 112)
    print(temporal[["Year", "Policy", "Trades", "Expectancy R", "Total R", "Profit Factor", "Max DD R"]].to_string(index=False))

    print("\nCOST SENSITIVITY")
    print("=" * 112)
    print(costs[["Policy", "Cost R / Trade", "Trades", "Expectancy R", "Total R", "Profit Factor", "Max DD R"]].to_string(index=False))

    print("\nMONTHLY POLICY PERFORMANCE")
    print("=" * 112)
    print(monthly[["Clean Month", "Policy", "Trades", "Expectancy R", "Total R", "Profit Factor", "Max DD R"]].to_string(index=False))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "policy_summary.csv": summary,
        "policy_difference_bootstrap.csv": differences,
        "leave_one_month_out.csv": lomo,
        "leave_one_symbol_out.csv": loso,
        "temporal_split.csv": temporal,
        "cost_sensitivity.csv": costs,
        "monthly_policy_performance.csv": monthly,
    }
    for name, frame in files.items():
        frame.to_csv(out_dir / name, index=False)

    xlsx_path = out_dir / "crt_m5_bias_policy_robustness.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Policy Summary", index=False)
            differences.to_excel(writer, sheet_name="Policy Differences", index=False)
            lomo.to_excel(writer, sheet_name="Leave One Month", index=False)
            loso.to_excel(writer, sheet_name="Leave One Symbol", index=False)
            temporal.to_excel(writer, sheet_name="Temporal Split", index=False)
            costs.to_excel(writer, sheet_name="Cost Sensitivity", index=False)
            monthly.to_excel(writer, sheet_name="Monthly Policy", index=False)
        excel_msg = str(xlsx_path)
    except (ImportError, ModuleNotFoundError):
        excel_msg = "not written (install openpyxl)"

    print("\nSaved:")
    for name in files:
        print(f"  {out_dir / name}")
    print(f"  Excel: {excel_msg}")
    print("\nNOTE: ALIGNED_ONLY is a pre-existing hypothesis. NON_NEUTRAL is exploratory and requires independent confirmation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
