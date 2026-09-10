from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.v4_7_frozen_validation import FrozenValidationConfig, run_v4_7_frozen_validation


def main():
    p = argparse.ArgumentParser(description="V4.7 frozen validation for OTE 0.705 and 0.79")
    p.add_argument("--trades", required=True, help="V4.2 causal trades CSV")
    p.add_argument("--m5", required=True, help="M5 OHLC CSV")
    p.add_argument("--retest-bars", type=int, default=24)
    p.add_argument("--max-hold-hours", type=int, default=24)
    p.add_argument("--cost-r", type=float, default=0.04, help="All-in round-trip execution cost in R")
    p.add_argument("--risk", type=float, default=0.0025, help="Account risk fraction per setup")
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--step-days", type=int, default=60)
    p.add_argument("--min-trades-window", type=int, default=8)
    p.add_argument("--mc-runs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="data/research/v4_7_frozen_validation")
    a = p.parse_args()

    trades_path = Path(a.trades)
    m5_path = Path(a.m5)
    if not trades_path.exists():
        raise FileNotFoundError(f"V4.2 trades CSV not found: {trades_path}")
    if not m5_path.exists():
        raise FileNotFoundError(f"M5 CSV not found: {m5_path}")

    cfg = FrozenValidationConfig(
        retest_window_bars=a.retest_bars,
        max_holding_hours=a.max_hold_hours,
        all_in_cost_r=a.cost_r,
        account_risk_fraction=a.risk,
        validation_window_days=a.window_days,
        validation_step_days=a.step_days,
        min_trades_per_window=a.min_trades_window,
        monte_carlo_runs=a.mc_runs,
        monte_carlo_seed=a.seed,
    )

    summary, windows, mc, segments, sensitivity, filled = run_v4_7_frozen_validation(
        pd.read_csv(trades_path), pd.read_csv(m5_path), a.output_dir, cfg
    )

    print("CRT V4.7 — FROZEN VALIDATION")
    print("Frozen candidates: OTE 0.705 and OTE 0.79")
    print(f"Retest window: {a.retest_bars} M5 bars | holding horizon: {a.max_hold_hours}h | all-in cost: {a.cost_r:.3f}R")
    print(f"Account risk simulation: {a.risk*100:.3f}% per setup | Monte Carlo runs: {a.mc_runs}\n")

    print("FROZEN SUMMARY")
    print(summary.to_string(index=False))

    print("\nMONTE CARLO")
    print(mc.to_string(index=False))

    print("\nCOST SENSITIVITY")
    print(sensitivity.to_string(index=False))

    print("\nKEY SEGMENTS")
    if segments.empty:
        print("No segment rows.")
    else:
        key = segments.loc[segments["dimension"].isin(["entry_model", "direction", "year", "quarter"])]
        print(key.to_string(index=False))

    out = Path(a.output_dir)
    print(f"\nReports: {out / 'v4_7_summary.csv'}")
    print(f"         {out / 'v4_7_validation_windows.csv'}")
    print(f"         {out / 'v4_7_monte_carlo.csv'}")
    print(f"         {out / 'v4_7_segments.csv'}")
    print(f"         {out / 'v4_7_cost_sensitivity.csv'}")
    print(f"         {out / 'v4_7_account_curve.csv'}")
    print(f"Frozen filled trades: {len(filled)}")
    print("Research-only. Passing this study does not enable MT5 execution.")


if __name__ == "__main__":
    main()
