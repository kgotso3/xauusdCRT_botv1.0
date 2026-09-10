from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.v4_5_regime import RegimeConfig, run_v4_5_regime_study


def main():
    p = argparse.ArgumentParser(description="V4.5 causal regime modelling study")
    p.add_argument("--trades", required=True, help="V4.2 causal trades CSV")
    p.add_argument("--h1", required=True, help="Raw H1 OHLC CSV")
    p.add_argument("--variant", default="FULL_M5_CAUSAL")
    p.add_argument("--train-days", type=int, default=180)
    p.add_argument("--test-days", type=int, default=90)
    p.add_argument("--step-days", type=int, default=60)
    p.add_argument("--min-train-trades", type=int, default=25)
    p.add_argument("--min-test-trades", type=int, default=8)
    p.add_argument("--output-dir", default="data/research/v4_5_regime")
    a = p.parse_args()

    trades_path = Path(a.trades)
    h1_path = Path(a.h1)
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {trades_path}")
    if not h1_path.exists():
        raise FileNotFoundError(f"H1 CSV not found: {h1_path}")

    cfg = RegimeConfig(
        train_days=a.train_days,
        test_days=a.test_days,
        step_days=a.step_days,
        min_train_trades=a.min_train_trades,
        min_test_trades=a.min_test_trades,
    )
    summary, folds, selections, segments, frame = run_v4_5_regime_study(
        pd.read_csv(trades_path),
        pd.read_csv(h1_path),
        a.output_dir,
        a.variant,
        cfg,
    )

    print("CRT V4.5 — CAUSAL REGIME MODELLING")
    print(f"Variant: {a.variant} | train={a.train_days}d | test={a.test_days}d | step={a.step_days}d")
    print(f"Regime-labelled trades: {len(frame)}\n")
    print("SUMMARY")
    print(summary.to_string(index=False))
    print("\nWALK-FORWARD FOLDS")
    print(folds.to_string(index=False) if not folds.empty else "No eligible folds.")
    print("\nTOP REGIME SEGMENTS")
    if segments.empty:
        print("No regime segments.")
    else:
        print(segments.sort_values(["expectancy_r", "trades"], ascending=[False, False]).head(20).to_string(index=False))
    print("\nSELECTED TRAINING RULES")
    print(selections.to_string(index=False) if not selections.empty else "No regime rules selected in training folds.")

    out = Path(a.output_dir)
    print(f"\nReports: {out / 'v4_5_summary.csv'}")
    print(f"         {out / 'v4_5_walk_forward_folds.csv'}")
    print(f"         {out / 'v4_5_selected_rules.csv'}")
    print(f"         {out / 'v4_5_regime_segments.csv'}")
    print(f"         {out / 'v4_5_regime_trades.csv'}")
    print("Research-only. Regimes are not an MT5 execution gate.")


if __name__ == "__main__":
    main()
