from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.session_crt_split import SplitTargetConfig
from backtesting.v4_2_causal import CausalConfirmationConfig, run_v4_2_causal_study
from backtesting.v4_7_frozen_validation import FrozenValidationConfig, run_v4_7_frozen_validation


def main() -> None:
    p = argparse.ArgumentParser(description="Rebuild corrected causal benchmark for V5.1")
    p.add_argument("--h1", required=True)
    p.add_argument("--m15", required=True)
    p.add_argument("--m5", required=True)
    p.add_argument("--cost-r", type=float, default=0.04)
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--output-dir", default="data/research/v5_1_corrected_benchmark")
    a = p.parse_args()

    h1 = pd.read_csv(a.h1)
    m15 = pd.read_csv(a.m15)
    m5 = pd.read_csv(a.m5)
    out = Path(a.output_dir)
    causal_dir = out / "v4_2_corrected"
    frozen_dir = out / "v4_7_rebuilt"

    split_cfg = SplitTargetConfig(max_holding_bars=24)
    conf_cfg = CausalConfirmationConfig(m15_window_hours=3, m5_window_hours=2, structure_lookback=3)
    causal_summary, _, causal_trades = run_v4_2_causal_study(
        h1, m15, m5, causal_dir, split_cfg, conf_cfg
    )

    frozen_cfg = FrozenValidationConfig(
        retest_window_bars=24,
        max_holding_hours=24,
        all_in_cost_r=a.cost_r,
        account_risk_fraction=a.risk,
        validation_window_days=90,
        validation_step_days=60,
        min_trades_per_window=8,
        monte_carlo_runs=5000,
        monte_carlo_seed=42,
    )
    frozen_summary, windows, mc, segments, sensitivity, filled = run_v4_7_frozen_validation(
        causal_trades, m5, frozen_dir, frozen_cfg
    )

    print("CRT V5.1 — CORRECTED CAUSAL BENCHMARK REBUILD")
    print("M15/M5 structure lookback now includes only already-completed pre-signal context.")
    print("No future confirmation bar may be used before its close.\n")
    print("CAUSAL SUMMARY")
    print(causal_summary.to_string(index=False))
    print("\nFROZEN OTE SUMMARY")
    print(frozen_summary.to_string(index=False))
    print("\nMONTE CARLO")
    print(mc.to_string(index=False))
    print(f"\nCorrected filled trades: {len(filled)}")
    print(f"Reports written under: {out}")
    print("This rebuild is historical robustness research, not strict out-of-sample validation.")


if __name__ == "__main__":
    main()
