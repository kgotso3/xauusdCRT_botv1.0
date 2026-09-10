from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.session_crt_split import SplitTargetConfig
from backtesting.v4_2_causal import CausalConfirmationConfig, run_v4_2_causal_study


def main() -> None:
    p = argparse.ArgumentParser(description="V4.2 causal unrestricted CRT confirmation study")
    p.add_argument("--h1", required=True, help="H1 historical CSV")
    p.add_argument("--m15", required=True, help="M15 historical CSV")
    p.add_argument("--m5", default=None, help="Optional M5 historical CSV for final refinement")
    p.add_argument("--max-hold", type=int, default=24)
    p.add_argument("--m15-window-hours", type=int, default=3)
    p.add_argument("--m5-window-hours", type=int, default=2)
    p.add_argument("--structure-lookback", type=int, default=3)
    p.add_argument("--output-dir", default="data/research/v4_2_causal")
    a = p.parse_args()

    h1p, m15p = Path(a.h1), Path(a.m15)
    if not h1p.exists():
        raise FileNotFoundError(f"H1 CSV not found: {h1p}")
    if not m15p.exists():
        raise FileNotFoundError(f"M15 CSV not found: {m15p}")
    m5 = None
    if a.m5:
        m5p = Path(a.m5)
        if not m5p.exists():
            raise FileNotFoundError(f"M5 CSV not found: {m5p}")
        m5 = pd.read_csv(m5p)

    split_cfg = SplitTargetConfig(max_holding_bars=a.max_hold)
    conf_cfg = CausalConfirmationConfig(
        m15_window_hours=a.m15_window_hours,
        m5_window_hours=a.m5_window_hours,
        structure_lookback=a.structure_lookback,
    )
    summary, breakdown, trades = run_v4_2_causal_study(
        pd.read_csv(h1p), pd.read_csv(m15p), m5,
        output_dir=a.output_dir,
        split_config=split_cfg,
        confirmation_config=conf_cfg,
    )

    print("CRT V4.2 — CAUSAL CONFIRMATION STUDY")
    print("24-hour H1 scan | confirmation known only at candle close | entry occurs after confirmation")
    print("50/50 risk split | TP1=1R | TP2=2R | runner keeps original stop")
    print(f"Raw H1 CRT candidates: {len(trades)}\n")
    print(summary.to_string(index=False))
    print("\nBreakdowns saved by New York hour, direction, premium/discount correctness, and year.")
    print(f"Reports: {Path(a.output_dir) / 'v4_2_causal_summary.csv'}")
    print(f"         {Path(a.output_dir) / 'v4_2_causal_breakdown.csv'}")
    print(f"         {Path(a.output_dir) / 'v4_2_causal_trades.csv'}")
    print("Research-only. No MT5 execution enabled.")


if __name__ == "__main__":
    main()
