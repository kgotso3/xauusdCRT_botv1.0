from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from shadow.v5_engine import FROZEN_BENCHMARK, comparison_frame, load_state, shadow_metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Report V5.0 prospective shadow performance vs V4.7 OTE 0.79")
    p.add_argument("--state", default="data/shadow/v5_shadow_state.json")
    a = p.parse_args()
    path = Path(a.state)
    if not path.exists():
        raise FileNotFoundError(f"Shadow state not found: {path}")
    state = load_state(path)
    m = shadow_metrics(state)
    print("CRT V5.0 — SHADOW VS V4.7 BENCHMARK")
    print(f"Cohort started: {state['started_at_utc']}")
    print(f"Setups={m['prospective_setups']} fills={m['prospective_fills']} closed={m['closed_fills']}")
    print(f"Average observed spread={m['avg_spread_r']:.4f}R" if pd.notna(m['avg_spread_r']) else "Average observed spread=n/a")
    print(f"Average shadow slippage={m['avg_shadow_slippage_r']:.4f}R" if pd.notna(m['avg_shadow_slippage_r']) else "Average shadow slippage=n/a")
    print("\nBENCHMARK COMPARISON")
    print(comparison_frame(m).to_string(index=False))
    if m["closed_fills"] < 30:
        print("\nSTATUS: cohort immature — do not judge strategy yet.")
    elif m["closed_fills"] < 40:
        print("\nSTATUS: minimum cohort reached; continuing toward the default 40-fill target is preferred.")
    else:
        print("\nSTATUS: target-sized prospective cohort reached; ready for formal V5.0 review.")
