from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SHADOW = Path("data/shadow/v5_1/v5_shadow_journal.csv")
DEMO = Path("data/demo/v5_1_demo_state.json")


def _demo_frame() -> pd.DataFrame:
    if not DEMO.exists():
        return pd.DataFrame()
    state = json.loads(DEMO.read_text(encoding="utf-8"))
    return pd.DataFrame(list(state.get("signals", {}).values()))


def main() -> None:
    shadow = pd.read_csv(SHADOW) if SHADOW.exists() else pd.DataFrame()
    demo = _demo_frame()

    print("CRT V5.1 — SHADOW / DEMO EXECUTION COMPARISON")
    print("Corrected causal engine | GOLD | OTE 0.79 | 1% setup risk\n")

    print(f"Shadow setups: {len(shadow)}")
    print(f"Demo signals:  {len(demo)}")
    if demo.empty:
        print("No V5.1 demo signal has been validated yet.")
        return

    cols = [c for c in [
        "signal_id", "direction", "confirmation_time_utc", "entry", "stop", "tp1", "tp2",
        "status", "order_tickets", "dry_run"
    ] if c in demo.columns]
    print("\nDEMO JOURNAL")
    print(demo[cols].to_string(index=False))

    if shadow.empty:
        print("\nNo corrected V5.1 shadow rows yet; keep the parallel shadow runner active.")
        return

    keys = ["confirmation_time_utc", "direction"]
    if all(k in shadow.columns and k in demo.columns for k in keys):
        left = shadow.copy()
        right = demo.copy()
        left["confirmation_time_utc"] = pd.to_datetime(left["confirmation_time_utc"], utc=True, errors="coerce")
        right["confirmation_time_utc"] = pd.to_datetime(right["confirmation_time_utc"], utc=True, errors="coerce")
        merged = left.merge(right, on=keys, how="outer", suffixes=("_shadow", "_demo"), indicator=True)
        show = [c for c in [
            "confirmation_time_utc", "direction", "status_shadow", "status_demo",
            "entry_shadow", "entry_demo", "fill_price", "spread_r_at_fill",
            "shadow_slippage_r", "gross_r"
        ] if c in merged.columns]
        print("\nSIGNAL MATCHING")
        print(merged[show + ["_merge"]].to_string(index=False))

    print("\nInterpretation: dry-run rows validate request construction only. Actual fill/slippage comparison begins after DEMO orders are explicitly enabled and broker fills exist.")


if __name__ == "__main__":
    main()
