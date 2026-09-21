from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import mt5_crt_m5_performance_test as base
from mt5_robust_history import load_m15_chunked, load_m5_chunked, resolve_frozen_symbols


# Patch only the data-access layer. All strategy rules, entry logic,
# management logic and performance accounting remain the frozen baseline.
base.load_m15 = load_m15_chunked
base.resolve_symbols = resolve_frozen_symbols


def diagnostic_load_m5(symbol, start_day, end_day):
    """Load chunked M5 history and surface the exact coverage used by the frozen model."""
    df = load_m5_chunked(symbol, start_day, end_day)
    if df.empty:
        print(f"M5 COVERAGE {symbol:<18} bars=0 | EMPTY")
        return df

    first_ny = df.iloc[0]["time_ny"]
    last_ny = df.iloc[-1]["time_ny"]
    print(
        f"M5 COVERAGE {symbol:<18} bars={len(df):>8,} | "
        f"first={first_ny} | last={last_ny}"
    )
    return df


base.load_m5 = diagnostic_load_m5


def _arg_value(name: str, default: str) -> str:
    try:
        i = sys.argv.index(name)
        return sys.argv[i + 1]
    except (ValueError, IndexError):
        return default


def print_setup_diagnostics(output_dir: Path) -> None:
    trades_path = output_dir / "crt_m5_performance_trades.csv"
    if not trades_path.exists():
        return

    try:
        results = pd.read_csv(trades_path)
    except Exception as exc:
        print(f"\nDIAGNOSTIC WARNING: could not read {trades_path}: {exc}")
        return

    if results.empty:
        print("\nSETUP STATUS DIAGNOSTICS: no setup rows were written.")
        return

    print("\nSETUP STATUS DIAGNOSTICS")
    print("=" * 118)
    print(
        f"{'Symbol':<10} {'CRTs':>5} {'C3>=min':>8} {'Raid':>6} {'MSS':>6} "
        f"{'FVG':>6} {'Entry':>7}  Status counts"
    )
    print("-" * 118)

    for logical, part in results.groupby("logical_symbol", sort=False):
        total = len(part)
        c3_ok = int((pd.to_numeric(part.get("c3_bars"), errors="coerce").fillna(0) >= 36).sum()) if "c3_bars" in part else 0
        raids = int(part["raid"].fillna(False).astype(bool).sum()) if "raid" in part else 0
        mss = int(part["mss"].fillna(False).astype(bool).sum()) if "mss" in part else 0
        fvg = int(part["fvg"].fillna(False).astype(bool).sum()) if "fvg" in part else 0
        entries = int(part["retest_entry"].fillna(False).astype(bool).sum()) if "retest_entry" in part else 0

        counts = part["trade_status"].fillna("BLANK").astype(str).value_counts()
        status_text = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(
            f"{str(logical):<10} {total:>5} {c3_ok:>8} {raids:>6} {mss:>6} "
            f"{fvg:>6} {entries:>7}  {status_text}"
        )

    print("\nInterpretation:")
    print("  NO_C3_DATA / INSUFFICIENT_C3 => historical M5 coverage problem, not a strategy result.")
    print("  NO_RAID => M5 data is present but the frozen liquidity-raid condition did not occur.")
    print("  NO_MSS / NO_FVG / NO_RETEST => the setup reached that stage and then failed the next rule.")


if __name__ == "__main__":
    exit_code = base.main()
    if exit_code == 0:
        output_dir = Path(_arg_value("--output", "results/crt_m5_performance"))
        print_setup_diagnostics(output_dir)
    sys.exit(exit_code)
