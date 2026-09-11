from __future__ import annotations

import argparse
from argparse import Namespace

from run_v5_shadow import run


def main() -> None:
    p = argparse.ArgumentParser(description="V5.1 corrected causal NASDAQ shadow runner")
    p.add_argument("--target-fills", type=int, default=40, choices=range(30, 51))
    p.add_argument("--risk", type=float, default=0.01)
    p.add_argument("--poll-seconds", type=int, default=5)
    p.add_argument("--symbol", default="US100Cash")
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    run(Namespace(
        target_fills=a.target_fills,
        risk=a.risk,
        poll_seconds=a.poll_seconds,
        symbol=a.symbol,
        state="data/shadow/v5_1_nasdaq_shadow_state.json",
        output_dir="data/shadow/v5_1_nasdaq",
        once=a.once,
    ))


if __name__ == "__main__":
    main()
