from __future__ import annotations

import sys

import mt5_crt_m5_performance_test as base
from mt5_robust_history import load_m15_chunked, load_m5_chunked, resolve_frozen_symbols


# Patch only the data-access layer. All strategy rules, entry logic,
# management logic and performance accounting remain the frozen baseline.
base.load_m15 = load_m15_chunked
base.load_m5 = load_m5_chunked
base.resolve_symbols = resolve_frozen_symbols


if __name__ == "__main__":
    sys.exit(base.main())
