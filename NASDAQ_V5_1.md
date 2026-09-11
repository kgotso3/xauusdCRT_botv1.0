# CRT V5.1 NASDAQ validation branch

Branch: `nasdaq-v5.1`

Purpose: run the frozen GOLD V5.1 strategy unchanged on the Nasdaq-100 as an external-validation experiment. This branch must not tune the strategy to improve NASDAQ results.

## Frozen strategy specification

- H1 CRT sweep + reclaim
- M15 causal MSS
- M15 FVG
- M5 causal MSS
- M5 FVG
- OTE entry: 0.79
- Retest window: 2 hours
- Holding window: 24 hours
- Setup risk: 1.00%
- Split target structure: 50% volume to 1R, 50% volume to 2R
- Maximum open setups: 1
- Daily loss guard: 2R
- Consecutive-loss guard: 3
- MT5 DEMO account hard lock remains mandatory

## Instrument

Default requested symbol: `US100Cash`.

The broker symbol resolver also recognizes common NASDAQ aliases including `US100`, `NAS100`, `USTEC`, `NASDAQ100`, and broker suffix variants.

## Separate prospective data

NASDAQ shadow state:
`data/shadow/v5_1_nasdaq_shadow_state.json`

NASDAQ shadow reports:
`data/shadow/v5_1_nasdaq/`

NASDAQ demo state:
`data/demo/v5_1_nasdaq_demo_state.json`

These are deliberately separate from GOLD so cohorts cannot be mixed.

## Shadow diagnostic

```powershell
python run_v5_1_shadow.py --once
```

## Continuous NASDAQ shadow

```powershell
python run_v5_1_shadow.py --target-fills 40 --risk 0.01 --poll-seconds 5
```

## NASDAQ demo dry-run

```powershell
python run_v5_1_demo.py
```

Do not use `--execute-demo` until the same execution-hardening gates required for GOLD have passed.

## Comparison objective

Compare GOLD and NASDAQ prospectively using the same frozen strategy. Primary comparison metrics:

- setup count
- fill rate
- expectancy (R)
- profit factor
- win rate
- max drawdown (R)
- spread/slippage
- frequency of TP1 / TP2 / stop / timeout outcomes
- actual monetary risk vs intended 1% risk once demo execution is enabled

No NASDAQ-specific parameter optimization is permitted during the initial comparison cohort. Any later NASDAQ-specific strategy would be treated as a separate research version rather than part of this external-validation test.
