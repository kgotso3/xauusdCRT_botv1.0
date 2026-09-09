# CRT V2 Research Pipeline

This pipeline deliberately separates discovery, deterministic benchmarking and ML modelling. The 1R and 1.5R outcomes are always treated as separate targets.

## Stage 1 — Historical coverage diagnostics

Use the raw history CSVs produced by `run_research.py`.

```powershell
python run_coverage_diagnostics.py `
  --h1 data\research\GOLD_H1_2023-01-01_2026-09-01.csv `
  --m15 data\research\GOLD_M15_2023-01-01_2026-09-01.csv `
  --m5 data\research\GOLD_M5_2023-01-01_2026-09-01.csv
```

Inspect `common_start` and `common_end`. Do not describe the research dataset as covering the requested filename period unless all three timeframes genuinely overlap over that interval.

## Stage 2 — Causal V2 feature dataset

```powershell
python build_v2_features.py --dataset data\research\GOLD_crt_occurrences_2023-01-01_2026-09-01.csv
```

Key safeguards:

- Rolling ATR percentile uses current/past rows only.
- No full-dataset quantile is used as an ML feature.
- MTF bias, sweep, candle structure, EMA/ATR distances, time and killzone features come from information available by decision time.
- Outcome columns remain labels, not predictors.

## Stage 3 — Feature analysis

```powershell
python run_feature_analysis.py --dataset data\research\GOLD_crt_occurrences_2023-01-01_2026-09-01.csv
```

Feature discovery uses only the first 80% chronological development sample. The final 20% is excluded from feature discovery. Rankings are descriptive research, not proof of edge.

## Stage 4 — Deterministic benchmark

```powershell
python run_deterministic_benchmark.py --dataset data\research\GOLD_crt_occurrences_2023-01-01_2026-09-01.csv
```

The benchmark contains predeclared transparent rules such as all CRT, killzone-only, alignment, sweep-size, directional RSI and fixed combinations. ML should be compared against this benchmark rather than against no baseline.

## Stage 5 — ML V1

Install the new dependencies after pulling:

```powershell
pip install -r requirements.txt
```

Train:

```powershell
python run_ml_v1.py --dataset data\research\GOLD_crt_occurrences_2023-01-01_2026-09-01.csv
```

ML V1 trains two independent binary models:

1. `P(hit 1R before stop)`
2. `P(hit 1.5R before stop)`

Candidate algorithms:

- Logistic Regression — interpretable baseline.
- Random Forest — nonlinear challenger.

The chronological split is 60% TRAIN / 20% VALIDATION / 20% TEST. Model selection uses VALIDATION only. TEST is report-only and must not be used to invent new features, thresholds or rules after inspection.

Primary metrics:

- ROC-AUC — ranking/discrimination.
- Brier score — probability quality/calibration error.
- Log loss — penalizes overconfident errors.
- Precision/recall — behaviour at the default 0.50 decision threshold.

A model is not promoted to XM demo just because TEST metrics are attractive. It must first beat the deterministic benchmark in a meaningful way and then survive a new forward period.

## Promotion sequence

`coverage -> causal features -> feature analysis -> deterministic benchmark -> ML V1 -> forward holdout -> probability threshold/risk policy -> XM demo`

The existing V2 execution engine remains disabled until an explicit validated rule/model promotion is made.
