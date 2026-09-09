# Frozen Forward Validation

## Purpose

Forward validation compares frozen ML V1 and ML V1.2 models only on CRT occurrences at or after `2026-09-01T00:00:00Z`.

The runner does not retrain models, refit feature selection, or optimize thresholds. Existing historical rows before the cutoff are retained only to provide causal rolling-feature history.

## Frozen models

- V1 1R: `data/models/crt_v2_1R_random_forest.joblib`, threshold 0.50
- V1 1.5R: `data/models/crt_v2_1_5R_random_forest.joblib`, threshold 0.55
- V1.2 1R: `data/models/crt_v2_1R_v1_2_extra_trees.joblib`, threshold 0.50
- V1.2 1.5R: `data/models/crt_v2_1_5R_v1_2_extra_trees.joblib`, threshold 0.50

Thresholds are frozen research settings, not live-trading authorization.

## Outcome maturity

A new CRT occurrence can be scored immediately. Its label remains pending until one of these happens:

1. the target is reached;
2. the stop is reached; or
3. the full 24-H1-bar research horizon completes.

This prevents incomplete recent events from being counted as failures.

## Workflow

Refresh the occurrence dataset through the latest available MT5 history:

```powershell
python run_research.py --start 2023-01-01 --end 2026-09-09
```

Then run frozen forward validation:

```powershell
python run_forward_validation.py --dataset data\research\GOLD_crt_occurrences_2023-01-01_2026-09-09.csv
```

Outputs:

- `data/research/forward/forward_predictions.csv`
- `data/research/forward/forward_metrics.csv`

The prediction journal is upserted by model version, target, signal time, and direction. Re-running the research dataset later updates pending labels without duplicating predictions.

## Metrics

The report includes total forward predictions, pending and labelled predictions, positive rate, ROC-AUC, Brier score, frozen-threshold selected trades, coverage, hit rate, expectancy in R, and net R.

AUC is only shown when both outcome classes are present. Timeout outcomes after a complete 24H horizon contribute 0R to the research expectancy; stop outcomes contribute -1R and target outcomes contribute +1R or +1.5R.

## Research discipline

Do not change model structure or thresholds based on early forward outcomes. Accumulate a meaningful future sample first. Any model changes create a new model version and require a new forward cohort/cutoff rather than rewriting the frozen benchmark.
