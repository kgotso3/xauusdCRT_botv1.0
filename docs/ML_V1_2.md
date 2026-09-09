# CRT V2 — ML V1.2

ML V1.2 expands the causal research dataset and compares stronger models while preserving chronological validation.

## What changed

The updated research dataset adds completed-H1 information available at decision time only:

- MACD, signal and histogram
- 1H / 4H / 8H returns
- 4H ATR change
- 4H EMA20 slope
- 4H / 8H / 24H recent high and low context
- distance from recent highs/lows normalized by ATR
- range position over 4H / 8H / 24H
- previous New York trading-day high/low and ATR-normalized distance
- intrabar ambiguity audit flag for outcome labels

The V1.2 feature layer derives additional interactions including:

- RSI distance from 50
- direction-adjusted RSI
- normalized MACD values
- direction-adjusted MACD histogram
- direction-adjusted 4H / 8H momentum
- wick imbalance
- prior-day sweep flags
- direction-adjusted prior-day liquidity sweep

## Model ladder

Each target is trained separately:

- 1R: hit 1R before stop
- 1.5R: hit 1.5R before stop

Candidate models:

1. Logistic Regression
2. Random Forest
3. Extra Trees
4. Gradient Boosting

Target-specific `SelectKBest(mutual_info_classif)` is fitted inside the pipeline on TRAIN only. Model selection uses VALIDATION ROC-AUC with Brier score as the tie-break. TEST is report-only and must not be used to tune V1.2.

## Required rebuild

Pull the repository and run the tests:

```powershell
git pull origin main
pytest -q
```

Rebuild the occurrence dataset so the new raw V1.2 fields are present:

```powershell
python run_research.py --start 2023-01-01 --end 2026-09-01
```

The broker may not provide data all the way back to the requested start date. Use the coverage diagnostics to verify the real common H1/M15/M5 range.

Then train V1.2 using the regenerated occurrence CSV:

```powershell
python run_ml_v1_2.py --dataset data\research\GOLD_crt_occurrences_2023-01-01_2026-09-01.csv
```

Outputs:

- `data/research/ml_v1_2/ml_v1_2_candidates.csv`
- `data/research/ml_v1_2/ml_v1_2_selected.csv`
- `data/research/ml_v1_2/ml_v1_2_selected_features.csv`
- selected model files under `data/models/`

## Promotion rule

V1.2 remains research-only. A better TEST result is not a reason to tune further on TEST or promote to XM demo. Any model/rule changes after reviewing TEST require a new future forward holdout period.
