# XAUUSD CRT Bot - Project Guide

## Purpose

This document is the technical reference for the GOLD/XAUUSD CRT trading project. It explains the architecture, research methodology, implementation, testing, assumptions, current results, and the V1-to-V2 roadmap.

The project is designed for XM MetaTrader 5 demo execution first. Live deployment must not be enabled until the strategy, risk controls, execution checks, and out-of-sample validation are complete.

## Core design

The system is separated into four responsibilities:

1. Prediction/research - what market conditions historically followed a CRT occurrence?
2. Strategy - does the current setup satisfy the chosen rule set?
3. Risk - is the setup safe to trade and what size is allowed?
4. Execution - send only a validated demo order to MT5/XM.

GitHub is used for version control and CI. MT5 and the Python process should eventually run on a Windows VPS for continuous operation.

## Instrument and timeframes

- Broker instrument: XM `GOLD`
- Primary timeframe: H1
- Confirmation timeframes: M15 and M5
- Timezone logic: UTC market timestamps converted with `America/New_York` for DST-aware session features
- Initial risk per trade: 0.25% of account equity
- Current research targets: 1R and 1.5R only

## CRT definition used by V1/V2 research

The detector compares the current completed H1 candle to the previous completed H1 candle.

Bullish CRT:

- current low trades below the previous low;
- current close returns inside the previous candle range.

Bearish CRT:

- current high trades above the previous high;
- current close returns inside the previous candle range.

Only completed H1 candles are used. Bar position 0 in MT5 is the currently forming bar and is excluded from signal decisions.

## Causal data rule

All features are frozen at H1 decision time. Future H1 bars are used only for outcome labels such as whether price reached 1R or 1.5R before the stop.

This separation is mandatory for avoiding look-ahead bias.

## Research dataset

The research layer records every detected H1 CRT occurrence, not only setups approved by the strategy.

Important fields include:

- signal/decision/entry UTC timestamps
- New York local date/hour
- direction and CRT direction
- H1/M15/M5 trend bias
- alignment count
- candle range and body percentage
- sweep size and sweep/ATR ratio
- directional wick size
- EMA20, EMA50, EMA200 and distances
- RSI14
- ATR14 and ATR percentage
- entry, stop and risk distance
- 1R and 1.5R hit labels
- bars-to-target and bars-to-stop
- maximum favourable excursion (MFE) in R
- maximum adverse excursion (MAE) in R

## Entry and outcome assumptions

Research entry is the next H1 candle open after a completed CRT signal.

For BUY:

- stop = CRT signal candle low.

For SELL:

- stop = CRT signal candle high.

Outcome evaluation uses H1 OHLC data. If a stop and target are touched in the same H1 bar, the engine uses the conservative assumption that the stop occurred first because the intrabar path is unknown.

## Historical data loader

Long MT5 ranges are downloaded in chunks because large M5/M15 requests can return `Terminal: Invalid params`.

The loader concatenates chunks, sorts them chronologically, and removes duplicate timestamps.

Historical coverage must always be checked after download. A requested start date does not guarantee the broker terminal has that entire lower-timeframe history.

## Current historical research result

Requested range: 2023-01-01 to 2026-09-01.

Usable CRT dataset:

- first CRT signal: 2025-04-11 13:00 UTC
- last CRT signal: 2026-08-31 21:00 UTC
- CRT occurrences: 2,965
- BUY: 1,682
- SELL: 1,283
- historical NY 08:00-13:00 occurrences: 672

Legacy research results before narrowing targets:

- hit 1R before stop: 43.31%
- hit 1.5R before stop: 36.05%
- hit 2R before stop: 31.20%
- hit 3R before stop: 24.11%
- average MFE: 3.43R
- average MAE: -2.66R

V2 will optimize and validate only 1R and 1.5R. The 2R/3R figures remain historical diagnostics, not V2 targets.

## Why 1R and 1.5R

At 1R, break-even win rate before costs is 50%.

Expected R per trade:

`expectancy_1R = hit_rate * 1 - (1 - hit_rate)`

At 1.5R, break-even win rate before costs is 40%.

`expectancy_1.5R = hit_rate * 1.5 - (1 - hit_rate)`

The goal is not merely a high win rate. A candidate must maintain positive expectancy after realistic spread/slippage assumptions are later introduced.

## V2 killzones

V2 research uses three configurable killzones expressed in New York local time:

- Asia: 20:00-00:00
- London: 02:00-05:00
- New York: 08:00-11:00

Asia wraps across midnight. The session definitions are configuration defaults, not permanent assumptions. They must be revalidated when changed.

Because UTC timestamps are converted through `America/New_York`, the local New York clock is DST-aware.

## Segmentation engine

The segmentation layer compares CRT outcomes by:

- killzone
- killzone + direction
- killzone + direction + alignment
- New York hour
- direction
- weekday
- alignment count
- direction + alignment
- RSI band
- volatility regime
- sweep/ATR band
- New York hour + direction
- New York direction + RSI
- New York direction + sweep size
- New York direction + volatility

The ranking focus is now 1.5R hit rate, then 1R hit rate, while always retaining sample counts.

Small high-performing groups are treated as hypotheses, not conclusions.

## Walk-forward validation

V2 uses a chronological split:

- TRAIN: first 60%
- VALIDATION: next 20%
- TEST: final 20%

Candidate rules are compared on TRAIN and VALIDATION. TEST is an untouched chronological holdout and must not be used to tune rules.

The walk-forward report evaluates deterministic candidate filters such as:

- Asia/London/New York
- killzone + BUY/SELL
- killzone + BUY/SELL + alignment 0/1/2/3
- direction alone
- alignment alone

For each split the report records:

- sample count
- 1R hit rate and expectancy
- 1.5R hit rate and expectancy

A candidate is considered sample-stable only if it has the configured minimum number of observations in every chronological split.

The report also produces monthly and quarterly stability tables.

## V1 backtest benchmark

The earlier V1 approved-strategy backtest used next-H1-open entries, a structural H1 stop, 2R target, one position at a time, 24-bar maximum holding period, and no execution costs.

Six-month result:

- trades: 40
- wins/losses: 11/29
- win rate: 27.50%
- net: -7R
- expectancy: -0.175R
- profit factor: 0.76
- max drawdown: 12R

This result showed that the initial hand-built approval rules did not create an adequate edge and justified moving to occurrence-level research rather than optimizing the 40 accepted trades.

## MT5 safety implementation

### Connection

`mt5/connection.py` initializes the local MT5 terminal, logs in using environment variables, checks the account and verifies demo mode/trading availability.

Secrets belong only in the local `.env` file. `.env` is excluded from Git. `.env.example` contains placeholders only.

### Market data

`mt5/market_data.py` resolves the XM gold symbol, enables it in Market Watch, loads completed bars and provides spread/tick information.

Completed-only retrieval excludes the forming candle and can verify bar completion against an aware UTC timestamp.

### Position sizing

`risk/position_size.py` sizes from account equity and risk percentage.

It uses MT5 `order_calc_profit()` to estimate one-lot stop loss in the actual account currency, floors volume to the broker step, respects min/max volume and returns zero if broker minimum size would exceed the risk budget.

### Execution validation

`mt5/execution.py` validates BUY/SELL stop and target geometry, symbol digits/point, minimum stop distance and freeze level before constructing a market order.

Demo-only checks remain mandatory during development.

### Journal and duplicate prevention

`database/journal.py` records signal IDs and execution states in SQLite.

A deterministic signal ID is based on symbol, H1 bar and direction. Signals already marked PENDING or SENT are blocked from duplicate demo execution.

## Tests

The test suite covers:

- session/time logic
- position sizing
- journal duplicate protection
- CRT detection
- signal-direction mapping
- research occurrence creation and leakage-safe timing
- killzone classification
- chronological train/validation/test splitting
- 1R/1.5R walk-forward metric generation

Run:

```powershell
pytest -q
```

The suite must pass before research or demo-execution changes are accepted.

## Main commands

Refresh code:

```powershell
git pull origin main
```

Run tests:

```powershell
pytest -q
```

Build occurrence research:

```powershell
python run_research.py --start 2023-01-01 --end 2026-09-01
```

Segment the dataset:

```powershell
python analyze_research.py --dataset data\research\GOLD_crt_occurrences_2023-01-01_2026-09-01.csv
```

Run V2 chronological validation:

```powershell
python run_walkforward.py --dataset data\research\GOLD_crt_occurrences_2023-01-01_2026-09-01.csv
```

## V2 acceptance philosophy

A V2 rule should not be promoted just because it has the highest full-dataset win rate.

Preferred evidence:

1. useful sample size;
2. positive or improving expectancy at 1R/1.5R;
3. similar behavior in TRAIN and VALIDATION;
4. acceptable untouched TEST behavior;
5. monthly/quarterly stability;
6. robustness to spread/slippage and execution constraints;
7. no look-ahead leakage.

## Planned next stages

1. Run the V2 walk-forward report on the current occurrence dataset.
2. Identify stable Asia/London/New York candidate filters.
3. Add transaction-cost and spread assumptions to research/backtests.
4. Improve historical M5 coverage where possible.
5. Add richer causal features such as normalized EMA distances, recent high/low distance, spread, volatility regime and liquidity context.
6. Freeze deterministic V2 benchmark rules.
7. Build ML only after the benchmark and dataset are stable.
8. Compare logistic regression baseline against tree models using chronological/walk-forward validation.
9. Add risk-manager kill switches and order margin/filling-mode checks.
10. Run XM demo execution before any live-capital consideration.

## ML roadmap

The later model should predict probabilities such as BUY/SELL/NO-TRADE or probability that a CRT occurrence reaches 1R/1.5R before stop.

Candidate features must be causal and known at decision time. Random train/test splitting is prohibited for time-series validation.

A simple interpretable baseline should be built before complex models. Tree models are acceptable only when they improve out-of-sample performance and stability.

## Deployment roadmap

GitHub is the source-control and CI layer. It is not the preferred 24/7 MT5 runtime.

Final deployment architecture:

GitHub -> Windows VPS -> Python process -> local MT5 terminal -> XM demo/live account

The VPS phase should add persistent H1 scheduling, structured logs, health monitoring, notifications and safe restart behavior.

## Known gaps before unattended demo automation

- broker filling-mode handling still requires final audit
- daily loss limit and consecutive-loss kill switch
- margin pre-check
- maximum trades per session/day
- persistent new-H1-bar runner
- spread calibration and execution-cost backtests
- deeper historical coverage validation
- out-of-sample V2 rule confirmation

Do not bypass these gates for unattended or live execution.

## Development principle

The project should progress from evidence to automation:

`raw CRT occurrences -> causal features -> segmentation -> chronological validation -> deterministic V2 benchmark -> ML comparison -> risk controls -> XM demo -> monitored deployment`

This prevents the system from becoming an automated version of an unvalidated trading idea.
