# CRT Scanner V2 — Bias-Aligned Shadow Validation

## Purpose

CRT Scanner V2 promotes the pre-defined Daily Bias alignment rule into a prospective research candidate while preserving V1 as the control.

No component in this phase sends, modifies or closes broker orders.

## Frozen research models

### V1 control

Primary cohort only:

- XAUUSD
- US500
- US30

Eligibility:

- valid CRT

### V2 candidate

Same primary cohort and same frozen CRT/M5 rules as V1, plus:

- valid CRT
- CRT direction equals frozen D1/H4 bias

Bias score remains unchanged:

- D1 close vs EMA50: +/-2
- D1 EMA20 vs EMA50: +/-1
- D1 EMA50 slope: +/-1
- H4 close vs EMA50: +/-1
- H4 EMA20 vs EMA50: +/-1
- H4 close vs EMA20: +/-1
- BULLISH >= +4
- BEARISH <= -4
- otherwise NEUTRAL

## Historical benchmark frozen before prospective collection

Coverage-safe primary cohort baseline:

- 101 clean trades
- expectancy +0.160R
- total +16.17R
- profit factor 1.463
- max drawdown -10.09R
- 7/8 positive clean months

Bias-aligned historical candidate:

- 35 trades
- expectancy +0.502R
- total +17.56R
- profit factor 2.912
- max drawdown -3.67R
- trade-bootstrap 95% mean-R interval approximately +0.126R to +0.886R

These values are research benchmarks, not live-performance targets or guarantees.

## Architecture

1. `crt_scanner_v1/tradingview/crt_scanner_v2.pine`
   - separate V2 Pine source
   - V1 Pine remains unchanged
   - sends the same CRT/bias fields with payload version `2.0.0`

2. `crt_scanner_v1/webhook_api.py`
   - receives TradingView scans
   - creates prospective shadow rows
   - exposes authenticated pending-shadow feed
   - accepts research outcome updates

3. `crt_scanner_v1/storage.py`
   - preserves existing V1 scan tables
   - adds `crt_shadow_setups`
   - every valid primary-cohort CRT becomes a V1 control row
   - aligned rows are additionally flagged as V2 candidates

4. `crt_scanner_v1/app.py`
   - displays V1-control and V2-candidate status
   - shows prospective shadow ledger
   - supports manual audit resolution if needed

5. `mt5_crt_v2_shadow_runner.py`
   - runs on the local Windows machine with XM/MT5
   - reads pending setups after C3 completes
   - applies the already-frozen M5 raid/MSS/FVG/retest/management model
   - writes `NO_ENTRY`, `STOP`, `BE`, `TP2`, `TIMEOUT` or `AMBIGUOUS` back to the ledger
   - contains no broker order-execution logic

## First-time setup

Pull the branch:

```powershell
cd C:\Users\Kgotso.Maila\Documents\xauusdCRT_botv1.0
git checkout crt-scanner-v1
git pull origin crt-scanner-v1
```

Run the scanner tests:

```powershell
cd crt_scanner_v1
python -m pytest -q
cd ..
```

Render should redeploy the existing webhook/dashboard services from `crt_scanner_v1` if automatic deployment is enabled. The new shadow table is additive and is created by SQLAlchemy at startup.

## TradingView

Use:

`crt_scanner_v1/tradingview/crt_scanner_v2.pine`

Keep the host chart on 15 minutes and set the same webhook secret used by the deployed webhook.

Create/recreate the alert using **Any alert() function call** and keep the existing webhook URL ending in `/tradingview`.

V1 Pine is intentionally retained separately as the frozen source control.

## Local prospective shadow runner

After the webhook service has redeployed and V2 scans are reaching the dashboard, run one research pass after C3 has completed:

```powershell
python mt5_crt_v2_shadow_runner.py `
    --api-url https://YOUR-WEBHOOK-SERVICE.onrender.com `
    --secret YOUR_TV_WEBHOOK_SECRET
```

To leave the research runner open and poll every five minutes:

```powershell
python mt5_crt_v2_shadow_runner.py `
    --api-url https://YOUR-WEBHOOK-SERVICE.onrender.com `
    --secret YOUR_TV_WEBHOOK_SECRET `
    --watch `
    --poll-seconds 300
```

The runner waits until the relevant C3 window is complete plus a default 10-minute grace period before resolving a setup.

## Data-source note

Live CRT/bias classification originates from the configured TradingView feeds. M5 shadow confirmation/outcomes originate from the local XM/MT5 feed. Provider differences can therefore create small price/candle differences. The shadow ledger records this as prospective research rather than treating it as identical-provider execution evidence.

## Research discipline

Do not change the following during the first V2 shadow cohort:

- primary symbols
- CRT validity
- bias score/thresholds
- M5 pivot definition
- raid/MSS/FVG rules
- first-edge retrace entry
- 10-pip SL buffer
- +1R break-even rule
- +2R TP rule
- C3 timeout policy

Further filters should be tested as separate hypotheses instead of being silently folded into V2.
