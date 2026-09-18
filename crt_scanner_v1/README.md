# CRT Scanner V1 — TradingView / GitHub Edition

Analysis-only scanner for seven symbols:

- XAUUSD
- BTCUSD
- GBPUSD
- EURGBP
- EURUSD
- USDJPY
- USDCAD

## CRT definition

New York time (`America/New_York`):

- AM: C1 01:00–05:00, C2 05:00–09:00, C3 starts 09:00.
- PM: C1 13:00–17:00, C2 17:00–21:00, C3 starts 21:00.
- Valid CRT: C2 sweeps exactly one side of C1 and closes back inside C1.
- Direction: low sweep -> bullish; high sweep -> bearish.
- Daily bias is a separate deterministic D1 + H4 trend classifier.

## Architecture

1. `tradingview/crt_scanner_v1.pine` scans all seven symbols on a 15-minute host chart.
2. TradingView sends one consolidated JSON webhook at the AM/PM C2 close.
3. `webhook_api.py` validates the shared secret and stores the scan.
4. PostgreSQL is the shared production store; SQLite is the local fallback.
5. `app.py` is the Streamlit dashboard.
6. For valid CRT setups, M15/M5 screenshots can optionally be reviewed with OpenAI vision for approximate MSS, Order Block and FVG overlays.

## Local run

Create a virtual environment, then:

```bash
pip install -r requirements.txt
```

Copy `.env.example` values into your shell/environment.

Webhook API:

```bash
uvicorn webhook_api:app --host 0.0.0.0 --port 8000
```

Streamlit:

```bash
streamlit run app.py
```

Health check:

```text
GET /health
```

TradingView webhook endpoint:

```text
POST /tradingview
```

## TradingView setup

1. Open a 15-minute chart.
2. Add `tradingview/crt_scanner_v1.pine` to Pine Editor and compile it.
3. Add the indicator to the chart.
4. Set the same long random value in the Pine `Webhook secret` input and the webhook deployment's `TV_WEBHOOK_SECRET` environment variable.
5. Create an alert on **Any alert() function call**.
6. Set the webhook URL to your public HTTPS API URL ending in `/tradingview`.

The script fires after the 15-minute bar that closes at 09:00 or 21:00 New York time. It does not place orders.

## One-click-style Render deployment

`render.yaml` defines:

- `crt-scanner-v1-webhook` — FastAPI TradingView receiver.
- `crt-scanner-v1-dashboard` — Streamlit dashboard.
- `crt-scanner-v1-db` — shared PostgreSQL database.

When creating the Blueprint, point Render to this repository, branch `crt-scanner-v1`, and blueprint path `crt_scanner_v1/render.yaml`.

Render generates `TV_WEBHOOK_SECRET` for the webhook service. Copy that generated value into the Pine indicator's **Webhook secret** setting.

`OPENAI_API_KEY` is optional. If omitted, the 7-symbol scanner and dashboard still work; only AI screenshot annotation is disabled.

## Streamlit Community Cloud alternative

You can also deploy `crt_scanner_v1/app.py` directly from branch `crt-scanner-v1`. The requirements file is next to the entrypoint, which Community Cloud supports.

If using Community Cloud, supply a PostgreSQL `DATABASE_URL` reachable from outside the webhook host, plus optional `OPENAI_API_KEY` and `OPENAI_VISION_MODEL=gpt-5.6-luna` secrets.

## Important limitations

- Screenshot structure analysis is a review aid, not deterministic market-structure ground truth.
- Confirm MSS, OB and FVG directly on TradingView before acting.
- No MT5 integration and no automatic execution are included in V1.
