# XAUUSD CRT AI Trader — V1

A modular MetaTrader 5 trading system for XAUUSD, initially focused on deterministic CRT signals, risk management, backtesting and **XM MT5 DEMO execution**. The ML layer is intentionally deferred until the V1 strategy has been validated.

## V1 status

- Mode: `DEMO`
- Instrument: `XAUUSD`
- Primary timeframe: `H1`
- Confirmation: `M15` + `M5`
- Session: New York
- Initial risk: `0.25%` per trade
- ML: disabled in V1
- Live trading: hard-blocked

## Architecture

`MT5 -> Market Data -> Features -> CRT Signal -> Risk Engine -> Demo Execution -> Trade Log`

## Security

Never commit `.env`, MT5 passwords, API keys or account credentials. The credentials supplied during setup are **not stored in this repository**.

Because an MT5 password was shared in chat during setup, rotate that password before using the account for anything beyond testing.

## Local setup

1. Install Python 3.11+.
2. Install and log into the XM MT5 demo terminal on Windows.
3. Confirm Algo Trading is enabled in MT5.
4. Copy `.env.example` to `.env` and fill in your demo credentials locally.
5. Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

6. Run the connectivity test:

```powershell
python main.py --mode test
```

7. Run signal monitoring without execution:

```powershell
python main.py --mode signal
```

8. Run V1 demo execution:

```powershell
python main.py --mode demo
```

## Safety controls

V1 refuses live accounts, requires an explicit `DEMO` mode, validates the symbol, checks spread and trading permissions, applies a maximum daily loss guard, limits open positions, and calls MT5 order validation before execution.

## Roadmap

- V1.0: deterministic CRT + demo execution
- V1.1: historical data collection and backtesting
- V1.2: expanded feature dataset
- V2.0: ML probability model
- V3.0: walk-forward model selection and regime detection
- V4.0: VPS + GitHub deployment + monitoring
