from __future__ import annotations

import math

import pandas as pd


def calculate_metrics(trades: pd.DataFrame) -> dict:
    """Calculate core strategy metrics from a trade ledger.

    The ledger is expected to contain ``r_multiple`` and ``equity_r`` columns.
    Metrics are expressed in R where appropriate so they remain comparable
    across account sizes and broker currencies.
    """
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "net_r": 0.0,
            "average_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "expectancy_r": 0.0,
        }

    r = trades["r_multiple"].astype(float)
    wins = r[r > 0]
    losses = r[r < 0]
    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else math.inf

    equity = r.cumsum()
    peaks = equity.cummax().clip(lower=0)
    drawdown = equity - peaks

    return {
        "trades": int(len(trades)),
        "wins": int((r > 0).sum()),
        "losses": int((r < 0).sum()),
        "win_rate": float((r > 0).mean()),
        "net_r": float(r.sum()),
        "average_r": float(r.mean()),
        "profit_factor": float(profit_factor),
        "max_drawdown_r": abs(float(drawdown.min())),
        "expectancy_r": float(r.mean()),
    }
