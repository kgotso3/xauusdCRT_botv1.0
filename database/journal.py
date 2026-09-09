from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB = Path("data") / "trade_journal.sqlite3"


def _connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id TEXT NOT NULL UNIQUE,
            created_at_utc TEXT NOT NULL,
            symbol TEXT NOT NULL,
            h1_bar_time TEXT NOT NULL,
            direction TEXT NOT NULL,
            score REAL NOT NULL,
            approved INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            execution_status TEXT NOT NULL DEFAULT 'NOT_SENT',
            order_ticket TEXT,
            deal_ticket TEXT,
            error_text TEXT
        )
        """
    )
    conn.commit()
    return conn


def make_signal_id(symbol: str, h1_bar_time: Any, direction: str) -> str:
    if hasattr(h1_bar_time, "isoformat"):
        bar_key = h1_bar_time.isoformat()
    else:
        bar_key = str(h1_bar_time)
    return f"{symbol}|H1|{bar_key}|{direction.upper()}"


def signal_seen(signal_id: str, db_path: Path = DEFAULT_DB) -> bool:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT 1 FROM signals WHERE signal_id = ? LIMIT 1", (signal_id,)).fetchone()
    return row is not None


def record_signal(
    signal_id: str,
    symbol: str,
    h1_bar_time: Any,
    signal: dict,
    db_path: Path = DEFAULT_DB,
) -> bool:
    """Insert a signal once. Returns False when signal_id already exists."""
    payload = json.dumps(signal, default=str, sort_keys=True)
    created = datetime.now(timezone.utc).isoformat()
    h1_time = h1_bar_time.isoformat() if hasattr(h1_bar_time, "isoformat") else str(h1_bar_time)

    try:
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO signals (
                    signal_id, created_at_utc, symbol, h1_bar_time,
                    direction, score, approved, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal_id,
                    created,
                    symbol,
                    h1_time,
                    str(signal.get("direction", "NONE")),
                    float(signal.get("score", 0.0)),
                    int(bool(signal.get("approved", False))),
                    payload,
                ),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def mark_execution(
    signal_id: str,
    status: str,
    order_ticket: Any = None,
    deal_ticket: Any = None,
    error_text: str | None = None,
    db_path: Path = DEFAULT_DB,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE signals
            SET execution_status = ?, order_ticket = ?, deal_ticket = ?, error_text = ?
            WHERE signal_id = ?
            """,
            (
                status,
                None if order_ticket is None else str(order_ticket),
                None if deal_ticket is None else str(deal_ticket),
                error_text,
                signal_id,
            ),
        )
