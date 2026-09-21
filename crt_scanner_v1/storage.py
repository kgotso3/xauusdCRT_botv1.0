from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData, String, Table, Text, and_, create_engine, delete, desc, select, update
from sqlalchemy.engine import Engine

from v2_policy import classify_model_eligibility

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SQLITE = f"sqlite:///{(BASE_DIR / 'crt_scanner.db').as_posix()}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_SQLITE)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine_kwargs: dict[str, Any] = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine: Engine = create_engine(DATABASE_URL, **engine_kwargs)
metadata = MetaData()

scans = Table(
    "crt_scans", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("received_at", DateTime(timezone=True), nullable=False),
    Column("scan_type", String(8), nullable=False),
    Column("time_ny", String(64), nullable=False),
    Column("version", String(32), nullable=False),
    Column("raw_json", Text, nullable=False),
)

scan_symbols = Table(
    "crt_scan_symbols", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("scan_id", Integer, ForeignKey("crt_scans.id", ondelete="CASCADE"), nullable=False),
    Column("label", String(24), nullable=False),
    Column("ticker", String(64), nullable=False),
    Column("bias", String(16), nullable=False),
    Column("trend_score", Integer, nullable=False),
    Column("sweep", String(16), nullable=False),
    Column("close_inside", Boolean, nullable=False),
    Column("valid_crt", Boolean, nullable=False),
    Column("direction", String(16), nullable=False),
    Column("aligned", Boolean, nullable=False),
    Column("c1_high", Float),
    Column("c1_low", Float),
    Column("c1_mid", Float),
    Column("c2_close", Float),
)

# Prospective research ledger. A row is created for every valid CRT in the
# frozen primary cohort (US30, US500, XAUUSD). V1 is the control; V2 requires
# the same CRT plus the pre-defined daily-bias alignment rule.
shadow_setups = Table(
    "crt_shadow_setups", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("scan_id", Integer, ForeignKey("crt_scans.id", ondelete="CASCADE"), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("resolved_at", DateTime(timezone=True)),
    Column("scan_type", String(8), nullable=False),
    Column("time_ny", String(64), nullable=False),
    Column("label", String(24), nullable=False),
    Column("ticker", String(64), nullable=False),
    Column("bias", String(16), nullable=False),
    Column("trend_score", Integer, nullable=False),
    Column("direction", String(16), nullable=False),
    Column("aligned", Boolean, nullable=False),
    Column("v1_eligible", Boolean, nullable=False),
    Column("v2_eligible", Boolean, nullable=False),
    Column("outcome", String(24), nullable=False, default="PENDING"),
    Column("model_r", Float),
    Column("entry_time_ny", String(64)),
    Column("entry_price", Float),
    Column("stop_price", Float),
    Column("tp2_price", Float),
    Column("notes", Text),
)

OUTCOME_DEFAULT_R: dict[str, float] = {
    "STOP": -1.0,
    "BE": 0.0,
    "TP2": 2.0,
}


def init_db() -> None:
    # create_all is intentionally additive here. Existing V1 tables are left
    # untouched while the new V2 shadow ledger is created if missing.
    metadata.create_all(engine)


def save_scan(payload: dict[str, Any]) -> int:
    init_db()
    clean_payload = dict(payload)
    clean_payload.pop("secret", None)
    now = datetime.now(timezone.utc)

    with engine.begin() as conn:
        result = conn.execute(scans.insert().values(
            received_at=now,
            scan_type=payload["scan_type"],
            time_ny=payload["time_ny"],
            version=payload["version"],
            raw_json=json.dumps(clean_payload, separators=(",", ":")),
        ))
        scan_id = int(result.inserted_primary_key[0])
        rows = [{"scan_id": scan_id, **item} for item in payload["symbols"]]
        conn.execute(scan_symbols.insert(), rows)

        shadow_rows: list[dict[str, Any]] = []
        for item in payload["symbols"]:
            policy = classify_model_eligibility(item)
            if not policy["v1_eligible"]:
                continue
            shadow_rows.append({
                "scan_id": scan_id,
                "created_at": now,
                "scan_type": payload["scan_type"],
                "time_ny": payload["time_ny"],
                "label": item["label"],
                "ticker": item["ticker"],
                "bias": item["bias"],
                "trend_score": item["trend_score"],
                "direction": item["direction"],
                "aligned": item["aligned"],
                "v1_eligible": True,
                "v2_eligible": policy["v2_eligible"],
                "outcome": "PENDING",
            })
        if shadow_rows:
            conn.execute(shadow_setups.insert(), shadow_rows)
    return scan_id


def latest_scan() -> dict[str, Any] | None:
    init_db()
    with engine.connect() as conn:
        parent = conn.execute(select(scans).order_by(desc(scans.c.id)).limit(1)).mappings().first()
        if not parent:
            return None
        children = conn.execute(select(scan_symbols).where(scan_symbols.c.scan_id == parent["id"]).order_by(scan_symbols.c.id)).mappings().all()

    return {
        "id": parent["id"], "received_at": parent["received_at"],
        "scan_type": parent["scan_type"], "time_ny": parent["time_ny"],
        "version": parent["version"], "symbols": [dict(row) for row in children],
    }


def recent_scans(limit: int = 20) -> list[dict[str, Any]]:
    init_db()
    limit = max(1, min(int(limit), 200))
    with engine.connect() as conn:
        rows = conn.execute(select(scans).order_by(desc(scans.c.id)).limit(limit)).mappings().all()
    return [dict(r) for r in rows]


def recent_shadow_setups(limit: int = 100) -> list[dict[str, Any]]:
    init_db()
    limit = max(1, min(int(limit), 1000))
    with engine.connect() as conn:
        rows = conn.execute(select(shadow_setups).order_by(desc(shadow_setups.c.id)).limit(limit)).mappings().all()
    return [dict(r) for r in rows]


def pending_shadow_setups(limit: int = 100) -> list[dict[str, Any]]:
    """Return pending primary-cohort setups with the CRT levels needed by the local MT5 shadow runner."""
    init_db()
    limit = max(1, min(int(limit), 500))
    stmt = (
        select(
            *shadow_setups.c,
            scan_symbols.c.sweep,
            scan_symbols.c.c1_high,
            scan_symbols.c.c1_low,
            scan_symbols.c.c1_mid,
            scan_symbols.c.c2_close,
        )
        .join(
            scan_symbols,
            and_(
                scan_symbols.c.scan_id == shadow_setups.c.scan_id,
                scan_symbols.c.label == shadow_setups.c.label,
            ),
        )
        .where(shadow_setups.c.outcome == "PENDING")
        .order_by(shadow_setups.c.id)
        .limit(limit)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [dict(r) for r in rows]


def update_shadow_outcome(
    setup_id: int,
    outcome: str,
    model_r: float | None = None,
    entry_time_ny: str | None = None,
    entry_price: float | None = None,
    stop_price: float | None = None,
    tp2_price: float | None = None,
    notes: str | None = None,
) -> bool:
    init_db()
    outcome = str(outcome).upper()
    if model_r is None and outcome in OUTCOME_DEFAULT_R:
        model_r = OUTCOME_DEFAULT_R[outcome]
    if outcome in {"NO_ENTRY", "AMBIGUOUS", "PENDING"}:
        model_r = None

    values: dict[str, Any] = {
        "outcome": outcome,
        "model_r": model_r,
        "entry_time_ny": entry_time_ny,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "tp2_price": tp2_price,
        "notes": notes,
        "resolved_at": None if outcome == "PENDING" else datetime.now(timezone.utc),
    }
    with engine.begin() as conn:
        result = conn.execute(update(shadow_setups).where(shadow_setups.c.id == int(setup_id)).values(**values))
        return bool(result.rowcount)


def clear_local_data() -> None:
    init_db()
    with engine.begin() as conn:
        conn.execute(delete(shadow_setups))
        conn.execute(delete(scan_symbols))
        conn.execute(delete(scans))
