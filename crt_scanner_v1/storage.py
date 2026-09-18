from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData, String, Table, Text, create_engine, delete, desc, select
from sqlalchemy.engine import Engine

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


def init_db() -> None:
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


def clear_local_data() -> None:
    init_db()
    with engine.begin() as conn:
        conn.execute(delete(scan_symbols))
        conn.execute(delete(scans))
