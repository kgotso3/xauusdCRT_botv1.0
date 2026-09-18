from __future__ import annotations

import hmac
import os

from fastapi import FastAPI, HTTPException, Request, status

from schemas import TradingViewPayload
from storage import init_db, save_scan

app = FastAPI(title="CRT Scanner V1 Webhook", version="1.0.0")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "crt-scanner-v1-webhook"}


@app.post("/tradingview", status_code=status.HTTP_202_ACCEPTED)
async def tradingview_webhook(request: Request) -> dict[str, int | str]:
    expected = os.getenv("TV_WEBHOOK_SECRET", "")
    if not expected:
        raise HTTPException(status_code=503, detail="TV_WEBHOOK_SECRET is not configured")

    try:
        body = await request.json()
        payload = TradingViewPayload.model_validate(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid TradingView payload") from exc

    if not hmac.compare_digest(payload.secret, expected):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    scan_id = save_scan(payload.model_dump())
    return {"status": "accepted", "scan_id": scan_id}
