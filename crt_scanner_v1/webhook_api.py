from __future__ import annotations

import hmac
import os

from fastapi import FastAPI, HTTPException, Request, status

from schemas import ShadowOutcomePayload, TradingViewPayload
from storage import init_db, save_scan, update_shadow_outcome

app = FastAPI(title="CRT Scanner V2 Webhook", version="2.0.0")


def require_secret(value: str) -> None:
    expected = os.getenv("TV_WEBHOOK_SECRET", "")
    if not expected:
        raise HTTPException(status_code=503, detail="TV_WEBHOOK_SECRET is not configured")
    if not hmac.compare_digest(value, expected):
        raise HTTPException(status_code=401, detail="Invalid webhook secret")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "crt-scanner-v2-shadow-webhook"}


@app.post("/tradingview", status_code=status.HTTP_202_ACCEPTED)
async def tradingview_webhook(request: Request) -> dict[str, int | str]:
    try:
        body = await request.json()
        payload = TradingViewPayload.model_validate(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid TradingView payload") from exc

    require_secret(payload.secret)
    scan_id = save_scan(payload.model_dump())
    return {"status": "accepted", "scan_id": scan_id}


@app.post("/shadow/{setup_id}/outcome")
def resolve_shadow_setup(setup_id: int, payload: ShadowOutcomePayload) -> dict[str, int | str]:
    require_secret(payload.secret)
    values = payload.model_dump(exclude={"secret"})
    updated = update_shadow_outcome(setup_id=setup_id, **values)
    if not updated:
        raise HTTPException(status_code=404, detail="Shadow setup not found")
    return {"status": "updated", "setup_id": setup_id}
