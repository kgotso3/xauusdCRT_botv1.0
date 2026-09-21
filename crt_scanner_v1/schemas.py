from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

Bias = Literal["BULLISH", "BEARISH", "NEUTRAL"]
Sweep = Literal["HIGH", "LOW", "BOTH", "NONE"]
Direction = Literal["BULLISH", "BEARISH", "-"]
ShadowOutcome = Literal["PENDING", "NO_ENTRY", "STOP", "BE", "TP2", "TIMEOUT", "AMBIGUOUS"]


class SymbolScan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    ticker: str
    bias: Bias
    trend_score: int = Field(ge=-7, le=7)
    sweep: Sweep
    close_inside: bool
    valid_crt: bool
    direction: Direction
    aligned: bool
    c1_high: float | None = None
    c1_low: float | None = None
    c1_mid: float | None = None
    c2_close: float | None = None

    @field_validator("direction")
    @classmethod
    def direction_matches_validity(cls, value: str, info):
        valid = info.data.get("valid_crt")
        if valid is False and value != "-":
            raise ValueError("Invalid CRT rows must use direction '-' ")
        return value


class TradingViewPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    secret: str = Field(min_length=8, max_length=256)
    version: str
    scan_type: Literal["AM", "PM"]
    time_ny: str
    symbols: list[SymbolScan] = Field(min_length=1, max_length=10)


class ShadowOutcomePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    secret: str = Field(min_length=8, max_length=256)
    outcome: ShadowOutcome
    model_r: float | None = Field(default=None, ge=-10.0, le=10.0)
    entry_time_ny: str | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    tp2_price: float | None = None
    notes: str | None = Field(default=None, max_length=4000)
