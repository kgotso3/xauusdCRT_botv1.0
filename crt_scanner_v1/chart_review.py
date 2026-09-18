from __future__ import annotations

import base64
import io
import os
from typing import Literal

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field


class NormalizedLine(BaseModel):
    present: bool
    direction: Literal["BULLISH", "BEARISH", "NONE"]
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)
    label: str = "MSS"


class NormalizedBox(BaseModel):
    kind: Literal["ORDER_BLOCK", "FVG"]
    direction: Literal["BULLISH", "BEARISH"]
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)
    label: str


class ChartAnalysis(BaseModel):
    timeframe_detected: str
    mss: NormalizedLine
    zones: list[NormalizedBox]
    confidence: float = Field(ge=0, le=1)
    notes: list[str]


PROMPT = """You are reviewing a TradingView candlestick screenshot for a CRT setup.
Analyze only visible price action. Do not invent off-screen candles or price levels.
Identify, if clearly visible:
1. Market Structure Shift (MSS): a break/close through a meaningful recent swing after the CRT sweep.
2. Order Block: the final opposing candle/range before the displacement causing the MSS.
3. Fair Value Gap (FVG): a visible three-candle imbalance created by displacement.
Return approximate normalized image coordinates from 0 to 1 for each line/box so an overlay can be drawn.
If evidence is weak, mark MSS present=false and omit uncertain zones. This is analysis for manual review, not a trade instruction.
"""


def analyze_screenshot(image_bytes: bytes, mime_type: str, context: str) -> ChartAnalysis:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    model = os.getenv("OPENAI_VISION_MODEL", "gpt-5.6-luna")
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"{PROMPT}\nSetup context: {context}"},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{encoded}",
                        "detail": "high",
                    },
                ],
            }
        ],
        text_format=ChartAnalysis,
    )
    if response.output_parsed is None:
        raise RuntimeError("No structured chart analysis was returned")
    return response.output_parsed


def annotate_image(image_bytes: bytes, analysis: ChartAnalysis) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    font = ImageFont.load_default()

    def pt(x: float, y: float) -> tuple[int, int]:
        return int(x * w), int(y * h)

    if analysis.mss.present:
        draw.line([pt(analysis.mss.x1, analysis.mss.y1), pt(analysis.mss.x2, analysis.mss.y2)], width=max(2, w // 500))
        x, y = pt(analysis.mss.x1, analysis.mss.y1)
        draw.text((x + 4, max(0, y - 14)), f"MSS {analysis.mss.direction}", font=font)

    for zone in analysis.zones:
        x1, y1 = pt(zone.x1, zone.y1)
        x2, y2 = pt(zone.x2, zone.y2)
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        draw.rectangle((left, top, right, bottom), width=max(2, w // 500))
        draw.text((left + 4, max(0, top - 14)), zone.label, font=font)

    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
