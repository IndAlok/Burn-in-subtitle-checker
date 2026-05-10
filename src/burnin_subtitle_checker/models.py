from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AudioSegment:
    id: int
    start: float
    end: float
    midpoint: float
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class OcrSegment:
    id: int
    timestamp: float
    subtitle_text: str
    confidence: float | None
    status: str
    crop_region: dict[str, int] | None = None
    ocr_variant: str | None = None


@dataclass(frozen=True)
class MatchResult:
    id: int
    timestamp: float
    audio_text: str
    subtitle_text: str
    score: float
    status: str
    reason: str
    start: float | None = None
    end: float | None = None
    metrics: dict[str, Any] | None = None


def as_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {k: as_jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {k: as_jsonable(v) for k, v in value.items()}
    return value
