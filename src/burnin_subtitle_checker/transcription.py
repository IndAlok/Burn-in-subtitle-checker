from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .languages import ASR_LANGUAGE_CHOICES, LANGUAGES, language_spec
from .media import probe_duration
from .models import AudioSegment, as_jsonable

log = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = set(ASR_LANGUAGE_CHOICES)
LANGUAGE_MODEL_DEFAULTS = {code: spec.default_model for code, spec in LANGUAGES.items()}
INITIAL_PROMPTS = {
    code: spec.initial_prompt
    for code, spec in LANGUAGES.items()
    if spec.initial_prompt
}

_HALLUCINATION_PHRASES = {
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "please like and subscribe",
    "subscribe to my channel",
    "subtitles by",
    "amara.org",
}


def _recommended_model(language: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    return LANGUAGE_MODEL_DEFAULTS.get(language, "small")


def _normalize_temperature(temperature: float) -> float:
    value = float(temperature)
    if not 0.0 <= value <= 1.0:
        raise ValueError("temperature must be between 0.0 and 1.0")
    return round(value, 3)


def seconds_to_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        raise ValueError("SRT timestamps cannot be negative")
    ms_total = int(round(seconds * 1000))
    ms = ms_total % 1000
    total_sec = ms_total // 1000
    sec = total_sec % 60
    total_min = total_sec // 60
    minutes = total_min % 60
    hours = total_min // 60
    return f"{hours:02}:{minutes:02}:{sec:02},{ms:03}"


def load_model(
    model_size: str,
    *,
    device: str = "auto",
    compute_type: str = "int8",
    cpu_threads: int | None = None,
) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper not installed. Run: pip install -r requirements.txt") from exc

    threads = cpu_threads or os.cpu_count() or 4
    log.info("Loading Whisper model '%s' (%s, %d threads)...", model_size, compute_type, threads)
    return WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=threads)


def _is_hallucination(text: str, *, language: str = "auto") -> bool:
    cleaned = text.strip()
    if not cleaned:
        return False

    lower = cleaned.lower()
    if any(phrase in lower for phrase in _HALLUCINATION_PHRASES):
        return True

    if len(cleaned) < 20:
        return False

    for window in range(2, 6):
        pattern = cleaned[:window]
        repeated = pattern * (len(cleaned) // len(pattern) + 1)
        if cleaned in repeated:
            return True

    words = cleaned.split()
    if len(words) > 5 and len(set(words)) <= max(2, int(len(words) * 0.15)):
        return True

    # Whisper sometimes loops half a sentence during silence. This catches exact loops
    # without penalizing normal long Indic sentences.
    if len(cleaned) > 60:
        half = len(cleaned) // 2
        if cleaned[:half] == cleaned[half:half + len(cleaned[:half])]:
            return True

    return False


def _build_segments(raw_segments: list[Any], *, language: str = "auto") -> list[AudioSegment]:
    segments: list[AudioSegment] = []
    for idx, raw in enumerate(raw_segments, start=1):
        start = round(float(raw.start), 3)
        end = round(float(raw.end), 3)
        text = str(raw.text).strip()
        confidence = None
        if hasattr(raw, "avg_logprob"):
            try:
                confidence = round(float(raw.avg_logprob), 4)
            except (TypeError, ValueError):
                pass

        if _is_hallucination(text, language=language):
            log.warning("Segment %d (%.1fs-%.1fs): hallucination detected, discarding", idx, start, end)
            text = ""

        if confidence is not None and confidence < -1.5 and len(text) < 10:
            log.debug("Segment %d: very low confidence (%.2f), discarding", idx, confidence)
            text = ""

        segments.append(
            AudioSegment(
                id=idx,
                start=start,
                end=end,
                midpoint=round((start + end) / 2, 3),
                text=text,
                confidence=confidence,
            )
        )
    return segments


def _filter_noise(segments: list[AudioSegment]) -> list[AudioSegment]:
    cleaned = []
    for seg in segments:
        duration = seg.end - seg.start
        if not seg.text and duration < 0.5:
            continue
        cleaned.append(seg)
    return [
        AudioSegment(
            id=i,
            start=s.start,
            end=s.end,
            midpoint=s.midpoint,
            text=s.text,
            confidence=s.confidence,
        )
        for i, s in enumerate(cleaned, start=1)
    ]


def transcribe_media(
    media_path: str | Path,
    *,
    language: str = "auto",
    model_size: str | None = None,
    beam_size: int = 2,
    device: str = "auto",
    compute_type: str = "int8",
    cpu_threads: int | None = None,
    vad_filter: bool = True,
    temperature: float = 0.0,
) -> dict[str, Any]:
    path = Path(media_path)
    if not path.exists():
        raise FileNotFoundError(f"Media not found: {path}")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{language}'. Use: {sorted(SUPPORTED_LANGUAGES)}")

    effective_model = _recommended_model(language, model_size)
    temperature = _normalize_temperature(temperature)
    model = load_model(effective_model, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
    prompt = INITIAL_PROMPTS.get(language)

    options: dict[str, Any] = {
        "beam_size": beam_size,
        "temperature": temperature,
        "vad_filter": vad_filter,
        "vad_parameters": {"min_silence_duration_ms": 300, "speech_pad_ms": 400},
        "condition_on_previous_text": False,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
    }
    if prompt:
        options["initial_prompt"] = prompt
    if language != "auto":
        options["language"] = language_spec(language).whisper_code

    log.info("Transcribing '%s' with model '%s'...", path.name, effective_model)
    raw_segments, info = model.transcribe(str(path), **options)

    segments = _filter_noise(_build_segments(list(raw_segments), language=language))
    empty = sum(1 for s in segments if not s.text)
    valid = len(segments) - empty
    log.info("Done: %d segments (%d valid, %d empty/discarded)", len(segments), valid, empty)

    duration = probe_duration(path)
    return {
        "metadata": {
            "input_file": str(path),
            "duration_seconds": duration or round(float(info.duration), 3),
            "model": effective_model,
            "language_requested": language,
            "language_detected": info.language,
            "language_probability": round(float(info.language_probability), 4),
            "segment_count": len(segments),
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "engine": "faster-whisper",
            "device": device,
            "compute_type": compute_type,
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "temperature": temperature,
        },
        "segments": as_jsonable(segments),
    }


_MIN_SRT_GAP = 0.1
_MIN_SRT_DURATION = 0.3


def _fix_srt_overlaps(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [s for s in segments if s.get("text", "").strip()]
    valid.sort(key=lambda s: float(s["start"]))
    fixed: list[dict[str, Any]] = []
    for seg in valid:
        start = float(seg["start"])
        end = float(seg["end"])
        if end - start < _MIN_SRT_DURATION:
            continue
        if fixed:
            prev_end = float(fixed[-1]["end"])
            if start < prev_end + _MIN_SRT_GAP:
                fixed[-1] = {**fixed[-1], "end": round(start - _MIN_SRT_GAP, 3)}
        fixed.append({**seg, "start": round(start, 3), "end": round(end, 3)})
    return fixed


def render_srt(segments: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, seg in enumerate(_fix_srt_overlaps(segments), start=1):
        start = seconds_to_srt_timestamp(float(seg["start"]))
        end = seconds_to_srt_timestamp(float(seg["end"]))
        blocks.append(f"{idx}\n{start} --> {end}\n{seg['text']}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_text(payload: dict[str, Any]) -> str:
    meta = payload["metadata"]
    lines = [
        f"# Input: {meta['input_file']}",
        f"# Language: {meta.get('language_detected') or meta['language_requested']}",
        f"# Model: {meta['model']}",
        "",
    ]
    for seg in payload["segments"]:
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {text}")
    return "\n".join(lines).rstrip() + "\n"


def write_transcription(payload: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": directory / "transcription.json",
        "srt": directory / "transcription.srt",
        "text": directory / "transcription.txt",
    }
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["srt"].write_text(render_srt(payload["segments"]), encoding="utf-8")
    paths["text"].write_text(render_text(payload), encoding="utf-8")
    return paths
