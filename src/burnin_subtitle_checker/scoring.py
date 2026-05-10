from __future__ import annotations

import re
import unicodedata
from typing import Any

from .models import MatchResult, as_jsonable

ZERO_WIDTH = "\u200b\u200c\u200d\ufeff"
PUNCT = ".,!?;:\"'()[]{}<>|/\\`~@#$%^&*_+=-\u2014\u2013\u2026\u0964\u0965"

_DEVANAGARI_EQUIV = str.maketrans({
    "\u0901": "\u0902",  # chandrabindu -> anusvara
    "\u093c": None,      # nukta is often dropped by OCR/ASR
    "\u0949": "\u094b",  # short-o matra confusion in OCR
})

_KANNADA_EQUIV = str.maketrans({"\u0CBC": None})
_TELUGU_EQUIV = str.maketrans({"\u0C3C": None})


def normalize_text(text: str | None) -> str:
    val = unicodedata.normalize("NFC", text or "")
    val = val.translate({ord(ch): None for ch in ZERO_WIDTH})
    val = val.replace("\n", " ").replace("\r", " ").casefold()
    val = val.translate(_DEVANAGARI_EQUIV)
    val = val.translate(_KANNADA_EQUIV)
    val = val.translate(_TELUGU_EQUIV)
    val = val.translate({ord(ch): " " for ch in PUNCT})
    return re.sub(r"\s+", " ", val).strip()


def _base_similarity_breakdown(a: str, b: str) -> dict[str, float]:
    if not a and not b:
        return {
            "char": 1.0,
            "token_sort": 1.0,
            "partial": 1.0,
            "token_set": 1.0,
            "length_coverage": 1.0,
            "score": 1.0,
        }
    if not a or not b:
        return {
            "char": 0.0,
            "token_sort": 0.0,
            "partial": 0.0,
            "token_set": 0.0,
            "length_coverage": 0.0,
            "score": 0.0,
        }

    length_coverage = min(len(a), len(b)) / max(len(a), len(b))
    try:
        from rapidfuzz import fuzz

        token = fuzz.token_sort_ratio(a, b) / 100.0
        char = fuzz.ratio(a, b) / 100.0
        partial = fuzz.partial_ratio(a, b) / 100.0
        token_set = fuzz.token_set_ratio(a, b) / 100.0
    except ImportError:
        from difflib import SequenceMatcher

        char = SequenceMatcher(None, a, b).ratio()
        token = SequenceMatcher(None, " ".join(sorted(a.split())), " ".join(sorted(b.split()))).ratio()
        partial = char
        token_set = token

    weighted = (
        0.30 * token
        + 0.22 * char
        + 0.22 * partial
        + 0.16 * token_set
        + 0.10 * length_coverage
    )
    containment_rescue = partial * (0.88 if length_coverage >= 0.65 else 0.72)
    relaxed_indic_rescue = 0.0
    if partial >= 0.80 and token_set >= 0.60 and length_coverage >= 0.50:
        relaxed_indic_rescue = min(0.84, partial * 0.96)
    score = max(weighted, containment_rescue, relaxed_indic_rescue)
    return {
        "char": round(char, 4),
        "token_sort": round(token, 4),
        "partial": round(partial, 4),
        "token_set": round(token_set, 4),
        "length_coverage": round(length_coverage, 4),
        "score": round(min(score, 1.0), 4),
    }


def _best_window_similarity(a: str, b: str) -> tuple[dict[str, float] | None, str]:
    a_tokens = a.split()
    b_tokens = b.split()
    if not a_tokens or len(b_tokens) <= len(a_tokens) + 1:
        return None, ""

    target = len(a_tokens)
    min_size = max(1, int(target * 0.55))
    max_size = min(len(b_tokens), max(target + 4, int(target * 1.45)))
    best: dict[str, float] | None = None
    best_text = ""
    for size in range(min_size, max_size + 1):
        for start in range(0, len(b_tokens) - size + 1):
            text = " ".join(b_tokens[start:start + size])
            metrics = _base_similarity_breakdown(a, text)
            if best is None or metrics["score"] > best["score"]:
                best = metrics
                best_text = text
    return best, best_text


def _similarity_breakdown(a: str, b: str) -> dict[str, float]:
    metrics = _base_similarity_breakdown(a, b)
    window_metrics, window_text = _best_window_similarity(a, b)
    if window_metrics and window_metrics["score"] > metrics["score"]:
        metrics = dict(metrics)
        metrics["window"] = round(window_metrics["score"], 4)
        metrics["matched_window_text"] = window_text
        metrics["score"] = round(min(1.0, window_metrics["score"]), 4)
    return metrics


def _similarity(a: str, b: str) -> float:
    return _similarity_breakdown(a, b)["score"]


def _timestamp(seg: dict[str, Any]) -> float:
    for key in ("timestamp", "midpoint", "mid"):
        if key in seg:
            return float(seg[key])
    if "start" in seg and "end" in seg:
        return (float(seg["start"]) + float(seg["end"])) / 2
    raise ValueError(f"Segment {seg.get('id', '?')} has no timestamp")


def align_segments(
    audio: list[dict[str, Any]],
    subtitle: list[dict[str, Any]],
    *,
    tolerance: float = 1.5,
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
    used: set[int] = set()
    sub_by_id = {int(s["id"]): i for i, s in enumerate(subtitle) if s.get("id") is not None}
    pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []

    for a in audio:
        matched: int | None = None
        aid = a.get("id")
        if aid is not None and int(aid) in sub_by_id:
            candidate = sub_by_id[int(aid)]
            if candidate not in used:
                matched = candidate
        if matched is None:
            a_ts = _timestamp(a)
            best_dist = tolerance + 0.001
            for i, s in enumerate(subtitle):
                if i in used:
                    continue
                dist = abs(a_ts - _timestamp(s))
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    matched = i
        if matched is None:
            pairs.append((a, None))
        else:
            used.add(matched)
            pairs.append((a, subtitle[matched]))

    for i, s in enumerate(subtitle):
        if i not in used:
            pairs.append((None, s))
    return sorted(pairs, key=lambda p: _timestamp(p[0] or p[1]))


def _subtitle_text(segment: dict[str, Any] | None) -> str:
    if not segment:
        return ""
    return str(segment.get("subtitle_text") or segment.get("ocr_text") or segment.get("text") or "")


def _audio_text(segment: dict[str, Any] | None) -> str:
    if not segment:
        return ""
    return str(segment.get("text") or segment.get("audio_text") or "")


def _segment_bounds(segment: dict[str, Any] | None, timestamp: float) -> tuple[float | None, float | None]:
    if not segment:
        return None, None
    start = float(segment["start"]) if "start" in segment else None
    end = float(segment["end"]) if "end" in segment else None
    if start is None and end is None:
        return None, None
    if start is None:
        start = timestamp
    if end is None:
        end = timestamp
    return round(start, 3), round(end, 3)


def _nearby_subtitles(
    subtitle_segments: list[dict[str, Any]],
    *,
    center_id: int | None,
    center_timestamp: float,
    start: float | None = None,
    end: float | None = None,
    window_ids: int = 2,
    window_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    segment_start = min(start, end) if start is not None and end is not None else None
    segment_end = max(start, end) if start is not None and end is not None else None
    for sub in subtitle_segments:
        text = _subtitle_text(sub).strip()
        if not text:
            continue
        timestamp = _timestamp(sub)
        distance = abs(timestamp - center_timestamp)
        id_distance = abs(int(sub.get("id", -10_000)) - center_id) if center_id is not None and sub.get("id") is not None else None
        in_segment = (
            segment_start is not None
            and segment_end is not None
            and segment_start - window_seconds <= timestamp <= segment_end + window_seconds
        )
        if in_segment or distance <= window_seconds or (id_distance is not None and id_distance <= window_ids):
            candidates.append({
                "timestamp": timestamp,
                "text": text,
                "source": "nearby",
            })
    candidates.sort(key=lambda item: item["timestamp"])
    merged: list[dict[str, Any]] = []
    segment_span = (segment_end - segment_start) if segment_start is not None and segment_end is not None else 0.0
    max_join = 4 if segment_span > 6 else 2
    max_span = max(window_seconds * 2, segment_span + window_seconds)
    for i in range(len(candidates)):
        for size in range(2, max_join + 1):
            group = candidates[i:i + size]
            if len(group) < size:
                continue
            if group[-1]["timestamp"] - group[0]["timestamp"] > max_span:
                continue
            merged.append({
                "timestamp": group[0]["timestamp"],
                "text": " ".join(item["text"] for item in group),
                "source": f"nearby_{size}x",
            })
    return candidates + merged


def compare_segments(
    audio_segments: list[dict[str, Any]],
    subtitle_segments: list[dict[str, Any]],
    *,
    threshold: float = 0.80,
    tolerance: float = 1.5,
    context_window_seconds: float = 2.0,
) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    results: list[MatchResult] = []
    previous_subtitle = ""
    for idx, (audio, subtitle) in enumerate(
        align_segments(audio_segments, subtitle_segments, tolerance=tolerance),
        start=1,
    ):
        a_text = _audio_text(audio)
        center_sub_text = _subtitle_text(subtitle)
        ts = _timestamp(audio or subtitle)
        audio_norm = normalize_text(a_text)

        start, end = _segment_bounds(audio, ts)

        aligned_metrics = _similarity_breakdown(audio_norm, normalize_text(center_sub_text))
        best_score = aligned_metrics["score"]
        best_metrics = dict(aligned_metrics)
        best_metrics["match_source"] = "aligned"
        best_sub_text = center_sub_text
        if audio and audio_norm:
            aid = int(audio["id"]) if audio.get("id") is not None else None
            for candidate in _nearby_subtitles(
                subtitle_segments,
                center_id=aid,
                center_timestamp=ts,
                start=start,
                end=end,
                window_seconds=context_window_seconds,
            ):
                metrics = _similarity_breakdown(audio_norm, normalize_text(candidate["text"]))
                score = metrics["score"]
                if score > best_score:
                    best_score = score
                    best_sub_text = candidate["text"]
                    best_metrics = dict(metrics)
                    best_metrics["match_source"] = candidate["source"]

        repeated_subtitle = best_sub_text.strip() and best_sub_text.strip() == previous_subtitle and normalize_text(a_text) != normalize_text(best_sub_text)
        if a_text.strip() and not best_sub_text.strip():
            reason = "missing_subtitle"
        elif best_sub_text.strip() and not a_text.strip():
            reason = "missing_speech"
        elif repeated_subtitle and best_score < threshold:
            reason = "duplicate"
        elif best_score < threshold:
            reason = "low_similarity"
        else:
            reason = "ok"

        previous_subtitle = best_sub_text.strip() or previous_subtitle
        results.append(
            MatchResult(
                id=idx,
                timestamp=round(ts, 3),
                audio_text=a_text,
                subtitle_text=best_sub_text,
                score=best_score,
                status="OK" if best_score >= threshold else "REVIEW",
                reason=reason,
                start=start,
                end=end,
                metrics=best_metrics,
            )
        )

    flagged = sum(1 for r in results if r.status == "REVIEW")
    scores = [r.score for r in results]
    return {
        "metadata": {
            "threshold": threshold,
            "segment_count": len(results),
            "flagged_count": flagged,
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "missing_subtitle_count": sum(1 for r in results if r.reason == "missing_subtitle"),
            "missing_speech_count": sum(1 for r in results if r.reason == "missing_speech"),
            "duplicate_count": sum(1 for r in results if r.reason == "duplicate"),
            "alignment_tolerance_seconds": tolerance,
            "context_window_seconds": context_window_seconds,
        },
        "segments": as_jsonable(results),
    }
