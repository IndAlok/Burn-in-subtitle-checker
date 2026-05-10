from __future__ import annotations

import json
import logging
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .languages import OCR_LANGUAGE_CHOICES, script_char_ratio, tesseract_language
from .models import OcrSegment, as_jsonable

log = logging.getLogger(__name__)

LANGUAGE_TO_TESSERACT = {
    code: tesseract_language(code)
    for code in OCR_LANGUAGE_CHOICES
}
WORD_CONFIDENCE_FLOOR = 20.0
OCR_EFFORT_CHOICES = ("fast", "balanced", "accurate")
_EFFORT_VARIANTS = {
    "fast": {"clahe_gray", "otsu", "inv_otsu"},
    "balanced": {"clahe_gray", "otsu", "inv_otsu", "adaptive", "joined", "inv_joined"},
    "accurate": {"clahe_gray", "otsu", "inv_otsu", "adaptive", "joined", "inv_joined"},
}
_EFFORT_PSMS = {
    "fast": (6,),
    "balanced": (6, 7),
    "accurate": (6, 7, 13),
}


def _dependency_error(import_name: str, package_name: str | None = None) -> RuntimeError:
    package = package_name or import_name
    return RuntimeError(
        f"Missing Python dependency '{import_name}'. Install project dependencies with: "
        f"python -m pip install -e .  (or: python -m pip install {package})"
    )


def _import_pytesseract() -> Any:
    try:
        import pytesseract
    except ImportError as exc:
        raise _dependency_error("pytesseract") from exc
    return pytesseract


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise _dependency_error("cv2", "opencv-python") from exc
    return cv2


def tesseract_lang(language: str) -> str:
    try:
        return LANGUAGE_TO_TESSERACT[language]
    except KeyError as exc:
        raise ValueError(f"Unsupported OCR language '{language}'. Use: {sorted(LANGUAGE_TO_TESSERACT)}") from exc


def check_tesseract_languages(language: str) -> list[str]:
    pytesseract = _import_pytesseract()

    requested = tesseract_lang(language).split("+")
    try:
        installed = set(pytesseract.get_languages(config=""))
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR executable not found. Install Tesseract and ensure the "
            "'tesseract' command is available on PATH, then install language packs "
            "for eng/hin/kan/tam/tel as needed."
        ) from exc
    return [code for code in requested if code not in installed]


def read_segments_json(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("segments", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Segment JSON must contain a list or {'segments': [...]}")
    return rows


def crop_subtitle_region(frame: Any, *, crop_ratio: float = 0.15) -> Any:
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise ValueError("frame must be a valid image array")
    if not 0 < crop_ratio <= 1:
        raise ValueError("crop_ratio must be in (0, 1]")
    h = frame.shape[0]
    y_start = int(round(h * (1.0 - crop_ratio)))
    return frame[y_start:h, :]


def _fixed_subtitle_bounds(frame: Any, *, crop_ratio: float = 0.15) -> tuple[int, int, int, int]:
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise ValueError("frame must be a valid image array")
    if not 0 < crop_ratio <= 1:
        raise ValueError("crop_ratio must be in (0, 1]")
    h, w = frame.shape[:2]
    y_start = int(round(h * (1.0 - crop_ratio)))
    return 0, y_start, w, h - y_start


def _region_dict(bounds: tuple[int, int, int, int]) -> dict[str, int]:
    x, y, width, height = bounds
    return {"x": x, "y": y, "width": width, "height": height}


def _crop_by_bounds(frame: Any, bounds: tuple[int, int, int, int]) -> Any:
    x, y, width, height = bounds
    return frame[y:y + height, x:x + width]


def _dynamic_subtitle_bounds(frame: Any, *, crop_ratio: float = 0.15, search_ratio: float = 0.45) -> tuple[int, int, int, int]:
    import numpy as np
    cv2 = _import_cv2()

    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise ValueError("frame must be a valid image array")
    h, w = frame.shape[:2]
    if h < 20 or w < 20:
        return _fixed_subtitle_bounds(frame, crop_ratio=crop_ratio)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    lower_start = int(h * (1.0 - search_ratio))
    lower = gray[lower_start:h, :]
    edges = cv2.Canny(lower, 80, 180)
    row_density = edges.mean(axis=1)
    threshold = max(float(row_density.mean() + row_density.std()), 8.0)
    rows = np.where(row_density > threshold)[0]
    if len(rows) < 3:
        return _fixed_subtitle_bounds(frame, crop_ratio=crop_ratio)

    y_min = max(0, lower_start + int(rows.min()) - 30)
    y_max = min(h, lower_start + int(rows.max()) + 35)
    min_height = int(h * crop_ratio)
    if y_max - y_min < min_height:
        center = (y_min + y_max) // 2
        y_min = max(0, center - min_height // 2)
        y_max = min(h, y_min + min_height)
    return 0, y_min, w, y_max - y_min


def _dynamic_subtitle_crop(frame: Any, *, crop_ratio: float = 0.15, search_ratio: float = 0.45) -> Any:
    bounds = _dynamic_subtitle_bounds(frame, crop_ratio=crop_ratio, search_ratio=search_ratio)
    return _crop_by_bounds(frame, bounds)


def _preprocess_variants(crop: Any) -> list[tuple[str, Any]]:
    cv2 = _import_cv2()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    h, w = gray.shape[:2]
    scale = max(1, min(3, 900 // max(h, 1)))
    if scale > 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (0, 0), 1.6)
    sharpened = cv2.addWeighted(clahe, 1.45, blurred, -0.45, 0)
    denoised = cv2.bilateralFilter(sharpened, 7, 50, 50)

    otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inv = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    block = max(11, (gray.shape[0] // 8) | 1)
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        4,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    joined = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    inv_joined = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel)
    return [
        ("clahe_gray", clahe),
        ("otsu", otsu),
        ("inv_otsu", inv),
        ("adaptive", adaptive),
        ("joined", joined),
        ("inv_joined", inv_joined),
    ]


def _is_garbage(text: str, *, language: str = "multi") -> bool:
    cleaned = text.strip()
    if len(cleaned) < 2:
        return True
    alnum_count = sum(1 for c in cleaned if c.isalnum())
    if len(cleaned) > 0 and (alnum_count / len(cleaned)) < 0.4:
        return True
    if script_char_ratio(cleaned, language) < 0.25:
        return True
    return False


def _ocr_image(image: Any, *, language: str, psm: int = 6) -> tuple[str, float | None]:
    pytesseract = _import_pytesseract()

    config = f"--oem 3 --psm {psm} -l {tesseract_lang(language)}"
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    words: list[str] = []
    confs: list[float] = []
    for text, conf in zip(data.get("text", []), data.get("conf", []), strict=False):
        cleaned = str(text).strip()
        if not cleaned:
            continue
        try:
            c = float(conf)
        except (TypeError, ValueError):
            c = -1
        if 0 <= c < WORD_CONFIDENCE_FLOOR:
            continue
        if c >= WORD_CONFIDENCE_FLOOR:
            confs.append(c)
        words.append(cleaned)
    text = " ".join(words)
    avg_conf = round(statistics.mean(confs) / 100.0, 4) if confs else None
    return text, avg_conf


def _ocr_crop(crop: Any, *, language: str, effort: str = "balanced") -> tuple[str, float | None, str | None]:
    if effort not in OCR_EFFORT_CHOICES:
        raise ValueError(f"Unsupported OCR effort '{effort}'. Use: {OCR_EFFORT_CHOICES}")

    best_text = ""
    best_conf: float | None = None
    best_variant: str | None = None
    best_score = -1.0

    for variant_name, image in _preprocess_variants(crop):
        if variant_name not in _EFFORT_VARIANTS[effort]:
            continue
        for psm in _EFFORT_PSMS[effort]:
            try:
                text, conf = _ocr_image(image, language=language, psm=psm)
            except RuntimeError:
                raise
            except Exception:
                continue
            text = " ".join(text.split())
            if not text or _is_garbage(text, language=language):
                continue
            script_score = script_char_ratio(text, language)
            score = (conf if conf is not None else 0.01) + min(len(text), 80) / 400 + script_score * 0.15
            if score > best_score:
                best_text = text
                best_conf = conf
                best_variant = f"{variant_name}/psm{psm}"
                best_score = score
            if conf is not None and conf > 0.72 and script_score > 0.70:
                return best_text, best_conf, best_variant
    return best_text, best_conf, best_variant


def _segment_midpoint(segment: dict[str, Any]) -> float:
    for key in ("timestamp", "midpoint", "mid"):
        if key in segment:
            return float(segment[key])
    if "start" in segment and "end" in segment:
        return (float(segment["start"]) + float(segment["end"])) / 2
    raise ValueError(f"Segment {segment.get('id', '?')} has no timestamp")


def _candidate_timestamps(mid: float, *, window_seconds: float, samples_per_segment: int) -> list[float]:
    if window_seconds <= 0 or samples_per_segment <= 1:
        return [round(max(0.0, mid), 3)]
    step = (2 * window_seconds) / (samples_per_segment - 1)
    return [round(max(0.0, mid - window_seconds + step * i), 3) for i in range(samples_per_segment)]


def _candidate_timestamps_for_segment(
    segment: dict[str, Any],
    *,
    window_seconds: float,
    samples_per_segment: int,
) -> list[float]:
    mid = _segment_midpoint(segment)
    if samples_per_segment <= 1 or "start" not in segment or "end" not in segment:
        return _candidate_timestamps(mid, window_seconds=window_seconds, samples_per_segment=samples_per_segment)

    start = max(0.0, float(segment["start"]))
    end = max(start, float(segment["end"]))
    duration = end - start
    if duration <= max(0.8, window_seconds * 2):
        return _candidate_timestamps(mid, window_seconds=window_seconds, samples_per_segment=samples_per_segment)

    margin = min(max(duration * 0.15, 0.2), 0.9)
    usable_start = min(end, start + margin)
    usable_end = max(usable_start, end - margin)
    if samples_per_segment == 2:
        candidates = [usable_start, usable_end]
    else:
        step = (usable_end - usable_start) / (samples_per_segment - 1)
        candidates = [usable_start + step * i for i in range(samples_per_segment)]
        if all(abs(mid - t) > 0.05 for t in candidates):
            candidates[len(candidates) // 2] = mid

    deduped: list[float] = []
    for timestamp in sorted(round(max(0.0, t), 3) for t in candidates):
        if not deduped or abs(timestamp - deduped[-1]) > 0.05:
            deduped.append(timestamp)
    return deduped


def _sharpness(frame: Any) -> float:
    cv2 = _import_cv2()

    if frame is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _transcript_similarity(audio_text: str, subtitle_text: str) -> float:
    if not audio_text.strip() or not subtitle_text.strip():
        return 0.0
    from .scoring import _similarity_breakdown, normalize_text

    return _similarity_breakdown(normalize_text(audio_text), normalize_text(subtitle_text))["score"]


def _process_item(
    item: tuple[int, float, list[tuple[float, Any | None]]] | tuple[int, float, str, list[tuple[float, Any | None]]],
    *,
    language: str,
    crop_ratio: float,
    dynamic_region: bool,
    ocr_effort: str = "balanced",
) -> OcrSegment:
    if len(item) == 3:
        seg_id, mid, candidates = item
        audio_text = ""
    else:
        seg_id, mid, audio_text, candidates = item
    best_text = ""
    best_conf: float | None = None
    best_variant: str | None = None
    best_region: dict[str, int] | None = None
    best_score = -1.0
    best_similarity = 0.0
    best_timestamp = mid
    saw_frame = False
    accepted_texts: list[str] = []

    for timestamp, frame in candidates:
        if frame is None:
            continue
        saw_frame = True
        base_bounds = _dynamic_subtitle_bounds(frame, crop_ratio=crop_ratio) if dynamic_region else _fixed_subtitle_bounds(frame, crop_ratio=crop_ratio)
        base_crop = _crop_by_bounds(frame, base_bounds)
        text, conf, variant = _ocr_crop(base_crop, language=language, effort=ocr_effort)
        region = _region_dict(base_bounds)
        if not text.strip() and crop_ratio < 0.28:
            wider_ratio = min(crop_ratio + 0.12, 0.35)
            wider_bounds = _dynamic_subtitle_bounds(frame, crop_ratio=wider_ratio) if dynamic_region else _fixed_subtitle_bounds(frame, crop_ratio=wider_ratio)
            wider = _crop_by_bounds(frame, wider_bounds)
            text2, conf2, variant2 = _ocr_crop(wider, language=language, effort=ocr_effort)
            if text2.strip():
                text, conf, variant = text2, conf2, variant2
                region = _region_dict(wider_bounds)

        if text.strip():
            accepted_texts.append(text.strip())

        transcript_score = _transcript_similarity(audio_text, text)
        quality = (conf if conf is not None else (0.02 if text else 0.0))
        quality += min(_sharpness(frame) / 10000.0, 0.08)
        quality += script_char_ratio(text, language) * 0.10
        quality += transcript_score * 0.70
        if quality > best_score:
            best_text = text
            best_conf = conf
            best_variant = variant
            best_region = region
            best_score = quality
            best_similarity = transcript_score
            best_timestamp = timestamp

    if not saw_frame:
        return OcrSegment(id=seg_id, timestamp=round(mid, 3), subtitle_text="", confidence=None, status="no_frame")

    merged_text = _merge_unique_texts(accepted_texts)
    merged_similarity = _transcript_similarity(audio_text, merged_text)
    should_merge = False
    if merged_text and audio_text.strip():
        should_merge = merged_similarity > best_similarity + 0.04
    elif merged_text and not best_text.strip():
        should_merge = True
    elif merged_text and len(merged_text) <= max(len(best_text.strip()) + 24, int(len(best_text.strip()) * 1.4)):
        should_merge = True

    if should_merge:
        best_text = merged_text

    return OcrSegment(
        id=seg_id,
        timestamp=round(best_timestamp, 3),
        subtitle_text=best_text.strip(),
        confidence=best_conf,
        status="merged" if should_merge and len(accepted_texts) > 1 else ("ok" if best_text.strip() else "empty"),
        crop_region=best_region,
        ocr_variant=best_variant,
    )


def _merge_unique_texts(texts: list[str]) -> str:
    unique: list[str] = []
    normalized: list[str] = []
    for text in texts:
        collapsed = " ".join(text.split())
        if not collapsed:
            continue
        key = collapsed.casefold()
        if any(key == prev or key in prev or prev in key for prev in normalized):
            continue
        unique.append(collapsed)
        normalized.append(key)
    return " ".join(unique)


def check_subtitle_presence(
    video_path: str | Path,
    *,
    language: str = "multi",
    sample_count: int = 8,
    crop_ratio: float = 0.15,
) -> bool:
    cv2 = _import_cv2()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return True

    total_ms = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1) * 1000
    if total_ms <= 0:
        cap.release()
        return True

    step = total_ms / (sample_count + 1)
    detected = 0
    try:
        for i in range(1, sample_count + 1):
            cap.set(cv2.CAP_PROP_POS_MSEC, step * i)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            crop = crop_subtitle_region(frame, crop_ratio=max(crop_ratio, 0.22))
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            edges = cv2.Canny(gray, 80, 180)
            edge_density = float((edges > 0).mean())
            contrast = float(gray.std())
            if edge_density > 0.012 and contrast > 18:
                detected += 1
                if detected >= 2:
                    return True
    finally:
        cap.release()
    return detected > 0


def extract_subtitles(
    video_path: str | Path,
    audio_segments: list[dict[str, Any]],
    *,
    language: str = "multi",
    crop_ratio: float = 0.15,
    window_seconds: float = 0.35,
    samples_per_segment: int = 3,
    dynamic_region: bool = True,
    ocr_effort: str = "balanced",
) -> dict[str, Any]:
    cv2 = _import_cv2()

    tesseract_lang(language)
    if ocr_effort not in OCR_EFFORT_CHOICES:
        raise ValueError(f"Unsupported OCR effort '{ocr_effort}'. Use: {OCR_EFFORT_CHOICES}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = len(audio_segments)
    frame_data: list[tuple[int, float, str, list[tuple[float, Any | None]]]] = []
    try:
        for idx, segment in enumerate(audio_segments, start=1):
            seg_id = int(segment.get("id", idx))
            mid = _segment_midpoint(segment)
            audio_text = str(segment.get("text") or segment.get("audio_text") or "")
            candidates: list[tuple[float, Any | None]] = []
            for timestamp in _candidate_timestamps_for_segment(segment, window_seconds=window_seconds, samples_per_segment=samples_per_segment):
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = cap.read()
                candidates.append((timestamp, frame if ok else None))
            frame_data.append((seg_id, mid, audio_text, candidates))
    finally:
        cap.release()

    log.info("Extracted candidate frames for %d segments, running OCR...", total)
    workers = min(4, os.cpu_count() or 2)

    def _do_ocr(item: tuple[int, float, str, list[tuple[float, Any | None]]]) -> OcrSegment:
        return _process_item(
            item,
            language=language,
            crop_ratio=crop_ratio,
            dynamic_region=dynamic_region,
            ocr_effort=ocr_effort,
        )

    results: list[OcrSegment] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(_do_ocr, frame_data), start=1):
            results.append(result)
            if i % 25 == 0 or i == total:
                log.info("OCR progress: %d/%d", i, total)

    return {
        "metadata": {
            "video_file": str(video_path),
            "language": language,
            "tesseract_language": tesseract_lang(language),
            "segment_count": len(results),
            "crop_ratio": crop_ratio,
            "window_seconds": window_seconds,
            "samples_per_segment": samples_per_segment,
            "dynamic_region": dynamic_region,
            "ocr_effort": ocr_effort,
        },
        "segments": as_jsonable(results),
    }


def write_ocr_output(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
