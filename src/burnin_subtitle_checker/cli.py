from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .media import (
    YtDlpAuth,
    is_url,
    iter_media_files,
    list_ytdlp_formats,
    resolve_media,
    safe_output_stem,
)
from .ocr import (
    OCR_EFFORT_CHOICES,
    check_subtitle_presence,
    check_tesseract_languages,
    extract_subtitles,
    read_segments_json,
    write_ocr_output,
)
from .reporting import read_json_payload, write_csv_report, write_html_report, write_json
from .scoring import compare_segments
from .transcription import _normalize_temperature, _recommended_model, transcribe_media, write_transcription
from .languages import ASR_LANGUAGE_CHOICES, OCR_LANGUAGE_CHOICES

log = logging.getLogger(__name__)

_COOKIES_HELP = (
    "Browser to load cookies from for YouTube bot detection "
    "(examples: chrome, firefox, edge, brave, firefox:ProfileName)"
)


def _add_ytdlp_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cookies-from-browser", default=None, help=_COOKIES_HELP)
    parser.add_argument("--cookies", default=None, help="Mozilla/Netscape cookies.txt file for yt-dlp")
    parser.add_argument("--user-agent", default=None, help="Browser user-agent string to pass to yt-dlp")
    parser.add_argument("--referer", default=None, help="Referer URL to pass to yt-dlp")
    parser.add_argument("--add-header", action="append", default=[], help="Extra yt-dlp HTTP header, e.g. 'Accept-Language: en-US,en;q=0.9'")
    parser.add_argument("--extractor-args", action="append", default=[], help="Raw yt-dlp extractor args, e.g. 'youtube:player_client=mweb'")
    parser.add_argument("--proxy", default=None, help="Proxy URL passed to yt-dlp")
    parser.add_argument("--source-address", default=None, help="Client-side IP address passed to yt-dlp")
    parser.add_argument("--sleep-requests", type=float, default=None, help="Seconds to sleep between yt-dlp requests")
    parser.add_argument("--retries", type=int, default=3, help="yt-dlp retry count")
    parser.add_argument("--ytdlp-format", default="bv*+ba/bestvideo+bestaudio/best/b", help="yt-dlp format selector")
    parser.add_argument("--format-sort", action="append", default=[], help="yt-dlp format sort rule, repeatable")
    parser.add_argument("--js-runtimes", action="append", default=[], help="yt-dlp JavaScript runtime, e.g. deno or node")
    parser.add_argument("--remote-components", action="append", default=[], help="yt-dlp remote component source, e.g. ejs:npm or ejs:github")
    parser.add_argument("--prefer-yt-dlp-cli", action="store_true", help="Use yt-dlp CLI before the Python API")
    parser.add_argument("--list-formats", action="store_true", help="Print yt-dlp formats for the URL and exit")


def _add_media_batch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recursive", action="store_true", help="When input is a directory, scan nested folders too")
    parser.add_argument("--include-all-files", action="store_true", help="Batch mode: try every file instead of filtering known media extensions")
    parser.add_argument("--continue-on-error", action="store_true", help="Batch mode: keep processing remaining files after one failure")


def _ytdlp_auth_from_args(args: argparse.Namespace) -> YtDlpAuth:
    return YtDlpAuth(
        cookies_from_browser=getattr(args, "cookies_from_browser", None),
        cookies=getattr(args, "cookies", None),
        user_agent=getattr(args, "user_agent", None),
        referer=getattr(args, "referer", None),
        add_headers=tuple(getattr(args, "add_header", []) or []),
        extractor_args=tuple(getattr(args, "extractor_args", []) or []),
        proxy=getattr(args, "proxy", None),
        source_address=getattr(args, "source_address", None),
        sleep_requests=getattr(args, "sleep_requests", None),
        retries=getattr(args, "retries", 3),
        format_selector=getattr(args, "ytdlp_format", "bv*+ba/bestvideo+bestaudio/best/b"),
        format_sort=tuple(getattr(args, "format_sort", []) or []),
        js_runtimes=tuple(getattr(args, "js_runtimes", []) or []),
        remote_components=tuple(getattr(args, "remote_components", []) or []),
        prefer_cli=bool(getattr(args, "prefer_yt_dlp_cli", False)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect mismatches between audio and burned-in subtitles.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed logging")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="Transcribe audio via Whisper ASR.")
    t.add_argument("media", help="Local path or URL")
    t.add_argument("--language", default="auto", choices=list(ASR_LANGUAGE_CHOICES))
    t.add_argument("--model", default=None, help="Whisper model size (default: medium for Indic languages, small for English/auto)")
    t.add_argument("--device", default="auto", help="faster-whisper device: auto, cpu, cuda")
    t.add_argument("--compute-type", default="int8", help="faster-whisper compute type, e.g. int8, int8_float16, float16")
    t.add_argument("--beam-size", type=int, default=2)
    t.add_argument("--temperature", type=float, default=0.0, help="Whisper decoding temperature. 0.0 is deterministic and recommended for QA.")
    t.add_argument("--cpu-threads", type=int, default=None)
    t.add_argument("--no-vad", action="store_true", help="Disable VAD filtering")
    t.add_argument("--output-dir", default="outputs/transcription")
    t.add_argument("--download-dir", default="downloads")
    _add_ytdlp_args(t)
    _add_media_batch_args(t)

    o = sub.add_parser("ocr", help="Extract burned-in subtitle text via OCR.")
    o.add_argument("video", help="Local video path or URL")
    o.add_argument("segments_json", help="Transcription JSON with segment timestamps")
    o.add_argument("--language", default="multi", choices=list(OCR_LANGUAGE_CHOICES))
    o.add_argument("--crop-ratio", type=float, default=0.15)
    o.add_argument("--window-seconds", type=float, default=0.35)
    o.add_argument("--samples-per-segment", type=int, default=3)
    o.add_argument("--ocr-effort", default="balanced", choices=list(OCR_EFFORT_CHOICES))
    o.add_argument("--fixed-region", action="store_true", help="Disable dynamic subtitle band detection")
    o.add_argument("--output", default="outputs/ocr/ocr_segments.json")
    o.add_argument("--download-dir", default="downloads")
    _add_ytdlp_args(o)
    _add_media_batch_args(o)

    c = sub.add_parser("compare", help="Compare ASR vs OCR and generate report.")
    c.add_argument("audio_json")
    c.add_argument("ocr_json")
    c.add_argument("--threshold", type=float, default=0.80)
    c.add_argument("--tolerance", type=float, default=1.5)
    c.add_argument("--context-window-seconds", type=float, default=2.0)
    c.add_argument("--video", default=None, help="Local video path or URL to link in the HTML report")
    c.add_argument("--asset-mode", default="link", choices=["link", "copy", "preview"], help="Link video, copy it, or create a browser-safe MP4 preview beside the report")
    c.add_argument("--output-json", default="outputs/report/results.json")
    c.add_argument("--output-html", default="outputs/report/report.html")
    c.add_argument("--output-csv", default="outputs/report/results.csv")
    c.add_argument("--continue-on-error", action="store_true", help="Batch mode: keep processing remaining JSON pairs after one failure")

    report = sub.add_parser("report", help="Generate an interactive HTML report from existing results JSON.")
    report.add_argument("results_json")
    report.add_argument("--video", default=None, help="Local video path or URL to enable timestamp review")
    report.add_argument("--asset-mode", default="link", choices=["link", "copy", "preview"], help="Link video, copy it, or create a browser-safe MP4 preview beside the report")
    report.add_argument("--output-html", default="outputs/report/report.html")
    report.add_argument("--output-csv", default=None)
    report.add_argument("--continue-on-error", action="store_true", help="Batch mode: keep processing remaining reports after one failure")

    r = sub.add_parser("run", help="Full pipeline: transcribe -> OCR -> compare -> report.")
    r.add_argument("media", help="Local path or URL")
    r.add_argument("--language", default="auto", choices=list(ASR_LANGUAGE_CHOICES))
    r.add_argument("--ocr-language", default=None, choices=list(OCR_LANGUAGE_CHOICES))
    r.add_argument("--model", default=None)
    r.add_argument("--device", default="auto")
    r.add_argument("--compute-type", default="int8")
    r.add_argument("--beam-size", type=int, default=2)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--cpu-threads", type=int, default=None)
    r.add_argument("--no-vad", action="store_true")
    r.add_argument("--threshold", type=float, default=0.80)
    r.add_argument("--crop-ratio", type=float, default=0.15)
    r.add_argument("--window-seconds", type=float, default=0.35)
    r.add_argument("--samples-per-segment", type=int, default=3)
    r.add_argument("--ocr-effort", default="balanced", choices=list(OCR_EFFORT_CHOICES))
    r.add_argument("--fixed-region", action="store_true")
    r.add_argument("--tolerance", type=float, default=1.5)
    r.add_argument("--context-window-seconds", type=float, default=2.0)
    r.add_argument("--output-dir", default="outputs")
    r.add_argument("--asset-mode", default="link", choices=["link", "copy", "preview"])
    r.add_argument("--resume", action="store_true", help="Reuse matching outputs already present in --output-dir")
    r.add_argument("--reuse-transcription", default=None, help="Use an existing transcription JSON instead of running ASR")
    r.add_argument("--reuse-ocr", default=None, help="Use an existing OCR JSON instead of running OCR")
    r.add_argument("--download-dir", default="downloads")
    _add_ytdlp_args(r)
    _add_media_batch_args(r)

    return parser


def _pick_ocr_language(asr_lang: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    return asr_lang if asr_lang in {"hi", "kn", "ta", "te", "en"} else "multi"


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def _read_json_file(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "segments" not in payload:
        raise ValueError(f"JSON payload must contain a 'segments' list: {path}")
    return payload


def _metadata_matches(payload: dict, expected: dict) -> bool:
    metadata = payload.get("metadata") or {}
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if isinstance(expected_value, float):
            try:
                if abs(float(actual) - expected_value) > 1e-6:
                    return False
            except (TypeError, ValueError):
                return False
        elif actual != expected_value:
            return False
    return True


def _load_reusable_payload(path: Path, expected_metadata: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = _read_json_file(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if _metadata_matches(payload, expected_metadata) else None


def _print_ytdlp_formats(url: str, ytdlp_auth: YtDlpAuth) -> int:
    if not is_url(url):
        raise ValueError("--list-formats is only valid for URLs")
    print(list_ytdlp_formats(url, ytdlp_auth))
    return 0


def _batch_artifact_root(output_value: str | Path) -> Path:
    path = Path(output_value)
    return path.parent if path.suffix else path


def _batch_json_output(output_value: str | Path, media_file: str | Path, filename: str) -> Path:
    return _batch_artifact_root(output_value) / safe_output_stem(media_file) / filename


def _discover_transcription_json(root: str | Path, media_file: str | Path) -> Path:
    base = Path(root)
    if base.is_file():
        return base
    stem = safe_output_stem(media_file)
    media_stem = Path(media_file).stem
    candidates = [
        base / stem / "transcription" / "transcription.json",
        base / stem / "transcription.json",
        base / media_stem / "transcription" / "transcription.json",
        base / media_stem / "transcription.json",
        base / f"{media_stem}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No transcription JSON found for {media_file} under {base}")


def _discover_ocr_json(root: str | Path, audio_json: str | Path) -> Path:
    base = Path(root)
    if base.is_file():
        return base
    audio_path = Path(audio_json)
    parent_stem = audio_path.parent.parent.name if audio_path.parent.name == "transcription" else audio_path.parent.name
    candidates = [
        base / parent_stem / "ocr" / "ocr_segments.json",
        base / parent_stem / "ocr_segments.json",
        base / parent_stem / "ocr.json",
        base / f"{parent_stem}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No OCR JSON found for {audio_json} under {base}")


def _json_batch_stem(path: str | Path) -> str:
    json_path = Path(path)
    if json_path.name == "transcription.json" and json_path.parent.name == "transcription":
        return safe_output_stem(json_path.parent.parent.name)
    if json_path.name in {"transcription.json", "ocr_segments.json", "results.json"}:
        return safe_output_stem(json_path.parent.name)
    return safe_output_stem(json_path.stem)


def _iter_audio_jsons(root: str | Path) -> list[Path]:
    base = Path(root)
    if base.is_file():
        return [base]
    if not base.exists():
        raise FileNotFoundError(f"Audio JSON path not found: {base}")
    files = list(base.rglob("transcription.json"))
    files.extend(
        candidate
        for candidate in base.glob("*.json")
        if candidate.name not in {"run_manifest.json", "ocr_segments.json", "results.json"}
    )
    return sorted(set(files), key=lambda item: str(item).casefold())


def _iter_result_jsons(root: str | Path) -> list[Path]:
    base = Path(root)
    if base.is_file():
        return [base]
    if not base.exists():
        raise FileNotFoundError(f"Results JSON path not found: {base}")
    files = list(base.rglob("results.json"))
    files.extend(candidate for candidate in base.glob("*.json") if candidate.name != "run_manifest.json")
    return sorted(set(files), key=lambda item: str(item).casefold())


def _discover_video_for_json(video: str | None, audio_json: str | Path) -> str | Path | None:
    if not video:
        return None
    if is_url(video):
        return video
    base = Path(video)
    if base.is_file():
        return base
    if not base.is_dir():
        return video
    stem = _json_batch_stem(audio_json)
    for candidate in iter_media_files(base, recursive=True):
        if safe_output_stem(candidate) == stem or safe_output_stem(candidate.stem) == stem:
            return candidate
    return None


def _run_compare_once(
    args: argparse.Namespace,
    audio_json: str | Path,
    ocr_json: str | Path,
    output_json: str | Path,
    output_html: str | Path,
    output_csv: str | Path,
    video: str | Path | None,
) -> None:
    audio = read_segments_json(audio_json)
    ocr = read_segments_json(ocr_json)
    payload = compare_segments(
        audio,
        ocr,
        threshold=args.threshold,
        tolerance=args.tolerance,
        context_window_seconds=args.context_window_seconds,
    )
    if video:
        payload["metadata"]["input_file"] = str(video)
    j = write_json(payload, output_json)
    h = write_html_report(payload, output_html, video=video, asset_mode=args.asset_mode)
    csv_path = write_csv_report(payload, output_csv)
    print(f"Flagged: {payload['metadata']['flagged_count']}/{payload['metadata']['segment_count']}")
    print(f"  JSON: {j}\n  CSV: {csv_path}\n  HTML: {h}")


def _run_report_once(
    args: argparse.Namespace,
    results_json: str | Path,
    output_html: str | Path,
    output_csv: str | Path | None,
    video: str | Path | None,
) -> None:
    payload = read_json_payload(results_json)
    if video:
        payload["metadata"]["input_file"] = str(video)
    h = write_html_report(payload, output_html, video=video, asset_mode=args.asset_mode)
    print(f"HTML: {h}")
    if output_csv:
        csv_path = write_csv_report(payload, output_csv)
        print(f"CSV: {csv_path}")


def _run_transcribe_once(args: argparse.Namespace, media_input: str | Path, output_dir: str | Path, ytdlp_auth: YtDlpAuth) -> None:
    media = resolve_media(str(media_input), args.download_dir, ytdlp_auth=ytdlp_auth)
    t0 = time.monotonic()
    payload = transcribe_media(
        media,
        language=args.language,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        cpu_threads=args.cpu_threads,
        vad_filter=not args.no_vad,
        temperature=args.temperature,
    )
    elapsed = time.monotonic() - t0
    paths = write_transcription(payload, output_dir)
    print(f"Segments: {payload['metadata']['segment_count']} ({_fmt_time(elapsed)})")
    for k, p in paths.items():
        print(f"  {k}: {p}")


def _run_ocr_once(
    args: argparse.Namespace,
    video_input: str | Path,
    segments_json: str | Path,
    output_path: str | Path,
    ytdlp_auth: YtDlpAuth,
) -> None:
    video = resolve_media(str(video_input), args.download_dir, ytdlp_auth=ytdlp_auth)
    segments = read_segments_json(segments_json)
    missing = check_tesseract_languages(args.language)
    if missing:
        raise RuntimeError(f"Missing Tesseract packs: {', '.join(missing)}")
    t0 = time.monotonic()
    payload = extract_subtitles(
        video,
        segments,
        language=args.language,
        crop_ratio=args.crop_ratio,
        window_seconds=args.window_seconds,
        samples_per_segment=args.samples_per_segment,
        ocr_effort=args.ocr_effort,
        dynamic_region=not args.fixed_region,
    )
    elapsed = time.monotonic() - t0
    out = write_ocr_output(payload, output_path)
    print(f"OCR: {payload['metadata']['segment_count']} segments ({_fmt_time(elapsed)})")
    print(f"  Output: {out}")


def _run_full_pipeline(
    args: argparse.Namespace,
    media_input: str | Path,
    output_dir: str | Path,
    ytdlp_auth: YtDlpAuth,
) -> None:
    pipeline_start = time.monotonic()
    media = resolve_media(str(media_input), args.download_dir, ytdlp_auth=ytdlp_auth)
    out = Path(output_dir)
    temperature = _normalize_temperature(args.temperature)
    transcription_dir = out / "transcription"
    transcription_json = transcription_dir / "transcription.json"
    ocr_json = out / "ocr" / "ocr_segments.json"
    report_json = out / "report" / "results.json"
    report_csv = out / "report" / "results.csv"
    report_html = out / "report" / "report.html"

    print("Step 1/3: Transcribing audio...")
    t_elapsed = 0.0
    t_reused = False
    effective_model = _recommended_model(args.language, args.model)
    transcription_expected = {
        "input_file": str(media),
        "model": effective_model,
        "language_requested": args.language,
        "engine": "faster-whisper",
        "device": args.device,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "vad_filter": not args.no_vad,
        "temperature": temperature,
    }
    if args.reuse_transcription:
        audio_payload = _read_json_file(args.reuse_transcription)
        t_reused = True
        print(f"  Reused transcription: {args.reuse_transcription}")
    else:
        audio_payload = _load_reusable_payload(transcription_json, transcription_expected) if args.resume else None
        if audio_payload is not None:
            t_reused = True
            print(f"  Reused matching transcription: {transcription_json}")
        else:
            t0 = time.monotonic()
            audio_payload = transcribe_media(
                media,
                language=args.language,
                model_size=args.model,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
                cpu_threads=args.cpu_threads,
                vad_filter=not args.no_vad,
                temperature=temperature,
            )
            t_elapsed = time.monotonic() - t0

    t_paths = write_transcription(audio_payload, transcription_dir)
    seg_count = audio_payload.get("metadata", {}).get("segment_count", len(audio_payload.get("segments", [])))
    model = audio_payload.get("metadata", {}).get("model", "reused")
    print(f"  {seg_count} segments via {model} ({'reused' if t_reused else _fmt_time(t_elapsed)})")

    ocr_lang = _pick_ocr_language(args.language, args.ocr_language)
    missing = check_tesseract_languages(ocr_lang)
    if missing:
        raise RuntimeError(f"Missing Tesseract packs: {', '.join(missing)}")

    if not check_subtitle_presence(media, language=ocr_lang, crop_ratio=args.crop_ratio):
        log.warning("No burned-in subtitles detected; OCR may return empty results")
        print("  WARNING: No subtitles detected in sampled frames")

    print("Step 2/3: Extracting subtitles via OCR...")
    o_elapsed = 0.0
    o_reused = False
    ocr_resume_expected = {
        "video_file": str(media),
        "language": ocr_lang,
        "crop_ratio": args.crop_ratio,
        "window_seconds": args.window_seconds,
        "samples_per_segment": args.samples_per_segment,
        "dynamic_region": not args.fixed_region,
        "ocr_effort": args.ocr_effort,
    }
    if args.reuse_ocr:
        ocr_payload = _read_json_file(args.reuse_ocr)
        o_reused = True
        print(f"  Reused OCR: {args.reuse_ocr}")
    else:
        ocr_payload = _load_reusable_payload(ocr_json, ocr_resume_expected) if args.resume else None
        if ocr_payload is not None:
            o_reused = True
            print(f"  Reused matching OCR: {ocr_json}")
        else:
            t0 = time.monotonic()
            ocr_payload = extract_subtitles(
                media,
                audio_payload["segments"],
                language=ocr_lang,
                crop_ratio=args.crop_ratio,
                window_seconds=args.window_seconds,
                samples_per_segment=args.samples_per_segment,
                ocr_effort=args.ocr_effort,
                dynamic_region=not args.fixed_region,
            )
            o_elapsed = time.monotonic() - t0
    ocr_path = write_ocr_output(ocr_payload, ocr_json)
    ocr_count = ocr_payload.get("metadata", {}).get("segment_count", len(ocr_payload.get("segments", [])))
    print(f"  {ocr_count} segments ({'reused' if o_reused else _fmt_time(o_elapsed)})")

    print("Step 3/3: Comparing and generating report...")
    report = compare_segments(
        audio_payload["segments"],
        ocr_payload["segments"],
        threshold=args.threshold,
        tolerance=args.tolerance,
        context_window_seconds=args.context_window_seconds,
    )

    report["metadata"]["input_source"] = str(media_input)
    report["metadata"]["input_file"] = str(media)
    report["metadata"]["duration_seconds"] = audio_payload["metadata"].get("duration_seconds")
    report["metadata"]["model"] = model

    j = write_json(report, report_json)
    csv_path = write_csv_report(report, report_csv)
    h = write_html_report(report, report_html, video=media, asset_mode=args.asset_mode)

    manifest = {
        "metadata": {
            "input_source": str(media_input),
            "resolved_media": str(media),
            "language": args.language,
            "ocr_language": ocr_lang,
            "model": model,
            "temperature": temperature,
            "transcription_reused": t_reused,
            "ocr_reused": o_reused,
            "total_seconds": round(time.monotonic() - pipeline_start, 3),
        },
        "artifacts": {
            "transcription_json": str(t_paths["json"]),
            "transcription_srt": str(t_paths["srt"]),
            "transcription_txt": str(t_paths["text"]),
            "ocr_json": str(ocr_path),
            "results_json": str(j),
            "results_csv": str(csv_path),
            "report_html": str(h),
        },
    }
    manifest_path = write_json(manifest, out / "run_manifest.json")

    total_elapsed = time.monotonic() - pipeline_start
    print(f"\nTranscription: {t_paths['json']}")
    print(f"OCR: {ocr_path}")
    print(f"CSV: {csv_path}")
    print(f"Report: {h}")
    print(f"Manifest: {manifest_path}")
    print(f"Flagged: {report['metadata']['flagged_count']}/{report['metadata']['segment_count']}")
    print(f"Total time: {_fmt_time(total_elapsed)}")


def _run_batch(items: list[Path], args: argparse.Namespace, runner) -> int:
    if not items:
        raise RuntimeError("No matching media files found for batch processing")
    failures: list[dict[str, str]] = []
    for index, item in enumerate(items, start=1):
        print(f"\nBatch {index}/{len(items)}: {item}")
        try:
            runner(item)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            message = str(exc)
            failures.append({"input": str(item), "error": message})
            print(f"ERROR: {message}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    ytdlp_auth = _ytdlp_auth_from_args(args)
    try:
        return _dispatch(args, ytdlp_auth)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace, ytdlp_auth: YtDlpAuth) -> int:
    if args.command == "transcribe":
        if args.list_formats:
            return _print_ytdlp_formats(args.media, ytdlp_auth)
        if not is_url(args.media) and Path(args.media).is_dir():
            files = iter_media_files(args.media, recursive=args.recursive, include_all_files=args.include_all_files)
            return _run_batch(
                files,
                args,
                lambda item: _run_transcribe_once(
                    args,
                    item,
                    Path(args.output_dir) / safe_output_stem(item),
                    ytdlp_auth,
                ),
            )
        _run_transcribe_once(args, args.media, args.output_dir, ytdlp_auth)
        return 0

    if args.command == "ocr":
        if args.list_formats:
            return _print_ytdlp_formats(args.video, ytdlp_auth)
        if not is_url(args.video) and Path(args.video).is_dir():
            files = iter_media_files(args.video, recursive=args.recursive, include_all_files=args.include_all_files)
            return _run_batch(
                files,
                args,
                lambda item: _run_ocr_once(
                    args,
                    item,
                    _discover_transcription_json(args.segments_json, item),
                    _batch_json_output(args.output, item, "ocr_segments.json"),
                    ytdlp_auth,
                ),
            )
        _run_ocr_once(args, args.video, args.segments_json, args.output, ytdlp_auth)
        return 0

    if args.command == "compare":
        if Path(args.audio_json).is_dir() or Path(args.ocr_json).is_dir():
            audio_files = _iter_audio_jsons(args.audio_json)
            return _run_batch(
                audio_files,
                args,
                lambda audio_json: _run_compare_once(
                    args,
                    audio_json,
                    _discover_ocr_json(args.ocr_json, audio_json),
                    _batch_artifact_root(args.output_json) / _json_batch_stem(audio_json) / "results.json",
                    _batch_artifact_root(args.output_html) / _json_batch_stem(audio_json) / "report.html",
                    _batch_artifact_root(args.output_csv) / _json_batch_stem(audio_json) / "results.csv",
                    _discover_video_for_json(args.video, audio_json),
                ),
            )
        _run_compare_once(
            args,
            args.audio_json,
            args.ocr_json,
            args.output_json,
            args.output_html,
            args.output_csv,
            args.video,
        )
        return 0

    if args.command == "report":
        if Path(args.results_json).is_dir():
            result_files = _iter_result_jsons(args.results_json)
            return _run_batch(
                result_files,
                args,
                lambda result_json: _run_report_once(
                    args,
                    result_json,
                    _batch_artifact_root(args.output_html) / _json_batch_stem(result_json) / "report.html",
                    (
                        _batch_artifact_root(args.output_csv) / _json_batch_stem(result_json) / "results.csv"
                        if args.output_csv else None
                    ),
                    _discover_video_for_json(args.video, result_json),
                ),
            )
        _run_report_once(args, args.results_json, args.output_html, args.output_csv, args.video)
        return 0

    if args.command == "run":
        if args.list_formats:
            return _print_ytdlp_formats(args.media, ytdlp_auth)
        if not is_url(args.media) and Path(args.media).is_dir():
            files = iter_media_files(args.media, recursive=args.recursive, include_all_files=args.include_all_files)
            return _run_batch(
                files,
                args,
                lambda item: _run_full_pipeline(
                    args,
                    item,
                    Path(args.output_dir) / safe_output_stem(item),
                    ytdlp_auth,
                ),
            )
        _run_full_pipeline(args, args.media, args.output_dir, ytdlp_auth)
        return 0

    raise RuntimeError(f"Unknown command: {args.command}")
