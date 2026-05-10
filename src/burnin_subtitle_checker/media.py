from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_YTDLP_DOMAINS = {"youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com"}
MEDIA_EXTENSIONS = {
    ".3gp", ".aac", ".aiff", ".alac", ".asf", ".avi", ".f4v", ".flac", ".flv",
    ".m2ts", ".m4a", ".m4v", ".mkv", ".mov", ".mp3", ".mp4", ".mpeg", ".mpg",
    ".mxf", ".oga", ".ogg", ".ogv", ".opus", ".rm", ".rmvb", ".ts", ".vob",
    ".wav", ".webm", ".wma", ".wmv",
}


class _YtDlpLogger:
    def debug(self, message: str) -> None:
        log.debug("yt-dlp: %s", message)

    def warning(self, message: str) -> None:
        log.debug("yt-dlp warning: %s", message)

    def error(self, message: str) -> None:
        log.debug("yt-dlp error: %s", message)


@dataclass(frozen=True)
class YtDlpAuth:
    cookies_from_browser: str | None = None
    cookies: str | None = None
    user_agent: str | None = None
    referer: str | None = None
    add_headers: tuple[str, ...] = ()
    extractor_args: tuple[str, ...] = ()
    proxy: str | None = None
    source_address: str | None = None
    sleep_requests: float | None = None
    retries: int = 3
    format_selector: str = "bv*+ba/bestvideo+bestaudio/best/b"
    format_sort: tuple[str, ...] = ()
    js_runtimes: tuple[str, ...] = ()
    remote_components: tuple[str, ...] = ()
    prefer_cli: bool = False

    def has_auth(self) -> bool:
        return bool(self.cookies_from_browser or self.cookies)


def is_url(value: str) -> bool:
    return urllib.parse.urlparse(value).scheme in {"http", "https"}


def _needs_ytdlp(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    return host in _YTDLP_DOMAINS


def _validate_cookies_file(path: str | Path) -> Path:
    cookie_path = Path(path).expanduser()
    if not cookie_path.exists():
        raise FileNotFoundError(f"Cookie file not found: {cookie_path}")
    try:
        first_line = cookie_path.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
    except IndexError as exc:
        raise ValueError(f"Cookie file is empty: {cookie_path}") from exc
    valid_headers = {"# HTTP Cookie File", "# Netscape HTTP Cookie File"}
    if first_line not in valid_headers:
        raise ValueError(
            "Cookie file must be Mozilla/Netscape cookies.txt format and start with "
            "'# HTTP Cookie File' or '# Netscape HTTP Cookie File'"
        )
    return cookie_path


def _build_ytdlp_cli_args(auth: YtDlpAuth) -> list[str]:
    args: list[str] = []
    if auth.cookies_from_browser:
        args.extend(["--cookies-from-browser", auth.cookies_from_browser])
    if auth.cookies:
        args.extend(["--cookies", str(_validate_cookies_file(auth.cookies))])
    if auth.user_agent:
        args.extend(["--user-agent", auth.user_agent])
    if auth.referer:
        args.extend(["--referer", auth.referer])
    for header in auth.add_headers:
        args.extend(["--add-header", header])
    for extractor_arg in auth.extractor_args:
        args.extend(["--extractor-args", extractor_arg])
    if auth.proxy:
        args.extend(["--proxy", auth.proxy])
    if auth.source_address:
        args.extend(["--source-address", auth.source_address])
    if auth.sleep_requests is not None:
        args.extend(["--sleep-requests", str(auth.sleep_requests)])
    if auth.retries is not None:
        args.extend(["--retries", str(auth.retries)])
    for selector in auth.format_sort:
        args.extend(["--format-sort", selector])
    for runtime in auth.js_runtimes:
        args.extend(["--js-runtimes", runtime])
    for component in auth.remote_components:
        args.extend(["--remote-components", component])
    return args


def _build_ytdlp_python_opts(auth: YtDlpAuth) -> dict[str, Any]:
    opts: dict[str, Any] = {"format": auth.format_selector}
    if auth.cookies_from_browser:
        opts["cookiesfrombrowser"] = tuple(auth.cookies_from_browser.split(":"))
    if auth.cookies:
        opts["cookiefile"] = str(_validate_cookies_file(auth.cookies))
    if auth.user_agent:
        opts["user_agent"] = auth.user_agent
    if auth.referer:
        opts["referer"] = auth.referer
    if auth.add_headers:
        headers: dict[str, str] = {}
        for item in auth.add_headers:
            name, sep, value = item.partition(":")
            if sep:
                headers[name.strip()] = value.strip()
        if headers:
            opts["http_headers"] = headers
    if auth.extractor_args:
        parsed: dict[str, dict[str, list[str]]] = {}
        for item in auth.extractor_args:
            extractor, sep, raw_args = item.partition(":")
            if not sep or not extractor or not raw_args:
                continue
            values: dict[str, list[str]] = {}
            for pair in raw_args.split(";"):
                key, pair_sep, value = pair.partition("=")
                if pair_sep and key:
                    values[key] = [part for part in value.split(",") if part]
            if values:
                parsed[extractor] = values
        if parsed:
            opts["extractor_args"] = parsed
    if auth.proxy:
        opts["proxy"] = auth.proxy
    if auth.source_address:
        opts["source_address"] = auth.source_address
    if auth.sleep_requests is not None:
        opts["sleep_interval_requests"] = auth.sleep_requests
    if auth.retries is not None:
        opts["retries"] = auth.retries
    if auth.format_sort:
        opts["format_sort"] = list(auth.format_sort)
    return opts


def _needs_ytdlp_cli(auth: YtDlpAuth) -> bool:
    return auth.prefer_cli or bool(auth.js_runtimes or auth.remote_components)


def _append_error(errors: list[str] | None, message: str) -> None:
    if errors is not None and message.strip():
        errors.append(message.strip())


def _ytdlp_cli_download(url: str, output_dir: Path, auth: YtDlpAuth, errors: list[str] | None = None) -> Path | None:
    ytdlp_bin = shutil.which("yt-dlp")
    if not ytdlp_bin:
        _append_error(errors, "yt-dlp CLI executable not found on PATH")
        return None
    template = str(output_dir / "%(id)s.%(ext)s")
    cmd = [
        ytdlp_bin, "--no-playlist", "--format", auth.format_selector,
        "--merge-output-format", "mp4", "--print", "after_move:filepath",
        "-o", template,
    ]
    cmd.extend(_build_ytdlp_cli_args(auth))
    cmd.append(url)
    result = _run(cmd)
    if result.returncode == 0:
        lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
        if lines:
            path = Path(lines[-1])
            if path.exists():
                return path
    if result.returncode != 0 and result.stderr:
        message = result.stderr.strip()
        _append_error(errors, message)
        log.debug("yt-dlp CLI failed: %s", message[:500])
    return None


def list_ytdlp_formats(url: str, auth: YtDlpAuth | None = None) -> str:
    auth = auth or YtDlpAuth()
    ytdlp_bin = shutil.which("yt-dlp")
    if not ytdlp_bin:
        raise RuntimeError("yt-dlp CLI executable not found on PATH")
    cmd = [ytdlp_bin, "--no-playlist", "--list-formats"]
    cmd.extend(_build_ytdlp_cli_args(auth))
    cmd.append(url)
    result = _run(cmd)
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp could not list formats.\n{describe_ytdlp_failure(url, auth, [output])}")
    return output


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass
    raise RuntimeError(
        "ffmpeg not found. Install via: pip install imageio-ffmpeg\n"
        "Or get the system package from https://ffmpeg.org/download.html"
    )


def find_ffprobe() -> str | None:
    path = shutil.which("ffprobe")
    if path:
        return path
    try:
        ffmpeg = find_ffmpeg()
        candidate = Path(ffmpeg).parent / ("ffprobe.exe" if Path(ffmpeg).suffix == ".exe" else "ffprobe")
        return str(candidate) if candidate.exists() else None
    except RuntimeError:
        return None


def _run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False, **kwargs)


def _safe_filename(name: str, max_len: int = 80) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    safe = re.sub(r'[^\x20-\x7E]', '', safe)
    safe = re.sub(r'_+', '_', safe).strip('_. ')
    return safe[:max_len] if safe else "media"


def safe_output_stem(value: str | Path) -> str:
    path = Path(str(value))
    stem = path.stem if path.stem else str(value)
    if is_url(str(value)):
        return stable_name(str(value))
    return _safe_filename(stem)


def iter_media_files(path: str | Path, *, recursive: bool = False, include_all_files: bool = False) -> list[Path]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {root}")
    if root.is_file():
        return [root]
    pattern = "**/*" if recursive else "*"
    files = [
        candidate
        for candidate in root.glob(pattern)
        if candidate.is_file()
        and (include_all_files or candidate.suffix.lower() in MEDIA_EXTENSIONS)
    ]
    return sorted(files, key=lambda item: str(item).casefold())


def stable_name(value: str, suffix: str = "") -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    stem = Path(urllib.parse.urlparse(value).path).stem or "media"
    return f"{_safe_filename(stem)}_{digest}{suffix}"


def _ytdlp_download(
    url: str,
    output_dir: Path,
    *,
    cookies_from_browser: str | None = None,
    auth: YtDlpAuth | None = None,
    errors: list[str] | None = None,
) -> Path | None:
    auth = auth or YtDlpAuth(cookies_from_browser=cookies_from_browser)
    if _needs_ytdlp_cli(auth):
        cli_result = _ytdlp_cli_download(url, output_dir, auth, errors)
        if cli_result:
            return cli_result

    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        return _ytdlp_cli_download(url, output_dir, auth, errors)

    ffmpeg_dir = None
    try:
        ffmpeg_dir = str(Path(find_ffmpeg()).parent)
    except RuntimeError:
        pass

    opts: dict[str, Any] = {
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "logger": _YtDlpLogger(),
        "retries": auth.retries,
    }
    opts.update(_build_ytdlp_python_opts(auth))
    if ffmpeg_dir:
        opts["ffmpeg_location"] = ffmpeg_dir
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info:
                filepath = ydl.prepare_filename(info)
                for ext in [".mp4", ".mkv", ".webm", ""]:
                    candidate = Path(filepath).with_suffix(ext) if ext else Path(filepath)
                    if candidate.exists():
                        return candidate
    except Exception as exc:
        message = str(exc)
        _append_error(errors, message)
        log.debug("yt-dlp download failed: %s", message)
    return _ytdlp_cli_download(url, output_dir, auth, errors)


def describe_ytdlp_failure(url: str, auth: YtDlpAuth, errors: list[str] | None = None) -> str:
    details = "\n".join(errors or [])
    lower = details.casefold()
    lines = [
        "yt-dlp failed to access a playable media stream for this URL.",
        "First checks:",
        '  1. Update yt-dlp and its default EJS solver package: python -m pip install -U "yt-dlp[default]"',
        "  2. Install Deno 2+ or Node 20+ if the error mentions the YouTube n/EJS challenge.",
        "  3. Use fresh cookies from the same browser/session/network that can play the video.",
        "  4. Run again with --list-formats to confirm yt-dlp can see video/audio formats before ASR/OCR.",
    ]
    if not auth.has_auth():
        lines.extend([
            "Cookie options:",
            "  --cookies-from-browser chrome",
            "  --cookies-from-browser firefox:PROFILE",
            "  --cookies cookies.txt",
        ])
    if "n challenge" in lower or "ejs" in lower or "javascript runtime" in lower:
        lines.extend([
            "n/EJS challenge options:",
            "  --js-runtimes deno",
            "  --remote-components ejs:npm",
            "  --remote-components ejs:github",
        ])
    if "requested format is not available" in lower or "only images are available" in lower:
        lines.extend([
            "Format diagnostics:",
            f"  --list-formats --ytdlp-format \"{auth.format_selector}\"",
            "  If that list only contains image/storyboard rows, YouTube did not expose a playable stream to yt-dlp.",
        ])
    if "po token" in lower or "pot" in lower:
        lines.append("Some YouTube clients/formats may require PO-token extractor args supplied through --extractor-args.")
    if details:
        lines.extend(["Last yt-dlp message:", details[-1200:]])
    lines.extend([
        "References:",
        "  https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp",
        "  https://github.com/yt-dlp/yt-dlp/wiki/EJS",
        "  https://github.com/yt-dlp/yt-dlp/wiki/Extractors#youtube",
    ])
    return "\n".join(lines)


def download_url(
    url: str,
    download_dir: str | Path,
    *,
    cookies_from_browser: str | None = None,
    ytdlp_auth: YtDlpAuth | None = None,
) -> Path:
    directory = Path(download_dir)
    directory.mkdir(parents=True, exist_ok=True)
    auth = ytdlp_auth or YtDlpAuth(cookies_from_browser=cookies_from_browser)
    errors: list[str] = []
    result = _ytdlp_download(url, directory, auth=auth, errors=errors)
    if result:
        return result
    if _needs_ytdlp(url):
        raise RuntimeError(describe_ytdlp_failure(url, auth, errors))
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".mp4"
    path = directory / stable_name(url, suffix=suffix)
    log.info("Downloading %s via urllib...", url)
    urllib.request.urlretrieve(url, path)
    return path


def resolve_media(
    input_value: str,
    download_dir: str | Path = "downloads",
    *,
    cookies_from_browser: str | None = None,
    ytdlp_auth: YtDlpAuth | None = None,
) -> Path:
    if is_url(input_value):
        return download_url(
            input_value,
            download_dir,
            cookies_from_browser=cookies_from_browser,
            ytdlp_auth=ytdlp_auth,
        )
    path = Path(input_value)
    if not path.exists():
        raise FileNotFoundError(f"Input media not found: {path}")
    return path


def extract_audio(media_path: str | Path, output_path: str | Path | None = None) -> Path:
    ffmpeg = find_ffmpeg()
    source = Path(media_path)
    if not source.exists():
        raise FileNotFoundError(f"Media file not found: {source}")

    target = Path(output_path) if output_path else source.with_suffix(".wav")
    target.parent.mkdir(parents=True, exist_ok=True)

    result = _run([
        ffmpeg, "-y", "-i", str(source),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(target),
    ])
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr.strip()[:500]}")
    if not target.exists() or target.stat().st_size < 1000:
        raise RuntimeError(f"Audio extraction produced empty file: {target}")
    return target


def load_audio_array(wav_path: str | Path) -> Any:
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise RuntimeError(f"Expected 16-bit mono WAV, got {wf.getnchannels()}ch {wf.getsampwidth()*8}bit")
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) < 16000:
        raise RuntimeError(f"Audio too short ({len(audio)} samples, need at least 1s)")
    return audio


def probe_duration(media_path: str | Path) -> float | None:
    ffprobe = find_ffprobe()
    if not ffprobe:
        return None
    result = _run([
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ])
    if result.returncode != 0:
        return None
    try:
        return round(float(result.stdout.strip()), 3)
    except ValueError:
        return None
