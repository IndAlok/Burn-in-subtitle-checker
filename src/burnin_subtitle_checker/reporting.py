from __future__ import annotations

import csv
import html
import json
import shutil
import subprocess
import urllib.parse
from pathlib import Path
from typing import Any


def read_json_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "metadata" not in payload or "segments" not in payload:
        raise ValueError("Report JSON must contain {'metadata': ..., 'segments': [...]}")
    return payload


def write_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_csv_report(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "timestamp",
        "start",
        "end",
        "status",
        "reason",
        "score",
        "audio_text",
        "subtitle_text",
        "match_source",
        "char",
        "token_sort",
        "partial",
        "token_set",
        "window",
        "length_coverage",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in payload.get("segments", []):
            metrics = row.get("metrics") or {}
            writer.writerow({
                "id": row.get("id"),
                "timestamp": row.get("timestamp"),
                "start": row.get("start"),
                "end": row.get("end"),
                "status": row.get("status"),
                "reason": row.get("reason"),
                "score": row.get("score"),
                "audio_text": row.get("audio_text"),
                "subtitle_text": row.get("subtitle_text"),
                "match_source": metrics.get("match_source", ""),
                "char": metrics.get("char", ""),
                "token_sort": metrics.get("token_sort", ""),
                "partial": metrics.get("partial", ""),
                "token_set": metrics.get("token_set", ""),
                "window": metrics.get("window", ""),
                "length_coverage": metrics.get("length_coverage", ""),
            })
    return path


def _score_color(score: float) -> str:
    if score >= 0.9:
        return "#1a7f37"
    if score >= 0.8:
        return "#2da44e"
    if score >= 0.65:
        return "#9a6700"
    return "#cf222e"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def _fmt_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return ""
    minutes, sec = divmod(float(seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:05.2f}"
    return f"{minutes:02d}:{sec:05.2f}"


def _metric_pct(value: Any) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return ""


def _is_url(value: str) -> bool:
    return urllib.parse.urlparse(value).scheme in {"http", "https"}


def _is_youtube_url(value: str) -> bool:
    host = urllib.parse.urlparse(value).hostname or ""
    return host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def _youtube_video_id(value: str) -> str | None:
    parsed = urllib.parse.urlparse(value)
    if parsed.hostname == "youtu.be":
        return parsed.path.strip("/") or None
    if parsed.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        query = urllib.parse.parse_qs(parsed.query)
        return (query.get("v") or [None])[0]
    return None


def _relative_asset(target: Path, output_html: Path) -> str:
    return target.relative_to(output_html.parent).as_posix()


def _browser_preview(source: Path, output_html: Path) -> dict[str, str]:
    from .media import find_ffmpeg

    assets = output_html.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    target = assets / f"{source.stem}_preview.mp4"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return {
            "kind": "html5",
            "src": _relative_asset(target, output_html),
            "label": f"{source.name} (MP4 preview)",
            "note": "Using a browser-safe MP4 preview generated from the source media.",
        }
    try:
        ffmpeg = find_ffmpeg()
    except RuntimeError as exc:
        return {
            "kind": "html5",
            "src": source.as_uri(),
            "label": source.name,
            "warning": f"Preview transcode unavailable: {exc}",
        }

    result = subprocess.run([
        ffmpeg,
        "-y",
        "-i", str(source),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(target),
    ], capture_output=True, text=True, check=False)
    if result.returncode != 0 or not target.exists():
        message = (result.stderr or result.stdout or "ffmpeg returned no details").strip()[:500]
        return {
            "kind": "html5",
            "src": source.as_uri(),
            "label": source.name,
            "warning": f"Preview transcode failed; using original file link. {message}",
        }
    return {
        "kind": "html5",
        "src": _relative_asset(target, output_html),
        "label": f"{source.name} (MP4 preview)",
        "note": "Using a browser-safe MP4 preview generated from the source media.",
    }


def _video_reference(video: str | Path | None, output_html: Path, *, asset_mode: str) -> dict[str, str] | None:
    if not video:
        return None
    value = str(video)
    if _is_youtube_url(value):
        video_id = _youtube_video_id(value)
        if video_id:
            return {
                "kind": "youtube",
                "src": f"https://www.youtube.com/embed/{html.escape(video_id, quote=True)}",
                "label": value,
            }
    if _is_url(value):
        return {"kind": "html5", "src": value, "label": value}

    source = Path(value).expanduser()
    if not source.exists():
        return {"kind": "missing", "src": "", "label": value}
    source = source.resolve()
    if asset_mode == "preview":
        return _browser_preview(source, output_html)
    if asset_mode == "copy":
        assets = output_html.parent / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        target = assets / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        src = _relative_asset(target, output_html)
    else:
        src = source.as_uri()
    return {"kind": "html5", "src": src, "label": source.name}


def _metrics_html(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "<span class='muted'>-</span>"
    parts = []
    source = metrics.get("match_source")
    if source:
        parts.append(f"<span>src {html.escape(str(source))}</span>")
    for key, label in [
        ("char", "c"),
        ("token_sort", "tok"),
        ("partial", "part"),
        ("token_set", "set"),
        ("window", "win"),
        ("length_coverage", "cov"),
    ]:
        value = _metric_pct(metrics.get(key))
        if value:
            parts.append(f"<span>{label} {value}</span>")
    return "".join(parts) if parts else "<span class='muted'>-</span>"


def _row_html(row: dict[str, Any]) -> str:
    status = str(row.get("status", ""))
    reason = str(row.get("reason", ""))
    score = float(row.get("score", 0.0))
    timestamp = float(row.get("timestamp", 0.0))
    start = row.get("start")
    end = row.get("end")
    start_value = "" if start is None else str(float(start))
    end_value = "" if end is None else str(float(end))
    audio = str(row.get("audio_text") or "")
    subtitle = str(row.get("subtitle_text") or "")
    metrics = row.get("metrics") or {}
    pct = int(max(0, min(score, 1.0)) * 100)
    color = _score_color(score)
    search = " ".join([
        str(row.get("id", "")),
        _fmt_timestamp(timestamp),
        status,
        reason,
        audio,
        subtitle,
        " ".join(str(v) for v in metrics.values()),
    ])
    css_class = "review" if status == "REVIEW" else "ok"
    reason_label = html.escape(reason.replace("_", " "))
    audio_html = html.escape(audio) or "<span class='empty'>(no speech)</span>"
    subtitle_html = html.escape(subtitle) or "<span class='empty'>(no subtitle)</span>"
    return (
        f"<tr class='{css_class}' data-status='{html.escape(status, quote=True)}' "
        f"data-reason='{html.escape(reason, quote=True)}' data-score='{score:.4f}' "
        f"data-time='{timestamp:.3f}' data-start='{html.escape(start_value, quote=True)}' "
        f"data-end='{html.escape(end_value, quote=True)}' "
        f"data-search='{html.escape(search.casefold(), quote=True)}'>"
        f"<td><button class='time-link' type='button' onclick='seekTo({timestamp:.3f}, this)'>"
        f"{html.escape(_fmt_timestamp(timestamp))}</button></td>"
        f"<td class='text-cell'>{audio_html}</td>"
        f"<td class='text-cell'>{subtitle_html}</td>"
        f"<td data-sort-value='{score:.4f}'><div class='score-bar'>"
        f"<div class='score-fill' style='width:{pct}%;background:{color}'></div>"
        f"<span class='score-label'>{score:.2f}</span></div></td>"
        f"<td><span class='badge badge-{css_class}'>{html.escape(status)}</span></td>"
        f"<td>{reason_label}</td>"
        f"<td class='metrics'>{_metrics_html(metrics)}</td>"
        "</tr>"
    )


def _video_panel(video_ref: dict[str, str] | None) -> str:
    if not video_ref:
        return (
            "<section class='player-shell empty-player'>"
            "<div class='player-empty'>No video linked. Re-run report generation with --video to enable timestamp review.</div>"
            "</section>"
        )
    kind = video_ref["kind"]
    label = html.escape(video_ref.get("label", ""))
    if kind == "missing":
        return (
            "<section class='player-shell empty-player'>"
            f"<div class='player-empty'>Video file not found: {label}</div>"
            "</section>"
        )
    if kind == "youtube":
        src = html.escape(video_ref["src"], quote=True)
        return (
            "<section class='player-shell'>"
            f"<iframe id='youtubePlayer' src='{src}' data-base='{src}' "
            "allow='accelerometer; autoplay; encrypted-media; picture-in-picture' allowfullscreen></iframe>"
            "<div class='player-controls youtube-note'>Timestamp clicks reload the embedded YouTube player at the selected time.</div>"
            "</section>"
        )
    src = html.escape(video_ref["src"], quote=True)
    note = video_ref.get("note") or video_ref.get("warning") or ""
    note_html = f"<div class='player-warning'>{html.escape(note)}</div>" if note else ""
    return (
        "<section class='player-shell'>"
        f"<video id='reviewVideo' controls preload='metadata' src='{src}'></video>"
        f"{note_html}"
        "<div class='player-controls'>"
        "<button type='button' onclick='skipBy(-5)'>-5s</button>"
        "<button type='button' onclick='togglePlay()'>Play/Pause</button>"
        "<button type='button' onclick='skipBy(5)'>+5s</button>"
        "<label>Rate <select id='rateSelect' onchange='setRate(this.value)'>"
        "<option value='0.5'>0.5x</option><option value='0.75'>0.75x</option>"
        "<option value='1' selected>1x</option><option value='1.25'>1.25x</option>"
        "<option value='1.5'>1.5x</option><option value='2'>2x</option>"
        "</select></label>"
        "<label><input type='checkbox' id='loopSegment'> Loop active segment</label>"
        "</div>"
        "</section>"
    )


def render_html_report(
    payload: dict[str, Any],
    *,
    video: str | Path | None = None,
    asset_mode: str = "link",
    prepared_video_ref: dict[str, str] | None = None,
) -> str:
    metadata = payload["metadata"]
    output_anchor = Path.cwd() / "report.html"
    video_ref = prepared_video_ref or _video_reference(
        video or metadata.get("input_file") or metadata.get("input_source"),
        output_anchor,
        asset_mode=asset_mode,
    )
    rows = [_row_html(row) for row in payload.get("segments", [])]
    reasons = sorted({str(row.get("reason", "")) for row in payload.get("segments", []) if row.get("reason")})
    reason_options = "".join(
        f"<option value='{html.escape(reason, quote=True)}'>{html.escape(reason.replace('_', ' '))}</option>"
        for reason in reasons
    )

    meta_parts = [f"Threshold: {float(metadata.get('threshold', 0)):.2f}"]
    if metadata.get("input_file"):
        meta_parts.append(f"Input: {Path(str(metadata['input_file'])).name}")
    if metadata.get("input_source") and metadata.get("input_source") != metadata.get("input_file"):
        meta_parts.append(f"Source: {metadata['input_source']}")
    if metadata.get("duration_seconds"):
        meta_parts.append(f"Duration: {_fmt_duration(float(metadata['duration_seconds']))}")
    if metadata.get("model"):
        meta_parts.append(f"Model: {metadata['model']}")
    meta_line = html.escape(" | ".join(meta_parts))
    avg_score = float(metadata.get("avg_score", 0.0))
    total = int(metadata.get("segment_count", len(payload.get("segments", []))))
    flagged = int(metadata.get("flagged_count", 0))
    ok_count = max(0, total - flagged)

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Audio-Subtitle Mismatch Report</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; color: #1f2328; background: #f6f8fa; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; }
    .app { max-width: 1500px; margin: 0 auto; padding: 20px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 16px; }
    h1 { margin: 0 0 6px; font-size: 1.35rem; }
    .meta { color: #656d76; font-size: 0.9rem; margin: 0; }
    .layout { display: grid; grid-template-columns: minmax(280px, 0.30fr) minmax(0, 0.70fr); gap: 16px; align-items: start; }
    aside, main { min-width: 0; }
    .panel { min-width: 0; background: #fff; border: 1px solid #d0d7de; border-radius: 8px; overflow: hidden; }
    .player-shell { background: #0d1117; border-radius: 8px; overflow: hidden; border: 1px solid #30363d; }
    video, iframe { display: block; width: 100%; aspect-ratio: 16 / 9; border: 0; background: #000; }
    .player-controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; padding: 10px; color: #c9d1d9; background: #161b22; }
    .player-warning { padding: 9px 10px; color: #f0f6fc; background: #6e2c00; font-size: 0.84rem; }
    button, select, input { font: inherit; }
    button, select { border: 1px solid #d0d7de; border-radius: 6px; background: #fff; color: #1f2328; padding: 6px 10px; }
    button { cursor: pointer; }
    button:hover { background: #f6f8fa; }
    .youtube-note { font-size: 0.85rem; }
    .empty-player { background: #fff; border: 1px dashed #8c959f; }
    .player-empty { padding: 18px; color: #656d76; }
    .stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }
    .stat-card { background: #fff; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
    .stat-value { font-size: 1.45rem; font-weight: 700; }
    .stat-value.flagged { color: #cf222e; }
    .stat-label { color: #656d76; font-size: 0.8rem; margin-top: 2px; }
    .filters { padding: 12px; display: grid; grid-template-columns: minmax(180px, 1.8fr) repeat(4, minmax(110px, 1fr)); gap: 8px; border-bottom: 1px solid #d0d7de; }
    .filters input, .filters select { width: 100%; border: 1px solid #d0d7de; border-radius: 6px; padding: 7px 9px; background: #fff; }
    .table-wrap { width: 100%; max-height: calc(100vh - 220px); overflow-y: auto; overflow-x: hidden; }
    table { border-collapse: collapse; width: 100%; table-layout: fixed; background: #fff; }
    col.time { width: 10%; }
    col.audio { width: 21%; }
    col.subtitle { width: 28%; }
    col.score { width: 10%; }
    col.status { width: 9%; }
    col.reason { width: 10%; }
    col.metrics { width: 12%; }
    th, td { border-bottom: 1px solid #d0d7de; padding: 9px 8px; text-align: left; vertical-align: top; overflow-wrap: anywhere; word-break: break-word; }
    th { background: #f6f8fa; color: #57606a; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.02em; position: sticky; top: 0; z-index: 1; }
    th.sortable { cursor: pointer; }
    tr.review { background: #fff8f7; }
    tr.ok { background: #f6ffed; }
    tr.active { outline: 2px solid #0969da; outline-offset: -2px; }
    tr.hidden { display: none; }
    .text-cell { white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; line-height: 1.42; }
    .metrics { color: #57606a; font-size: 0.82rem; overflow-wrap: anywhere; word-break: break-word; }
    .metrics span { display: inline-block; margin: 0 8px 3px 0; }
    .empty, .muted { color: #8c959f; font-style: italic; }
    .time-link { color: #0969da; background: transparent; border: 0; padding: 0; font-weight: 650; white-space: nowrap; }
    .score-bar { position: relative; height: 22px; min-width: 78px; background: #eaeef2; border-radius: 999px; overflow: hidden; }
    .score-fill { height: 100%; }
    .score-label { position: absolute; top: 2px; left: 8px; color: #1f2328; font-size: 0.78rem; font-weight: 700; }
    .badge { display: inline-block; white-space: nowrap; text-align: center; border-radius: 999px; padding: 2px 8px; font-size: 0.78rem; font-weight: 700; }
    .badge-review { background: #ffebe9; color: #cf222e; }
    .badge-ok { background: #dafbe1; color: #1a7f37; }
    .toolbar { display: flex; gap: 8px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
    .visible-count { color: #656d76; font-size: 0.9rem; }
    @media (max-width: 1050px) {
      .layout { grid-template-columns: 1fr; }
      .table-wrap { max-height: none; }
      .filters { grid-template-columns: 1fr 1fr; }
    }
    @media print {
      .player-shell, .filters, .toolbar { display: none; }
      body { background: #fff; }
      .app { max-width: none; padding: 0; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1>Audio-Subtitle Mismatch Report</h1>
        <p class="meta">__META_LINE__</p>
      </div>
      <div class="toolbar">
        <span class="visible-count" id="visibleCount"></span>
        <button type="button" onclick="exportVisibleCsv()">Export visible CSV</button>
      </div>
    </header>
    <div class="layout">
      <aside>
        __VIDEO_PANEL__
        <div class="stats">
          <div class="stat-card"><div class="stat-value">__TOTAL__</div><div class="stat-label">Total</div></div>
          <div class="stat-card"><div class="stat-value flagged">__FLAGGED__</div><div class="stat-label">Review</div></div>
          <div class="stat-card"><div class="stat-value">__OK__</div><div class="stat-label">OK</div></div>
          <div class="stat-card"><div class="stat-value">__AVG__</div><div class="stat-label">Avg Similarity</div></div>
          <div class="stat-card"><div class="stat-value">__MISSING_SUB__</div><div class="stat-label">Missing Subtitles</div></div>
          <div class="stat-card"><div class="stat-value">__MISSING_SPEECH__</div><div class="stat-label">Missing Speech</div></div>
        </div>
      </aside>
      <main class="panel">
        <div class="filters">
          <input id="searchBox" type="search" placeholder="Search timestamp, text, reason, metric" oninput="applyFilters()">
          <select id="statusFilter" onchange="applyFilters()">
            <option value="">All statuses</option>
            <option value="REVIEW">Review</option>
            <option value="OK">OK</option>
          </select>
          <select id="reasonFilter" onchange="applyFilters()">
            <option value="">All reasons</option>
            __REASON_OPTIONS__
          </select>
          <input id="minScore" type="number" min="0" max="1" step="0.05" placeholder="Min score" oninput="applyFilters()">
          <input id="maxScore" type="number" min="0" max="1" step="0.05" placeholder="Max score" oninput="applyFilters()">
        </div>
        <div class="table-wrap">
          <table id="resultsTable">
            <colgroup>
              <col class="time">
              <col class="audio">
              <col class="subtitle">
              <col class="score">
              <col class="status">
              <col class="reason">
              <col class="metrics">
            </colgroup>
            <thead>
              <tr>
                <th class="sortable" onclick="sortTable(0, 'number')">Time</th>
                <th class="sortable" onclick="sortTable(1, 'text')">Audio</th>
                <th class="sortable" onclick="sortTable(2, 'text')">Subtitle</th>
                <th class="sortable" onclick="sortTable(3, 'number')">Score</th>
                <th class="sortable" onclick="sortTable(4, 'text')">Status</th>
                <th class="sortable" onclick="sortTable(5, 'text')">Reason</th>
                <th>Metrics</th>
              </tr>
            </thead>
            <tbody>
              __ROWS__
            </tbody>
          </table>
        </div>
      </main>
    </div>
  </div>
  <script>
    var activeRow = null;
    var loopStart = null;
    var loopEnd = null;
    var sortState = { index: -1, dir: 1 };

    function videoEl() { return document.getElementById('reviewVideo'); }

    function seekTo(seconds, button) {
      var row = button ? button.closest('tr') : null;
      if (row) {
        if (activeRow) activeRow.classList.remove('active');
        activeRow = row;
        activeRow.classList.add('active');
        var start = parseFloat(row.dataset.start);
        var end = parseFloat(row.dataset.end);
        loopStart = Number.isFinite(start) ? start : Math.max(0, seconds - 1.5);
        loopEnd = Number.isFinite(end) && end > loopStart ? end : seconds + 1.5;
      }
      var video = videoEl();
      if (video) {
        video.currentTime = Math.max(0, seconds);
        video.play().catch(function() {});
        return;
      }
      var yt = document.getElementById('youtubePlayer');
      if (yt) {
        yt.src = yt.dataset.base + '?start=' + Math.floor(seconds) + '&autoplay=1';
      }
    }

    function skipBy(delta) {
      var video = videoEl();
      if (video) video.currentTime = Math.max(0, video.currentTime + delta);
    }

    function togglePlay() {
      var video = videoEl();
      if (!video) return;
      if (video.paused) video.play().catch(function() {});
      else video.pause();
    }

    function setRate(value) {
      var video = videoEl();
      if (video) video.playbackRate = parseFloat(value);
    }

    var video = videoEl();
    if (video) {
      video.addEventListener('timeupdate', function() {
        var loop = document.getElementById('loopSegment');
        if (loop && loop.checked && loopStart !== null && loopEnd !== null && video.currentTime > loopEnd) {
          video.currentTime = loopStart;
          video.play().catch(function() {});
        }
      });
    }

    function applyFilters() {
      var query = document.getElementById('searchBox').value.trim().toLowerCase();
      var status = document.getElementById('statusFilter').value;
      var reason = document.getElementById('reasonFilter').value;
      var minScore = parseFloat(document.getElementById('minScore').value);
      var maxScore = parseFloat(document.getElementById('maxScore').value);
      var visible = 0;
      document.querySelectorAll('#resultsTable tbody tr').forEach(function(row) {
        var score = parseFloat(row.dataset.score);
        var show = true;
        if (query && !row.dataset.search.includes(query)) show = false;
        if (status && row.dataset.status !== status) show = false;
        if (reason && row.dataset.reason !== reason) show = false;
        if (Number.isFinite(minScore) && score < minScore) show = false;
        if (Number.isFinite(maxScore) && score > maxScore) show = false;
        row.classList.toggle('hidden', !show);
        if (show) visible += 1;
      });
      document.getElementById('visibleCount').textContent = visible + ' visible';
    }

    function sortTable(index, type) {
      var tbody = document.querySelector('#resultsTable tbody');
      var rows = Array.from(tbody.querySelectorAll('tr'));
      var dir = sortState.index === index ? -sortState.dir : 1;
      sortState = { index: index, dir: dir };
      rows.sort(function(a, b) {
        var av;
        var bv;
        if (index === 0) {
          av = parseFloat(a.dataset.time);
          bv = parseFloat(b.dataset.time);
        } else if (index === 3) {
          av = parseFloat(a.dataset.score);
          bv = parseFloat(b.dataset.score);
        } else {
          av = a.children[index].innerText.toLowerCase();
          bv = b.children[index].innerText.toLowerCase();
        }
        if (type === 'number') return (av - bv) * dir;
        return av.localeCompare(bv) * dir;
      });
      rows.forEach(function(row) { tbody.appendChild(row); });
    }

    function csvEscape(value) {
      value = String(value).replace(/"/g, '""');
      return '"' + value + '"';
    }

    function exportVisibleCsv() {
      var rows = [['Timestamp', 'Start', 'End', 'Status', 'Reason', 'Score', 'Audio Text', 'Subtitle Text', 'Metrics']];
      document.querySelectorAll('#resultsTable tbody tr:not(.hidden)').forEach(function(row) {
        rows.push([
          row.dataset.time,
          row.dataset.start || '',
          row.dataset.end || '',
          row.dataset.status,
          row.dataset.reason,
          row.dataset.score,
          row.children[1].innerText,
          row.children[2].innerText,
          row.children[6].innerText.replace(/\\s+/g, ' ')
        ]);
      });
      var csv = rows.map(function(row) { return row.map(csvEscape).join(','); }).join('\\n');
      var blob = new Blob(["\uFEFF" + csv], { type: 'text/csv;charset=utf-8' });
      var link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'visible_mismatch_rows.csv';
      link.click();
      URL.revokeObjectURL(link.href);
    }

    applyFilters();
  </script>
</body>
</html>
"""
    return (
        template
        .replace("__META_LINE__", meta_line)
        .replace("__VIDEO_PANEL__", _video_panel(video_ref))
        .replace("__TOTAL__", str(total))
        .replace("__FLAGGED__", str(flagged))
        .replace("__OK__", str(ok_count))
        .replace("__AVG__", f"{avg_score:.0%}")
        .replace("__MISSING_SUB__", str(metadata.get("missing_subtitle_count", 0)))
        .replace("__MISSING_SPEECH__", str(metadata.get("missing_speech_count", 0)))
        .replace("__REASON_OPTIONS__", reason_options)
        .replace("__ROWS__", "\n".join(rows))
    )


def write_html_report(
    payload: dict[str, Any],
    output_path: str | Path,
    *,
    video: str | Path | None = None,
    asset_mode: str = "link",
) -> Path:
    if asset_mode not in {"link", "copy", "preview"}:
        raise ValueError("asset_mode must be 'link', 'copy', or 'preview'")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    video_value = video or payload.get("metadata", {}).get("input_file") or payload.get("metadata", {}).get("input_source")
    video_ref = _video_reference(video_value, path, asset_mode=asset_mode)

    metadata = dict(payload.get("metadata", {}))
    if video_value:
        metadata.setdefault("input_file", str(video_value))
    render_payload = {**payload, "metadata": metadata}
    html_text = render_html_report(render_payload, prepared_video_ref=video_ref)
    path.write_text(html_text, encoding="utf-8")
    return path
