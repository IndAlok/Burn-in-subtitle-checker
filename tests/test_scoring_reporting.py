import pytest

from burnin_subtitle_checker.scoring import normalize_text, _similarity, compare_segments, align_segments
from burnin_subtitle_checker.reporting import render_html_report, write_csv_report, write_html_report, _fmt_duration


class TestNormalizeText:
    def test_basic(self):
        assert normalize_text("  Hello,  World!  ") == "hello world"

    def test_none(self):
        assert normalize_text(None) == ""

    def test_hindi_chandrabindu(self):
        assert normalize_text("हँस") == normalize_text("हंस")

    def test_nukta_stripped(self):
        assert normalize_text("ज़") == normalize_text("ज")

    def test_devanagari_danda(self):
        assert "।" not in normalize_text("वाक्य समाप्त।")

    def test_kannada_text(self):
        result = normalize_text("ನಮಸ್ಕಾರ!")
        assert "!" not in result
        assert "ನಮಸ್ಕಾರ" in result

    def test_mixed_whitespace(self):
        assert normalize_text("a\n\nb\r\nc") == "a b c"


class TestSimilarity:
    def test_identical(self):
        assert _similarity("hello world", "hello world") == 1.0

    def test_empty_both(self):
        assert _similarity("", "") == 1.0

    def test_one_empty(self):
        assert _similarity("hello", "") == 0.0

    def test_partial_match(self):
        score = _similarity("the quick brown fox", "the quick brown")
        assert 0.5 < score < 1.0

    def test_window_match_rescues_noisy_ocr_context(self):
        audio = "the quick brown fox jumps"
        subtitle = "credits noise the quick brown fox jumps footer numbers"
        assert _similarity(audio, subtitle) > 0.85

    def test_word_reorder(self):
        score = _similarity("hello beautiful world", "world beautiful hello")
        assert score > 0.7

    def test_single_char_diff(self):
        score = _similarity("नमस्ते", "नमस्तै")
        assert score > 0.6

    def test_hindi_identical(self):
        score = _similarity("यह एक परीक्षण है", "यह एक परीक्षण है")
        assert score == 1.0


class TestCompareSegments:
    def test_perfect_match(self):
        audio = [{"id": 1, "start": 0, "end": 2, "text": "hello"}]
        subs = [{"id": 1, "timestamp": 1.0, "subtitle_text": "hello"}]
        result = compare_segments(audio, subs)
        assert result["metadata"]["flagged_count"] == 0
        assert result["segments"][0]["status"] == "OK"
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 2.0
        assert result["segments"][0]["metrics"]["match_source"] == "aligned"
        assert result["segments"][0]["metrics"]["char"] == 1.0

    def test_mismatch_flagged(self):
        audio = [{"id": 1, "start": 0, "end": 2, "text": "completely different text"}]
        subs = [{"id": 1, "timestamp": 1.0, "subtitle_text": "xyz abc"}]
        result = compare_segments(audio, subs)
        assert result["metadata"]["flagged_count"] == 1
        assert result["segments"][0]["status"] == "REVIEW"

    def test_missing_subtitle(self):
        audio = [{"id": 1, "start": 0, "end": 2, "text": "hello"}]
        subs = [{"id": 1, "timestamp": 1.0, "subtitle_text": ""}]
        result = compare_segments(audio, subs)
        assert result["segments"][0]["reason"] == "missing_subtitle"
        assert result["metadata"]["missing_subtitle_count"] == 1

    def test_missing_speech(self):
        audio = [{"id": 1, "start": 0, "end": 2, "text": ""}]
        subs = [{"id": 1, "timestamp": 1.0, "subtitle_text": "hello"}]
        result = compare_segments(audio, subs)
        assert result["segments"][0]["reason"] == "missing_speech"
        assert result["metadata"]["missing_speech_count"] == 1

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            compare_segments([], [], threshold=1.5)

    def test_hindi_comparison(self):
        audio = [{"id": 1, "start": 0, "end": 3, "text": "नमस्ते दुनिया"}]
        subs = [{"id": 1, "timestamp": 1.5, "subtitle_text": "नमस्ते दुनिया"}]
        result = compare_segments(audio, subs)
        assert result["segments"][0]["score"] >= 0.9

    def test_kannada_comparison(self):
        audio = [{"id": 1, "start": 0, "end": 3, "text": "ನಮಸ್ಕಾರ ಪ್ರಪಂಚ"}]
        subs = [{"id": 1, "timestamp": 1.5, "subtitle_text": "ನಮಸ್ಕಾರ ಪ್ರಪಂಚ"}]
        result = compare_segments(audio, subs)
        assert result["segments"][0]["score"] >= 0.9

    def test_empty_segments(self):
        result = compare_segments([], [])
        assert result["metadata"]["segment_count"] == 0
        assert result["metadata"]["flagged_count"] == 0
        assert result["metadata"]["avg_score"] == 0.0

    def test_multiple_segments(self):
        audio = [
            {"id": 1, "start": 0, "end": 2, "text": "hello"},
            {"id": 2, "start": 3, "end": 5, "text": "world"},
        ]
        subs = [
            {"id": 1, "timestamp": 1.0, "subtitle_text": "hello"},
            {"id": 2, "timestamp": 4.0, "subtitle_text": "xyz"},
        ]
        result = compare_segments(audio, subs)
        assert result["metadata"]["segment_count"] == 2
        assert result["segments"][0]["status"] == "OK"
        assert result["segments"][1]["status"] == "REVIEW"

    def test_avg_score_computed(self):
        audio = [{"id": 1, "start": 0, "end": 2, "text": "hello"}]
        subs = [{"id": 1, "timestamp": 1.0, "subtitle_text": "hello"}]
        result = compare_segments(audio, subs)
        assert result["metadata"]["avg_score"] > 0.9

    def test_duplicate_detection(self):
        audio = [
            {"id": 1, "start": 0, "end": 2, "text": "hello"},
            {"id": 2, "start": 3, "end": 5, "text": "world"},
        ]
        subs = [
            {"id": 1, "timestamp": 1.0, "subtitle_text": "same text"},
            {"id": 2, "timestamp": 4.0, "subtitle_text": "same text"},
        ]
        result = compare_segments(audio, subs)
        assert result["segments"][1]["reason"] == "duplicate"
        assert result["metadata"]["duplicate_count"] == 1

    def test_metadata_has_all_fields(self):
        result = compare_segments([], [])
        meta = result["metadata"]
        assert "avg_score" in meta
        assert "missing_subtitle_count" in meta
        assert "missing_speech_count" in meta
        assert "duplicate_count" in meta

    def test_context_match_marks_match_source(self):
        audio = [{"id": 1, "start": 0, "end": 2, "text": "hello world"}]
        subs = [
            {"id": 1, "timestamp": 1.0, "subtitle_text": "wrong"},
            {"id": 2, "timestamp": 1.4, "subtitle_text": "hello world"},
        ]
        result = compare_segments(audio, subs, context_window_seconds=1.0)
        assert result["segments"][0]["status"] == "OK"
        assert result["segments"][0]["metrics"]["match_source"] == "nearby"

    def test_hindi_window_match_handles_neighboring_ocr_noise(self):
        audio_text = "\u0936\u093e\u0902\u0924\u093f \u0914\u0930 \u0905\u0930\u0941\u0923 \u0905\u091a\u094d\u091b\u0947 \u092e\u093f\u0924\u094d\u0930 \u0925\u0947"
        noisy_subtitle = "\u096b \u096d \u0932\u0947\u0916\u0928 \u0936\u0947\u0930\u093f\u0932 \u0930\u093e\u0935 " + audio_text + " \u096e \u092b\u094d\u0930\u0947\u092e \u0936\u094b\u0930"
        result = compare_segments(
            [{"id": 1, "start": 0, "end": 3, "text": audio_text}],
            [{"id": 1, "timestamp": 1.5, "subtitle_text": noisy_subtitle}],
        )
        assert result["segments"][0]["status"] == "OK"
        assert result["segments"][0]["metrics"]["window"] > 0.9

    def test_long_segment_can_match_joined_subtitle_sequence(self):
        audio = [{"id": 1, "start": 10, "end": 20, "text": "alpha beta gamma delta"}]
        subs = [
            {"id": 1, "timestamp": 11.0, "subtitle_text": "alpha beta"},
            {"id": 2, "timestamp": 15.0, "subtitle_text": "gamma delta"},
        ]
        result = compare_segments(audio, subs, threshold=0.80)
        assert result["segments"][0]["status"] == "OK"
        assert result["segments"][0]["metrics"]["match_source"] == "nearby_2x"


class TestAlignSegments:
    def test_id_matching(self):
        audio = [{"id": 1, "start": 0, "end": 2}]
        subs = [{"id": 1, "timestamp": 1.0}]
        pairs = align_segments(audio, subs)
        assert len(pairs) == 1
        assert pairs[0][0]["id"] == 1
        assert pairs[0][1]["id"] == 1

    def test_unmatched_audio(self):
        audio = [{"id": 1, "start": 0, "end": 2}]
        subs = [{"id": 99, "timestamp": 50.0}]
        pairs = align_segments(audio, subs, tolerance=1.0)
        assert len(pairs) == 2

    def test_timestamp_fallback(self):
        audio = [{"id": 1, "start": 10.0, "end": 12.0}]
        subs = [{"id": 99, "timestamp": 11.0}]
        pairs = align_segments(audio, subs, tolerance=2.0)
        assert len(pairs) == 1
        assert pairs[0][0] is not None
        assert pairs[0][1] is not None

    def test_sorted_output(self):
        audio = [
            {"id": 2, "start": 5, "end": 7},
            {"id": 1, "start": 0, "end": 2},
        ]
        subs = [
            {"id": 1, "timestamp": 1.0},
            {"id": 2, "timestamp": 6.0},
        ]
        pairs = align_segments(audio, subs)
        assert pairs[0][0]["id"] == 1
        assert pairs[1][0]["id"] == 2


class TestFmtDuration:
    def test_none(self):
        assert _fmt_duration(None) == ""

    def test_seconds(self):
        assert _fmt_duration(45) == "0m 45s"

    def test_minutes(self):
        assert _fmt_duration(125) == "2m 5s"

    def test_hours(self):
        assert _fmt_duration(3661) == "1h 1m 1s"


class TestHtmlReport:
    def test_renders_valid_html(self):
        payload = {
            "metadata": {"threshold": 0.75, "segment_count": 1, "flagged_count": 1},
            "segments": [{
                "id": 1, "timestamp": 1.0,
                "audio_text": "hello", "subtitle_text": "helo",
                "score": 0.5, "status": "REVIEW", "reason": "low_similarity",
            }],
        }
        result = render_html_report(payload)
        assert "<!doctype html>" in result
        assert "REVIEW" in result
        assert "hello" in result

    def test_renders_with_extra_metadata(self):
        payload = {
            "metadata": {
                "threshold": 0.75, "segment_count": 1, "flagged_count": 0,
                "input_file": "test_video.mp4", "duration_seconds": 120.0, "model": "medium",
                "avg_score": 0.95, "missing_subtitle_count": 0,
                "missing_speech_count": 0, "duplicate_count": 0,
            },
            "segments": [{
                "id": 1, "timestamp": 1.0,
                "audio_text": "hello", "subtitle_text": "hello",
                "score": 1.0, "status": "OK", "reason": "ok",
            }],
        }
        result = render_html_report(payload)
        assert "test_video.mp4" in result
        assert "2m 0s" in result
        assert "medium" in result
        assert "Avg Similarity" in result
        assert "95%" in result

    def test_hindi_text_in_report(self):
        payload = {
            "metadata": {"threshold": 0.75, "segment_count": 1, "flagged_count": 0},
            "segments": [{
                "id": 1, "timestamp": 1.0,
                "audio_text": "नमस्ते", "subtitle_text": "नमस्ते",
                "score": 1.0, "status": "OK", "reason": "ok",
            }],
        }
        result = render_html_report(payload)
        assert "नमस्ते" in result

    def test_missing_stats_cards(self):
        payload = {
            "metadata": {
                "threshold": 0.75, "segment_count": 2, "flagged_count": 1,
                "missing_subtitle_count": 1, "missing_speech_count": 0,
            },
            "segments": [
                {"id": 1, "timestamp": 1.0, "audio_text": "hello", "subtitle_text": "",
                 "score": 0.0, "status": "REVIEW", "reason": "missing_subtitle"},
                {"id": 2, "timestamp": 3.0, "audio_text": "world", "subtitle_text": "world",
                 "score": 1.0, "status": "OK", "reason": "ok"},
            ],
        }
        result = render_html_report(payload)
        assert "Missing Subtitles" in result
        assert "Missing Speech" in result

    def test_report_has_player_search_sort_and_timestamp_seek(self):
        payload = {
            "metadata": {"threshold": 0.8, "segment_count": 1, "flagged_count": 1},
            "segments": [{
                "id": 1, "timestamp": 12.5, "start": 11.0, "end": 14.0,
                "audio_text": "hello", "subtitle_text": "wrong",
                "score": 0.2, "status": "REVIEW", "reason": "low_similarity",
                "metrics": {"char": 0.2, "token_sort": 0.2, "partial": 0.2, "match_source": "aligned"},
            }],
        }
        result = render_html_report(payload, video="https://example.com/video.mp4")
        assert "reviewVideo" in result
        assert "searchBox" in result
        assert "sortTable" in result
        assert "seekTo(12.500" in result
        assert "src aligned" in result

    def test_youtube_report_uses_embed_player(self):
        payload = {"metadata": {"threshold": 0.8, "segment_count": 0, "flagged_count": 0}, "segments": []}
        result = render_html_report(payload, video="https://www.youtube.com/watch?v=abc123")
        assert "youtubePlayer" in result
        assert "youtube.com/embed/abc123" in result

    def test_write_csv_report(self, tmp_path):
        payload = {
            "metadata": {"threshold": 0.8, "segment_count": 1, "flagged_count": 0},
            "segments": [{
                "id": 1, "timestamp": 1.0, "start": 0.0, "end": 2.0,
                "audio_text": "hello", "subtitle_text": "hello",
                "score": 1.0, "status": "OK", "reason": "ok",
                "metrics": {"match_source": "aligned", "char": 1.0},
            }],
        }
        path = write_csv_report(payload, tmp_path / "report.csv")
        text = path.read_text(encoding="utf-8")
        assert "audio_text" in text
        assert "aligned" in text

    def test_preview_asset_mode_falls_back_when_ffmpeg_missing(self, tmp_path, monkeypatch):
        import burnin_subtitle_checker.media as media

        payload = {"metadata": {"threshold": 0.8, "segment_count": 0, "flagged_count": 0}, "segments": []}
        video = tmp_path / "clip.mkv"
        video.write_bytes(b"not a real video")
        monkeypatch.setattr(media, "find_ffmpeg", lambda: (_ for _ in ()).throw(RuntimeError("missing ffmpeg")))

        path = write_html_report(payload, tmp_path / "report.html", video=video, asset_mode="preview")
        text = path.read_text(encoding="utf-8")

        assert "Preview transcode unavailable" in text
        assert "reviewVideo" in text
