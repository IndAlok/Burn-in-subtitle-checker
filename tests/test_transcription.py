import pytest

from burnin_subtitle_checker.models import AudioSegment
from burnin_subtitle_checker.transcription import (
    INITIAL_PROMPTS,
    _MIN_SRT_DURATION,
    _MIN_SRT_GAP,
    _filter_noise,
    _fix_srt_overlaps,
    _is_hallucination,
    _normalize_temperature,
    _recommended_model,
    render_srt,
    render_text,
    seconds_to_srt_timestamp,
)


class TestSrtTimestamp:
    def test_zero(self):
        assert seconds_to_srt_timestamp(0) == "00:00:00,000"

    def test_standard(self):
        assert seconds_to_srt_timestamp(3661.5) == "01:01:01,500"

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            seconds_to_srt_timestamp(-1)


class TestHallucinationDetection:
    def test_short_text_not_hallucination(self):
        assert not _is_hallucination("hello world")

    def test_repetitive_chars(self):
        assert _is_hallucination("वा" * 30)

    def test_word_repetition(self):
        assert _is_hallucination("नमस्ते " * 10)

    def test_half_duplication(self):
        base = "यह एक लंबा वाक्य है जो बार बार दोहराया जाता है"
        assert _is_hallucination(base + base)

    def test_normal_hindi(self):
        assert not _is_hallucination("आज मौसम अच्छा है और हम बाहर जा रहे हैं")

    def test_common_english_artifacts(self):
        assert _is_hallucination("Thank you for watching this video")
        assert _is_hallucination("Please like and subscribe to my channel")

    def test_mixed_script_ok(self):
        assert not _is_hallucination("यह एक test है", language="hi")

    def test_empty_not_hallucination(self):
        assert not _is_hallucination("")


class TestModelRecommendation:
    def test_indic_defaults_medium(self):
        for code in ("hi", "kn", "ta", "te"):
            assert _recommended_model(code, None) == "medium"

    def test_english_defaults_small(self):
        assert _recommended_model("en", None) == "small"

    def test_explicit_overrides(self):
        assert _recommended_model("hi", "large-v3") == "large-v3"

    def test_auto_defaults_small(self):
        assert _recommended_model("auto", None) == "small"


class TestTemperature:
    def test_default_temperature_is_deterministic(self):
        assert _normalize_temperature(0.0) == 0.0

    def test_rounds_temperature(self):
        assert _normalize_temperature(0.23456) == 0.235

    def test_rejects_invalid_temperature(self):
        with pytest.raises(ValueError):
            _normalize_temperature(-0.1)
        with pytest.raises(ValueError):
            _normalize_temperature(1.1)


class TestInitialPrompts:
    def test_hindi_prompt_is_devanagari(self):
        prompt = INITIAL_PROMPTS["hi"]
        assert sum(1 for ch in prompt if "\u0900" <= ch <= "\u097F") > 10

    def test_kannada_prompt_is_kannada(self):
        prompt = INITIAL_PROMPTS["kn"]
        assert sum(1 for ch in prompt if "\u0C80" <= ch <= "\u0CFF") > 10

    def test_tamil_prompt_is_tamil(self):
        prompt = INITIAL_PROMPTS["ta"]
        assert sum(1 for ch in prompt if "\u0B80" <= ch <= "\u0BFF") > 10

    def test_telugu_prompt_is_telugu(self):
        prompt = INITIAL_PROMPTS["te"]
        assert sum(1 for ch in prompt if "\u0C00" <= ch <= "\u0C7F") > 10

    def test_english_has_no_prompt(self):
        assert "en" not in INITIAL_PROMPTS


class TestFilterNoise:
    def test_keeps_valid_segments(self):
        segs = [AudioSegment(id=1, start=0.0, end=3.0, midpoint=1.5, text="hello")]
        assert len(_filter_noise(segs)) == 1

    def test_removes_short_empty(self):
        segs = [AudioSegment(id=1, start=0.0, end=0.3, midpoint=0.15, text="")]
        assert len(_filter_noise(segs)) == 0

    def test_keeps_long_empty(self):
        segs = [AudioSegment(id=1, start=0.0, end=2.0, midpoint=1.0, text="")]
        assert len(_filter_noise(segs)) == 1

    def test_renumbers(self):
        segs = [
            AudioSegment(id=1, start=0.0, end=0.2, midpoint=0.1, text=""),
            AudioSegment(id=2, start=1.0, end=3.0, midpoint=2.0, text="hello"),
            AudioSegment(id=3, start=4.0, end=6.0, midpoint=5.0, text="world"),
        ]
        filtered = _filter_noise(segs)
        assert len(filtered) == 2
        assert filtered[0].id == 1
        assert filtered[1].id == 2


class TestFixSrtOverlaps:
    def test_no_overlaps(self):
        segs = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello"},
            {"id": 2, "start": 3.0, "end": 5.0, "text": "World"},
        ]
        assert len(_fix_srt_overlaps(segs)) == 2

    def test_trims_overlapping(self):
        segs = [
            {"id": 1, "start": 0.0, "end": 3.0, "text": "Hello"},
            {"id": 2, "start": 2.5, "end": 5.0, "text": "World"},
        ]
        fixed = _fix_srt_overlaps(segs)
        assert len(fixed) == 2
        assert float(fixed[0]["end"]) < float(fixed[1]["start"])

    def test_discards_short_entries(self):
        segs = [
            {"id": 1, "start": 0.0, "end": 0.1, "text": "Too short"},
            {"id": 2, "start": 1.0, "end": 3.0, "text": "OK"},
        ]
        fixed = _fix_srt_overlaps(segs)
        assert len(fixed) == 1
        assert fixed[0]["text"] == "OK"

    def test_empty_text_skipped(self):
        segs = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": ""},
            {"id": 2, "start": 3.0, "end": 5.0, "text": "Valid"},
        ]
        assert len(_fix_srt_overlaps(segs)) == 1

    def test_sorts_by_start(self):
        segs = [
            {"id": 2, "start": 3.0, "end": 5.0, "text": "Second"},
            {"id": 1, "start": 0.0, "end": 2.0, "text": "First"},
        ]
        assert _fix_srt_overlaps(segs)[0]["text"] == "First"

    def test_enforces_minimum_gap(self):
        segs = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello"},
            {"id": 2, "start": 2.05, "end": 4.0, "text": "World"},
        ]
        fixed = _fix_srt_overlaps(segs)
        gap = float(fixed[1]["start"]) - float(fixed[0]["end"])
        assert gap >= _MIN_SRT_GAP - 0.001

    def test_constants_are_sane(self):
        assert _MIN_SRT_DURATION > _MIN_SRT_GAP


class TestRenderSrt:
    def test_basic(self):
        segments = [
            {"id": 1, "start": 0.0, "end": 2.0, "text": "Hello"},
            {"id": 2, "start": 3.0, "end": 5.0, "text": "World"},
        ]
        srt = render_srt(segments)
        assert "1\n00:00:00,000 --> 00:00:02,000\nHello" in srt
        assert "2\n00:00:03,000 --> 00:00:05,000\nWorld" in srt

    def test_skips_empty(self):
        assert render_srt([{"id": 1, "start": 0.0, "end": 1.0, "text": ""}]) == ""

    def test_renumbers_after_overlap_fix(self):
        segments = [
            {"id": 5, "start": 0.0, "end": 2.0, "text": "Hello"},
            {"id": 10, "start": 3.0, "end": 5.0, "text": "World"},
        ]
        srt = render_srt(segments)
        assert srt.startswith("1\n")
        assert "2\n" in srt


class TestRenderText:
    def test_output_format(self):
        payload = {
            "metadata": {
                "input_file": "test.mp4",
                "language_requested": "hi",
                "language_detected": "hi",
                "model": "medium",
            },
            "segments": [{"start": 1.0, "end": 3.0, "text": "नमस्ते"}],
        }
        text = render_text(payload)
        assert "test.mp4" in text
        assert "नमस्ते" in text
