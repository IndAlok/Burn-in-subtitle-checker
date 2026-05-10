import numpy as np
import pytest
import builtins

from burnin_subtitle_checker.ocr import (
    OCR_EFFORT_CHOICES,
    WORD_CONFIDENCE_FLOOR,
    _candidate_timestamps,
    _candidate_timestamps_for_segment,
    _dynamic_subtitle_bounds,
    _fixed_subtitle_bounds,
    _is_garbage,
    _merge_unique_texts,
    _preprocess_variants,
    _process_item,
    _segment_midpoint,
    check_tesseract_languages,
    crop_subtitle_region,
    tesseract_lang,
)


class TestTesseractLang:
    def test_required_languages(self):
        assert tesseract_lang("en") == "eng"
        assert tesseract_lang("hi") == "hin"
        assert tesseract_lang("kn") == "kan"
        assert tesseract_lang("ta") == "tam"
        assert tesseract_lang("te") == "tel"
        assert tesseract_lang("multi") == "eng+hin+kan+tam+tel"

    def test_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported"):
            tesseract_lang("fr")

    def test_missing_pytesseract_has_install_hint(self, monkeypatch):
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pytesseract":
                raise ImportError("missing")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError, match="python -m pip install -e"):
            check_tesseract_languages("hi")


class TestCropSubtitleRegion:
    def test_default_15_percent(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = crop_subtitle_region(frame)
        assert crop.shape[0] == 15
        assert crop.shape[1] == 200

    def test_custom_ratio(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        crop = crop_subtitle_region(frame, crop_ratio=0.25)
        assert crop.shape[0] == 50

    def test_invalid_ratio(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            crop_subtitle_region(frame, crop_ratio=0)
        with pytest.raises(ValueError):
            crop_subtitle_region(frame, crop_ratio=1.5)

    def test_none_frame(self):
        with pytest.raises(ValueError):
            crop_subtitle_region(None)

    def test_grayscale_frame(self):
        frame = np.zeros((100, 200), dtype=np.uint8)
        crop = crop_subtitle_region(frame, crop_ratio=0.20)
        assert crop.shape[0] == 20

    def test_fixed_bounds_report_original_coordinates(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        assert _fixed_subtitle_bounds(frame, crop_ratio=0.20) == (0, 80, 200, 20)

    def test_dynamic_bounds_stay_inside_frame(self):
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        frame[92:98, 40:200] = 255
        x, y, width, height = _dynamic_subtitle_bounds(frame, crop_ratio=0.15)
        assert x == 0
        assert width == 240
        assert 0 <= y < 120
        assert height > 0
        assert y + height <= 120


class TestSegmentMidpoint:
    def test_midpoint_key(self):
        assert _segment_midpoint({"midpoint": 5.5}) == 5.5

    def test_timestamp_key(self):
        assert _segment_midpoint({"timestamp": 3.0}) == 3.0

    def test_start_end(self):
        assert _segment_midpoint({"start": 2.0, "end": 4.0}) == 3.0

    def test_missing_raises(self):
        with pytest.raises(ValueError):
            _segment_midpoint({"id": 1})


class TestIsGarbage:
    def test_empty(self):
        assert _is_garbage("")

    def test_single_char(self):
        assert _is_garbage("x")

    def test_all_symbols(self):
        assert _is_garbage("!@#$%^&*()")

    def test_valid_languages(self):
        assert not _is_garbage("Hello World", language="en")
        assert not _is_garbage("नमस्ते दुनिया", language="hi")
        assert not _is_garbage("ನಮಸ್ಕಾರ", language="kn")
        assert not _is_garbage("வணக்கம் உலகம்", language="ta")
        assert not _is_garbage("నమస్తే ప్రపంచం", language="te")

    def test_mostly_numbers(self):
        assert _is_garbage("12345678")


class TestPreprocessVariants:
    def test_returns_multiple_variants(self):
        crop = np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)
        variants = _preprocess_variants(crop)
        names = {name for name, _ in variants}
        assert {"clahe_gray", "otsu", "inv_otsu", "adaptive"}.issubset(names)

    def test_grayscale_input(self):
        crop = np.random.randint(0, 255, (50, 200), dtype=np.uint8)
        variants = _preprocess_variants(crop)
        assert len(variants) >= 4

    def test_upscales_small_crops(self):
        crop = np.zeros((20, 100, 3), dtype=np.uint8)
        variants = _preprocess_variants(crop)
        for _, img in variants:
            assert img.shape[0] > 20


class TestCandidateTimestamps:
    def test_windowed_sampling(self):
        assert _candidate_timestamps(10.0, window_seconds=0.5, samples_per_segment=3) == [9.5, 10.0, 10.5]

    def test_single_sample(self):
        assert _candidate_timestamps(10.0, window_seconds=0.5, samples_per_segment=1) == [10.0]

    def test_long_segment_samples_inside_segment(self):
        segment = {"start": 10.0, "end": 20.0, "midpoint": 15.0}
        result = _candidate_timestamps_for_segment(segment, window_seconds=0.5, samples_per_segment=3)
        assert result == [10.9, 15.0, 19.1]

    def test_ocr_effort_choices(self):
        assert OCR_EFFORT_CHOICES == ("fast", "balanced", "accurate")


class TestMergeUniqueTexts:
    def test_deduplicates_and_keeps_order(self):
        assert _merge_unique_texts(["hello", "hello", "world"]) == "hello world"

    def test_ignores_contained_text(self):
        assert _merge_unique_texts(["hello world", "hello"]) == "hello world"


class TestProcessItem:
    def test_returns_crop_metadata_for_best_candidate(self, monkeypatch):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        def fake_ocr_crop(crop, *, language, effort="balanced"):
            return "Hello", 0.9, "otsu/psm6"

        monkeypatch.setattr("burnin_subtitle_checker.ocr._ocr_crop", fake_ocr_crop)
        monkeypatch.setattr("burnin_subtitle_checker.ocr._sharpness", lambda frame: 10.0)

        result = _process_item(
            (1, 5.0, [(5.0, frame)]),
            language="en",
            crop_ratio=0.20,
            dynamic_region=False,
        )

        assert result.subtitle_text == "Hello"
        assert result.crop_region == {"x": 0, "y": 80, "width": 200, "height": 20}
        assert result.ocr_variant == "otsu/psm6"

    def test_transcript_similarity_prefers_matching_middle_frame(self, monkeypatch):
        previous = np.full((100, 200, 3), 10, dtype=np.uint8)
        middle = np.full((100, 200, 3), 80, dtype=np.uint8)
        next_frame = np.full((100, 200, 3), 160, dtype=np.uint8)

        def fake_ocr_crop(crop, *, language, effort="balanced"):
            mean = int(crop.mean())
            if mean < 40:
                return "previous caption noise", 0.95, "otsu/psm6"
            if mean < 120:
                return "hello world", 0.70, "otsu/psm6"
            return "next caption noise", 0.95, "otsu/psm6"

        monkeypatch.setattr("burnin_subtitle_checker.ocr._ocr_crop", fake_ocr_crop)
        monkeypatch.setattr("burnin_subtitle_checker.ocr._sharpness", lambda frame: 10.0)

        result = _process_item(
            (1, 5.0, "hello world", [(4.8, previous), (5.0, middle), (5.2, next_frame)]),
            language="en",
            crop_ratio=0.20,
            dynamic_region=False,
        )

        assert result.subtitle_text == "hello world"
        assert result.status == "ok"


class TestWordConfidenceFloor:
    def test_floor_value(self):
        assert WORD_CONFIDENCE_FLOOR == 20.0
