import pytest

from burnin_subtitle_checker.languages import (
    ASR_LANGUAGE_CHOICES,
    OCR_LANGUAGE_CHOICES,
    LANGUAGES,
    language_spec,
    script_char_ratio,
    tesseract_language,
)


def test_supported_language_registry_is_complete():
    assert set(LANGUAGES) == {"en", "hi", "kn", "ta", "te"}
    assert ASR_LANGUAGE_CHOICES == ("auto", "en", "hi", "kn", "ta", "te")
    assert OCR_LANGUAGE_CHOICES == ("en", "hi", "kn", "ta", "te", "multi")


def test_tesseract_language_mapping():
    assert tesseract_language("en") == "eng"
    assert tesseract_language("hi") == "hin"
    assert tesseract_language("kn") == "kan"
    assert tesseract_language("ta") == "tam"
    assert tesseract_language("te") == "tel"
    assert tesseract_language("multi") == "eng+hin+kan+tam+tel"


@pytest.mark.parametrize(
    ("code", "text"),
    [
        ("en", "This is English"),
        ("hi", "यह हिंदी है"),
        ("kn", "ಇದು ಕನ್ನಡ"),
        ("ta", "இது தமிழ்"),
        ("te", "ఇది తెలుగు"),
    ],
)
def test_script_char_ratio_accepts_supported_scripts(code, text):
    assert script_char_ratio(text, code) > 0.8


def test_language_spec_rejects_unknown_code():
    with pytest.raises(ValueError):
        language_spec("mr")
