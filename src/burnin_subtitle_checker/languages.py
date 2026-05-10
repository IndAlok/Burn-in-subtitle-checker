from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    name: str
    whisper_code: str
    tesseract_code: str
    unicode_ranges: tuple[tuple[int, int], ...]
    default_model: str
    initial_prompt: str | None = None


LANGUAGES: dict[str, LanguageSpec] = {
    "en": LanguageSpec(
        code="en",
        name="English",
        whisper_code="en",
        tesseract_code="eng",
        unicode_ranges=((0x0041, 0x005A), (0x0061, 0x007A)),
        default_model="small",
    ),
    "hi": LanguageSpec(
        code="hi",
        name="Hindi",
        whisper_code="hi",
        tesseract_code="hin",
        unicode_ranges=((0x0900, 0x097F),),
        default_model="medium",
        initial_prompt="नमस्ते, यह हिंदी में बातचीत है। कृपया देवनागरी हिंदी लिपि में ही लिखें।",
    ),
    "kn": LanguageSpec(
        code="kn",
        name="Kannada",
        whisper_code="kn",
        tesseract_code="kan",
        unicode_ranges=((0x0C80, 0x0CFF),),
        default_model="medium",
        initial_prompt="ನಮಸ್ಕಾರ, ಇದು ಕನ್ನಡದಲ್ಲಿನ ಸಂಭಾಷಣೆ. ದಯವಿಟ್ಟು ಕನ್ನಡ ಲಿಪಿಯಲ್ಲೇ ಬರೆಯಿರಿ.",
    ),
    "ta": LanguageSpec(
        code="ta",
        name="Tamil",
        whisper_code="ta",
        tesseract_code="tam",
        unicode_ranges=((0x0B80, 0x0BFF),),
        default_model="medium",
        initial_prompt="வணக்கம், இது தமிழில் பேசப்படும் உரை. தயவுசெய்து தமிழ் எழுத்தில் மட்டும் எழுதுங்கள்.",
    ),
    "te": LanguageSpec(
        code="te",
        name="Telugu",
        whisper_code="te",
        tesseract_code="tel",
        unicode_ranges=((0x0C00, 0x0C7F),),
        default_model="medium",
        initial_prompt="నమస్తే, ఇది తెలుగులో మాట్లాడిన సంభాషణ. దయచేసి తెలుగు లిపిలోనే వ్రాయండి.",
    ),
}

ASR_LANGUAGE_CHOICES = ("auto", "en", "hi", "kn", "ta", "te")
OCR_LANGUAGE_CHOICES = ("en", "hi", "kn", "ta", "te", "multi")


def language_spec(code: str) -> LanguageSpec:
    try:
        return LANGUAGES[code]
    except KeyError as exc:
        raise ValueError(f"Unsupported language '{code}'. Use one of: {sorted(LANGUAGES)}") from exc


def tesseract_language(code: str) -> str:
    if code == "multi":
        return "+".join(spec.tesseract_code for spec in LANGUAGES.values())
    return language_spec(code).tesseract_code


def script_char_ratio(text: str, code: str) -> float:
    if not text:
        return 0.0
    if code == "multi":
        ranges = tuple(r for spec in LANGUAGES.values() for r in spec.unicode_ranges)
    else:
        ranges = language_spec(code).unicode_ranges

    script_count = 0
    alpha_count = 0
    for char in text:
        if not char.isalpha():
            continue
        alpha_count += 1
        point = ord(char)
        if any(start <= point <= end for start, end in ranges):
            script_count += 1
    if alpha_count == 0:
        return 0.0
    return script_count / alpha_count
