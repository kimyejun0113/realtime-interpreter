import io
import wave
from dataclasses import dataclass

import numpy as np


class RateLimitError(Exception):
    """Provider quota exhausted (HTTP 429)."""


@dataclass
class Transcript:
    text: str
    lang: str  # ISO 639-1 code, e.g. "en", "ko"


# Whisper often invents these on silence/noise
HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thank you for watching.", "you",
    "시청해주셔서 감사합니다.", "구독과 좋아요 부탁드립니다.", "감사합니다.",
    "mbc 뉴스 이덕영입니다.",
}

WHISPER_LANG_CODES = {
    "english": "en", "korean": "ko", "japanese": "ja", "chinese": "zh",
    "spanish": "es", "french": "fr", "german": "de", "russian": "ru",
    "vietnamese": "vi", "thai": "th", "indonesian": "id", "portuguese": "pt",
    "italian": "it", "arabic": "ar", "hindi": "hi",
}


def normalize_lang(lang: str) -> str:
    lang = (lang or "").strip().lower()
    return WHISPER_LANG_CODES.get(lang, lang[:2] if len(lang) > 2 else lang)


def is_hallucination(text: str) -> bool:
    text = text.strip().lower()
    # punctuation-only output ("." / "...") comes from noise
    return text in HALLUCINATIONS or not any(ch.isalnum() for ch in text)


def pcm_to_wav(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


class STT:
    def transcribe(self, pcm: np.ndarray, prompt: str = "") -> Transcript | None:
        raise NotImplementedError
