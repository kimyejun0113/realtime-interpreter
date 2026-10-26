import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


@dataclass
class Config:
    groq_api_key: str = _env("GROQ_API_KEY", "")
    stt_model: str = _env("STT_MODEL", "whisper-large-v3-turbo")

    translate_provider: str = _env("TRANSLATE_PROVIDER", "groq").lower()
    translate_model: str = _env("TRANSLATE_MODEL", "openai/gpt-oss-120b")
    gemini_api_key: str = _env("GEMINI_API_KEY", "")
    gemini_model: str = _env("GEMINI_MODEL", "gemini-2.5-flash")

    main_target_lang: str = _env("MAIN_TARGET_LANG", "ko")
    other_lang: str = _env("OTHER_LANG", "en")

    sample_rate: int = 16000
    frame_ms: int = 30
    vad_silence_ms: int = int(_env("VAD_SILENCE_MS", "400"))
    max_segment_sec: float = float(_env("MAX_SEGMENT_SEC", "5"))
    vad_aggressiveness: int = int(_env("VAD_AGGRESSIVENESS", "2"))

    local_stt_fallback: bool = _env("LOCAL_STT_FALLBACK", "0") == "1"
    local_stt_model: str = _env("LOCAL_STT_MODEL", "small")

    save_audio: bool = _env("SAVE_AUDIO", "0") == "1"


LANG_NAMES = {
    "ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese",
    "es": "Spanish", "fr": "French", "de": "German", "ru": "Russian",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "pt": "Portuguese",
    "it": "Italian", "ar": "Arabic", "hi": "Hindi",
}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)
