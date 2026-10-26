import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


@dataclass
class Config:
    groq_api_key: str = _env("GROQ_API_KEY", "")

    # language being spoken: en (-> Korean) or ko (-> English); recognised locally, word by word
    input_lang: str = _env("INPUT_LANG", "en")
    # groq: polish each finished sentence with Whisper (punctuation, accuracy) while quota allows
    # local: never call an STT API
    final_stt: str = _env("FINAL_STT", "groq").lower()
    stt_model: str = _env("STT_MODEL", "whisper-large-v3-turbo")

    # finished sentences: Groq quality model (second model on Groq = separate quota, used as fallback)
    translate_model: str = _env("TRANSLATE_MODEL", "openai/gpt-oss-120b")
    partial_model: str = _env("PARTIAL_MODEL", "openai/gpt-oss-20b")
    # live (partial) re-translation while still speaking.
    # local: NLLB on this PC, unlimited. groq: PARTIAL_MODEL, uses quota
    live_translation: bool = _env("LIVE_TRANSLATION", "1") == "1"
    partial_engine: str = _env("PARTIAL_ENGINE", "local").lower()
    partial_interval_sec: float = float(_env("PARTIAL_INTERVAL_SEC", "0.6" if partial_engine == "local" else "2.0"))
    partial_min_new_words: int = int(_env("PARTIAL_MIN_NEW_WORDS", "1" if partial_engine == "local" else "4"))

    # optional extra fallback when Groq is rate-limited
    gemini_api_key: str = _env("GEMINI_API_KEY", "")
    gemini_model: str = _env("GEMINI_MODEL", "gemini-2.5-flash")

    main_target_lang: str = _env("MAIN_TARGET_LANG", "ko")
    other_lang: str = _env("OTHER_LANG", "en")

    sample_rate: int = 16000
    frame_ms: int = 100
    endpoint_silence_sec: float = float(_env("ENDPOINT_SILENCE_SEC", "0.6"))
    max_utterance_sec: float = float(_env("MAX_UTTERANCE_SEC", "30"))

    # Groq free tier requests/minute (not reported in headers)
    rpm_limits: dict = field(default_factory=lambda: {
        "whisper-large-v3-turbo": 20,
        "whisper-large-v3": 20,
        "openai/gpt-oss-120b": 30,
        "openai/gpt-oss-20b": 30,
    })

    save_audio: bool = _env("SAVE_AUDIO", "0") == "1"


LANG_NAMES = {
    "ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese",
    "es": "Spanish", "fr": "French", "de": "German", "ru": "Russian",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "pt": "Portuguese",
    "it": "Italian", "ar": "Arabic", "hi": "Hindi",
}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)
