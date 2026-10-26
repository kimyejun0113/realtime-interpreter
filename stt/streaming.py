"""Local streaming speech recognition (sherpa-onnx zipformer). Free, offline, word-by-word."""
from pathlib import Path

import numpy as np
import sherpa_onnx

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIRS = {
    "en": "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06",
    "ko": "sherpa-onnx-streaming-zipformer-korean-2024-06-16",
}


def model_available(lang: str) -> bool:
    return (MODELS_DIR / MODEL_DIRS.get(lang, "-")).is_dir()


def _find(model_dir: Path, part: str) -> str:
    matches = sorted(model_dir.glob(f"{part}*.int8.onnx")) or sorted(model_dir.glob(f"{part}*.onnx"))
    if not matches:
        raise FileNotFoundError(f"{part} model not found in {model_dir}")
    return str(matches[0])


def tidy(text: str, lang: str) -> str:
    """Some models output upper-case without punctuation; make that readable."""
    text = " ".join(text.split())
    if lang == "en" and text and text.isupper():
        text = text.lower()
        text = " ".join("I" if w == "i" else ("I" + w[1:] if w.startswith("i'") else w) for w in text.split())
        text = text[0].upper() + text[1:]
    return text


class StreamingRecognizer:
    def __init__(self, lang: str, endpoint_silence: float, max_utterance_sec: float):
        model_dir = MODELS_DIR / MODEL_DIRS[lang]
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"음성인식 모델이 없습니다: {model_dir.name}\n'python download_models.py {lang}' 를 실행하세요."
            )
        self.lang = lang
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(model_dir / "tokens.txt"),
            encoder=_find(model_dir, "encoder"),
            decoder=_find(model_dir, "decoder"),
            joiner=_find(model_dir, "joiner"),
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,           # nothing recognised yet
            rule2_min_trailing_silence=endpoint_silence,  # pause after speech -> utterance ends
            rule3_min_utterance_length=max_utterance_sec,
        )
        self.stream = self.recognizer.create_stream()

    def accept(self, pcm: np.ndarray) -> tuple[str, bool]:
        """Feed int16 audio. Returns (current text, utterance ended)."""
        self.stream.accept_waveform(16000, pcm.astype(np.float32) / 32768.0)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        text = tidy(self.recognizer.get_result(self.stream), self.lang)
        ended = self.recognizer.is_endpoint(self.stream)
        if ended:
            self.recognizer.reset(self.stream)
        return text, ended
