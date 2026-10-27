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
    def __init__(self, lang: str, endpoint_silence: float, max_utterance_sec: float, threads: int = 4):
        self.last_lang = lang  # same interface as DualRecognizer
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
            num_threads=threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,           # nothing recognised yet
            rule2_min_trailing_silence=endpoint_silence,  # pause after speech -> utterance ends
            rule3_min_utterance_length=max_utterance_sec,
        )
        self.stream = self.recognizer.create_stream()

    def feed(self, pcm: np.ndarray) -> tuple[str, bool]:
        """Feed int16 audio. Returns (current text, utterance ended); the caller resets."""
        self.stream.accept_waveform(16000, pcm.astype(np.float32) / 32768.0)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        return tidy(self.recognizer.get_result(self.stream), self.lang), self.recognizer.is_endpoint(self.stream)

    def accept(self, pcm: np.ndarray) -> tuple[str, bool]:
        """Feed int16 audio. Returns (current text, utterance ended) and starts afresh after an ending."""
        text, ended = self.feed(pcm)
        if ended:
            self.reset()
        return text, ended

    def reset(self) -> None:
        self.recognizer.reset(self.stream)

    def confidence(self, window: int | None = None) -> tuple[float, int]:
        """(mean log-probability, token count) of the current utterance, or of its last `window` tokens."""
        probs = list(self.recognizer.get_result_all(self.stream).ys_probs)
        if window:
            probs = probs[-window:]
        return (float(np.mean(probs)), len(probs)) if probs else (float("-inf"), 0)

    def text_without_unsure_tail(self, threshold: float = -0.7) -> str:
        """Current text minus trailing low-confidence tokens (another language misheard as this one)."""
        res = self.recognizer.get_result_all(self.stream)
        tokens, probs = list(res.tokens), list(res.ys_probs)
        while probs and probs[-1] < threshold:
            tokens.pop()
            probs.pop()
        return tidy("".join(tokens).replace("▁", " "), self.lang)


class DualRecognizer:
    """Runs the English and Korean recognisers side by side and follows whichever is more confident.

    Measured on test clips, the right language's model is clearly more confident (token log-prob
    ≈ -0.2…-0.6 vs -1.2…-1.8), but not always by much, so decisions need a margin and enough tokens,
    use only the most recent tokens (a long English stretch shouldn't outvote Korean that just
    started), and each new utterance starts in the previous one's language.
    `last_lang` is the language of the text returned by the latest `accept`.
    """

    WINDOW = 10            # judge on the latest tokens only
    SWITCH_MARGIN = 0.4    # mid-utterance switch: the other side must be this much more confident...
    MIN_TOKENS = 6         # ...over at least this many recent tokens
    ENDPOINT_MARGIN = 0.2  # when either side ends a sentence, a smaller lead decides whose sentence it is
    ENDPOINT_MIN_TOKENS = 4
    WEAK_CONF = -1.2       # a few low-confidence tokens are noise, not a sentence
    CLEAR_CONF = -0.8      # a short sentence is trusted alone only above this
    POOR_CONF = -0.9       # the followed side scoring below this is probably the wrong language...
    POOR_MARGIN = 0.1      # ...so this lead is enough to switch

    def __init__(self, recognizers: dict[str, StreamingRecognizer], initial: str = "en"):
        self.recognizers = recognizers
        self.lang = initial if initial in recognizers else next(iter(recognizers))
        self.last_lang = self.lang

    def accept(self, pcm: np.ndarray) -> tuple[str, bool]:
        # The two models cut sentences at different moments, so every ending is arbitrated:
        # the side that ends keeps the sentence only if it is (about) as confident as the other.
        results = {lang: r.feed(pcm) for lang, r in self.recognizers.items()}
        self.last_lang = self.lang

        switched = self._maybe_switch()
        if switched is not None:
            return switched

        for lang, (text, ended) in results.items():
            if not ended or lang == self.lang:
                continue
            if text and (self._leads(lang, self.lang, self.ENDPOINT_MARGIN, self.ENDPOINT_MIN_TOKENS)
                         or self._hears_nothing_better(lang)):
                self.lang = self.last_lang = lang  # the other language finished a sentence we were mishearing
                return self._finish(text)
            self.recognizers[lang].reset()  # a sentence in the wrong language: drop it

        text, ended = results[self.lang]
        if not ended:
            return text, False
        conf, n = self.recognizers[self.lang].confidence()
        if not text.strip() or (n < self.ENDPOINT_MIN_TOKENS and conf < self.WEAK_CONF):
            # nothing real (silence, or the onset of the other language misheard as a word):
            # restart only this side, so the other keeps what it has heard
            self.recognizers[self.lang].reset()
            return "", False
        for lang, (other_text, other_ended) in results.items():
            if lang != self.lang and not other_ended and                     self._leads(lang, self.lang, self.ENDPOINT_MARGIN, self.ENDPOINT_MIN_TOKENS):
                # we were following the wrong language and it cut early: keep listening in the other
                self.recognizers[self.lang].reset()
                self.lang = self.last_lang = lang
                return other_text, False
        return self._finish(text)

    def _finish(self, text: str) -> tuple[str, bool]:
        for r in self.recognizers.values():
            r.reset()
        return text, True

    def _leads(self, lang: str, over: str, margin: float, min_tokens: int) -> bool:
        conf, n = self.recognizers[lang].confidence(self.WINDOW)
        over_conf = self.recognizers[over].confidence(self.WINDOW)[0]
        if over_conf < self.POOR_CONF:
            # the side we follow is struggling (the right model scores > -0.6 on its own language,
            # the wrong one < -1.0), so a small lead is enough
            margin = min(margin, self.POOR_MARGIN)
        # must be decent on its own too: right after a reset the other side has no tokens (-inf),
        # which would make any noise look like a clear winner
        return n >= min_tokens and conf >= self.WEAK_CONF and conf - over_conf >= margin

    def _hears_nothing_better(self, lang: str) -> bool:
        """A short but clear sentence while the active side has picked up next to nothing (e.g. "지하철")."""
        conf, n = self.recognizers[lang].confidence()
        _, active_n = self.recognizers[self.lang].confidence(self.WINDOW)
        return active_n < self.ENDPOINT_MIN_TOKENS and n >= 2 and conf >= self.CLEAR_CONF

    def _maybe_switch(self) -> tuple[str, bool] | None:
        """Language changed mid-utterance: close what was said so far as its own sentence."""
        for lang in self.recognizers:
            if lang == self.lang or not self._leads(lang, self.lang, self.SWITCH_MARGIN, self.MIN_TOKENS):
                continue
            old = self.recognizers[self.lang]
            conf, n = old.confidence()
            # keep what was said only if it really was the old language (not noise misheard as it)
            said = old.text_without_unsure_tail() if n >= self.ENDPOINT_MIN_TOKENS and conf >= self.CLEAR_CONF else ""
            old.reset()
            self.lang = lang
            if said:
                return said, True  # last_lang still names the old language for this sentence
            self.last_lang = lang
            return None
        return None
