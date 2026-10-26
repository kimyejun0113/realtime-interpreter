import numpy as np

from .base import STT, Transcript, is_hallucination


class LocalSTT(STT):
    """CPU fallback via faster-whisper. Slower, but free and offline."""

    def __init__(self, model_size: str):
        from faster_whisper import WhisperModel  # optional dependency

        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, pcm: np.ndarray, prompt: str = "") -> Transcript | None:
        audio = pcm.astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(
            audio, beam_size=1, initial_prompt=prompt or None, vad_filter=False
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if not text or is_hallucination(text):
            return None
        return Transcript(text=text, lang=info.language)
