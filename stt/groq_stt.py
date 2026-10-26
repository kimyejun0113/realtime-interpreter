import httpx
import numpy as np

from quota import QuotaTracker

from .base import STT, RateLimitError, Transcript, is_hallucination, normalize_lang, pcm_to_wav

URL = "https://api.groq.com/openai/v1/audio/transcriptions"
NO_SPEECH_THRESHOLD = 0.6


class GroqSTT(STT):
    def __init__(self, api_key: str, model: str, quota: QuotaTracker | None = None):
        self.model = model
        self.quota = quota
        # one keep-alive client so each request skips the TLS handshake
        self.client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=15.0)

    def transcribe(self, pcm: np.ndarray, prompt: str = "", language: str = "") -> Transcript | None:
        data = {"model": self.model, "response_format": "verbose_json", "temperature": "0"}
        if prompt:
            data["prompt"] = prompt[-200:]
        if language:
            data["language"] = language
        if self.quota:
            self.quota.record_request(self.model)
        resp = self.client.post(URL, data=data, files={"file": ("audio.wav", pcm_to_wav(pcm), "audio/wav")})
        if self.quota:
            self.quota.update(self.model, resp)
        if resp.status_code == 429:
            raise RateLimitError(resp.text)
        resp.raise_for_status()
        body = resp.json()

        segments = body.get("segments") or []
        if segments and all(s.get("no_speech_prob", 0) > NO_SPEECH_THRESHOLD for s in segments):
            return None
        text = (body.get("text") or "").strip()
        if not text or is_hallucination(text):
            return None
        return Transcript(text=text, lang=normalize_lang(body.get("language", "")) or language)
