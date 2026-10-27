import json
from typing import Iterator

import httpx

from quota import QuotaTracker
from stt.base import RateLimitError

from .base import Translator, build_messages

GROQ_BASE = "https://api.groq.com/openai/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
OPENAI_BASE = "https://api.openai.com/v1"


class OpenAICompatTranslator(Translator):
    """Streaming translation over any OpenAI-compatible chat endpoint (Groq, Gemini, OpenAI, …)."""

    def __init__(self, base_url: str, api_key: str, model: str, quota: QuotaTracker | None = None,
                 provider: str = "groq", extra_headers: dict | None = None):
        self.url = f"{base_url}/chat/completions"
        self.model = model
        self.provider = provider
        # quota bookkeeping key; Groq keeps the bare model name so its RPM table applies
        self.name = model if provider == "groq" else f"{provider}:{model}"
        self.quota = quota
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}", **(extra_headers or {})}, timeout=20.0
        )

    def complete(self, messages: list[dict], max_tokens: int = 250) -> str:
        """One non-streamed answer (used for the conversation memo)."""
        payload = self._tune({"model": self.model, "messages": messages, "temperature": 0.1, "max_tokens": max_tokens})
        if self.quota:
            self.quota.record_request(self.name)
        resp = self.client.post(self.url, json=payload)
        if self.quota:
            self.quota.update(self.name, resp)
        if resp.status_code == 429:
            raise RateLimitError(resp.text)
        resp.raise_for_status()
        return (resp.json()["choices"][0]["message"].get("content") or "").strip()

    def stream(self, text, src, tgt, context, partial=False) -> Iterator[str]:
        payload = self._tune({
            "model": self.model,
            "messages": build_messages(text, src, tgt, context, partial),
            "temperature": 0.1,
            "max_tokens": 300 if partial else 500,
            "stream": True,
        })
        if self.quota:
            self.quota.record_request(self.name)
        with self.client.stream("POST", self.url, json=payload) as resp:
            if self.quota:
                self.quota.update(self.name, resp)
            if resp.status_code == 429:
                raise RateLimitError(resp.read().decode(errors="replace"))
            if resp.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"{resp.status_code}: {resp.read().decode(errors='replace')[:300]}",
                    request=resp.request, response=resp,
                )
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        yield delta

    def _tune(self, payload: dict) -> dict:
        model = self.model.rsplit("/", 1)[-1]
        # reasoning models think before answering; keep it minimal for latency and tokens
        if model.startswith("gpt-oss"):
            payload["reasoning_effort"] = "low"
        elif model.startswith("qwen3") and self.provider == "groq":
            payload["reasoning_effort"] = "none"
        if self.provider == "openai" and model.startswith(("gpt-5", "o1", "o3", "o4")):
            # OpenAI reasoning models: no custom temperature, token cap has a different name
            payload.pop("temperature")
            payload["max_completion_tokens"] = payload.pop("max_tokens") + 1000
            payload["reasoning_effort"] = "minimal" if model.startswith("gpt-5") else "low"
        return payload
