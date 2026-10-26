import json
from typing import Iterator

import httpx

from stt.base import RateLimitError

from .base import Translator, build_messages

GROQ_BASE = "https://api.groq.com/openai/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"


class OpenAICompatTranslator(Translator):
    """Streaming translation over any OpenAI-compatible chat endpoint (Groq, Gemini)."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.url = f"{base_url}/chat/completions"
        self.model = model
        self.client = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=20.0)

    def stream(self, text, src, tgt, context) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": build_messages(text, src, tgt, context),
            "temperature": 0.2,
            "max_tokens": 400,
            "stream": True,
        }
        # reasoning models think before answering; keep it minimal for latency
        if self.model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = "low"
        elif self.model.startswith("qwen/qwen3"):
            payload["reasoning_effort"] = "none"
        with self.client.stream("POST", self.url, json=payload) as resp:
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
