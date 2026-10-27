"""Dedicated machine-translation APIs (not LLMs): fast, generous free tiers, no prompt/tokens.

Free tiers (check the provider's site, they change): DeepL API Free 500k chars/month,
Azure Translator F0 2M chars/month. Roughly 50k English characters per hour of speech.
"""
from typing import Iterator

import httpx

from quota import QuotaTracker
from stt.base import RateLimitError

from .base import Translator

DEEPL_LANGS = {"en": "EN", "ko": "KO", "ja": "JA", "zh": "ZH", "es": "ES", "fr": "FR", "de": "DE",
               "ru": "RU", "id": "ID", "pt": "PT-BR", "it": "IT"}


def _check(resp: httpx.Response) -> None:
    if resp.status_code in (429, 456):  # 456: DeepL monthly quota used up
        raise RateLimitError(resp.text)
    resp.raise_for_status()


class DeepLTranslator(Translator):
    def __init__(self, api_key: str, quota: QuotaTracker | None = None):
        # free keys end in ":fx" and use a separate host
        host = "api-free.deepl.com" if api_key.endswith(":fx") else "api.deepl.com"
        self.url = f"https://{host}/v2/translate"
        self.model = self.name = "deepl"
        self.quota = quota
        self.client = httpx.Client(headers={"Authorization": f"DeepL-Auth-Key {api_key}"}, timeout=15.0)

    def stream(self, text, src, tgt, context, partial=False) -> Iterator[str]:
        payload = {
            "text": [text],
            "source_lang": DEEPL_LANGS.get(src, "EN").split("-")[0],
            "target_lang": DEEPL_LANGS.get(tgt, "KO"),
        }
        if context:  # earlier sentences help pronouns and terms; not translated or billed
            payload["context"] = " ".join(s for s, _ in context)
        resp = self.client.post(self.url, json=payload)
        if self.quota:
            self.quota.update(self.name, resp)
        _check(resp)
        yield resp.json()["translations"][0]["text"]


class AzureTranslator(Translator):
    URL = "https://api.cognitive.microsofttranslator.com/translate"

    def __init__(self, api_key: str, region: str, quota: QuotaTracker | None = None):
        self.model = self.name = "azure"
        self.quota = quota
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        if region:
            headers["Ocp-Apim-Subscription-Region"] = region
        self.client = httpx.Client(headers=headers, timeout=15.0)

    def stream(self, text, src, tgt, context, partial=False) -> Iterator[str]:
        resp = self.client.post(
            self.URL, params={"api-version": "3.0", "from": src, "to": tgt}, json=[{"Text": text}]
        )
        if self.quota:
            self.quota.update(self.name, resp)
        _check(resp)
        yield resp.json()[0]["translations"][0]["text"]
