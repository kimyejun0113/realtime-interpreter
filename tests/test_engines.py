import json

import httpx
import pytest

from quota import QuotaTracker
from stt.base import RateLimitError
from translate import engines
from translate.base import Context
from translate.mt_services import AzureTranslator, DeepLTranslator
from translate.openai_compat import OpenAICompatTranslator

ALL_KEYS = ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "GITHUB_MODELS_TOKEN", "CEREBRAS_API_KEY",
            "MISTRAL_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "DEEPL_API_KEY", "AZURE_TRANSLATOR_KEY"]


@pytest.fixture
def no_keys(monkeypatch):
    for k in ALL_KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_engines_without_keys_are_skipped_in_order(no_keys):
    no_keys.setenv("GROQ_API_KEY", "g")
    no_keys.setenv("DEEPL_API_KEY", "d:fx")
    built, skipped = engines.build("openai:gpt-4o-mini, groq:qwen/qwen3.8-27b, deepl, azure, bogus:x, groq:openai/gpt-oss-20b",
                                   QuotaTracker({}))
    assert [e.name for e in built] == ["qwen/qwen3.8-27b", "deepl", "openai/gpt-oss-20b"]
    assert skipped == ["openai:gpt-4o-mini", "azure", "bogus:x"]


def test_quota_names_are_unique_per_provider(no_keys):
    no_keys.setenv("GROQ_API_KEY", "g")
    no_keys.setenv("CEREBRAS_API_KEY", "c")
    built, _ = engines.build("groq:openai/gpt-oss-120b, cerebras:gpt-oss-120b", QuotaTracker({}))
    assert [e.name for e in built] == ["openai/gpt-oss-120b", "cerebras:gpt-oss-120b"]


def tuned(model, provider="groq"):
    t = OpenAICompatTranslator("https://x", "k", model, provider=provider)
    return t._tune({"model": model, "temperature": 0.1, "max_tokens": 500})


def test_tune_reasoning_settings():
    assert tuned("openai/gpt-oss-120b")["reasoning_effort"] == "low"
    assert tuned("qwen/qwen3.8-27b")["reasoning_effort"] == "none"
    assert "reasoning_effort" not in tuned("qwen-3-32b", provider="cerebras")
    plain = tuned("gpt-4o-mini", provider="openai")
    assert plain["temperature"] == 0.1 and plain["max_tokens"] == 500 and "reasoning_effort" not in plain
    r = tuned("gpt-5-nano", provider="openai")
    assert "temperature" not in r and "max_tokens" not in r
    assert r["max_completion_tokens"] == 1500 and r["reasoning_effort"] == "minimal"


def mocked(translator, handler):
    translator.client = httpx.Client(transport=httpx.MockTransport(handler), headers=translator.client.headers)
    return translator


def test_openai_compat_streams_sse_and_raises_on_429():
    sse = ('data: {"choices":[{"delta":{"content":"안녕"}}]}\n\n'
           'data: {"choices":[{"delta":{"content":"하세요"}}]}\n\ndata: [DONE]\n\n')
    t = mocked(OpenAICompatTranslator("https://x", "k", "m"), lambda r: httpx.Response(200, text=sse))
    assert "".join(t.stream("Hello", "en", "ko", Context())) == "안녕하세요"
    t = mocked(t, lambda r: httpx.Response(429, text="slow down", headers={"retry-after": "7"}))
    with pytest.raises(RateLimitError):
        list(t.stream("Hello", "en", "ko", Context()))


def test_openai_compat_complete():
    t = mocked(OpenAICompatTranslator("https://x", "k", "m"),
               lambda r: httpx.Response(200, json={"choices": [{"message": {"content": " 메모 "}}]}))
    assert t.complete([{"role": "user", "content": "x"}]) == "메모"


def test_deepl_request_format_and_quota_error():
    seen = {}

    def handler(req):
        seen.update(url=str(req.url), auth=req.headers["authorization"], body=json.loads(req.content))
        return httpx.Response(200, json={"translations": [{"text": "안녕하세요"}]})

    d = mocked(DeepLTranslator("abc:fx"), handler)
    out = list(d.stream("Hello", "en", "ko", Context(recent=[("Earlier.", "앞")], older=["Before."])))
    assert out == ["안녕하세요"]
    assert seen["url"] == "https://api-free.deepl.com/v2/translate"
    assert seen["auth"] == "DeepL-Auth-Key abc:fx"
    assert seen["body"] == {"text": ["Hello"], "source_lang": "EN", "target_lang": "KO", "context": "Before. Earlier."}
    d = mocked(d, lambda r: httpx.Response(456, text="Quota exceeded"))
    with pytest.raises(RateLimitError):
        list(d.stream("x", "en", "ko", Context()))


def test_deepl_pro_key_uses_pro_host():
    assert DeepLTranslator("abc").url == "https://api.deepl.com/v2/translate"


def test_azure_request_format():
    seen = {}

    def handler(req):
        seen.update(url=str(req.url), region=req.headers.get("ocp-apim-subscription-region"))
        return httpx.Response(200, json=[{"translations": [{"text": "안녕"}]}])

    a = mocked(AzureTranslator("k", "koreacentral"), handler)
    assert list(a.stream("Hi", "en", "ko", Context())) == ["안녕"]
    assert "from=en" in seen["url"] and "to=ko" in seen["url"] and seen["region"] == "koreacentral"
