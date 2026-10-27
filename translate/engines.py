"""Builds the ordered list of final-translation engines from TRANSLATE_ENGINES in .env.

Each entry is "provider:model" (LLM over an OpenAI-compatible API) or a bare service name
("deepl", "azure"). Entries whose API key is missing are skipped, so the same list works
whether or not you have signed up for a service. The local NLLB model is always the last resort.
"""
import os

from quota import QuotaTracker

from .base import Translator
from .mt_services import AzureTranslator, DeepLTranslator
from .openai_compat import OpenAICompatTranslator

# provider -> (base URL, env var holding the key)
LLM_PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "github": ("https://models.github.ai/inference", "GITHUB_MODELS_TOKEN"),
    "cerebras": ("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
}


def _key(var: str) -> str:
    return os.getenv(var, "").strip()


def parse(spec: str) -> list[str]:
    return [e.strip() for e in spec.split(",") if e.strip()]


def build(spec: str, quota: QuotaTracker) -> tuple[list[Translator], list[str]]:
    """Returns (engines ready to use, entries skipped for lack of a key or an unknown name)."""
    engines: list[Translator] = []
    skipped: list[str] = []
    for entry in parse(spec):
        name, _, model = entry.partition(":")
        name = name.lower()
        if name == "deepl" and _key("DEEPL_API_KEY"):
            engines.append(DeepLTranslator(_key("DEEPL_API_KEY"), quota))
        elif name == "azure" and _key("AZURE_TRANSLATOR_KEY"):
            engines.append(AzureTranslator(_key("AZURE_TRANSLATOR_KEY"), _key("AZURE_TRANSLATOR_REGION"), quota))
        elif name in LLM_PROVIDERS and model and _key(LLM_PROVIDERS[name][1]):
            base, var = LLM_PROVIDERS[name]
            extra = {"X-Title": "realtime-interpreter"} if name == "openrouter" else None
            engines.append(OpenAICompatTranslator(base, _key(var), model, quota, provider=name, extra_headers=extra))
        else:
            skipped.append(entry)
    return engines, skipped


def label(engine: Translator) -> str:
    """Short name for the quota readout, e.g. "qwen3.8-27b", "deepl"."""
    return getattr(engine, "model", "?").rsplit("/", 1)[-1]
