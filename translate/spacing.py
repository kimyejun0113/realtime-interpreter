"""Korean word spacing for display: the local Korean recogniser writes sentences with no spaces."""
from quota import QuotaTracker

from .openai_compat import OpenAICompatTranslator

PROMPT = ("Insert correct Korean word spacing (띄어쓰기) into the user's text. Do not add, remove or change any "
          "other character, including punctuation. Output only the spaced text.")


def needs_spacing(text: str) -> bool:
    hangul = sum("가" <= ch <= "힣" for ch in text)
    return hangul >= 8 and text.count(" ") < len(text) * 0.06


def space_korean(text: str, models: list[OpenAICompatTranslator], quota: QuotaTracker) -> str | None:
    """Spaced version of `text`, or None. Rejects any answer that changed more than spaces."""
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": text}]
    for model in models:
        if not quota.can_use(model.name, min_tokens=500, keep_day_ratio=0.2):
            continue
        try:
            spaced = " ".join(model.complete(messages, max_tokens=200).split())
        except Exception:
            continue
        if spaced.replace(" ", "") == text.replace(" ", ""):
            return spaced
    return None
