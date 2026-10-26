from typing import Iterator

from config import lang_name


class Translator:
    def stream(self, text: str, src: str, tgt: str, context: list[tuple[str, str]]) -> Iterator[str]:
        raise NotImplementedError


def build_messages(text: str, src: str, tgt: str, context: list[tuple[str, str]]) -> list[dict]:
    system = (
        f"You are a professional simultaneous interpreter translating {lang_name(src)} speech "
        f"into natural, spoken {lang_name(tgt)}.\n"
        "- The input is a speech-recognition fragment; it may be an incomplete sentence or contain "
        "recognition errors. Translate what is meant, smoothly, without adding anything.\n"
        "- Output ONLY the translation. No quotes, notes, romanization, or explanations.\n"
        "- Keep proper nouns, product names and numbers accurate."
    )
    if tgt == "ko":
        system += "\n- Use natural conversational Korean (해요체 unless the speaker is clearly formal)."
    if context:
        lines = "\n".join(f"- {s} => {t}" for s, t in context)
        system += f"\n\nRecent conversation (for context only, do not translate again):\n{lines}"
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]
