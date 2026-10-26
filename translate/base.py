from typing import Iterator

from config import lang_name


class Translator:
    model: str

    def stream(self, text: str, src: str, tgt: str, context: list[tuple[str, str]], partial: bool = False) -> Iterator[str]:
        raise NotImplementedError


def build_messages(text: str, src: str, tgt: str, context: list[tuple[str, str]], partial: bool = False) -> list[dict]:
    if partial:
        # short prompt: partial updates are frequent, so every token counts
        system = (
            f"Live interpreter. Translate the {lang_name(src)} speech fragment (still being spoken, "
            f"no punctuation) into natural {lang_name(tgt)}. Output only the translation."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    system = (
        f"You are a professional simultaneous interpreter translating {lang_name(src)} speech "
        f"into natural, spoken {lang_name(tgt)}.\n"
        "- The input is a speech-recognition transcript; it may lack punctuation or contain "
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
