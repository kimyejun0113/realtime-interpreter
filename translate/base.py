from typing import Iterator

from config import lang_name


class Translator:
    model: str
    name: str  # unique key for quota tracking

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

    # Tuned on real lecture/video logs: subtitle-style rewording, no filler words, no added
    # content, one consistent register, and graceful handling of half-finished sentences.
    system = (
        f"You translate live {lang_name(src)} speech into {lang_name(tgt)} subtitles for someone watching a video.\n"
        f"- Write natural spoken {lang_name(tgt)} the way a professional subtitle translator would, "
        f"not word-for-word. Reorder freely so it reads like {lang_name(tgt)}.\n"
        "- Drop verbal fillers and hedges that add nothing (you know, like, I mean, kind of, sort of, "
        "um, okay so, basically, actually) unless they carry meaning.\n"
        "- Never add information, explanations or answers that are not in the source. "
        "If the speaker asks a question, translate the question.\n"
        "- The input comes from speech recognition: it may lack punctuation, stop mid-sentence or "
        "contain misheard words. Use the context to infer the intended words.\n"
        "- If the input is an unfinished sentence, translate it as a natural unfinished phrase ending "
        'with "…" instead of inventing an ending.\n'
        "- Keep names, numbers and technical terms accurate.\n"
        "- Output only the subtitle text."
    )
    if tgt == "ko":
        system += (
            "\n- Always end sentences in 해요체 (…해요, …거예요, …죠, …있어요), even for lectures and formal talks. "
            "Never use 합니다체 (…합니다, …습니다, …겠습니다)."
        )
    if context:
        lines = "\n".join(f"- {s} => {t}" for s, t in context)
        system += f"\n\nPrevious subtitles (context only, do not repeat them):\n{lines}"
    # a couple of worked examples hold the style (register, filler removal) far better than rules alone
    examples = KO_EXAMPLES if tgt == "ko" else []
    shots = [m for src_text, out in examples for m in ({"role": "user", "content": src_text}, {"role": "assistant", "content": out})]
    return [{"role": "system", "content": system}, *shots, {"role": "user", "content": text}]


KO_EXAMPLES = [
    ("Okay, so, you know, let's just start with some admin and then we'll get into the actual content.",
     "자, 먼저 공지 사항부터 말하고 본론으로 들어갈게요."),
    ("However, I should note that I reserve the right to change the slides, and I often do.",
     "다만 슬라이드는 제가 언제든 바꿀 수 있고, 실제로 자주 바꿔요."),
]
