import os
from dataclasses import dataclass, field
from typing import Iterator

from config import lang_name


@dataclass
class Context:
    """What the translator knows about the conversation so far, sized to keep prompts small."""

    recent: list[tuple[str, str]] = field(default_factory=list)  # last sentences: (source, translation)
    older: list[str] = field(default_factory=list)               # sentences before those: source only
    memo: str = ""                                               # running notes: topic, people, terms

    def sources(self) -> list[str]:
        return [*self.older, *(s for s, _ in self.recent)]


class Translator:
    model: str
    name: str  # unique key for quota tracking

    def stream(self, text: str, src: str, tgt: str, context: Context, partial: bool = False) -> Iterator[str]:
        raise NotImplementedError


def build_messages(text: str, src: str, tgt: str, context: Context, partial: bool = False) -> list[dict]:
    if partial:
        # short prompt: partial updates are frequent, so every token counts
        system = (
            f"Live interpreter. Translate the {lang_name(src)} speech fragment (still being spoken, "
            f"no punctuation) into natural {lang_name(tgt)}. Output only the translation."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    # Tuned on real lecture/video logs: subtitle-style rewording, no filler words, no added
    # content, one consistent register, and graceful handling of half-finished sentences.
    # every final request pays for this text, so it is kept terse (same rules, ~half the tokens)
    system = (
        f"Translate live {lang_name(src)} speech into {lang_name(tgt)} video subtitles.\n"
        "- Natural, like a professional subtitler; reorder freely, not word-for-word.\n"
        "- Drop empty fillers (you know, like, I mean, kind of, um, basically, actually).\n"
        "- Add nothing that wasn't said; translate questions as questions.\n"
        "- Input is speech recognition: punctuation may be missing, words misheard; infer from context.\n"
        "- Input often stops mid-sentence. Then translate it as a natural unfinished phrase ending "
        'with "…"; never invent an ending.\n'
        "- Keep names, numbers, dates, weekdays and technical terms exactly.\n"
        "- Output only the subtitle."
    )
    style = KO_STYLES.get(KO_STYLE, KO_STYLES["합니다"])
    if tgt == "ko":
        system += "\n- " + style["rule"]
    if context.memo:
        system += f"\n\nConversation notes (keep names/terms consistent):\n{context.memo}"
    if context.older:
        system += "\n\nEarlier in the conversation (context only):\n" + " ".join(context.older)
    if context.recent:
        lines = "\n".join(f"- {s} => {t}" for s, t in context.recent)
        system += f"\n\nPrevious subtitles (context only, do not repeat them):\n{lines}"
    # a couple of worked examples hold the style (register, filler removal) far better than rules alone
    examples = style["examples"] if tgt == "ko" else []
    shots = [m for src_text, out in examples for m in ({"role": "user", "content": src_text}, {"role": "assistant", "content": out})]
    return [{"role": "system", "content": system}, *shots, {"role": "user", "content": text}]


# Korean register for finished subtitles (.env KO_STYLE): 합니다 (default), 해요, 한다
_EX_ADMIN = "Okay, so, you know, let's just start with some admin and then we'll get into the actual content."
_EX_SLIDES = "However, I should note that I reserve the right to change the slides, and I often do."
KO_STYLES = {
    "합니다": {
        "rule": "Always end sentences in 합니다체 (…합니다, …습니다, …겠습니다, …습니까?), never 해요체 or 반말.",
        "examples": [("and the more data we collect without checking the", "그리고 확인도 없이 데이터를 더 많이 모을수록…"),
                     (_EX_ADMIN, "먼저 공지 사항부터 말씀드리고 본론으로 들어가겠습니다."),
                     ("That's not a great habit, right? So what are we really talking about here?",
                      "좋은 습관은 아닙니다. 그렇다면 우리가 정말 이야기하려는 것은 무엇입니까?")],
    },
    "해요": {
        "rule": "Always end sentences in 해요체 (…해요, …거예요, …죠, …있어요), even for lectures and formal talks. "
                "Never use 합니다체 (…합니다, …습니다, …겠습니다).",
        "examples": [(_EX_ADMIN, "자, 먼저 공지 사항부터 말하고 본론으로 들어갈게요."),
                     (_EX_SLIDES, "다만 슬라이드는 제가 언제든 바꿀 수 있고, 실제로 자주 바꿔요.")],
    },
    "한다": {
        "rule": "Always end sentences in plain written style 한다체 (…한다, …이다, …있다, …인가?), "
                "like documentary subtitles. Never use 존댓말 endings (…요, …습니다).",
        "examples": [(_EX_ADMIN, "먼저 공지 사항부터 말하고 본론으로 들어간다."),
                     (_EX_SLIDES, "다만 슬라이드는 언제든 바꿀 수 있고, 실제로 자주 바꾼다.")],
    },
}
KO_STYLE = os.getenv("KO_STYLE", "").strip() or "합니다"
