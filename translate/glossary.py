"""User-written glossary: correct spellings (and optional translations) of names and terms.

Unlike the conversation memo, which is built from recognised text and inherits its misspellings,
this file is written by the user before a talk, so it is safe to feed to Whisper as a hint.
"""
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

from wordfreq import zipf_frequency

GLOSSARY_FILE = Path(__file__).resolve().parent.parent / "glossary.txt"
WHISPER_MAX_CHARS = 300   # Whisper's prompt window is small; the rest is left for recent sentences
PROMPT_MAX_TERMS = 40     # every translation request pays for these lines

TEMPLATE = """\
# 용어집: 강의·영상에 나올 이름과 용어를 올바른 철자로 한 줄에 하나씩 적어 주세요.
# "원어 = 번역"으로 적으면 번역도 그대로 맞춰요. 번역을 비워 두면 원어만 힌트로 써요.
# 저장하면 바로 적용돼요. #으로 시작하는 줄은 무시해요.
#
# 예시:
# Tilly = 틸리
# Centillion = 센틸리언
# George Orwell = 조지 오웰
# Kubernetes
"""


def parse(text: str) -> list[tuple[str, str]]:
    """(term, translation or "") pairs, in file order, without duplicates."""
    terms, seen = [], set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        term, _, translation = (p.strip() for p in line.partition("="))
        if term and term.lower() not in seen:
            seen.add(term.lower())
            terms.append((term, translation))
    return terms


class Glossary:
    """Reads glossary.txt, re-reading it whenever the file changes (edits apply without a restart)."""

    def __init__(self, path: Path = GLOSSARY_FILE):
        self.path = path
        self._mtime: float | None = None
        self._terms: list[tuple[str, str]] = []

    @property
    def terms(self) -> list[tuple[str, str]]:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            self._mtime, self._terms = None, []
            return self._terms
        if mtime != self._mtime:
            try:
                self._terms = parse(self.path.read_text(encoding="utf-8-sig"))
                self._mtime = mtime
            except (OSError, UnicodeDecodeError):
                pass  # half-saved file: keep the previous terms and retry next time
        return self._terms

    def whisper_hint(self, lang: str) -> str:
        """Correct spellings written in the speech language, e.g. "Tilly, Centillion, George Orwell."."""
        script = re.compile(r"[가-힣]" if lang == "ko" else r"[A-Za-z]")
        words, size = [], 0
        for term, translation in self.terms:
            # Whisper hears one language, so it gets whichever side of "Tilly = 틸리" is written in it
            word = next((w for w in (term, translation) if script.search(w)), None)
            if word is None:
                continue
            size += len(word) + 2
            if size > WHISPER_MAX_CHARS:
                break
            words.append(word)
        return ", ".join(words) + "." if words else ""

    def prompt_lines(self) -> list[str]:
        return [f"{t} = {tr}" if tr else t for t, tr in self.terms[:PROMPT_MAX_TERMS]]

    def suspects(self, text: str) -> list[tuple[str, str]]:
        """Words in the input spelled like a glossary term but not it ("Killy", "Stillion" for Tilly,
        Centillion). Only pointed out to the translator, which decides from context: told just
        "watch for sound-alikes", it almost never acted on them."""
        latin = [(t, _key(t)) for t, _ in self.terms if len(_key(t)) >= 4]
        if not latin:
            return []
        # (word, capitalised mid-sentence) - the recogniser capitalises what it takes for a name
        words = [(m[0], m[0][0].isupper() and m.start() > 0 and not re.search(r"[.?!]\s*$", text[:m.start()]))
                 for m in re.finditer(r"[A-Za-z']+", text)]
        # ordinary words ("silly", "thrill") must not be flagged: the translator then forces the term
        # into a normal sentence ("it was truly a thrill" -> "정말 틸리였습니다")
        plausible = [not _ordinary(w, named) for w, named in words]
        found = []
        for n in (1, 2):  # a name can come out split in two ("Saint Illin")
            for i in range(len(words) - n + 1):
                part = [w for w, _ in words[i:i + n]]
                if not all(plausible[i:i + n]) or (n == 2 and not all(named for _, named in words[i:i + n])):
                    continue
                heard = " ".join(part)
                key = _key(heard)
                if len(key) < 4 or any(_key(w) == tk for w in part for _, tk in latin):
                    continue
                for term, term_key in latin:
                    if key != term_key and _sounds_like(key, term_key) and (heard, term) not in found:
                        found.append((heard, term))
        return found[:8]


def _ordinary(word: str, named: bool) -> bool:
    """A real English word used as one. Written like a name mid-sentence ("Jenny says Silly"),
    only very common words count, since the recogniser capitalised it for a reason."""
    zipf = zipf_frequency(word.lower().removesuffix("'s"), "en")
    return zipf >= (5.0 if named else 3.0)


def _key(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


def _skeleton(s: str) -> str:
    """Rough sound key: similar consonants merged, vowels dropped after the first letter."""
    for a, b in (("ph", "f"), ("ck", "k"), ("c", "k"), ("q", "k"), ("z", "s"), ("x", "ks"), ("w", "")):
        s = s.replace(a, b)
    s = s[0] + re.sub(r"[aeiouyh]", "", s[1:])
    return re.sub(r"(.)\1+", r"\1", s)


def _sounds_like(heard: str, term: str) -> bool:
    if abs(len(heard) - len(term)) > max(3, len(term) // 2):
        return False
    spelled = SequenceMatcher(None, heard, term).ratio()
    sounded = SequenceMatcher(None, _skeleton(heard), _skeleton(term)).ratio()
    return spelled >= 0.72 or (sounded >= 0.8 and spelled >= 0.5)


def open_in_editor(path: Path = GLOSSARY_FILE) -> None:
    if not path.exists():
        path.write_text(TEMPLATE, encoding="utf-8")
    os.startfile(path)
