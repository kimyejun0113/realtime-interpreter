"""Running conversation notes (topic, people, term translations) that keep long talks consistent.

Rather than sending many earlier sentences with every translation, a cheap model condenses them
into a few lines every MEMO_EVERY sentences, in the background, and those lines ride along with
each translation request (~80 tokens).
"""
import threading

from config import lang_name
from quota import QuotaTracker

from .openai_compat import OpenAICompatTranslator

MEMO_EVERY = 8      # update the notes after this many finished sentences
MEMO_MAX_CHARS = 500

PROMPT = """You keep short running notes for an interpreter translating a live {src} talk into {tgt} subtitles.
Update the notes with the new lines. Keep only what helps translate what comes next:
주제: what the talk is about and the setting (lecture, meeting, interview, video…), one line
인물: people/roles named so far and who "he/she/they/we" usually refers to
용어: recurring names and technical terms with the {tgt} rendering to keep using (term = 번역), at most 10
At most 6 short lines in exactly that format. Drop what is no longer relevant. Output only the notes."""


class MemoKeeper:
    def __init__(self, models: list[OpenAICompatTranslator], quota: QuotaTracker, on_update=lambda memo: None):
        self.models = models
        self.quota = quota
        self.on_update = on_update
        self.memo = ""
        self._pending: list[tuple[str, str]] = []
        self._langs = ("en", "ko")
        self._busy = threading.Lock()

    def reset(self) -> None:
        self.memo = ""
        self._pending.clear()

    def add(self, source: str, translation: str, src: str, tgt: str) -> None:
        self._pending.append((source, translation))
        self._pending = self._pending[-2 * MEMO_EVERY:]  # if updates keep failing, don't grow forever
        self._langs = (src, tgt)
        if len(self._pending) >= MEMO_EVERY and not self._busy.locked():
            threading.Thread(target=self._update, daemon=True).start()

    def _update(self) -> None:
        with self._busy:
            batch = list(self._pending)
            src, tgt = self._langs
            lines = "\n".join(f"- {s} => {t}" for s, t in batch)
            messages = [
                {"role": "system", "content": PROMPT.format(src=lang_name(src), tgt=lang_name(tgt))},
                {"role": "user", "content": f"Current notes:\n{self.memo or '(none yet)'}\n\nNew lines:\n{lines}"},
            ]
            for model in self.models:
                if not self.quota.can_use(model.name, min_tokens=1000, keep_day_ratio=0.2):
                    continue
                try:
                    memo = model.complete(messages, max_tokens=250)
                except Exception:
                    continue
                if memo:
                    self.memo = memo[:MEMO_MAX_CHARS]
                    del self._pending[: len(batch)]
                    self.on_update(self.memo)
                return
