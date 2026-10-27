import json
import queue
from concurrent.futures import ThreadPoolExecutor
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from audio.capture import (
    APP_PREFIX, FILE_PREFIX, SYSTEM_AUDIO, AppAudioCapture, FileCapture, MicCapture, SystemAudioCapture,
)
from config import Config
from quota import QuotaTracker
from stt.base import RateLimitError, pcm_to_wav
from stt.groq_stt import GroqSTT
from stt.streaming import DualRecognizer, StreamingRecognizer
from translate import local_mt
from translate.base import Translator
from translate import engines, spacing
from translate.base import Context
from translate.memo import MemoKeeper
from translate.openai_compat import GROQ_BASE, OpenAICompatTranslator

LOG_DIR = Path(__file__).with_name("logs")
AUTO = "auto"            # input language mode: detect English/Korean per sentence
AUTO_LANGS = ("en", "ko")


class Events:
    """Callbacks the UI subscribes to. Called from worker threads."""

    def __init__(self):
        self.on_live: Callable[[int, str], None] = lambda seq, text: None             # words as they are heard
        self.on_partial: Callable[[int, str], None] = lambda seq, text: None          # tentative translation
        self.on_source: Callable[[int, str, str], None] = lambda seq, text, lang: None  # final source text
        self.on_final: Callable[[int, str], None] = lambda seq, text: None            # final translation
        self.on_merged: Callable[[int], None] = lambda seq: None                      # seq folded into the next
        self.on_status: Callable[[str, str], None] = lambda msg, kind: None           # kind: live/paused/warn
        self.on_level: Callable[[float], None] = lambda level: None
        self.on_quota: Callable[[str], None] = lambda text: None
        self.on_lang: Callable[[str], None] = lambda lang: None                       # language now heard


@dataclass
class Utterance:
    seq: int
    text: str
    lang: str
    pcm: np.ndarray
    t_end: float


_SENTENCE_END = re.compile(r"[.?!](?=\s+\S)")
_SENTENCE_FINAL = re.compile(r"[.?!][\"')\]]*\s*$")
_CLAUSE_END = re.compile(r"[,;:](?=\s+\S)")
# Korean puts the verb last, so translating half an English sentence reads badly: prefer whole
# sentences (the local live translation covers the wait) and only split genuinely long run-ons.
SOFT_CUT_WORDS = 25  # past this, a comma is good enough to finalise
HARD_CUT_WORDS = 40  # past this, cut anyway (fast speech with no punctuation)
SHORT_FRAGMENT_WORDS = 6  # an unfinished piece shorter than this is joined to the next one
MERGE_WAIT_SEC = 3.0      # ...if the next one arrives within this time


def _commit_point(text: str) -> int | None:
    """Where to finalise the running text, if anywhere: index just past the cut.

    Sentence end followed by more words -> cut there. Long run-on speech -> cut at the
    last comma, or failing that keep only the last few words open, so a sentence never
    grows past what the subtitle window and the translators handle quickly.
    """
    words = text.split()
    matches = list(_SENTENCE_END.finditer(text))
    if matches and len(text[: matches[-1].end()].split()) >= 3:
        return matches[-1].end()
    if len(words) > SOFT_CUT_WORDS:
        clauses = [m.end() for m in _CLAUSE_END.finditer(text) if len(text[: m.end()].split()) >= 8]
        if clauses:
            return clauses[-1]
    if len(words) > HARD_CUT_WORDS:
        keep_open = 4  # the newest words may still be revised by the recogniser
        return len(text) - len(" ".join(words[-keep_open:]))
    return None


def _written_in(text: str, lang: str) -> bool:
    """Whether the script matches the language: mostly Hangul for Korean, almost none otherwise."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    hangul = sum("가" <= ch <= "힣" for ch in letters) / len(letters)
    return hangul >= 0.5 if lang == "ko" else hangul <= 0.1


def _similar(a: str, b: str, threshold: float = 0.5) -> bool:
    """Same sentence, give or take spelling and punctuation (compared letter by letter, ignoring spaces)."""
    def norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())
    return SequenceMatcher(None, norm(a), norm(b)).ratio() >= threshold


class TranscriptLogger:
    """Appends every interpreted utterance to logs/YYYY-MM-DD.jsonl (optionally with audio)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        LOG_DIR.mkdir(exist_ok=True)
        self.session = datetime.now().strftime("%Y%m%d-%H%M%S")

    def write(self, utt: Utterance, source: str, translation: str, tgt: str, timings: dict) -> None:
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "session": self.session,
            "seq": utt.seq,
            "src_lang": utt.lang,
            "tgt_lang": tgt,
            "source": source,
            "source_local": utt.text,
            "translation": translation,
            "duration_sec": round(len(utt.pcm) / 16000, 2),
            **timings,
        }
        if self.cfg.save_audio:
            audio_dir = LOG_DIR / "audio" / self.session
            audio_dir.mkdir(parents=True, exist_ok=True)
            path = audio_dir / f"{utt.seq:05d}.wav"
            path.write_bytes(pcm_to_wav(utt.pcm))
            record["audio"] = str(path.relative_to(LOG_DIR))
        day_file = LOG_DIR / f"{datetime.now():%Y-%m-%d}.jsonl"
        with day_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Pipeline:
    """mic -> local streaming ASR -> (live) throttled re-translation, (sentence end) final translation."""

    def __init__(self, cfg: Config, events: Events):
        self.cfg = cfg
        self.events = events
        # None: default mic, int: mic index, SYSTEM_AUDIO: PC sound, "app:<exe>": one app's own audio
        self.device: int | str | None = None
        self.lang = cfg.input_lang
        self.frames: queue.Queue = queue.Queue()
        self.finals: queue.Queue = queue.Queue()
        self.history: deque[tuple[str, str]] = deque(maxlen=cfg.context_recent + cfg.context_older)
        self.quota = QuotaTracker(cfg.rpm_limits)
        self.stt = GroqSTT(cfg.groq_api_key, cfg.stt_model, self.quota)
        groq_partial = OpenAICompatTranslator(GROQ_BASE, cfg.groq_api_key, cfg.partial_model, self.quota)
        self.partial_translator: Translator | None = None if cfg.partial_engine == "local" else groq_partial
        # tried in order for finished sentences; engines without an API key are skipped
        self.final_translators, skipped = engines.build(cfg.translate_engines, self.quota)
        if skipped:
            print(f"translation engines skipped (no API key or unknown): {', '.join(skipped)}", flush=True)
        memo_models = [e for e in engines.build(cfg.memo_engines, self.quota)[0] if isinstance(e, OpenAICompatTranslator)]
        self.memo = MemoKeeper(memo_models, self.quota, on_update=lambda m: print(f"[memo]\n{m}", flush=True))
        self.logger = TranscriptLogger(cfg)

        self.recognizer: StreamingRecognizer | DualRecognizer | None = None
        self._recognizer_mode = ""  # the self.lang the recognizer was built for
        self.mic: MicCapture | SystemAudioCapture | None = None
        self.output_name = ""  # current speaker/earphones when listening to PC sound
        self.running = threading.Event()
        self._loading = threading.Lock()
        if local_mt.available():
            threading.Thread(target=self._load_local_mt, daemon=True).start()
        self.finalized: set[int] = set()
        self._live = (0, "")          # latest (seq, text) for the partial translator
        self._live_cond = threading.Condition()

        for target in (self._asr_loop, self._partial_loop, self._final_loop, self._quota_loop):
            threading.Thread(target=target, daemon=True).start()

    # ---- control -------------------------------------------------------
    @property
    def target_lang(self) -> str:
        return self.target_for(self.current_lang)

    @property
    def current_lang(self) -> str:
        """Language being heard right now ("auto" mode follows the recogniser's pick)."""
        if self.recognizer is not None:
            return self.recognizer.last_lang
        return self.cfg.other_lang if self.lang == AUTO else self.lang

    def target_for(self, lang: str) -> str:
        return self.cfg.other_lang if lang == self.cfg.main_target_lang else self.cfg.main_target_lang

    def _build_recognizer(self) -> StreamingRecognizer | DualRecognizer:
        silence, max_sec = self.cfg.endpoint_silence_sec, self.cfg.max_utterance_sec
        if self.lang != AUTO:
            return StreamingRecognizer(self.lang, silence, max_sec)
        # both models side by side (2 threads each); load them in parallel to halve the wait
        with ThreadPoolExecutor(2) as pool:
            models = dict(zip(AUTO_LANGS, pool.map(
                lambda lang: StreamingRecognizer(lang, silence, max_sec, threads=2), AUTO_LANGS)))
        previous = self.recognizer.last_lang if self.recognizer is not None else self.cfg.other_lang
        return DualRecognizer(models, initial=previous)

    def _load_local_mt(self) -> None:
        try:
            translator = local_mt.LocalTranslator()
        except Exception as e:
            self.events.on_status(f"로컬 번역 모델 로드 실패: {e}", "warn")
            return
        self.final_translators.append(translator)  # last resort when every API is rate-limited
        if self.cfg.partial_engine == "local":
            self.partial_translator = translator

    @property
    def listening_label(self) -> str:
        if isinstance(self.device, str) and self.device.startswith(APP_PREFIX):
            app = self.device[len(APP_PREFIX):].removesuffix(".exe")
            return f"{app} 소리 듣는 중" if self.output_name else f"{app} 실행 대기 중"
        if self.device != SYSTEM_AUDIO:
            return "마이크 듣는 중"
        return f"PC 소리 듣는 중 · {self.output_name}" if self.output_name else "PC 소리 듣는 중"

    def _on_output_device(self, name: str) -> None:
        changed = name != self.output_name
        self.output_name = name
        if self.running.is_set() and changed:
            self.events.on_status(self.listening_label, "live" if name else "warn")

    def start(self) -> None:
        """Loads the speech model off the UI thread (takes a few seconds), then opens the mic."""
        if self.running.is_set() or self._loading.locked():
            return
        threading.Thread(target=self._start, daemon=True).start()

    def _start(self) -> None:
        with self._loading:
            try:
                if self.recognizer is None or self._recognizer_mode != self.lang:
                    self.events.on_status("음성인식 모델 불러오는 중…", "paused")
                    self.recognizer = self._build_recognizer()
                    self._recognizer_mode = self.lang
                if isinstance(self.device, str) and self.device.startswith(FILE_PREFIX):
                    self.mic = FileCapture(self.frames, self.cfg.sample_rate, self.cfg.frame_ms, self.device[len(FILE_PREFIX):])
                elif isinstance(self.device, str) and self.device.startswith(APP_PREFIX):
                    self.mic = AppAudioCapture(
                        self.frames, self.cfg.sample_rate, self.cfg.frame_ms,
                        self.device[len(APP_PREFIX):], on_device=self._on_output_device,
                    )
                elif self.device == SYSTEM_AUDIO:
                    self.mic = SystemAudioCapture(
                        self.frames, self.cfg.sample_rate, self.cfg.frame_ms, on_device=self._on_output_device
                    )
                else:
                    self.mic = MicCapture(self.frames, self.cfg.sample_rate, self.cfg.frame_ms, self.device)
                self.mic.start()
            except Exception as e:
                self.events.on_status(f"시작 실패: {e}", "warn")
                return
            self.running.set()
            self.events.on_status(self.listening_label, "live")

    def stop(self) -> None:
        self.running.clear()
        if self.mic:
            self.mic.stop()
            self.mic = None
        self.output_name = ""
        self.events.on_status("일시정지", "paused")

    def _restart(self, apply: Callable[[], None]) -> None:
        was_running = self.running.is_set()
        self.stop()
        apply()
        if was_running:
            self.start()

    def set_device(self, device: int | str | None) -> None:
        self._restart(lambda: setattr(self, "device", device))

    def set_lang(self, lang: str) -> None:
        def apply() -> None:
            self.lang = lang
            self.history.clear()
            self.memo.reset()
        self._restart(apply)

    OLDER_MAX_CHARS = 600  # cap on the source-only context, newest sentences kept

    def _context(self) -> Context:
        items = list(self.history)
        n = self.cfg.context_recent
        recent, older = items[-n:] if n else [], items[:-n] if n else items
        older_src, size = [], 0
        for src, _ in reversed(older):
            size += len(src)
            if size > self.OLDER_MAX_CHARS:
                break
            older_src.insert(0, src)
        return Context(recent=recent, older=older_src, memo=self.memo.memo if self.cfg.conversation_memo else "")

    # ---- workers -------------------------------------------------------
    def _asr_loop(self) -> None:
        seq = 1
        last_text = ""
        committed = ""  # start of the recogniser's running text already finalised sentence by sentence
        heard_lang = ""
        audio: list[np.ndarray] = []
        max_frames = int((self.cfg.max_utterance_sec + 3) * 1000 / self.cfg.frame_ms)

        def finalize(text: str, pcm: np.ndarray) -> None:
            nonlocal seq
            self.finalized.add(seq)
            self.finals.put(Utterance(seq, text, self.recognizer.last_lang, pcm, time.perf_counter()))
            seq += 1

        while True:
            frame = self.frames.get()
            if not self.running.is_set() or self.recognizer is None:
                continue
            self.events.on_level(min(1.0, float(np.abs(frame.astype(np.int32)).mean()) / 3000))
            audio.append(frame)
            audio = audio[-max_frames:]
            full, ended = self.recognizer.accept(frame)
            if self.recognizer.last_lang != heard_lang:
                heard_lang = self.recognizer.last_lang
                self.events.on_lang(heard_lang)
            text = full[len(committed):].strip()

            # the model punctuates: as soon as a new sentence starts, finalise the finished one(s)
            # instead of waiting for a pause, so long run-on speech still gets quick final translations
            boundary = None if ended else _commit_point(text)
            if boundary:
                head = text[:boundary].strip()
                committed = full[: len(full) - len(text) + boundary]
                text = text[boundary:].strip()
                self.events.on_live(seq, head)
                finalize(head, np.zeros(0, np.int16))  # no clean audio cut mid-stream
                last_text = ""

            if text and text != last_text:
                last_text = text
                self.events.on_live(seq, text)
                with self._live_cond:
                    self._live = (seq, text)
                    self._live_cond.notify()
            if ended:
                if last_text:
                    # the audio also holds sentences already finalised; Whisper would repeat them
                    finalize(last_text, np.zeros(0, np.int16) if committed else np.concatenate(audio))
                last_text = ""
                committed = ""
                audio = []

    def _partial_loop(self) -> None:
        done_seq, done_words = 0, 0
        last_call = 0.0
        while True:
            with self._live_cond:
                self._live_cond.wait(timeout=0.3)
                seq, text = self._live
            if not self.cfg.live_translation or not text or seq in self.finalized:
                continue
            words = len(text.split())
            if seq == done_seq and words - done_words < self.cfg.partial_min_new_words:
                continue
            if time.time() - last_call < self.cfg.partial_interval_sec:
                continue
            translator = self.partial_translator
            if translator is None:  # local model still loading
                continue
            # API engine: keep 30% of the daily quota and per-minute headroom for finished sentences
            if isinstance(translator, OpenAICompatTranslator) and not self.quota.can_use(
                translator.name, min_tokens=1500, keep_day_ratio=0.3
            ):
                continue
            last_call = time.time()
            done_seq, done_words = seq, words
            out = ""
            try:
                lang = self.current_lang
                for piece in translator.stream(text, lang, self.target_for(lang), Context(), partial=True):
                    if seq in self.finalized:
                        break
                    out += piece
                    self.events.on_partial(seq, out)
            except Exception:
                pass  # partials are best-effort; the final translation will follow

    def _refine_source(self, utt: Utterance) -> tuple[str, int]:
        """Polish the locally recognised sentence with Whisper when quota allows. Returns (text, ms)."""
        if (
            self.cfg.final_stt != "groq"
            or len(utt.pcm) < 16000 * 0.5
            or not self.quota.can_use(self.cfg.stt_model, rpm_headroom=2)
        ):
            return utt.text, 0
        t0 = time.perf_counter()
        try:
            # earlier sentences in another language push Whisper into translating instead of transcribing
            prompt = " ".join(s for s, _ in self.history if _written_in(s, utt.lang))
            tr = self.stt.transcribe(utt.pcm, prompt, language=utt.lang)
            # Whisper polishes; it must not rewrite (it can echo its prompt or translate instead)
            if tr and _written_in(tr.text, utt.lang) and _similar(tr.text, utt.text):
                return tr.text, round((time.perf_counter() - t0) * 1000)
        except Exception:
            pass
        return utt.text, round((time.perf_counter() - t0) * 1000)

    def _next_final(self) -> Utterance:
        """Next utterance to translate, with short unfinished pieces ("which I") joined to what follows."""
        utt: Utterance = self.finals.get()
        while len(utt.text.split()) < SHORT_FRAGMENT_WORDS and not _SENTENCE_FINAL.search(utt.text):
            try:
                nxt: Utterance = self.finals.get(timeout=MERGE_WAIT_SEC)
            except queue.Empty:
                break  # speaker stopped: translate the piece on its own
            self.events.on_merged(utt.seq)
            pcm = np.concatenate([utt.pcm, nxt.pcm]) if len(utt.pcm) and len(nxt.pcm) else np.zeros(0, np.int16)
            utt = Utterance(nxt.seq, f"{utt.text} {nxt.text}", nxt.lang, pcm, nxt.t_end)
        return utt

    def _fix_spacing(self, seq: int, source: str) -> None:
        spaced = spacing.space_korean(source, self.memo.models, self.quota)
        if spaced:
            self.events.on_source(seq, spaced, "ko")

    def _final_loop(self) -> None:
        while True:
            utt = self._next_final()
            source, stt_ms = self._refine_source(utt)
            self.events.on_source(utt.seq, source, utt.lang)
            if utt.lang == "ko" and spacing.needs_spacing(source):
                # display only: translate right away, put the spaced text on screen when it arrives
                threading.Thread(target=self._fix_spacing, args=(utt.seq, source), daemon=True).start()
            tgt = self.target_for(utt.lang)

            t0 = time.perf_counter()
            out, used, first_token_ms = "", None, None
            for translator in self.final_translators:
                if self.quota.cooldown_left(translator.name) > 0:
                    continue
                try:
                    for piece in translator.stream(source, utt.lang, tgt, self._context()):
                        if first_token_ms is None:
                            first_token_ms = round((time.perf_counter() - t0) * 1000)
                        out += piece
                        self.events.on_partial(utt.seq, out)
                    used = translator.model
                    break
                except RateLimitError:
                    out = ""
                    continue
                except Exception as e:
                    self.events.on_status(f"번역 오류: {str(e)[:80]}", "warn")
                    out = ""
                    continue
            out = out.strip()
            self.events.on_final(utt.seq, out)
            if used is None:
                self.events.on_status("번역 한도 초과 · 원문만 표시 중", "warn")
                continue
            if used == local_mt.LocalTranslator.model:
                self.events.on_status("Groq 한도 초과 · 로컬 번역 사용 중", "warn")
            elif self.running.is_set():
                self.events.on_status(self.listening_label, "live")
            self.history.append((source, out))
            if self.cfg.conversation_memo:
                self.memo.add(source, out, utt.lang, tgt)
            timings = {
                "stt_ms": stt_ms,
                "first_token_ms": first_token_ms or 0,
                "translate_ms": round((time.perf_counter() - t0) * 1000),
                "latency_ms": round((t0 - utt.t_end) * 1000 + (first_token_ms or 0)),
                "model": used,
            }
            print(f"[{utt.seq}] {utt.lang}->{tgt} {timings}  {source!r} => {out!r}", flush=True)
            try:
                self.logger.write(utt, source, out, tgt, timings)
            except OSError as e:
                self.events.on_status(f"로그 저장 실패: {e}", "warn")

    def _quota_loop(self) -> None:
        labels = {e.name: engines.label(e) for e in self.final_translators[:3]}
        labels[self.cfg.stt_model] = "인식"
        while True:
            time.sleep(2)
            self.events.on_quota(self.quota.summary(labels))
