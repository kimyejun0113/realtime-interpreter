import json
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from audio.capture import APP_PREFIX, SYSTEM_AUDIO, AppAudioCapture, MicCapture, SystemAudioCapture
from config import Config
from quota import QuotaTracker
from stt.base import RateLimitError, pcm_to_wav
from stt.groq_stt import GroqSTT
from stt.streaming import StreamingRecognizer
from translate import local_mt
from translate.base import Translator
from translate import engines
from translate.openai_compat import GROQ_BASE, OpenAICompatTranslator

LOG_DIR = Path(__file__).with_name("logs")


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
        self.context: deque[tuple[str, str]] = deque(maxlen=3)
        self.quota = QuotaTracker(cfg.rpm_limits)
        self.stt = GroqSTT(cfg.groq_api_key, cfg.stt_model, self.quota)
        groq_partial = OpenAICompatTranslator(GROQ_BASE, cfg.groq_api_key, cfg.partial_model, self.quota)
        self.partial_translator: Translator | None = None if cfg.partial_engine == "local" else groq_partial
        # tried in order for finished sentences; engines without an API key are skipped
        self.final_translators, skipped = engines.build(cfg.translate_engines, self.quota)
        if skipped:
            print(f"translation engines skipped (no API key or unknown): {', '.join(skipped)}", flush=True)
        self.logger = TranscriptLogger(cfg)

        self.recognizer: StreamingRecognizer | None = None
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
        return self.cfg.other_lang if self.lang == self.cfg.main_target_lang else self.cfg.main_target_lang

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
                if self.recognizer is None or self.recognizer.lang != self.lang:
                    self.events.on_status("음성인식 모델 불러오는 중…", "paused")
                    self.recognizer = StreamingRecognizer(self.lang, self.cfg.endpoint_silence_sec, self.cfg.max_utterance_sec)
                if isinstance(self.device, str) and self.device.startswith(APP_PREFIX):
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
        self._restart(lambda: setattr(self, "lang", lang))

    # ---- workers -------------------------------------------------------
    def _asr_loop(self) -> None:
        seq = 1
        last_text = ""
        committed = ""  # start of the recogniser's running text already finalised sentence by sentence
        audio: list[np.ndarray] = []
        max_frames = int((self.cfg.max_utterance_sec + 3) * 1000 / self.cfg.frame_ms)

        def finalize(text: str, pcm: np.ndarray) -> None:
            nonlocal seq
            self.finalized.add(seq)
            self.finals.put(Utterance(seq, text, self.recognizer.lang, pcm, time.perf_counter()))
            seq += 1

        while True:
            frame = self.frames.get()
            if not self.running.is_set() or self.recognizer is None:
                continue
            self.events.on_level(min(1.0, float(np.abs(frame.astype(np.int32)).mean()) / 3000))
            audio.append(frame)
            audio = audio[-max_frames:]
            full, ended = self.recognizer.accept(frame)
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
                for piece in translator.stream(text, self.lang, self.target_lang, [], partial=True):
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
            prompt = " ".join(s for s, _ in self.context)
            tr = self.stt.transcribe(utt.pcm, prompt, language=utt.lang)
            if tr:
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

    def _final_loop(self) -> None:
        while True:
            utt = self._next_final()
            source, stt_ms = self._refine_source(utt)
            self.events.on_source(utt.seq, source, utt.lang)
            tgt = self.cfg.other_lang if utt.lang == self.cfg.main_target_lang else self.cfg.main_target_lang

            t0 = time.perf_counter()
            out, used, first_token_ms = "", None, None
            for translator in self.final_translators:
                if self.quota.cooldown_left(translator.name) > 0:
                    continue
                try:
                    for piece in translator.stream(source, utt.lang, tgt, list(self.context)):
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
            self.context.append((source, out))
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
