import json
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

from audio.capture import MicCapture
from audio.segmenter import Segment, VadSegmenter
from config import Config
from stt.base import STT, RateLimitError, Transcript, pcm_to_wav
from stt.groq_stt import GroqSTT
from translate.base import Translator
from translate.openai_compat import GEMINI_BASE, GROQ_BASE, OpenAICompatTranslator

LOG_DIR = Path(__file__).with_name("logs")


class Events:
    """Callbacks the UI subscribes to. Called from worker threads."""

    def __init__(self):
        self.on_source: Callable[[int, str, str], None] = lambda seq, text, lang: None
        self.on_partial: Callable[[int, str], None] = lambda seq, text: None
        self.on_final: Callable[[int, str], None] = lambda seq, text: None
        self.on_status: Callable[[str], None] = lambda msg: None
        self.on_level: Callable[[float], None] = lambda level: None


def make_translator(cfg: Config) -> Translator:
    if cfg.translate_provider == "gemini":
        return OpenAICompatTranslator(GEMINI_BASE, cfg.gemini_api_key, cfg.gemini_model)
    return OpenAICompatTranslator(GROQ_BASE, cfg.groq_api_key, cfg.translate_model)


class TranscriptLogger:
    """Appends every interpreted utterance to logs/YYYY-MM-DD.jsonl (optionally with audio)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        LOG_DIR.mkdir(exist_ok=True)
        self.session = datetime.now().strftime("%Y%m%d-%H%M%S")

    def write(self, seg: Segment, tr: Transcript, translation: str, tgt: str, timings: dict) -> None:
        record = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "session": self.session,
            "seq": seg.seq,
            "src_lang": tr.lang,
            "tgt_lang": tgt,
            "source": tr.text,
            "translation": translation,
            "duration_sec": round(seg.duration, 2),
            "stt_model": self.cfg.stt_model,
            "translate_model": self.cfg.gemini_model if self.cfg.translate_provider == "gemini" else self.cfg.translate_model,
            **timings,
        }
        if self.cfg.save_audio:
            audio_dir = LOG_DIR / "audio" / self.session
            audio_dir.mkdir(parents=True, exist_ok=True)
            path = audio_dir / f"{seg.seq:05d}.wav"
            path.write_bytes(pcm_to_wav(seg.pcm))
            record["audio"] = str(path.relative_to(LOG_DIR))
        day_file = LOG_DIR / f"{datetime.now():%Y-%m-%d}.jsonl"
        with day_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class Pipeline:
    def __init__(self, cfg: Config, events: Events):
        self.cfg = cfg
        self.events = events
        self.device: int | None = None
        self.frames: queue.Queue = queue.Queue()
        self.segments: queue.Queue = queue.Queue()
        self.transcripts: queue.Queue = queue.Queue()
        self.context: deque[tuple[str, str]] = deque(maxlen=3)
        self.stt: STT = GroqSTT(cfg.groq_api_key, cfg.stt_model)
        self.translator = make_translator(cfg)
        self.logger = TranscriptLogger(cfg)
        self.mic: MicCapture | None = None
        self.running = threading.Event()
        threading.Thread(target=self._segment_loop, daemon=True).start()
        threading.Thread(target=self._stt_loop, daemon=True).start()
        threading.Thread(target=self._translate_loop, daemon=True).start()

    # ---- control -------------------------------------------------------
    def start(self) -> None:
        if self.running.is_set():
            return
        self.mic = MicCapture(self.frames, self.cfg.sample_rate, self.cfg.frame_ms, self.device)
        self.mic.start()
        self.running.set()
        self.events.on_status("듣는 중…")

    def stop(self) -> None:
        self.running.clear()
        if self.mic:
            self.mic.stop()
            self.mic = None
        self.events.on_status("일시정지")

    def set_device(self, device: int | None) -> None:
        was_running = self.running.is_set()
        self.stop()
        self.device = device
        if was_running:
            self.start()

    # ---- workers -------------------------------------------------------
    def _segment_loop(self) -> None:
        cfg = self.cfg
        seg = VadSegmenter(cfg.sample_rate, cfg.frame_ms, cfg.vad_silence_ms, cfg.max_segment_sec, cfg.vad_aggressiveness)
        level_tick = 0
        while True:
            frame = self.frames.get()
            if not self.running.is_set():
                continue
            level_tick += 1
            if level_tick % 3 == 0:
                self.events.on_level(min(1.0, float(abs(frame.astype("int32")).mean()) / 3000))
            out = seg.push(frame)
            if out is not None:
                self.segments.put(out)

    def _stt_loop(self) -> None:
        while True:
            seg: Segment = self.segments.get()
            prompt = " ".join(s for s, _ in self.context)
            t0 = time.perf_counter()
            try:
                tr = self.stt.transcribe(seg.pcm, prompt)
            except RateLimitError:
                if self._switch_to_local_stt():
                    self.segments.put(seg)  # retry this segment locally
                else:
                    self.events.on_status("⚠ Groq 무료 한도 초과 - 잠시 후 다시 시도됩니다")
                continue
            except Exception as e:  # network errors must not kill the worker
                self.events.on_status(f"⚠ 음성인식 오류: {e}")
                continue
            if tr is None:
                continue
            stt_ms = (time.perf_counter() - t0) * 1000
            self.events.on_source(seg.seq, tr.text, tr.lang)
            self.transcripts.put((seg, tr, {"wait_ms": round((t0 - seg.t_end) * 1000), "stt_ms": round(stt_ms)}))

    def _switch_to_local_stt(self) -> bool:
        if not self.cfg.local_stt_fallback or not isinstance(self.stt, GroqSTT):
            return False
        try:
            from stt.local_stt import LocalSTT

            self.events.on_status("Groq 한도 초과 → 로컬 음성인식으로 전환 중…")
            self.stt = LocalSTT(self.cfg.local_stt_model)
            self.events.on_status("로컬 음성인식 사용 중 (느릴 수 있음)")
            return True
        except Exception as e:
            self.events.on_status(f"⚠ 로컬 음성인식 전환 실패: {e}")
            return False

    def _translate_loop(self) -> None:
        while True:
            seg, tr, timings = self.transcripts.get()
            tgt = self.cfg.other_lang if tr.lang == self.cfg.main_target_lang else self.cfg.main_target_lang
            t0 = time.perf_counter()
            first_token_ms = None
            out = ""
            try:
                for piece in self.translator.stream(tr.text, tr.lang, tgt, list(self.context)):
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - t0) * 1000
                    out += piece
                    self.events.on_partial(seg.seq, out)
            except RateLimitError:
                self.events.on_status("⚠ 번역 무료 한도 초과 - 잠시 후 다시 시도됩니다")
                continue
            except Exception as e:
                self.events.on_status(f"⚠ 번역 오류: {e}")
                continue
            out = out.strip()
            self.events.on_final(seg.seq, out)
            self.context.append((tr.text, out))
            timings.update(
                first_token_ms=round(first_token_ms or 0),
                translate_ms=round((time.perf_counter() - t0) * 1000),
                # speech end -> first translated word on screen
                latency_ms=round((t0 - seg.t_end) * 1000 + (first_token_ms or 0)),
            )
            print(f"[{seg.seq}] {tr.lang}->{tgt} {timings}  {tr.text!r} => {out!r}", flush=True)
            try:
                self.logger.write(seg, tr, out, tgt, timings)
            except OSError as e:
                self.events.on_status(f"⚠ 로그 저장 실패: {e}")
