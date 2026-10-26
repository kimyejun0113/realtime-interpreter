import time
from dataclasses import dataclass, field

import numpy as np
import webrtcvad


@dataclass
class Segment:
    seq: int
    pcm: np.ndarray  # int16 mono
    t_end: float = field(default_factory=time.perf_counter)  # when the segment was cut

    @property
    def duration(self) -> float:
        return len(self.pcm) / 16000


class VadSegmenter:
    """Turns a stream of 30ms frames into speech segments.

    A segment is cut as soon as `silence_ms` of silence follows speech, or when
    it reaches `max_sec` (cut at the quietest recent frame so words are not split).
    """

    PRE_ROLL_FRAMES = 10  # keep ~300ms before speech onset so the first word isn't clipped
    START_FRAMES = 3      # voiced frames needed to start a segment (filters clicks)
    MIN_SPEECH_SEC = 0.35

    def __init__(self, sample_rate: int, frame_ms: int, silence_ms: int, max_sec: float, aggressiveness: int):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.max_frames = int(max_sec * 1000 / frame_ms)
        self._seq = 0
        self._reset()

    def _reset(self) -> None:
        self.frames: list[np.ndarray] = []
        self.voiced_count = 0
        self.trailing_silence = 0
        self.in_speech = False
        self.pre_roll: list[np.ndarray] = []
        self.onset_run = 0

    def _emit(self, frames: list[np.ndarray]) -> Segment | None:
        pcm = np.concatenate(frames)
        if len(pcm) / self.sample_rate < self.MIN_SPEECH_SEC:
            return None
        self._seq += 1
        return Segment(self._seq, pcm)

    def push(self, frame: np.ndarray) -> Segment | None:
        voiced = self.vad.is_speech(frame.tobytes(), self.sample_rate)

        if not self.in_speech:
            self.pre_roll.append(frame)
            self.pre_roll = self.pre_roll[-self.PRE_ROLL_FRAMES:]
            self.onset_run = self.onset_run + 1 if voiced else 0
            if self.onset_run >= self.START_FRAMES:
                self.in_speech = True
                self.frames = list(self.pre_roll)
                self.pre_roll = []
                self.trailing_silence = 0
            return None

        self.frames.append(frame)
        self.trailing_silence = 0 if voiced else self.trailing_silence + 1

        if self.trailing_silence >= self.silence_frames:
            # drop most of the trailing silence; keep a little tail
            keep = len(self.frames) - self.trailing_silence + 3
            seg = self._emit(self.frames[:keep])
            self._reset()
            return seg

        if len(self.frames) >= self.max_frames:
            # forced cut: split at the lowest-energy frame in the last second
            window = min(len(self.frames) // 2, 1000 // self.frame_ms)
            tail = self.frames[-window:]
            energies = [float(np.abs(f.astype(np.int32)).mean()) for f in tail]
            cut = len(self.frames) - window + int(np.argmin(energies)) + 1
            seg = self._emit(self.frames[:cut])
            self.frames = self.frames[cut:]
            self.trailing_silence = 0
            return seg

        return None
