import ctypes
import queue
import threading
import warnings

import numpy as np
import sounddevice as sd


def list_input_devices() -> list[tuple[int, str]]:
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append((idx, dev["name"]))
    return devices


class MicCapture:
    """Pushes fixed-size int16 mono frames from the microphone into a queue."""

    def __init__(self, out_queue: queue.Queue, sample_rate: int, frame_ms: int, device: int | None = None):
        self.out_queue = out_queue
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self.device = device
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames, time_info, status) -> None:
        # blocksize == frame_samples, so each callback is exactly one VAD frame
        self.out_queue.put(indata[:, 0].copy())

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            channels=1,
            dtype="int16",
            device=self.device,
            latency="low",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


SYSTEM_AUDIO = "system"  # device id for "what the PC is playing" (WASAPI loopback)


class SystemAudioCapture:
    """Captures what the default speaker is playing (videos, calls) via WASAPI loopback.

    On many PCs the loopback signal follows the volume slider, so a quiet PC means a quiet
    signal; an automatic gain brings speech back to a level the recogniser handles well.
    """

    TARGET_RMS = 0.08
    MAX_GAIN = 30.0
    SILENCE_RMS = 3e-4  # below this it's silence/hiss: don't amplify

    def __init__(self, out_queue: queue.Queue, sample_rate: int, frame_ms: int):
        self.out_queue = out_queue
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._level, self._gain = 0.0, 1.0

    def _run(self) -> None:
        import soundcard as sc

        ctypes.windll.ole32.CoInitializeEx(None, 0)  # WASAPI needs COM on this thread
        warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)
        speaker = sc.default_speaker()
        loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        with loopback.recorder(samplerate=self.sample_rate, channels=1, blocksize=self.frame_samples) as rec:
            while self._running.is_set():
                self.out_queue.put(self.agc(rec.record(numframes=self.frame_samples)[:, 0]))

    def agc(self, data: np.ndarray) -> np.ndarray:
        """float32 [-1, 1] frame -> int16 frame with automatic gain."""
        rms = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0
        self._level = 0.9 * self._level + 0.1 * rms  # smoothed so gain doesn't pump word by word
        if self._level > self.SILENCE_RMS:
            self._gain = min(self.MAX_GAIN, max(1.0, self.TARGET_RMS / self._level))
        return (np.clip(data * self._gain, -1.0, 1.0) * 32767).astype(np.int16)

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
