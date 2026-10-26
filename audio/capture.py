import ctypes
import queue
import threading
import time
import warnings
from typing import Callable

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

    DEVICE_CHECK_SEC = 1.0  # how often to look for a new default output (AirPods connected, etc.)

    def __init__(self, out_queue: queue.Queue, sample_rate: int, frame_ms: int,
                 on_device: Callable[[str], None] = lambda name: None):
        self.out_queue = out_queue
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self.on_device = on_device  # called with the output device name whenever capture (re)starts
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._level, self._gain = 0.0, 1.0

    def _run(self) -> None:
        import soundcard as sc

        ctypes.windll.ole32.CoInitializeEx(None, 0)  # WASAPI needs COM on this thread
        warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)
        silence = np.zeros(self.frame_samples, np.int16)
        checks_every = max(1, int(self.DEVICE_CHECK_SEC * self.sample_rate / self.frame_samples))
        while self._running.is_set():
            # follow Windows' default output: switching to earphones/AirPods reopens the loopback
            try:
                speaker = sc.default_speaker()
                loopback = sc.get_microphone(id=speaker.id, include_loopback=True)
                with loopback.recorder(samplerate=self.sample_rate, channels=1, blocksize=self.frame_samples) as rec:
                    self.on_device(speaker.name)
                    frames = 0
                    while self._running.is_set():
                        self.out_queue.put(self.agc(rec.record(numframes=self.frame_samples)[:, 0]))
                        frames += 1
                        if frames % checks_every == 0 and sc.default_speaker().id != speaker.id:
                            break
            except Exception:
                # device unplugged / no output right now: keep the recogniser fed with silence
                # so the current sentence still ends, then try again
                for _ in range(5):
                    self.out_queue.put(silence)
                time.sleep(0.5)

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


APP_PREFIX = "app:"  # device id "app:chrome.exe" = that app's own audio
KNOWN_PLAYERS = {"chrome.exe", "msedge.exe", "whale.exe", "firefox.exe", "brave.exe", "opera.exe",
                 "zoom.exe", "ms-teams.exe", "discord.exe", "vlc.exe", "potplayermini64.exe", "spotify.exe"}


def list_audio_apps() -> list[str]:
    """Apps worth offering for per-app capture: anything with an audio session, plus running browsers/players."""
    import psutil

    names = set()
    try:
        from pycaw.pycaw import AudioUtilities

        for session in AudioUtilities.GetAllSessions():
            try:
                if session.Process:
                    names.add(session.Process.name().lower())
            except psutil.Error:
                pass
    except Exception:
        pass
    for p in psutil.process_iter(["name"]):
        name = (p.info["name"] or "").lower()
        if name in KNOWN_PLAYERS:
            names.add(name)
    # our own interpreter and audio/system services are never what the user wants to hear
    skip = {"python.exe", "pythonw.exe", "svchost.exe", "audiodg.exe", "explorer.exe"}
    return sorted(n for n in names if n not in skip and "service" not in n)


def find_app_root(exe: str) -> int | None:
    """PID of the top process of an app (browsers run audio in a child process; the tree covers it)."""
    import psutil

    candidates = []
    for p in psutil.process_iter(["name", "ppid", "create_time"]):
        if (p.info["name"] or "").lower() != exe:
            continue
        try:
            parent = psutil.Process(p.info["ppid"]).name().lower()
        except psutil.Error:
            parent = ""
        if parent != exe:
            candidates.append((p.info["create_time"], p.pid))
    return min(candidates)[1] if candidates else None


class AppAudioCapture(SystemAudioCapture):
    """Captures one app's own audio (Windows process loopback): works at volume 0 / muted speakers.

    The app only delivers audio while it plays, so silence is padded in to keep a real-time
    stream; otherwise pausing a video would leave the current sentence hanging.
    """

    def __init__(self, out_queue: queue.Queue, sample_rate: int, frame_ms: int, exe: str,
                 on_device: Callable[[str], None] = lambda name: None):
        super().__init__(out_queue, sample_rate, frame_ms, on_device)
        self.exe = exe.lower()

    def _run(self) -> None:
        import psutil

        from .app_loopback import AppLoopback

        silence = np.zeros(self.frame_samples, np.int16)
        while self._running.is_set():
            pid = find_app_root(self.exe)
            if pid is None:  # app not running (yet): wait for it
                self.on_device("")
                for _ in range(10):
                    self.out_queue.put(silence)
                time.sleep(1)
                continue
            try:
                cap = AppLoopback(pid, self.sample_rate)
            except Exception:
                time.sleep(1)
                continue
            self.on_device(self.exe)
            buffered = np.zeros(0, np.int16)
            next_frame = time.perf_counter()
            last_check = next_frame
            frame_sec = self.frame_samples / self.sample_rate
            try:
                while self._running.is_set():
                    buffered = np.concatenate([buffered, cap.read()])
                    now = time.perf_counter()
                    # emit one frame per frame period: real audio if we have it, silence otherwise
                    while now >= next_frame:
                        if len(buffered) >= self.frame_samples:
                            frame, buffered = buffered[: self.frame_samples], buffered[self.frame_samples:]
                        else:
                            frame = silence
                        self.out_queue.put(self.agc(frame.astype(np.float32) / 32768.0))
                        next_frame += frame_sec
                    if len(buffered) > self.sample_rate:  # never lag more than 1 s behind
                        buffered = buffered[-self.frame_samples:]
                    if now - last_check > 2:
                        last_check = now
                        if not psutil.pid_exists(pid):  # app closed/restarted: find it again
                            break
                    time.sleep(0.02)
            finally:
                cap.close()
