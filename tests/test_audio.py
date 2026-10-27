import queue
import time

import numpy as np
import pytest

import audio.app_loopback as app_loopback
import audio.capture as capture
from audio.capture import AppAudioCapture, SystemAudioCapture
from stt.streaming import tidy


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def tone(amplitude, n=1600):
    return (amplitude * np.sin(np.linspace(0, 200 * np.pi, n))).astype(np.float32)


def test_agc_lifts_quiet_speech():
    cap = SystemAudioCapture(queue.Queue(), 16000, 100)
    out = [cap.agc(tone(0.005)) for _ in range(40)]  # let the smoothed level settle
    assert rms(out[-1]) > rms((tone(0.005) * 32767).astype(np.int16)) * 5


def test_agc_leaves_silence_alone_and_never_clips():
    cap = SystemAudioCapture(queue.Queue(), 16000, 100)
    silent = [cap.agc(tone(0.0001)) for _ in range(40)]
    assert np.abs(silent[-1]).max() < 50  # hiss is not amplified
    loud = [cap.agc(tone(0.9)) for _ in range(10)]
    assert np.abs(loud[-1]).max() <= 32767


class FakeLoopback:
    """Delivers 1 s of audio, then nothing — like an app that plays and then pauses."""

    def __init__(self, pid, sample_rate=16000):
        self.chunks = [np.full(1600, 1000, np.int16)] * 10

    def read(self):
        return self.chunks.pop(0) if self.chunks else np.zeros(0, np.int16)

    def close(self):
        pass


def test_app_capture_keeps_real_time_pace_with_silence(monkeypatch):
    monkeypatch.setattr(app_loopback, "AppLoopback", FakeLoopback)
    monkeypatch.setattr(capture, "find_app_root", lambda exe: 1234)
    import psutil
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: True)
    devices = []
    q = queue.Queue()
    cap = AppAudioCapture(q, 16000, 100, "chrome.exe", on_device=devices.append)
    cap.start()
    time.sleep(2.05)
    cap.stop()
    frames = [q.get() for _ in range(q.qsize())]
    assert devices == ["chrome.exe"]
    assert 17 <= len(frames) <= 23  # ~10 frames per second, played or not
    assert all(len(f) == 1600 for f in frames)
    assert np.abs(frames[-1]).max() == 0  # after the app went quiet: padded silence


def test_app_capture_waits_while_app_is_not_running(monkeypatch):
    monkeypatch.setattr(capture, "find_app_root", lambda exe: None)
    devices = []
    q = queue.Queue()
    cap = AppAudioCapture(q, 16000, 100, "chrome.exe", on_device=devices.append)
    cap.start()
    time.sleep(0.3)
    cap.stop()
    assert devices and devices[0] == "" and q.qsize() >= 10  # status "waiting" + silence keeps flowing


@pytest.mark.parametrize("raw,clean", [
    ("HELLO THERE I'M FINE", "Hello there I'm fine"),
    ("I THINK I SAW IT", "I think I saw it"),
    ("Hey, so I wanted to go.", "Hey, so I wanted to go."),  # already cased: untouched
    ("  spaced   out  ", "spaced out"),
])
def test_tidy_english(raw, clean):
    assert tidy(raw, "en") == clean


def test_tidy_leaves_korean_text():
    assert tidy("안녕하세요  반갑습니다", "ko") == "안녕하세요 반갑습니다"
