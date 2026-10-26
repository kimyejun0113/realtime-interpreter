import queue

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
