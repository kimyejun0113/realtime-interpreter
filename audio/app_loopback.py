"""Capture one app's audio (e.g. the browser) with Windows process loopback (Windows 10 2004+).

Unlike speaker loopback this taps the app's own stream, so it works even when the
PC volume is 0 or the speakers are muted.
"""
import ctypes
import time
from ctypes import POINTER, Structure, Union, byref, c_int, c_uint16, c_uint32, c_uint64, c_ulong, c_void_p, sizeof
from ctypes.wintypes import DWORD, LPCWSTR
from threading import Event

import comtypes
import numpy as np
from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

# ---- COM declarations -----------------------------------------------------
REFERENCE_TIME = ctypes.c_longlong


class WAVEFORMATEX(Structure):
    _fields_ = [
        ("wFormatTag", c_uint16), ("nChannels", c_uint16), ("nSamplesPerSec", c_uint32),
        ("nAvgBytesPerSec", c_uint32), ("nBlockAlign", c_uint16), ("wBitsPerSample", c_uint16), ("cbSize", c_uint16),
    ]


class AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(Structure):
    _fields_ = [("TargetProcessId", DWORD), ("ProcessLoopbackMode", c_int)]


class AUDIOCLIENT_ACTIVATION_PARAMS(Structure):
    _fields_ = [("ActivationType", c_int), ("ProcessLoopbackParams", AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS)]


class BLOB(Structure):
    _fields_ = [("cbSize", c_ulong), ("pBlobData", c_void_p)]


class _PropVariantValue(Union):
    _fields_ = [("blob", BLOB), ("pad", ctypes.c_byte * 16)]


class PROPVARIANT(Structure):
    _fields_ = [("vt", c_uint16), ("r1", c_uint16), ("r2", c_uint16), ("r3", c_uint16), ("value", _PropVariantValue)]


class IAudioCaptureClient(IUnknown):
    _iid_ = GUID("{C8ADBD64-E71E-48a0-A4DE-185C395CD317}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetBuffer", (["out"], POINTER(POINTER(ctypes.c_byte))), (["out"], POINTER(c_uint32)),
                  (["out"], POINTER(DWORD)), (["out"], POINTER(c_uint64)), (["out"], POINTER(c_uint64))),
        COMMETHOD([], HRESULT, "ReleaseBuffer", (["in"], c_uint32)),
        COMMETHOD([], HRESULT, "GetNextPacketSize", (["out"], POINTER(c_uint32))),
    ]


class IAudioClient(IUnknown):
    _iid_ = GUID("{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}")
    _methods_ = [
        COMMETHOD([], HRESULT, "Initialize", (["in"], c_int), (["in"], DWORD), (["in"], REFERENCE_TIME),
                  (["in"], REFERENCE_TIME), (["in"], POINTER(WAVEFORMATEX)), (["in"], POINTER(GUID))),
        COMMETHOD([], HRESULT, "GetBufferSize", (["out"], POINTER(c_uint32))),
        COMMETHOD([], HRESULT, "GetStreamLatency", (["out"], POINTER(REFERENCE_TIME))),
        COMMETHOD([], HRESULT, "GetCurrentPadding", (["out"], POINTER(c_uint32))),
        COMMETHOD([], HRESULT, "IsFormatSupported", (["in"], c_int), (["in"], POINTER(WAVEFORMATEX)), (["out"], POINTER(c_void_p))),
        COMMETHOD([], HRESULT, "GetMixFormat", (["out"], POINTER(c_void_p))),
        COMMETHOD([], HRESULT, "GetDevicePeriod", (["out"], POINTER(REFERENCE_TIME)), (["out"], POINTER(REFERENCE_TIME))),
        COMMETHOD([], HRESULT, "Start"),
        COMMETHOD([], HRESULT, "Stop"),
        COMMETHOD([], HRESULT, "Reset"),
        COMMETHOD([], HRESULT, "SetEventHandle", (["in"], c_void_p)),
        COMMETHOD([], HRESULT, "GetService", (["in"], POINTER(GUID)), (["out"], POINTER(c_void_p))),
    ]


class IActivateAudioInterfaceAsyncOperation(IUnknown):
    _iid_ = GUID("{72A22D78-CDE4-431D-B8CC-843A71199B6D}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetActivateResult", (["out"], POINTER(HRESULT)), (["out"], POINTER(POINTER(IUnknown)))),
    ]


class IActivateAudioInterfaceCompletionHandler(IUnknown):
    _iid_ = GUID("{41D949AB-9862-444A-80F6-C261334DA5EB}")
    _methods_ = [
        COMMETHOD([], HRESULT, "ActivateCompleted", (["in"], POINTER(IActivateAudioInterfaceAsyncOperation))),
    ]


class IAgileObject(IUnknown):
    _iid_ = GUID("{94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90}")
    _methods_ = []


class _CompletionHandler(comtypes.COMObject):
    # agile: Windows calls back on a worker thread
    _com_interfaces_ = [IActivateAudioInterfaceCompletionHandler, IAgileObject]

    def __init__(self):
        super().__init__()
        self.done = Event()

    def ActivateCompleted(self, operation):
        self.done.set()
        return 0


_mmdevapi = ctypes.WinDLL("Mmdevapi.dll")
_activate = _mmdevapi.ActivateAudioInterfaceAsync
_activate.restype = HRESULT
_activate.argtypes = [LPCWSTR, POINTER(GUID), POINTER(PROPVARIANT), c_void_p, POINTER(POINTER(IActivateAudioInterfaceAsyncOperation))]

VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback"
AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0
VT_BLOB = 65
AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM = 0x80000000
AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY = 0x08000000
AUDCLNT_BUFFERFLAGS_SILENT = 0x2


class AppLoopback:
    """Reads 16-bit mono PCM of one process tree at `sample_rate`. Use from a single (MTA) thread."""

    def __init__(self, pid: int, sample_rate: int = 16000):
        try:
            comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        except OSError:
            pass  # thread already initialised (e.g. STA main thread); the agile handler copes with both
        params = AUDIOCLIENT_ACTIVATION_PARAMS(
            AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
            AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(pid, PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE),
        )
        self._params = params  # must stay alive until activation completes
        pv = PROPVARIANT()
        pv.vt = VT_BLOB
        pv.value.blob = BLOB(sizeof(params), ctypes.cast(byref(params), c_void_p))

        handler = _CompletionHandler()
        handler_ptr = handler.QueryInterface(IActivateAudioInterfaceCompletionHandler)
        op = POINTER(IActivateAudioInterfaceAsyncOperation)()
        _activate(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK, byref(IAudioClient._iid_), byref(pv), handler_ptr, byref(op))
        if not handler.done.wait(5):
            raise TimeoutError("process loopback activation timed out")
        hr, unk = op.GetActivateResult()
        if hr < 0:
            raise OSError(f"process loopback activation failed: 0x{hr & 0xFFFFFFFF:08X}")
        self.client = unk.QueryInterface(IAudioClient)

        fmt = WAVEFORMATEX(1, 1, sample_rate, sample_rate * 2, 2, 16, 0)  # PCM, mono, 16-bit
        flags = AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM | AUDCLNT_STREAMFLAGS_SRC_DEFAULT_QUALITY
        self.client.Initialize(AUDCLNT_SHAREMODE_SHARED, flags, 2_000_000, 0, byref(fmt), None)  # 200 ms buffer
        svc = self.client.GetService(byref(IAudioCaptureClient._iid_))
        self.capture = ctypes.cast(svc, POINTER(IAudioCaptureClient))
        self.client.Start()

    def read(self) -> np.ndarray:
        """All audio captured since the last call (may be empty)."""
        chunks = []
        while self.capture.GetNextPacketSize() > 0:
            data, frames, flags, _, _ = self.capture.GetBuffer()
            if flags & AUDCLNT_BUFFERFLAGS_SILENT:
                chunks.append(np.zeros(frames, np.int16))
            else:
                chunks.append(np.ctypeslib.as_array(ctypes.cast(data, POINTER(ctypes.c_int16)), (frames,)).copy())
            self.capture.ReleaseBuffer(frames)
        return np.concatenate(chunks) if chunks else np.zeros(0, np.int16)

    def close(self) -> None:
        try:
            self.client.Stop()
        except Exception:
            pass
