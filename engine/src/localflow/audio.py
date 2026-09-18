"""Always-on microphone capture with a pre-roll ring buffer.

The input stream runs continuously (cheap) so that when the hotkey fires we can
include the ~500 ms *before* the press: people start talking as they press.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from localflow.config import AudioConfig

log = logging.getLogger(__name__)

BLOCK_MS = 20
COINIT_APARTMENTTHREADED = 0x2
RPC_E_CHANGED_MODE = 0x80010106


def ensure_com_initialized() -> None:
    """PortAudio's WASAPI backend needs COM initialised on the thread that opens the stream.
    Without it, opening from a worker thread fails with a misleading WDM-KS host error."""
    import ctypes

    hr = ctypes.windll.ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED) & 0xFFFFFFFF
    if hr not in (0x0, 0x1, RPC_E_CHANGED_MODE):  # S_OK, S_FALSE (already), other mode (fine)
        log.debug("CoInitializeEx returned 0x%08x", hr)


def list_input_devices() -> list[tuple[int, str, str]]:
    apis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append((i, d["name"], apis[d["hostapi"]]["name"]))
    return out


def resolve_input_device(pref: int | str | None) -> tuple[int | None, object | None]:
    """Pick a device index, preferring the WASAPI host API (lowest latency on Windows).

    Returns (device_index, extra_settings). MME truncates device names to 31 chars, so
    matching is done by prefix in both directions.
    """
    if isinstance(pref, int):
        return pref, None
    devices = sd.query_devices()
    apis = sd.query_hostapis()
    if pref is None:
        default_in = sd.default.device[0]
        if default_in is None or default_in < 0:
            return None, None
        wanted = devices[default_in]["name"]
    else:
        wanted = pref
    best_any: int | None = None
    for i, d in enumerate(devices):
        if d["max_input_channels"] <= 0:
            continue
        name = d["name"]
        w, n = wanted.lower(), name.lower()
        if not (n.startswith(w) or w.startswith(n) or w in n):
            continue
        if apis[d["hostapi"]]["name"] == "Windows WASAPI":
            try:
                return i, sd.WasapiSettings(auto_convert=True)
            except TypeError:  # older sounddevice without auto_convert
                return i, None
        if best_any is None:
            best_any = i
    return best_any, None


class Recorder:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.sample_rate = cfg.sample_rate
        self.blocksize = int(self.sample_rate * BLOCK_MS / 1000)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, cfg.preroll_ms // BLOCK_MS))
        self._chunks: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._max_blocks = int(cfg.max_seconds * 1000 / BLOCK_MS)
        self.overflowed = False
        self.level = 0.0  # current mic level 0..1 (-50 dBFS -> 0, -10 dBFS -> 1), read by the overlay
        # Called from the audio thread with each 20 ms block while recording (streams to the engine).
        self.on_block: Callable[[np.ndarray], None] | None = None

    # stream --------------------------------------------------------------------------
    def start_stream(self) -> None:
        ensure_com_initialized()
        device, extra = resolve_input_device(self.cfg.device)
        kwargs = dict(samplerate=self.sample_rate, channels=1, dtype="float32",
                      blocksize=self.blocksize, device=device, callback=self._callback)
        last_error: Exception | None = None
        for attempt in range(4):  # a device just released by a previous instance can refuse for ~1 s
            try:
                self._stream = sd.InputStream(extra_settings=extra, **kwargs) if extra else sd.InputStream(**kwargs)
                self._stream.start()
                last_error = None
                break
            except Exception as e:
                last_error = e
                log.debug("open device %s failed (attempt %d): %s", device, attempt + 1, e)
                time.sleep(0.5)
        if last_error is not None:  # WASAPI refused -> fall back to the default host API (MME)
            log.warning("Could not open device %s via WASAPI (%s); falling back to default device", device, last_error)
            self._stream = sd.InputStream(**{**kwargs, "device": None})
            self._stream.start()
        info = sd.query_devices(self._stream.device, "input")
        log.info("Mic: %s @ %d Hz (latency %.0f ms)", info["name"], self.sample_rate, self._stream.latency * 1000)

    def close(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.debug("audio status: %s", status)
        block = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(block * block))) + 1e-9
        self.level = min(1.0, max(0.0, (20.0 * np.log10(rms) + 50.0) / 40.0))
        stream_it = False
        with self._lock:
            self._preroll.append(block)
            if self._recording:
                self._chunks.append(block)
                stream_it = True
                if len(self._chunks) >= self._max_blocks:
                    self._recording = False
                    self.overflowed = True
        if stream_it and self.on_block is not None:
            try:
                self.on_block(block)
            except Exception:
                log.exception("on_block failed")

    # recording -------------------------------------------------------------------------
    def begin(self) -> np.ndarray:
        """Start a take. Returns the pre-roll audio (the ~500 ms before the press)."""
        with self._lock:
            preroll = list(self._preroll)
            self._chunks = preroll
            self._recording = True
            self.overflowed = False
        return np.concatenate(preroll) if preroll else np.zeros(0, dtype=np.float32)

    def end(self) -> np.ndarray:
        with self._lock:
            self._recording = False
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    @property
    def is_recording(self) -> bool:
        return self._recording

    @staticmethod
    def rms_db(audio: np.ndarray) -> float:
        if audio.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        return 20 * float(np.log10(max(rms, 1e-9)))
