"""Microphone helpers for the command line and the benchmarks: listing input devices and
opening one through WASAPI. Dictation's own capture lives in the shell (app/src-tauri/src/audio.rs).
"""

from __future__ import annotations

import logging

import sounddevice as sd

log = logging.getLogger(__name__)

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
