"""What the graphics card is doing right now: temperature, load, memory.

Read through NVML, the monitoring library every NVIDIA driver installs (nvml.dll on Windows),
called directly with ctypes so there is no package to bundle and nothing to go wrong on a
machine without an NVIDIA card: `GpuMonitor.sample()` simply returns None there.

An NVML query costs microseconds and does not wake the GPU, so polling every few seconds is
free - unlike `nvidia-smi`, which is a process launch each time.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from dataclasses import dataclass

log = logging.getLogger(__name__)

_NVML_TEMPERATURE_GPU = 0
_NVML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


@dataclass(frozen=True)
class GpuSample:
    temp_c: int
    util_pct: int  # share of the last sample period the GPU was busy, all processes
    mem_used_mb: int
    mem_total_mb: int
    slowdown_c: int | None  # where the driver starts throttling clocks

    def as_dict(self) -> dict:
        return {"temp_c": self.temp_c, "util_pct": self.util_pct, "mem_used_mb": self.mem_used_mb,
                "mem_total_mb": self.mem_total_mb, "slowdown_c": self.slowdown_c}


class GpuMonitor:
    """One NVML session for the process. Thread-safe; failures are sticky, so a machine without
    NVML pays for the attempt once."""

    def __init__(self, index: int = 0):
        self.index = index
        self._lib = None
        self._handle = None
        self._failed = False
        self._lock = threading.Lock()
        self.name: str | None = None

    def _open(self) -> bool:
        if self._handle is not None:
            return True
        if self._failed:
            return False
        try:
            if os.name == "nt":
                lib = ctypes.WinDLL("nvml.dll")
            else:
                lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                raise OSError("nvmlInit failed")
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(self.index), ctypes.byref(handle)) != 0:
                raise OSError(f"no NVIDIA GPU at index {self.index}")
            self._lib, self._handle = lib, handle
            buf = ctypes.create_string_buffer(96)
            if lib.nvmlDeviceGetName(handle, buf, ctypes.c_uint(len(buf))) == 0:
                self.name = buf.value.decode("utf-8", "replace").replace("NVIDIA GeForce ", "")
            return True
        except (OSError, AttributeError) as e:
            self._failed = True
            log.info("GPU monitoring unavailable (%s)", e)
            return False

    def driver_version(self) -> str | None:
        """The NVIDIA driver's version ("581.57"), or None with no NVIDIA card."""
        with self._lock:
            if not self._open():
                return None
            buf = ctypes.create_string_buffer(80)
            if self._lib.nvmlSystemGetDriverVersion(buf, ctypes.c_uint(len(buf))) != 0:
                return None
            return buf.value.decode("ascii", "replace") or None

    def sample(self) -> GpuSample | None:
        with self._lock:
            if not self._open():
                return None
            lib, h = self._lib, self._handle
            temp = ctypes.c_uint()
            util = _Utilization()
            mem = _Memory()
            if (lib.nvmlDeviceGetTemperature(h, _NVML_TEMPERATURE_GPU, ctypes.byref(temp)) != 0
                    or lib.nvmlDeviceGetUtilizationRates(h, ctypes.byref(util)) != 0
                    or lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(mem)) != 0):
                return None  # the GPU can be briefly unreachable (driver reset, Optimus hand-off)
            slowdown = ctypes.c_uint()
            has_slowdown = lib.nvmlDeviceGetTemperatureThreshold(
                h, _NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, ctypes.byref(slowdown)) == 0
            return GpuSample(
                temp_c=int(temp.value),
                util_pct=int(util.gpu),
                mem_used_mb=int(mem.used >> 20),
                mem_total_mb=int(mem.total >> 20),
                slowdown_c=int(slowdown.value) if has_slowdown and slowdown.value else None,
            )


_monitor: GpuMonitor | None = None


def monitor() -> GpuMonitor:
    global _monitor
    if _monitor is None:
        _monitor = GpuMonitor()
    return _monitor
