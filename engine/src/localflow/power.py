"""Power source detection (Windows)."""

from __future__ import annotations

import ctypes
from ctypes import wintypes


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", wintypes.DWORD),
        ("BatteryFullLifeTime", wintypes.DWORD),
    ]


def on_ac_power() -> bool:
    """True on mains (or when unknown, e.g. a desktop without a battery)."""
    try:
        status = _SYSTEM_POWER_STATUS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return True
        return status.ACLineStatus != 0  # 0 = offline (battery), 1 = online, 255 = unknown
    except Exception:
        return True
