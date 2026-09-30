"""What this computer has: processor, memory, graphics adapters and disk.

The capability report behind model choice and placement (and, later, the slim installer and
the low-memory checks). Everything is read with plain Windows calls through ctypes, so nothing
extra is bundled and nothing wakes a sleeping graphics card:

  * graphics adapters come from DXGI's adapter list - every vendor, not only NVIDIA - with the
    memory Windows says each one has: its own (dedicated) and what it may borrow from RAM
    (shared). Listing adapters creates no device, so an NVIDIA card asleep behind Optimus
    stays asleep.
  * instruction sets come from IsProcessorFeaturePresent. Windows versions older than the
    AVX constants answer "no" to all of them; that reads as unknown, not as absent.

Live NVIDIA readings (temperature, load, memory in use) stay in localflow/gpu.py.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x1022: "amd", 0x8086: "intel", 0x5143: "qualcomm"}
_MICROSOFT = 0x1414  # the Basic Render Driver and other software adapters
_DXGI_ADAPTER_FLAG_SOFTWARE = 2
_DXGI_ERROR_NOT_FOUND = -2005270526  # 0x887A0002 as a signed HRESULT


@dataclass(frozen=True)
class Gpu:
    name: str
    vendor: str  # nvidia | amd | intel | qualcomm | other
    dedicated_mb: int  # memory of its own (for integrated graphics, a small carve-out of RAM)
    shared_mb: int  # RAM it may borrow
    integrated: bool

    @property
    def usable_mb(self) -> int:
        """Memory a model could use on it: its own for a discrete card, own + borrowed for
        integrated graphics, whose "own" memory is only a slice of RAM anyway."""
        return self.dedicated_mb + (self.shared_mb if self.integrated else 0)

    def as_dict(self) -> dict:
        return {"name": self.name, "vendor": self.vendor, "dedicated_mb": self.dedicated_mb,
                "shared_mb": self.shared_mb, "integrated": self.integrated}


@dataclass(frozen=True)
class Cpu:
    name: str
    cores: int  # physical cores: inference scales with these, not threads
    threads: int
    isa: tuple[str, ...] | None  # e.g. ("avx", "avx2", "avx512"); None when Windows cannot say

    def as_dict(self) -> dict:
        return {"name": self.name, "cores": self.cores, "threads": self.threads,
                "isa": list(self.isa) if self.isa is not None else None}


@dataclass(frozen=True)
class Report:
    cpu: Cpu
    ram_gb: float
    gpus: tuple[Gpu, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {"cpu": self.cpu.as_dict(), "ram_gb": round(self.ram_gb, 1),
                "gpus": [g.as_dict() for g in self.gpus]}


# processor ---------------------------------------------------------------------------------------
def physical_cores() -> int:
    """Physical cores, not threads: inference scales with the former."""
    if os.name == "nt":
        try:
            from ctypes import wintypes

            class INFO(ctypes.Structure):  # SYSTEM_LOGICAL_PROCESSOR_INFORMATION, x64
                _fields_ = [("mask", ctypes.c_size_t), ("relationship", ctypes.c_int),
                            ("_pad", ctypes.c_int), ("union", ctypes.c_ubyte * 16)]

            size = wintypes.DWORD(0)
            k32 = ctypes.WinDLL("kernel32")
            k32.GetLogicalProcessorInformation(None, ctypes.byref(size))
            buf = (INFO * (size.value // ctypes.sizeof(INFO)))()
            if k32.GetLogicalProcessorInformation(buf, ctypes.byref(size)):
                cores = sum(1 for i in buf if i.relationship == 0)  # RelationProcessorCore
                if cores:
                    return cores
        except Exception:
            log.debug("could not count processor cores", exc_info=True)
    return max(1, (os.cpu_count() or 2) // 2)


# IsProcessorFeaturePresent constants (winnt.h)
_PF = {"sse4.2": 38, "avx": 39, "avx2": 40, "avx512": 41}


def instruction_sets() -> tuple[str, ...] | None:
    if os.name != "nt":
        return None
    try:
        present = ctypes.windll.kernel32.IsProcessorFeaturePresent
        have = tuple(name for name, pf in _PF.items() if present(pf))
    except Exception:
        log.debug("could not read processor features", exc_info=True)
        return None
    # Every x64 processor Windows 10 runs on has SSE4.2 in practice; a Windows too old to know
    # the constant answers no to all of them.
    return have or None


def cpu_name() -> str:
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return " ".join(str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).split())
        except OSError:
            pass
    import platform

    return platform.processor() or "unknown processor"


# memory ------------------------------------------------------------------------------------------
class _MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                ("pagefile", ctypes.c_ulonglong), ("avail_pagefile", ctypes.c_ulonglong),
                ("virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                ("ext", ctypes.c_ulonglong)]


def _memory() -> _MemoryStatus | None:
    if os.name != "nt":
        return None
    m = _MemoryStatus()
    m.length = ctypes.sizeof(_MemoryStatus)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return m
    except Exception:
        log.debug("could not read memory size", exc_info=True)
    return None


def ram_gb() -> float:
    m = _memory()
    return m.total / 2**30 if m else 16.0


def ram_free_gb() -> float | None:
    """RAM available right now (in use by nothing, or cache Windows would give up)."""
    m = _memory()
    return m.avail / 2**30 if m else None


# What loading a model needs (B5). Measured 2026-09-28 on the development machine: llama-server
# on the processor maps the model file and adds about 0.5 GB for its context; on the graphics
# card the file goes to the card and the process keeps about 1.4 GB of RAM (the CUDA runtime).
HEADROOM_GB = 0.75  # left free for Windows and the app being dictated into
LOW_MEMORY_GB = 9.0  # "8 GB" PCs report 7.4-7.9 GB; below this AI clean-up starts off


# Speech, measured 2026-09-28 (RAM added by loading and warming the model): Parakeet v3 on the
# processor 2.24 GB, Compact (int8) 0.78 GB; on the graphics card about 1 GB for any of them - the
# model lives on the card - plus the speech worker process itself.
SPEECH_CPU_RAM_GB = {"parakeet-v3": 2.3, "parakeet-v2": 2.3, "parakeet-v3-compact": 0.8}
SPEECH_CUDA_RAM_GB = 1.4


def speech_ram_gb(key: str, device: str, size_gb: float) -> float:
    """RAM speech model `key` takes on `device`; `size_gb` (its download) stands in for one
    not measured."""
    if device == "cuda":
        return SPEECH_CUDA_RAM_GB
    return SPEECH_CPU_RAM_GB.get(key, size_gb + 0.3)


def cleanup_ram_gb(model_gb: float, device: str) -> float:
    """RAM the bundled clean-up server takes for a model of `model_gb` on `device`."""
    return model_gb + 0.5 if device == "cpu" else 1.4


# disk --------------------------------------------------------------------------------------------
def disk_free_gb(folder: Path) -> float | None:
    """Free space on the drive that holds `folder`, which need not exist yet."""
    probe = Path(folder)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / 2**30
    except OSError:
        return None


# graphics ----------------------------------------------------------------------------------------
class _GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_ulong), ("d2", ctypes.c_ushort), ("d3", ctypes.c_ushort),
                ("d4", ctypes.c_ubyte * 8)]


class _AdapterDesc1(ctypes.Structure):  # DXGI_ADAPTER_DESC1
    _fields_ = [("description", ctypes.c_wchar * 128), ("vendor_id", ctypes.c_uint),
                ("device_id", ctypes.c_uint), ("subsys_id", ctypes.c_uint), ("revision", ctypes.c_uint),
                ("dedicated_video", ctypes.c_size_t), ("dedicated_system", ctypes.c_size_t),
                ("shared_system", ctypes.c_size_t), ("luid_low", ctypes.c_ulong),
                ("luid_high", ctypes.c_long), ("flags", ctypes.c_uint)]


# IID_IDXGIFactory1 {770aae78-f26f-4dba-a829-253c83d1b387}
_IID_FACTORY1 = _GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
# vtable slots: IUnknown 0-2, IDXGIObject 3-6, IDXGIFactory 7-11, IDXGIFactory1 12-13;
# IDXGIAdapter 7-9, IDXGIAdapter1 10
_RELEASE, _ENUM_ADAPTERS1, _GET_DESC1 = 2, 12, 10


def _com(obj: ctypes.c_void_p, slot: int, *argtypes):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[slot])


def _is_integrated(vendor: str, name: str, dedicated_mb: int) -> bool:
    """DXGI has no integrated flag, and asking D3D12 would wake the card, so: NVIDIA never is,
    a GPU with under 1 GB of its own memory always is, and otherwise the name decides (AMD's
    discrete cards are "RX"/"PRO"/"Instinct", Intel's "Arc A770"/"Arc B580")."""
    if vendor == "nvidia":
        return False
    if dedicated_mb < 1024:
        return True
    if vendor == "amd":
        return not re.search(r"\bRX\b|\bPro\b|\bInstinct\b|\bVega (56|64)\b|\bFirePro\b", name, re.I)
    if vendor == "intel":
        return not re.search(r"\bArc(\(TM\))? [AB]\d{3}", name, re.I)
    return vendor == "qualcomm"


def graphics_adapters() -> tuple[Gpu, ...]:
    """Every hardware graphics adapter, in the order Windows lists them (the one driving the
    main display first). Empty off Windows or when DXGI cannot be asked."""
    if os.name != "nt":
        return ()
    gpus: list[Gpu] = []
    factory = ctypes.c_void_p()
    try:
        dxgi = ctypes.WinDLL("dxgi")
        if dxgi.CreateDXGIFactory1(ctypes.byref(_IID_FACTORY1), ctypes.byref(factory)) != 0:
            return ()
        enum = _com(factory, _ENUM_ADAPTERS1, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        seen: set[tuple[int, int]] = set()
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            hr = enum(factory, i, ctypes.byref(adapter))
            if hr == _DXGI_ERROR_NOT_FOUND or hr != 0:
                break
            i += 1
            try:
                desc = _AdapterDesc1()
                if _com(adapter, _GET_DESC1, ctypes.POINTER(_AdapterDesc1))(adapter, ctypes.byref(desc)) != 0:
                    continue
                luid = (desc.luid_high, desc.luid_low)
                if (desc.flags & _DXGI_ADAPTER_FLAG_SOFTWARE or desc.vendor_id == _MICROSOFT
                        or luid in seen):
                    continue
                seen.add(luid)
                vendor = VENDORS.get(desc.vendor_id, "other")
                name = " ".join(desc.description.split())
                dedicated = int(desc.dedicated_video >> 20)
                gpus.append(Gpu(name=name, vendor=vendor, dedicated_mb=dedicated,
                                shared_mb=int(desc.shared_system >> 20),
                                integrated=_is_integrated(vendor, name, dedicated)))
            finally:
                _com(adapter, _RELEASE)(adapter)
    except Exception:
        log.debug("could not list graphics adapters", exc_info=True)
    finally:
        if factory:
            _com(factory, _RELEASE)(factory)
    return tuple(gpus)


# the report --------------------------------------------------------------------------------------
def detect() -> Report:
    return Report(
        cpu=Cpu(name=cpu_name(), cores=physical_cores(), threads=os.cpu_count() or 1,
                isa=instruction_sets()),
        ram_gb=ram_gb(),
        gpus=graphics_adapters(),
    )


def describe(report: Report, models_dir: Path | None = None) -> list[str]:
    """The report as lines a person can read (`localflow hardware`, the diagnostics file)."""
    c = report.cpu
    isa = ", ".join(s.upper() for s in c.isa) if c.isa else "unknown"
    lines = [f"Processor  {c.name}: {c.cores} cores, {c.threads} threads; instruction sets {isa}"]
    free = ram_free_gb()
    lines.append(f"Memory     {report.ram_gb:.1f} GB" + (f" ({free:.1f} GB free now)" if free is not None else ""))
    if not report.gpus:
        lines.append("Graphics   none found")
    for g in report.gpus:
        kind = "integrated, shares RAM" if g.integrated else "discrete"
        lines.append(f"Graphics   {g.name} ({g.vendor}, {kind}): {g.dedicated_mb / 1024:.1f} GB own, "
                     f"{g.shared_mb / 1024:.1f} GB shared")
    if models_dir is not None:
        disk = disk_free_gb(models_dir)
        if disk is not None:
            lines.append(f"Disk       {disk:.0f} GB free where models are kept ({models_dir})")
    return lines
