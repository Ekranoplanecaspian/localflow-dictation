"""Which model runs: chosen for this computer, best first.

The rule is quality first. Automatic picks the most accurate model that can do its job in a
reasonable time on the device it is going to run on, and only steps down when that device is
too slow for it. Heat and load are handled separately (localflow/placement.py moves models
between the graphics card and the processor); this module never trades accuracy for a cooler
GPU.

"Reasonable time" is judged from what this machine actually does. Every dictation's speech and
clean-up timings are recorded per model and device (perf.json beside the settings), and once
there are enough of them their median decides. Until then an estimate stands in: another model
measured on this machine and device, scaled by how the two compare; failing that, the numbers
measured on the machine this was developed on, scaled by processor cores for the processor.

A model the user picked by hand is never replaced: Automatic is a choice in the Hub like any
model, and choosing a model turns it off.
"""

from __future__ import annotations

import json
import logging
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from localflow import hwinfo

log = logging.getLogger(__name__)

# Measured 2026-09-22 on a Ryzen AI 9 HX 370 (12 cores) and an RTX 4060 laptop GPU.
# Speech: milliseconds per second of audio. Clean-up: median milliseconds per clean-up.
#
# Speech on the processor measured again 2026-09-29 (B4), the way the engine times it: the
# median per decode over the 30 own-voice dictations streamed at real time (scripts/weak_pc.py),
# on AC power. 2026-09-22's 163 and 124 were about 2.4x too slow, and scaled straight by cores
# they made a 4-core PC look six times slower than it is:
#
#                          12 cores   4 cores   2 cores     key-up -> final text (median)
#   Parakeet v3              67        82       190         327 / 462 / 794 ms
#   Parakeet v3 Compact      57        60       144         309 / 340 / 623 ms
#
# Parakeet v2 has v3's encoder, so its figure is v3's.
MEASURED_ON_CORES = 12
PRIORS: dict[tuple[str, str, str], float] = {
    ("speech", "parakeet-v3", "cuda"): 41, ("speech", "parakeet-v3", "cpu"): 67,
    ("speech", "parakeet-v2", "cuda"): 46, ("speech", "parakeet-v2", "cpu"): 67,
    ("speech", "parakeet-v3-compact", "cuda"): 220, ("speech", "parakeet-v3-compact", "cpu"): 57,
    ("speech", "whisper-turbo", "cuda"): 92, ("speech", "whisper-turbo", "cpu"): 1443,
    ("cleanup", "qwen3-4b", "cuda"): 381, ("cleanup", "qwen3-4b", "cpu"): 767,
    ("cleanup", "phi-4-mini", "cuda"): 185, ("cleanup", "phi-4-mini", "cpu"): 575,
    ("cleanup", "gemma-4-e2b", "cuda"): 208, ("cleanup", "gemma-4-e2b", "cpu"): 555,
    # AMD/Intel graphics through Vulkan (B3), from the Radeon 890M on 2026-09-28: Qwen3 4B took
    # 788 ms there against 1042 ms on the processor in the same session, so each model's
    # processor figure x 0.76. Not scaled by anything: built-in graphics vary more than cores
    # tell, so the first clean-ups there are what decides.
    ("cleanup", "qwen3-4b", "vulkan"): 580, ("cleanup", "phi-4-mini", "vulkan"): 435,
    ("cleanup", "gemma-4-e2b", "vulkan"): 420,
}

# What counts as reasonable. Speech: 250 ms per second of audio puts the final text of a
# 5-second dictation about a second after the key comes up. Clean-up: 1.5 s at the median.
SPEECH_BUDGET_MS_PER_S = 250
CLEANUP_BUDGET_MS = 1500
MIN_SAMPLES = 5  # measurements needed before they outrank the estimate
# Below this no real decode or clean-up ever comes (the quickest seen: speech on the graphics
# card, 15 ms per second of audio). A test's instant fake speech model once filled perf.json
# with zeros, which made the processor look free; such values are dropped, on load too.
IMPOSSIBLY_QUICK = 1.0
KEEP_SAMPLES = 30

# Automatic chooses among these, most accurate first. Parakeet v2 is English-only and was less
# accurate on real (accented) dictation than v3; Whisper Turbo is for languages Parakeet lacks,
# which the router already sends to it without a model change.
SPEECH_ORDER = ("parakeet-v3", "parakeet-v3-compact")
CLEANUP_ORDER = ("qwen3-4b", "phi-4-mini", "gemma-4-e2b")
# Downloaded without asking when Automatic wants it: small enough not to surprise anyone.
QUIET_DOWNLOAD_GB = 1.0
# When nothing is quick enough, speed alone would pick the least accurate model for a few tens of
# milliseconds. Accuracy still wins among those within this factor of the quickest.
NEAR_QUICKEST = 1.2


@dataclass(frozen=True)
class Hardware:
    cpu_cores: int
    ram_gb: float
    vram_mb: int | None  # None: no usable NVIDIA GPU
    report: hwinfo.Report | None = None  # the rest of what this machine has (B1)

    @property
    def nvidia(self) -> bool:
        """An NVIDIA card is present (from Windows' adapter list, not from NVML, whose first
        reading can fail during an Optimus hand-off)."""
        if self.report is None:
            return self.vram_mb is not None
        return any(g.vendor == "nvidia" for g in self.report.gpus)

    @property
    def other_gpu(self) -> hwinfo.Gpu | None:
        """The graphics the Vulkan build would use: the first that is not NVIDIA's (B3)."""
        if self.report is None:
            return None
        return next((g for g in self.report.gpus if g.vendor != "nvidia"), None)

    def as_dict(self) -> dict:
        return {"cpu_cores": self.cpu_cores, "ram_gb": round(self.ram_gb, 1), "vram_mb": self.vram_mb,
                **({"cpu": self.report.cpu.as_dict(), "gpus": [g.as_dict() for g in self.report.gpus]}
                   if self.report else {})}


def detect_hardware(vram_mb: int | None) -> Hardware:
    report = hwinfo.detect()
    return Hardware(cpu_cores=report.cpu.cores, ram_gb=report.ram_gb, vram_mb=vram_mb, report=report)


# measurements ----------------------------------------------------------------------------------
def _default_perf_path() -> Path:
    from localflow.config import CONFIG_DIR

    return CONFIG_DIR / "perf.json"


PERF_PATH: Path | None = _default_perf_path()  # None keeps measurements in memory only (tests)


class PerfLog:
    """Recent timings per (kind, model, device), kept across runs."""

    def __init__(self, path: Path | None):
        self.path = path
        self.samples: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._saved = 0.0
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.samples = {k: [float(x) for x in v if float(x) >= IMPOSSIBLY_QUICK][-KEEP_SAMPLES:]
                                for k, v in data.items()}
            except Exception:
                log.warning("ignoring unreadable %s", path)

    @staticmethod
    def _key(kind: str, model: str, device: str) -> str:
        return f"{kind}:{model}:{device}"

    def record(self, kind: str, model: str, device: str, value: float) -> None:
        if value < IMPOSSIBLY_QUICK:
            return
        with self._lock:
            xs = self.samples.setdefault(self._key(kind, model, device), [])
            xs.append(round(float(value), 1))
            del xs[:-KEEP_SAMPLES]
            if self.path is not None and time.monotonic() - self._saved > 30:
                self._save()

    def _save(self) -> None:
        from localflow.config import write_atomic

        try:
            write_atomic(self.path, json.dumps(self.samples))
            self._saved = time.monotonic()
        except Exception as e:
            log.debug("could not save timings: %s", e)

    def flush(self) -> None:
        with self._lock:
            if self.path is not None:
                self._save()

    def measured(self, kind: str, model: str, device: str) -> float | None:
        xs = self.samples.get(self._key(kind, model, device), [])
        return statistics.median(xs) if len(xs) >= MIN_SAMPLES else None

    def estimate(self, kind: str, model: str, device: str, hw: Hardware) -> tuple[float | None, bool]:
        """(milliseconds, measured on this machine?). Not measured yet: from another model that
        has been, on this device, scaled by how their priors compare - a machine twice as slow
        as the development one is twice as slow for every model (the processor's measurements
        above: v3 took 1.2x Compact at 12 cores, 1.4x at 4 and 1.3x at 2). Only failing that,
        the prior itself, scaled by cores on the processor.

        The sibling matters: an estimate too slow for the more accurate model used to be
        permanent, since a model never chosen is never measured (B4 found Automatic keeping a
        4-core PC on Parakeet Compact for good)."""
        seen = self.measured(kind, model, device)
        if seen is not None:
            return seen, True
        prior = PRIORS.get((kind, model, device))
        if prior is None:
            return None, False
        for (k, other, d), other_prior in PRIORS.items():
            if k == kind and d == device and other != model:
                other_seen = self.measured(kind, other, device)
                if other_seen is not None:
                    return prior * other_seen / other_prior, False
        if device == "cpu":
            prior *= cpu_scale(kind, hw.cpu_cores)
        return prior, False


def cpu_scale(kind: str, cores: int) -> float:
    """How much slower than on the 12-core development machine a job is with `cores`. Speech
    follows the 2026-09-29 measurements: hardly slower down to 4 cores (the model's work does not
    spread over more), then steeply so. Clean-up is not measured that way and keeps the plain
    ratio. Neither knows how fast each core is - that only this machine's own timings tell."""
    cores = max(1, cores)
    if kind != "speech":
        return MEASURED_ON_CORES / cores
    if cores >= 4:
        return 1 + 0.15 * max(0, MEASURED_ON_CORES - cores) / 8  # 12 cores: 1.0, 4 cores: 1.15
    return 1.15 * (1 + 1.4 * (4 / cores - 1))  # 2 cores: 2.76 (measured 2.8 and 2.5), 1 core: 6.0


# the choice ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Choice:
    key: str
    why: str


def choose_speech(device: str, hw: Hardware, perf: PerfLog, available: Callable[[str], bool],
                  short_of_memory: Callable[[str], str | None] = lambda key: None) -> Choice:
    """The most accurate speech model that fits in memory and keeps up on `device`.
    `short_of_memory(key)` says why a model would not fit right now ("Parakeet v3 needs about
    3.1 GB of free memory, and 1.9 GB is free"), or None when it does. Memory is a hard limit,
    not a trade: a model that does not fit is not loaded at all, so it is skipped (B5)."""
    return _choose("speech", SPEECH_ORDER, SPEECH_BUDGET_MS_PER_S, device, hw, perf, available, short_of_memory)


def choose_cleanup(device: str, hw: Hardware, perf: PerfLog, available: Callable[[str], bool]) -> Choice:
    """The most careful clean-up model that answers in reasonable time on `device`."""
    return _choose("cleanup", CLEANUP_ORDER, CLEANUP_BUDGET_MS, device, hw, perf, available)


def _choose(kind: str, order: tuple[str, ...], budget: float, device: str, hw: Hardware, perf: PerfLog,
            available: Callable[[str], bool],
            short_of_memory: Callable[[str], str | None] = lambda key: None) -> Choice:
    unit = "speech" if kind == "speech" else "clean-up"
    timed = []
    short: list[tuple[str, str]] = []  # (key, why it does not fit), most accurate first
    for key in order:  # most accurate first
        if not available(key):
            continue
        reason = short_of_memory(key)
        if reason:
            short.append((key, reason))
            continue
        ms, seen = perf.estimate(kind, key, device, hw)
        if ms is None:
            continue
        if ms <= budget:
            if short:
                return Choice(key, f"{short[0][1]}; this is the most accurate one that fits and keeps up "
                                   f"on the {_where(device)} ({_speed(ms, unit)}, {'measured' if seen else 'estimated'})")
            return Choice(key, _why(key, device, ms, seen, first=key == order[0], unit=unit))
        timed.append((key, ms, seen))
    if not timed:
        if short:  # nothing fits: the one that needs least, which is the last in the order
            return Choice(short[-1][0], f"{short[0][1]}; nothing fits, so the smallest")
        return Choice(order[0], "the default")
    quickest = min(ms for _, ms, _ in timed)
    key, ms, seen = next(t for t in timed if t[1] <= quickest * NEAR_QUICKEST)
    speed = _speed(ms, unit)
    prefix = f"{short[0][1]}; " if short else ""
    return Choice(key, f"{prefix}nothing is quick enough on this {_where(device)}; the most accurate of the quickest "
                       f"({speed}, {'measured' if seen else 'estimated'})")


# how a model suits this PC -------------------------------------------------------------------------
# What Windows and the apps being dictated into keep for themselves, for "too big for this PC".
# Stricter than the start-up check (HEADROOM_GB, against free memory right now): this rating is
# about the PC, not the moment, and a model that leaves Windows 2 GB makes everything else crawl.
KEEP_FOR_WINDOWS_GB = 4.0


@dataclass(frozen=True)
class Fit:
    rating: str  # good | slow | too-big
    why: str


def fit(kind: str, key: str, device: str, hw: Hardware, perf: PerfLog, *,
        ram_gb: float, vram_mb: int | None = None) -> Fit:
    """How model `key` would suit this PC on `device` (where it would run): too big for its
    memory, quick enough (the same budget Automatic uses), or working but slowly. `ram_gb` and
    `vram_mb` are what the model takes there."""
    unit = "speech" if kind == "speech" else "clean-up"
    where = device_words(device, hw)
    if device == "cuda" and vram_mb and hw.vram_mb and vram_mb > hw.vram_mb * 0.9:
        return Fit("too-big", f"needs about {vram_mb / 1024:.1f} GB on the graphics card, which has "
                              f"{hw.vram_mb / 1024:.0f} GB")
    if hw.ram_gb and ram_gb + KEEP_FOR_WINDOWS_GB > hw.ram_gb:
        return Fit("too-big", f"needs about {ram_gb:.1f} GB of memory, and this PC has {hw.ram_gb:.0f} GB")
    ms, seen = perf.estimate(kind, key, device, hw)
    if ms is None:
        return Fit("good", f"runs on {where}")
    how = "measured" if seen else "estimated"
    budget = SPEECH_BUDGET_MS_PER_S if kind == "speech" else CLEANUP_BUDGET_MS
    if ms <= budget:
        return Fit("good", f"quick on {where} ({_speed(ms, unit)}, {how})")
    return Fit("slow", f"{_speed(ms, unit)} on {where}, {how}")


def _where(device: str) -> str:
    return {"cuda": "graphics card", "vulkan": "built-in graphics"}.get(device, "processor")


def device_words(device: str, hw: Hardware | None) -> str:
    """"the graphics card", "the processor", or for Vulkan "the built-in graphics" or the card's
    name ("the AMD Radeon RX 7600")."""
    if device == "vulkan" and hw is not None and hw.other_gpu is not None and not hw.other_gpu.integrated:
        return f"the {hw.other_gpu.name}"
    return f"the {_where(device)}"


def _speed(ms: float, unit: str) -> str:
    return f"{ms / 1000:.2f} s per second of speech" if unit == "speech" else f"{ms / 1000:.1f} s per clean-up"


def _why(key: str, device: str, ms: float, seen: bool, *, first: bool, unit: str) -> str:
    how = "measured" if seen else "estimated"
    if first:
        return f"the most accurate, and quick enough on the {_where(device)} ({_speed(ms, unit)}, {how})"
    return f"the most accurate one that keeps up on this {_where(device)} ({_speed(ms, unit)}, {how})"
