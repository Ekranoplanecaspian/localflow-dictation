"""The performance record: one file per release, so a regression shows as a number.

`localflow bench release` measures, in one gentle pass on the machine it runs on:

* start-up - how long until speech, and then clean-up, are ready in a fresh engine;
* latency and accuracy - the own-voice set streamed at real-time pace (key-up to text, WER);
* memory - the engine and the clean-up server, and the graphics memory LocalFlow uses;
* clean-up quality - the clean-up set (exact matches, must-not violations, model latency).

It writes `docs/perf/v<version>.json` and compares it with the newest earlier record. Numbers
are only comparable on the same machine, so each record carries the machine's description and
the comparison says when they differ. It needs its own engine, so LocalFlow must be quit first.
"""

from __future__ import annotations

import json
import platform
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from localflow import __version__
from localflow.config import Config

PERF_DIR = Path(__file__).resolve().parents[3] / "docs" / "perf"
COOL_C = 60  # start only once the graphics card is this cool (the laptop runs hot)

# A change counts as a regression when it is worse by more than both: a share, and an amount -
# the second so that a 12 ms latency does not "regress" by 25 % over 3 ms of noise.
RULES: list[tuple[str, str, float, float]] = [
    # (path, direction, relative, absolute); direction "up" = higher is worse
    ("startup.speech_ready_s", "up", 0.20, 0.5),
    ("startup.cleanup_ready_s", "up", 0.20, 0.5),
    ("latency.keyup_to_text_p50_ms", "up", 0.15, 30),
    ("latency.keyup_to_text_p95_ms", "up", 0.20, 60),
    ("accuracy.wer_pct", "up", 0.0, 0.5),
    ("memory.engine_private_mb", "up", 0.10, 150),
    ("memory.cleanup_private_mb", "up", 0.10, 150),
    ("memory.vram_mb", "up", 0.10, 200),
    ("cleanup.exact", "down", 0.0, 1.5),
    ("cleanup.violations", "up", 0.0, 0.5),
    ("cleanup.llm_ms_p50", "up", 0.20, 40),
]


def _gpu() -> dict[str, Any] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,memory.used,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        name, total, driver, used, temp = [p.strip() for p in out.stdout.splitlines()[0].split(",")]
        return {"name": name, "vram_mb": int(total), "driver": driver, "used_mb": int(used), "temp_c": int(temp)}
    except Exception:
        return None


def _cpu_name() -> str:
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except Exception:
        return platform.processor()


def _commit() -> str | None:
    """The commit measured, marked when the working tree had changes beyond it."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=PERF_DIR.parent.parent).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                               timeout=5, cwd=PERF_DIR.parent.parent).stdout.strip()
        return (head + ("+dirty" if dirty else "")) or None
    except Exception:
        return None


def _machine() -> dict[str, Any]:
    import psutil

    gpu = _gpu()
    return {
        "cpu": _cpu_name(),
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30),
        "gpu": gpu["name"] if gpu else None,
        "vram_mb": gpu["vram_mb"] if gpu else None,
        "gpu_driver": gpu["driver"] if gpu else None,
        "windows": platform.version(),
    }


class Probe:
    """Samples the engine process, its clean-up server and the graphics memory once a second
    while the stream runs. `stream` calls `started(pid)`, `ready(what)` and `stop()`."""

    def __init__(self) -> None:
        gpu = _gpu()
        self.vram_before = gpu["used_mb"] if gpu else None
        self.t0 = time.perf_counter()
        self.times: dict[str, float] = {}
        self.peak = {"engine_private_mb": 0.0, "cleanup_private_mb": 0.0, "vram_mb": 0.0}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def started(self, pid: int | None) -> None:
        self.t0 = time.perf_counter()
        if pid:
            self._thread = threading.Thread(target=self._run, args=(pid,), name="perf-probe", daemon=True)
            self._thread.start()

    def ready(self, what: str) -> None:
        self.times.setdefault(what, time.perf_counter() - self.t0)

    def _run(self, pid: int) -> None:
        import psutil

        while not self._stop.wait(1.0):
            try:
                engine = psutil.Process(pid)
                self._keep("engine_private_mb", engine.memory_info().private / 2**20)
                cleanup = sum(c.memory_info().private for c in engine.children(recursive=True)
                              if c.name().lower() == "llama-server.exe")
                self._keep("cleanup_private_mb", cleanup / 2**20)
            except Exception:
                pass
            gpu = _gpu()
            if gpu and self.vram_before is not None:
                self._keep("vram_mb", gpu["used_mb"] - self.vram_before)

    def _keep(self, key: str, value: float) -> None:
        self.peak[key] = max(self.peak[key], value)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(3)


def _engine_running() -> bool:
    from localflow.service.server import _listed_pid, _pid_alive

    pid = _listed_pid()
    return pid is not None and _pid_alive(pid)


def _get(record: dict, path: str) -> float | None:
    cur: Any = record
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur if isinstance(cur, (int, float)) else None


def compare(new: dict, old: dict) -> list[tuple[str, float | None, float | None, bool]]:
    """(measure, old, new, regressed) for every rule both records can answer."""
    rows = []
    for path, direction, rel, absolute in RULES:
        a, b = _get(old, path), _get(new, path)
        if a is None or b is None:
            continue
        worse = (b - a) if direction == "up" else (a - b)
        regressed = worse > absolute and worse > abs(a) * rel
        rows.append((path, a, b, regressed))
    return rows


def _version_key(path: Path) -> tuple:
    return tuple(int(p) if p.isdigit() else p for p in path.stem.lstrip("v").replace("-", ".").split("."))


def previous(before: str, directory: Path = PERF_DIR) -> dict | None:
    """The newest record for a version other than `before`."""
    files = [p for p in directory.glob("v*.json") if p.stem != f"v{before}"]
    if not files:
        return None
    return json.loads(max(files, key=_version_key).read_text(encoding="utf-8"))


def release(cfg: Config) -> int:
    from localflow import bench

    if _engine_running():
        print("LocalFlow's engine is running. Quit LocalFlow first: the record needs an engine of its own, "
              "and two do not fit on the graphics card.")
        return 2
    while (gpu := _gpu()) and gpu["temp_c"] > COOL_C:
        print(f"graphics card at {gpu['temp_c']} C; waiting until it is under {COOL_C} C ...", flush=True)
        time.sleep(15)

    probe = Probe()
    run = bench.stream(cfg, "own", probe=probe)
    probe.stop()
    if run is None:
        return 1
    print("\nclean-up set ...", flush=True)
    quality = bench.cleanup(cfg, show=False)

    record = {
        "version": __version__,
        "commit": _commit(),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "machine": _machine(),
        "models": {"speech": run.model, "speech_device": run.device_used, "speech_precision": run.precision,
                   "cleanup": quality.get("model")},
        "startup": {"speech_ready_s": round(probe.times.get("speech", run.load_s), 2),
                    "cleanup_ready_s": round(probe.times["cleanup"], 2) if "cleanup" in probe.times else None},
        "latency": {"keyup_to_text_p50_ms": run.stt_ms_p50, "keyup_to_text_p95_ms": run.stt_ms_p95,
                    "takes": run.files, "audio_s": run.audio_s},
        "accuracy": {"wer_pct": run.wer, "set": "own"},
        "memory": {k: round(v) for k, v in probe.peak.items()},
        "cleanup": {k: quality.get(k) for k in ("cases", "exact", "violations", "word_error_pct",
                                                 "llm_ms_p50", "llm_ms_p95", "command_pass", "command_cases")},
    }
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    out = PERF_DIR / f"v{__version__}.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"\nperformance record written to {out}")
    return report(record, previous(__version__))


def report(record: dict, old: dict | None) -> int:
    if old is None:
        print("no earlier record to compare with: this one is the baseline")
        return 0
    print(f"\ncompared with v{old['version']} ({old.get('date', '?')}):")
    if old.get("machine", {}).get("gpu") != record["machine"].get("gpu") or \
            old.get("machine", {}).get("cpu") != record["machine"].get("cpu"):
        print("  (measured on a different machine: differences may be the hardware, not the code)")
    regressions = 0
    for path, a, b, bad in compare(record, old):
        regressions += bad
        print(f"  {'REGRESSED' if bad else 'ok':<9} {path:<30} {a:>9} -> {b:<9}")
    print(f"{regressions} regression(s)")
    return 1 if regressions else 0
