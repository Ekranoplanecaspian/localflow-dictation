"""Benchmark speech as on a smaller PC than this one (B4).

The process is pinned to N physical cores (2N logical CPUs: cores with hyper-threading, as
laptops have), onnxruntime is told to use N threads (LOCALFLOW_STT_THREADS; it would otherwise
size itself for every core this machine has), and CUDA is hidden. What it cannot do is make
each core slower: an older laptop's cores are perhaps half as fast, so read the 2-core numbers
of a fast machine as roughly an older 4-core one.

  run     `localflow bench run` in this process: whole files, accuracy and ms per audio second
  stream  `localflow bench stream`: a real engine process at real-time pace, key-up -> final
          latency, on settings of its own (processor only, the named model, no AI clean-up) in
          a scratch APPDATA, so your settings and your perf.json are untouched. The engine's
          own per-decode timings end up in that folder's perf.json.

usage (from the repository, with the virtualenv):
  .venv\\Scripts\\python scripts\\weak_pc.py run 4 parakeet-v3-compact
  .venv\\Scripts\\python scripts\\weak_pc.py stream 2 parakeet-v3 [--limit 10]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "engine"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("mode", choices=["run", "stream"])
    p.add_argument("cores", type=int, help="physical cores to keep")
    p.add_argument("speech", help="parakeet-v3, parakeet-v3-compact, parakeet-v2, ...")
    p.add_argument("--set", default=None, help="bench set (run: all, stream: own)")
    p.add_argument("--limit", default=None)
    a = p.parse_args()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "LOCALFLOW_STT_THREADS": str(a.cores),
           "PYTHONIOENCODING": "utf-8"}
    extra = ["--limit", a.limit] if a.limit else []
    if a.mode == "run":
        # in-process: the benchmark's own process is the one pinned and measured
        env["LOCALFLOW_SPEECH_IN_PROCESS"] = "1"
        args = ["bench", "run", "--set", a.set or "all", "--device", "cpu", "--speech", a.speech, *extra]
    else:
        appdata = Path(tempfile.mkdtemp(prefix=f"weak-pc-{a.speech}-{a.cores}-"))
        (appdata / "LocalFlow").mkdir()
        env["APPDATA"] = str(appdata)
        setup = (
            "from localflow.config import Config, CONFIG_PATH\n"
            "from localflow.stt import catalogue\n"
            "cfg = Config()\n"
            f"catalogue.get({a.speech!r}).apply(cfg.stt)\n"
            "cfg.stt.device = 'cpu'\n"
            "cfg.compute.mode = 'cpu'\n"
            "cfg.compute.auto_speech = cfg.compute.auto_cleanup = False\n"
            "cfg.postprocess.llm_cleanup = False\n"
            "cfg.save(CONFIG_PATH)\n"
        )
        subprocess.run([sys.executable, "-c", setup], env=env, check=True, cwd=ENGINE)
        print(f"[settings and timings in {appdata / 'LocalFlow'}]", flush=True)
        args = ["bench", "stream", "--set", a.set or "own", *extra]
    proc = subprocess.Popen([sys.executable, "-m", "localflow", *args], env=env, cwd=ENGINE)
    import psutil

    psutil.Process(proc.pid).cpu_affinity(list(range(2 * a.cores)))  # its engine inherits it
    print(f"[as a {a.cores}-core PC: logical CPUs 0-{2 * a.cores - 1}, {a.cores} speech threads, no CUDA]", flush=True)
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
