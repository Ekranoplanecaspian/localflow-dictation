"""Downloading one model in a process of its own, so that it can be stopped.

huggingface_hub downloads through Xet, which here is 7 to 15 times quicker than its plain HTTP
path (measured 2026-10-01 on the same file: 11-22 MB/s against 1.5-3). Xet cannot be
interrupted from inside: an exception raised from the progress callback is swallowed, and the
download runs on to the end. A process can be ended at any moment, so each model download runs
in one. Neither way resumes a partial file - Xet starts again from zero - so what a stopped
download leaves behind is deleted.

The child is `localflow fetch <kind> <key> <device>`, with one line per event on stdout:

    P <done> <total>        progress, in bytes written
    OK                      finished
    ERR <code><TAB><text>   failed: the problem code for its kind, and a sentence

Tests (and anything else that sets LOCALFLOW_FETCH_IN_PROCESS=1) download in the calling thread
instead, where the functions they replace are the ones that run.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from localflow import problems

log = logging.getLogger(__name__)

IN_PROCESS_ENV = "LOCALFLOW_FETCH_IN_PROCESS"
KINDS = ("speech", "cleanup")

Progress = Callable[[int, int], None]  # (bytes done, bytes total)


class Cancelled(Exception):
    """The download was stopped on request."""


# the child ------------------------------------------------------------------------------------------
def _download(kind: str, key: str, device: str, progress: Progress) -> None:
    """The download itself, wherever it runs. Checks for room first."""
    from localflow import net

    if kind == "speech":
        from localflow.stt import catalogue

        model = catalogue.get(key)
        net.wait_for_proxy()
        net.ensure_space(net.hf_cache_dir(), int(model.size_gb(device) * 1e9), model.label)
        log.info("Downloading speech model %s (~%.1f GB)", model.label, model.size_gb(device))
        catalogue.download(model, device, progress=progress)
    elif kind == "cleanup":
        from localflow.llm.server import ensure_model

        ensure_model(key, progress=lambda _name, done, total: total and progress(done, total))
    else:
        raise ValueError(f"unknown download kind {kind!r}")


def child_main(argv: list[str]) -> int:
    """`localflow fetch <kind> <key> <device>`: download, and say how it goes on stdout."""
    kind, key, device = (argv + ["", "", ""])[:3]

    def say(line: str) -> None:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    last = [0.0]

    def progress(done: int, total: int) -> None:
        now = time.monotonic()
        if total and (done >= total or now - last[0] >= 0.2):
            last[0] = now
            say(f"P {int(done)} {int(total)}")

    try:
        _download(kind, key, device or "cpu", progress)
    except Exception as e:
        code = problems.classify_speech(e) if kind == "speech" else problems.classify_cleanup(e, "bundled")
        say(f"ERR {code}\t{problems.detail(e)}".replace("\n", " "))
        return 1
    say("OK")
    return 0


# the engine's side ----------------------------------------------------------------------------------
def _command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "fetch"]
    return [sys.executable, "-m", "localflow", "fetch"]


def run(kind: str, key: str, device: str, progress: Progress, stop: threading.Event) -> None:
    """Download `key` of `kind` (speech | cleanup) for `device`. Returns once it is on disk.
    Raises Cancelled when `stop` is set first (and deletes what the download left), or
    problems.Classified when it fails."""
    if kind not in KINDS:
        raise ValueError(f"unknown download kind {kind!r}")
    if os.environ.get(IN_PROCESS_ENV) == "1":
        return _run_here(kind, key, device, progress, stop)
    from localflow import jobobject, net

    # The child takes the proxy and the mirror from the environment this process set up.
    net.wait_for_proxy()
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen([*_command(), kind, key, device], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                            creationflags=flags)
    jobobject.assign(proc, "download")  # it must never outlive the engine

    def watch() -> None:
        while proc.poll() is None:
            if stop.wait(0.2):
                proc.kill()
                return

    threading.Thread(target=watch, name=f"fetch-{kind}-watch", daemon=True).start()
    ok, failure = False, None
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.startswith("P "):
            try:
                _, done, total = line.split()
                progress(int(done), int(total))
            except ValueError:
                pass
        elif line == "OK":
            ok = True
        elif line.startswith("ERR "):
            code, _, text = line[4:].partition("\t")
            failure = problems.Classified(code, text or "The download failed.")
    code = proc.wait()
    if stop.is_set() and not ok:
        remove_partial(kind, key)
        raise Cancelled()
    if ok:
        return
    if failure is not None:
        raise failure
    generic = problems.SPEECH_DOWNLOAD_FAILED if kind == "speech" else problems.CLEANUP_DOWNLOAD_FAILED
    raise problems.Classified(generic, f"The download stopped unexpectedly (exit code {code}).")


def _run_here(kind: str, key: str, device: str, progress: Progress, stop: threading.Event) -> None:
    def checked(done: int, total: int) -> None:
        if stop.is_set():
            raise Cancelled()
        progress(done, total)

    if stop.is_set():
        raise Cancelled()
    try:
        _download(kind, key, device, checked)
    except Cancelled:
        remove_partial(kind, key)
        raise


def remove_partial(kind: str, key: str) -> int:
    """Delete the unfinished files a stopped download left (huggingface_hub's `*.incomplete`).
    Returns the bytes freed."""
    folders: list[Path] = []
    try:
        if kind == "speech":
            from localflow import net
            from localflow.stt import catalogue

            folders.append(net.hf_cache_dir() / ("models--" + catalogue.get(key).repo.replace("/", "--")))
        else:
            from localflow.llm import manifest as M

            folders.append(M.gguf_dir() / ".cache")
    except Exception:
        return 0
    freed = 0
    for folder in folders:
        if not folder.is_dir():
            continue
        for f in folder.rglob("*.incomplete"):
            try:
                size = f.stat().st_size
                f.unlink()
                freed += size
            except OSError:
                pass
    if freed:
        log.info("removed %.0f MB a stopped download left behind", freed / 2**20)
    return freed
