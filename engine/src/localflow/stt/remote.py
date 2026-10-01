"""Speech on the graphics card, in a process of its own.

Once a process has used CUDA, it keeps ~1.4 GB of memory (0.8 GB of it resident) and ~0.1-0.35 GB
of graphics memory until it exits: the CUDA context, cuDNN and cuBLAS state and their kernels.
Unloading the model does not give it back. Measured 2026-09-25 on the RTX 4060 laptop: speech on
the processor alone is 3.0 GB private / 2.5 GB resident; the same after one spell on the GPU,
4.3 GB / 3.3 GB. So the model that uses the GPU lives in a child process, and an idle release -
speech moved to the processor, in the engine itself - ends that process and all of it goes.

The child is `localflow speech-worker` (the frozen engine runs itself with that argument). It
starts at the very beginning of `serve`, before the engine has loaded anything, so its imports
and its CUDA check overlap the engine's own start-up. The two talk over the child's stdin and
stdout: length-prefixed pickles, one request and one reply at a time, with the child's log
records sent back ahead of each reply so they land in the engine's log. The child's own stdout is
pointed at stderr before anything is imported, so no library can write into the channel.

Every call is made from the engine's speech worker thread; the lock is for anything else.
"""

from __future__ import annotations

import logging
import os
import pickle
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import Future
from typing import Any

import numpy as np

from localflow.config import STTConfig

log = logging.getLogger(__name__)

IN_PROCESS_ENV = "LOCALFLOW_SPEECH_IN_PROCESS"  # "1": the old way, CUDA in the engine itself
_HEADER = struct.Struct("<Q")
#: How long `close` waits for the channel before it stops asking and ends the worker.
QUIT_WAIT_S = 2.0


def enabled() -> bool:
    return os.environ.get(IN_PROCESS_ENV) != "1"


class WorkerDied(RuntimeError):
    """The speech worker process ended, or its channel broke."""


def _command() -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "speech-worker"]
    return [sys.executable, "-m", "localflow", "speech-worker"]


def _read_exact(stream, n: int) -> bytes:
    parts, got = [], 0
    while got < n:
        chunk = stream.read(n - got)
        if not chunk:
            raise EOFError
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def _send(stream, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(data)) + data)
    stream.flush()


def _recv(stream) -> Any:
    (n,) = _HEADER.unpack(_read_exact(stream, _HEADER.size))
    return pickle.loads(_read_exact(stream, n))


# ---------------------------------------------------------------------------------------------
# the engine's side

class Worker:
    """One speech worker process."""

    def __init__(self) -> None:
        from localflow import jobobject

        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(_command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, creationflags=flags)
        jobobject.assign(self.proc, "speech worker")  # it must never outlive the engine
        self._lock = threading.Lock()
        self.started = time.perf_counter()
        # A load started before anyone asked (see `prestart`): the settings, and its reply.
        self.preload_cfg: STTConfig | None = None
        self.preload: Future | None = None

    def start_preload(self, cfg: STTConfig) -> None:
        self.preload_cfg = cfg
        self.preload = Future()

        def run() -> None:
            try:
                self.preload.set_result(self.call("load", cfg))
            except BaseException as e:  # noqa: BLE001 - handed to whoever takes the worker
                self.preload.set_exception(e)

        threading.Thread(target=run, name="speech-preload", daemon=True).start()

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def call(self, op: str, *args: Any) -> Any:
        with self._lock:
            try:
                _send(self.proc.stdin, (op, args))
                while True:
                    kind, *rest = _recv(self.proc.stdout)
                    if kind != "log":
                        break
                    level, name, message = rest
                    logging.getLogger(name).log(level, "%s", message)
            except (EOFError, OSError, pickle.UnpicklingError) as e:
                code = self.proc.poll()
                raise WorkerDied(f"the speech worker stopped ({'exit code %s' % code if code is not None else e})") from e
        if kind == "ok":
            return rest[0]
        raise rest[0]  # the model's own error, as it was raised in the worker

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        # Asked to quit only if the channel comes free soon. A call waiting for a reply holds
        # the lock for as long as the worker stays silent, and waiting for it here hung the
        # engine's shutdown behind a stalled worker, never reaching the kill below. Killing it
        # ends that call too: its read meets the end of the pipe.
        if self._lock.acquire(timeout=QUIT_WAIT_S):
            try:
                _send(self.proc.stdin, ("quit", ()))
            except Exception:
                pass
            finally:
                self._lock.release()
            try:
                self.proc.wait(3)
                return
            except Exception:
                pass
        self.proc.kill()
        # Ended, not just told to end: a busy worker (still loading its model) that was
        # killed went on holding its memory and files for a moment after close() returned.
        try:
            self.proc.wait(5)
        except Exception:
            pass


_spare: Worker | None = None
_spare_lock = threading.Lock()


def prestart(cfg: STTConfig | None = None) -> None:
    """Start a worker now, before it is needed, and if `cfg` is given have it load and warm that
    model straight away: the 2 s onnxruntime spends setting the model up then overlaps the rest
    of the engine's start-up. The first `RemoteTranscriber` takes it; one asking for different
    settings (the engine chose the processor, or another model) has the worker load those."""
    global _spare
    if not enabled():
        return
    with _spare_lock:
        if _spare is None:
            try:
                _spare = Worker()
                if cfg is not None:
                    _spare.start_preload(cfg)
            except Exception as e:
                log.warning("could not start the speech worker early: %s", e)


def _take_worker() -> Worker:
    global _spare
    with _spare_lock:
        w, _spare = _spare, None
    if w is not None and w.alive:
        return w
    return Worker()


def discard_spare() -> None:
    """The spare will not be needed (speech is going on the processor)."""
    global _spare
    with _spare_lock:
        w, _spare = _spare, None
    if w is not None:
        # Not `close`: that would wait for an early load still running.
        w.proc.kill()


class RemoteTranscriber:
    """A transcriber whose model lives in a speech worker process. Same interface as the
    in-process ones; `close` ends the process, and with it everything CUDA held."""

    sample_rate = 16000

    def __init__(self, cfg: STTConfig):
        t0 = time.perf_counter()
        self._worker = _take_worker()
        try:
            info = None
            if self._worker.preload is not None:
                try:
                    early = self._worker.preload.result()
                    if self._worker.preload_cfg == cfg:
                        info = early
                    else:
                        log.info("speech settings changed since the early load; loading as asked")
                except Exception as e:
                    log.info("the early speech load failed (%s); loading as asked", e)
                self._worker.preload = None
            if info is None and not self._worker.alive:
                self._worker = Worker()
            if info is None:
                info = self._worker.call("load", cfg)
        except BaseException:
            self._worker.close()
            raise
        self.name, self.device, self.precision = info["name"], info["device"], info["precision"]
        self.has_whisper_router = info["router"]
        log.info("speech worker ready (pid %s, %s) in %.1fs", self._worker.proc.pid, self.device,
                 time.perf_counter() - t0)

    @property
    def pid(self) -> int:
        return self._worker.proc.pid

    @property
    def alive(self) -> bool:
        return self._worker.alive

    def warmup(self) -> None:
        """Nothing to do: the worker warms every model it loads, just after saying it is loaded,
        so the engine can call itself ready meanwhile. A decode asked for during the warm-up
        simply waits for it."""

    def warm(self) -> None:
        self._worker.call("warm")

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        if audio.size == 0:
            return ""
        return self._worker.call("transcribe", np.ascontiguousarray(audio, dtype=np.float32), language)

    def drop_idle_whisper(self, *args: Any) -> bool:
        return bool(self.has_whisper_router and self._worker.call("drop_idle_whisper", *args))

    def close(self) -> None:
        self._worker.close()

    def __del__(self) -> None:
        try:
            self._worker.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------
# the worker's side

class _Forward(logging.Handler):
    def __init__(self, out) -> None:
        super().__init__()
        self.out = out

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _send(self.out, ("log", record.levelno, record.name, self.format(record)))
        except Exception:
            pass


def worker_main() -> int:
    """`localflow speech-worker`: serve one transcriber over stdin/stdout until told to quit or
    the engine goes away."""
    # The channel is the real stdout; anything else writing to "stdout" goes to stderr instead,
    # at the file-descriptor level too, so native libraries cannot corrupt it.
    channel = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    if sys.platform == "win32":
        import ctypes
        import msvcrt

        ctypes.windll.kernel32.SetStdHandle(-11, msvcrt.get_osfhandle(2))  # STD_OUTPUT_HANDLE
    sys.stdout = sys.stderr
    inp = sys.stdin.buffer

    import faulthandler

    from localflow.config import CONFIG_DIR

    try:
        fault = open(CONFIG_DIR / "speech-worker-fault.log", "w", encoding="utf-8")
        faulthandler.enable(file=fault, all_threads=True)
    except OSError:
        pass

    handler = _Forward(channel)
    handler.setFormatter(logging.Formatter("[speech worker] %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    # Import and check CUDA now, while the engine is still starting: the reason to start early.
    from localflow.stt.parakeet import cuda_available

    cuda_available()

    model = None
    while True:
        try:
            op, args = _recv(inp)
        except (EOFError, OSError):
            return 0  # the engine went away
        if op == "quit":
            _send(channel, ("ok", None))
            return 0
        try:
            if op == "load":
                from localflow.stt.base import build_transcriber

                model = None  # the previous one's memory first
                model = build_transcriber(args[0])
                result: Any = {
                    "name": getattr(model, "name", "?"),
                    "device": getattr(model, "device", "cpu"),
                    "precision": getattr(model, "precision", "?"),
                    "router": hasattr(model, "drop_idle_whisper"),
                }
                _send(channel, ("ok", result))
                model.warmup()  # after the reply: the next request waits for it, the engine does not
                continue
            elif op == "transcribe":
                result = model.transcribe(args[0], language=args[1])
            elif op == "warmup":
                result = model.warmup()
            elif op == "warm":
                result = model.warm() if hasattr(model, "warm") else None
            elif op == "drop_idle_whisper":
                result = model.drop_idle_whisper(*args)
            else:
                raise ValueError(f"unknown request {op!r}")
            _send(channel, ("ok", result))
        except Exception as e:
            try:
                pickle.dumps(e)
                err: BaseException = e
            except Exception:
                err = RuntimeError(f"{type(e).__name__}: {e}")
            _send(channel, ("err", err))
