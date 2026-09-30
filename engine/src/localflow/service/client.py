"""Synchronous client for the engine service, for shells written in Python (the tray app,
the CLI). Spawns the engine as a child process by default, or attaches to a running one.

Threads: a receiver thread dispatches engine events to callbacks; a sender thread drains
the audio queue so the microphone callback never blocks on the socket.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from localflow import jobobject
from localflow.config import CONFIG_DIR, LOG_PATH
from localflow.service import protocol as P
from localflow.service.server import ENGINE_INFO_PATH

log = logging.getLogger(__name__)

Event = dict[str, Any]


class EngineProcess:
    """Child `localflow serve` process with a stdout handshake."""

    def __init__(self, log_level: str = "INFO"):
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None
        self.token: str | None = None
        self.pid: int | None = None  # the engine's own, from the handshake (a launcher may sit in between)
        self.log_level = log_level

    def start(self, timeout: float = 30.0) -> tuple[int, str]:
        exe = Path(sys.executable)
        # keep the child windowless when the parent is windowless
        cmd = [str(exe), "-m", "localflow", "--log-level", self.log_level, "serve", "--handshake"]
        creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     stdin=subprocess.DEVNULL, text=True, creationflags=creation)
        # The engine must not outlive its shell. A shell killed from Task Manager never runs
        # stop(), and the engine it left behind held ~4 GB of VRAM (models plus llama-server)
        # until it was found by hand.
        jobobject.assign(self.proc, "engine")
        assert self.proc.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"engine exited during start (code {self.proc.returncode}); see {LOG_PATH}")
                continue
            try:
                info = json.loads(line)
                self.port, self.token = int(info["port"]), str(info["token"])
                self.pid = info.get("pid")
                return self.port, self.token
            except (ValueError, KeyError):
                continue
        raise RuntimeError("engine did not report a port in time")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        """Stop the engine and everything it started. Terminating the process we spawned was not
        enough: in development that is the virtualenv's launcher, and the engine under it - with
        its clean-up server holding graphics memory - ran on until this process exited."""
        if not (self.proc and self.proc.poll() is None):
            return
        try:
            import psutil

            family = psutil.Process(self.proc.pid).children(recursive=True)
        except Exception:
            family = []
        try:
            self.proc.terminate()
            self.proc.wait(5)
        except Exception:
            self.proc.kill()
        for p in family:
            try:
                p.kill()
            except Exception:
                pass


def _pid_alive(pid: int) -> bool:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def discover() -> tuple[int, str] | None:
    """Port and token of an already running engine. The info file can outlive a killed
    engine, so the pid in it is checked first."""
    try:
        info = json.loads(ENGINE_INFO_PATH.read_text(encoding="utf-8"))
        if not _pid_alive(int(info["pid"])):
            ENGINE_INFO_PATH.unlink(missing_ok=True)
            return None
        return int(info["port"]), str(info["token"])
    except Exception:
        return None


class EngineClient:
    def __init__(self, name: str = "tray"):
        self.name = name
        self.on_status: Callable[[Event], None] | None = None
        self.on_partial: Callable[[Event], None] | None = None
        self.on_final: Callable[[Event], None] | None = None
        self.on_error: Callable[[Event], None] | None = None
        self.on_disconnect: Callable[[], None] | None = None
        self.status: Event | None = None
        self._conn = None
        self._cm = None
        # One outbound queue for audio frames *and* control messages keeps them in order:
        # session.end must not overtake frames that are still queued.
        self._out: queue.Queue[bytes | str | None] = queue.Queue()
        self._recv_thread: threading.Thread | None = None
        self._send_thread: threading.Thread | None = None
        self.session_id: str | None = None
        self.connected = threading.Event()

    # connection ------------------------------------------------------------------------------
    def connect(self, port: int, token: str, timeout: float = 10.0) -> Event:
        from websockets.sync.client import connect

        # websockets >= 17 wants connect() used as a context manager; we hold it open for the
        # life of the client and exit it in close().
        self._cm = connect(f"ws://127.0.0.1:{port}", open_timeout=timeout, max_size=4 * 2**20)
        self._conn = self._cm.__enter__()
        self._conn.send(P.encode({"type": P.HELLO, "token": token, "client": self.name}))
        reply = P.decode(self._conn.recv(timeout=timeout))
        if reply.get("type") != P.HELLO_OK:
            raise RuntimeError(f"engine refused: {reply}")
        self.status = reply.get("status")
        self.connected.set()
        self._recv_thread = threading.Thread(target=self._recv_loop, name="engine-recv", daemon=True)
        self._send_thread = threading.Thread(target=self._send_loop, name="engine-send", daemon=True)
        self._recv_thread.start()
        self._send_thread.start()
        return self.status or {}

    def close(self) -> None:
        self.connected.clear()
        self._out.put(None)
        if self._conn is not None:
            try:
                self._cm.__exit__(None, None, None)
            except Exception:
                pass
            self._conn = None

    def _send(self, msg: Event) -> None:
        if self._conn is None:
            raise RuntimeError("not connected")
        self._out.put(P.encode(msg))

    def _send_loop(self) -> None:
        while True:
            item = self._out.get()
            if item is None:
                return
            conn = self._conn
            if conn is None:
                continue
            try:
                conn.send(item)
            except Exception as e:
                log.debug("send failed: %s", e)

    def _recv_loop(self) -> None:
        conn = self._conn
        try:
            while conn is not None:
                raw = conn.recv()
                if not isinstance(raw, str):
                    continue
                msg = P.decode(raw)
                t = msg.get("type")
                if t == P.STATUS:
                    self.status = msg
                    if self.on_status:
                        self.on_status(msg)
                elif t == P.PARTIAL:
                    if self.on_partial:
                        self.on_partial(msg)
                elif t == P.FINAL:
                    if self.on_final:
                        self.on_final(msg)
                elif t == P.ERROR:
                    log.warning("engine error %s: %s", msg.get("code"), msg.get("message"))
                    if self.on_error:
                        self.on_error(msg)
        except Exception as e:
            if self.connected.is_set():
                log.warning("engine connection lost: %s", e)
        finally:
            was_connected = self.connected.is_set()
            self.connected.clear()
            if was_connected and self.on_disconnect:
                self.on_disconnect()

    # sessions -------------------------------------------------------------------------------------
    def start_session(self, context: dict[str, Any] | None = None, language: str | None = None) -> str:
        self.session_id = uuid.uuid4().hex[:8]
        self._send({"type": P.SESSION_START, "id": self.session_id, "context": context or {}, "language": language})
        return self.session_id

    def send_audio(self, block: np.ndarray) -> None:
        """float32 [-1, 1] samples at 16 kHz -> int16 frame on the wire."""
        pcm = np.clip(block * 32767.0, -32768, 32767).astype("<i2").tobytes()
        self._out.put(pcm)

    def end_session(self) -> None:
        if self.session_id:
            self._send({"type": P.SESSION_END, "id": self.session_id})

    def cancel_session(self) -> None:
        if self.session_id:
            self._send({"type": P.SESSION_CANCEL, "id": self.session_id})
            self.session_id = None

    def request_status(self) -> None:
        self._send({"type": P.STATUS_GET})

    def set_settings(self, **settings: Any) -> None:
        self._send({"type": P.SETTINGS_SET, **settings})

    def shutdown_engine(self) -> None:
        try:
            self._send({"type": P.SHUTDOWN})
        except Exception:
            pass
