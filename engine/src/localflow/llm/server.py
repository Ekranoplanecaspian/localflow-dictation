"""Bundled llama-server: ensure the binaries and model exist, start it, watch it, stop it.

The engine owns this process. It is started on demand (when LLM clean-up is enabled), bound
to 127.0.0.1 with an API key, and killed with the engine.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from localflow import jobobject
from localflow.llm import manifest as M
from localflow.llm.downloader import Progress, download, extract_zip

log = logging.getLogger(__name__)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- keeping llama-server tied to our lifetime ------------------------------------------------
# A terminated engine used to leave its llama-server behind holding ~2.7 GB of VRAM; a few of
# those and the GPU is full. Two defences: the child joins a job object that kills it when this
# process dies (however it dies, see localflow.jobobject), and any orphan from a previous run is
# reaped at startup.


def reap_orphans(exe: Path) -> int:
    """Kill llama-server processes started from our own install whose parent is gone.

    Matched by path shape (`...\\LocalFlow\\bin\\llama\\...`) rather than an exact path: a
    packaged or containerised host can redirect %LOCALAPPDATA% writes elsewhere, and another
    app's llama-server (LM Studio, say) must never be touched."""
    import ctypes
    from ctypes import wintypes

    TH32CS_SNAPPROCESS, MAX_PATH = 0x2, 260
    PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_TERMINATE, STILL_ACTIVE = 0x1000, 0x1, 259

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", wintypes.WCHAR * MAX_PATH)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE

    def image_path(pid: int) -> str | None:
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return None
        try:
            size = wintypes.DWORD(MAX_PATH * 4)
            buf = ctypes.create_unicode_buffer(size.value)
            if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return buf.value
            return None
        finally:
            k32.CloseHandle(h)

    def alive(pid: int) -> bool:
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        try:
            code = wintypes.DWORD()
            k32.GetExitCodeProcess(h, ctypes.byref(code))
            return code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(h)

    killed = 0
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == wintypes.HANDLE(-1).value:
        return 0
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        marker = os.path.join("localflow", "bin", "llama").lower()
        while ok:
            if entry.szExeFile.lower() == exe.name.lower() and entry.th32ProcessID != os.getpid():
                path = image_path(entry.th32ProcessID)
                if path and marker in path.lower() and not alive(entry.th32ParentProcessID):
                    h = k32.OpenProcess(PROCESS_TERMINATE, False, entry.th32ProcessID)
                    if h:
                        try:
                            if k32.TerminateProcess(h, 1):
                                killed += 1
                                log.info("reaped orphaned llama-server (pid %d) from a previous run",
                                         entry.th32ProcessID)
                        finally:
                            k32.CloseHandle(h)
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return killed


def ensure_binaries(kind: str, progress: Progress | None = None) -> Path:
    """Download and unpack llama.cpp for `kind` ("cuda" | "cpu") if missing. Returns the exe."""
    exe = M.llama_server_exe(kind)
    marker = M.llama_dir(kind) / ".ok"
    if exe.exists() and marker.exists():
        return exe
    if not M.is_windows_x64():
        raise RuntimeError("bundled llama-server is only provided for Windows x64")
    for asset in M.LLAMA_ASSETS[kind]:
        archive = M.llama_archive_dir() / asset.name
        if not archive.exists():
            log.info("Downloading %s", asset.name)
            download(asset.url, archive, progress, asset.sha256)
        extract_zip(archive, M.llama_dir(kind))
    marker.write_text(time.strftime("%Y-%m-%d"), encoding="utf-8")
    if not exe.exists():
        raise RuntimeError(f"llama-server.exe not found after extracting into {M.llama_dir(kind)}")
    return exe


def ensure_model(key: str, progress: Progress | None = None) -> Path:
    if key not in M.CLEANUP_MODELS:
        log.warning("unknown bundled clean-up model %r; using %s", key, M.DEFAULT_CLEANUP_MODEL)
        key = M.DEFAULT_CLEANUP_MODEL
    model = M.CLEANUP_MODELS[key]
    path = M.gguf_path(key)
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    log.info("Downloading %s (%s, ~%.1f GB)", model.filename, model.repo, model.approx_gb)
    if progress:
        progress(model.filename, 0, None)
    got = hf_hub_download(model.repo, model.filename, local_dir=str(M.gguf_dir()))
    if progress:
        progress(model.filename, Path(got).stat().st_size, Path(got).stat().st_size)
    return Path(got)


class LlamaServer:
    def __init__(self, model_key: str = M.DEFAULT_CLEANUP_MODEL, device: str = "auto", context: int = 2048,
                 gpu_layers: int = 99, threads: int | None = None, extra_args: list[str] | None = None):
        self.model_key = model_key if model_key in M.CLEANUP_MODELS else M.DEFAULT_CLEANUP_MODEL
        self.device = device
        self.context = context
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.extra_args = list(extra_args or [])
        self.port: int | None = None
        self.api_key = secrets.token_hex(16)
        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, progress: Progress | None = None, timeout: float = 120.0) -> None:
        with self._lock:
            if self.alive():
                return
            kind = "cuda" if self.device in ("auto", "cuda") else "cpu"
            try:
                exe = ensure_binaries(kind, progress)
            except Exception:
                if kind == "cuda" and self.device == "auto":
                    log.warning("CUDA llama.cpp unavailable; using the CPU build")
                    kind, exe = "cpu", ensure_binaries("cpu", progress)
                else:
                    raise
            model = ensure_model(self.model_key, progress)
            reap_orphans(exe)  # a previous run may have been killed before it could stop its server
            self.kind = kind
            self.port = _free_port()
            self.log_path = M.llama_dir(kind) / "llama-server.log"
            cmd = [str(exe), "-m", str(model), "--host", "127.0.0.1", "--port", str(self.port),
                   "--api-key", self.api_key, "-c", str(self.context),
                   "-ngl", str(self.gpu_layers if kind == "cuda" else 0),
                   "-np", "1", "--no-webui", "--log-timestamps",
                   "--reasoning", "off",  # clean-up is a copy-edit task; thinking models must answer directly
                   *(["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"] if kind == "cuda" else []),  # smaller KV cache
                   *self.extra_args]
            if self.threads:
                cmd += ["-t", str(self.threads)]
            creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            logf = self.log_path.open("ab")
            self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         creationflags=creation, env=env, cwd=str(exe.parent))
            # Dies with us even if we are killed rather than shut down.
            jobobject.assign(self.proc, "llama-server")
            log.info("llama-server starting (%s, %s) on port %d", self.model_key, kind, self.port)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited with code {self.proc.returncode}; see {self.log_path}")
                if self.health():
                    log.info("llama-server ready")
                    return
                time.sleep(0.25)
            self.stop()
            raise RuntimeError("llama-server did not become healthy in time")

    def health(self) -> bool:
        if self.port is None:
            return False
        try:
            req = urllib.request.Request(f"{self.base_url}/health", headers={"Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=2) as r:
                return json.loads(r.read().decode()).get("status") == "ok"
        except Exception:
            return False

    def stop(self) -> None:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    self.proc.wait(5)
                except Exception:
                    self.proc.kill()
            self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
