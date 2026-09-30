"""Bundled llama-server: ensure the binaries and model exist, start it, watch it, stop it.

The engine owns this process. It is started on demand (when LLM clean-up is enabled), bound
to 127.0.0.1 with an API key, and killed with the engine.
"""

from __future__ import annotations

import json
import logging
import os
import re
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


LOG_MAX_BYTES = 10 << 20


_VULKAN_LINE = re.compile(r"^\s*Vulkan(\d+):\s*(.+?)\s*\((\d+) MiB")


def vulkan_device(exe: Path) -> tuple[int, str] | None:
    """The graphics adapter the Vulkan build should use: the first that is not NVIDIA's (an
    NVIDIA card runs the CUDA build, which is faster there). Its index and name, or None.

    Listing the devices is asked of llama-server once per install and kept beside it, since
    starting Vulkan touches every adapter, the NVIDIA card too. Starting the server itself is
    then limited to the chosen one (GGML_VK_VISIBLE_DEVICES), so the NVIDIA card is left alone."""
    cache = exe.parent / "vulkan-devices.json"
    devices: list[tuple[int, str, int]] | None = None
    try:
        devices = [tuple(d) for d in json.loads(cache.read_text(encoding="utf-8"))]
    except (OSError, ValueError):
        pass
    if devices is None:
        creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            out = subprocess.run([str(exe), "--list-devices"], capture_output=True, text=True, timeout=30,
                                 creationflags=creation, cwd=str(exe.parent)).stdout
        except (OSError, subprocess.SubprocessError) as e:
            log.info("could not list Vulkan devices: %s", e)
            return None
        devices = [(int(m[1]), m[2], int(m[3])) for m in map(_VULKAN_LINE.match, out.splitlines()) if m]
        try:
            cache.write_text(json.dumps(devices), encoding="utf-8")
        except OSError:
            pass
    for index, name, _mib in devices:
        if "nvidia" not in name.lower() and "llvmpipe" not in name.lower():
            return index, name
    return None


def _roll_log(path: Path) -> None:
    """Start the log afresh once it passes LOG_MAX_BYTES, keeping the previous one as `.old`.
    It is appended to by every server started, and was never trimmed."""
    try:
        if path.stat().st_size > LOG_MAX_BYTES:
            os.replace(path, path.with_name(path.name + ".old"))
    except OSError:
        pass  # no log yet, or a server still holds it: roll it next time


def ensure_binaries(kind: str, progress: Progress | None = None) -> Path:
    """Download and unpack llama.cpp for `kind` ("cuda" | "cpu" | "vulkan") if missing. Returns the exe."""
    exe = M.llama_server_exe(kind)
    marker = M.llama_dir(kind) / ".ok"
    if exe.exists() and marker.exists():
        return exe
    if not M.is_windows_x64():
        raise RuntimeError("bundled llama-server is only provided for Windows x64")
    from localflow import net

    for asset in M.LLAMA_ASSETS[kind]:
        archive = M.llama_archive_dir() / asset.name
        if not archive.exists():
            net.wait_for_proxy()
            # Unpacked beside the archive, so about twice its size; the CUDA build is ~0.6 GB.
            net.ensure_space(M.llama_archive_dir(), 2 * (asset.size or (400 << 20 if kind == "cuda" else 40 << 20)),
                             "the clean-up server")
            log.info("Downloading %s", asset.name)
            download(asset.url, archive, progress, asset.sha256)
        extract_zip(archive, M.llama_dir(kind))
    if not exe.exists():
        raise RuntimeError(f"llama-server.exe not found after extracting into {M.llama_dir(kind)}")
    marker.write_text(time.strftime("%Y-%m-%d"), encoding="utf-8")
    _drop_archives(kind)
    return exe


def _drop_archives(kind: str) -> int:
    """Delete the downloaded zips for `kind` once it is unpacked; they were kept for good, about
    half a gigabyte for the CUDA build. Returns the bytes freed."""
    freed = 0
    for asset in M.LLAMA_ASSETS[kind]:
        archive = M.llama_archive_dir() / asset.name
        try:
            size = archive.stat().st_size
            archive.unlink()
            freed += size
        except OSError:
            pass
    return freed


def tidy_downloads() -> int:
    """Free the disk LocalFlow no longer needs: archives of builds already unpacked (installs from
    before they were deleted straight away) and llama.cpp builds other than the current one,
    left behind by an update. A build still in use - its server running - cannot be renamed, so
    it is skipped and tried again at the next start. Returns the bytes freed."""
    import re
    import shutil

    freed = 0
    for kind in M.LLAMA_ASSETS:
        if (M.llama_dir(kind) / ".ok").exists():
            freed += _drop_archives(kind)
    root = M.BIN_DIR / "llama"
    if root.is_dir():
        for old in root.iterdir():
            if not old.is_dir() or old.is_symlink() or not re.fullmatch(r"b\d+", old.name) or old.name == M.LLAMA_BUILD:
                continue
            size = sum(f.stat().st_size for f in old.rglob("*") if f.is_file())
            doomed = old.with_name(old.name + ".delete")
            try:
                old.rename(doomed)  # fails while any file in it is open: it is still in use
            except OSError:
                log.info("llama.cpp %s is still in use; it will be removed at a later start", old.name)
                continue
            shutil.rmtree(doomed, ignore_errors=True)
            freed += size
            log.info("removed the previous llama.cpp build %s (%.0f MB)", old.name, size / 2**20)
        # A removal interrupted before: finish it.
        for doomed in root.glob("b*.delete"):
            shutil.rmtree(doomed, ignore_errors=True)
    if freed:
        log.info("freed %.0f MB of downloads no longer needed", freed / 2**20)
    return freed


def ensure_model(key: str, progress: Progress | None = None) -> Path:
    if key not in M.CLEANUP_MODELS:
        log.warning("unknown bundled clean-up model %r; using %s", key, M.DEFAULT_CLEANUP_MODEL)
        key = M.DEFAULT_CLEANUP_MODEL
    model = M.CLEANUP_MODELS[key]
    path = M.gguf_path(key)
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    from localflow import net
    from localflow.hfprogress import reporter

    net.wait_for_proxy()
    net.ensure_space(M.gguf_dir(), int(model.approx_gb * 1e9), model.label or model.key)
    log.info("Downloading %s (%s, ~%.1f GB)", model.filename, model.repo, model.approx_gb)
    kwargs = {}
    if progress:
        progress(model.filename, 0, None)
        kwargs["tqdm_class"] = reporter(lambda done, total: progress(model.filename, done, total))
    got = hf_hub_download(model.repo, model.filename, local_dir=str(M.gguf_dir()), **kwargs)
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
        # Set once the process has been launched (or launching has failed): from there it loads
        # on its own, whatever the Python side is doing.
        self.spawned = threading.Event()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, progress: Progress | None = None, timeout: float = 120.0) -> None:
        with self._lock:
            if self.alive():
                return
            kind = {"auto": "cuda", "cuda": "cuda", "vulkan": "vulkan"}.get(self.device, "cpu")
            try:
                exe = ensure_binaries(kind, progress)
            except Exception:
                if kind == "cuda" and self.device == "auto":
                    log.warning("CUDA llama.cpp unavailable; using the CPU build")
                    kind, exe = "cpu", ensure_binaries("cpu", progress)
                else:
                    raise
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            if kind == "vulkan":
                chosen = vulkan_device(exe)
                if chosen is None:
                    raise RuntimeError("no graphics adapter for the Vulkan build of llama-server")
                env["GGML_VK_VISIBLE_DEVICES"] = str(chosen[0])
                log.info("llama-server on %s (Vulkan%d)", chosen[1], chosen[0])
            gpu = kind in ("cuda", "vulkan")
            model = ensure_model(self.model_key, progress)
            reap_orphans(exe)  # a previous run may have been killed before it could stop its server
            self.kind = kind
            self.port = _free_port()
            self.log_path = M.llama_dir(kind) / "llama-server.log"
            cmd = [str(exe), "-m", str(model), "--host", "127.0.0.1", "--port", str(self.port),
                   "--api-key", self.api_key, "-c", str(self.context),
                   "-ngl", str(self.gpu_layers if gpu else 0),
                   "-np", "1", "--no-webui", "--log-timestamps",
                   "--reasoning", "off",  # clean-up is a copy-edit task; thinking models must answer directly
                   *(["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"] if gpu else []),  # smaller KV cache
                   # On the processor llama.cpp copies the weights into a repacked private buffer
                   # by default. Without it they stay memory-mapped from the file: measured on the
                   # Ryzen AI 9 with Qwen3 4B, private memory 2.2 GB -> 0.5 GB, and clean-up was
                   # faster too (1.25 s vs 1.37 s) and loaded in half the time.
                   *(["--no-repack"] if kind == "cpu" else []),
                   *self.extra_args]
            if self.threads:
                cmd += ["-t", str(self.threads)]
            creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            _roll_log(self.log_path)
            # The child gets its own handle to the log; ours is closed straight away. It used to
            # stay open for the life of the engine, one more with every server started (each
            # model switch and each move between the graphics card and the processor).
            with self.log_path.open("ab") as logf:
                self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                             creationflags=creation, env=env, cwd=str(exe.parent))
            self.spawned.set()
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
