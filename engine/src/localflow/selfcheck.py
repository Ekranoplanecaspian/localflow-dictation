"""The engine's half of "Check LocalFlow": what it can see that the shell cannot.

Quick checks (cheap; run when the engine starts and reported in its status): the NVIDIA driver,
the models folder's free space and whether it can be written, the clean-up server's program.
The full check adds what costs time or touches the network, and runs only when asked: every
model file hashed against the hashes its download recorded, and whether the download hosts can
be reached.

Model files are verified offline. The Hugging Face cache keeps, beside each snapshot, the file
listing it was downloaded from (`trees/<commit>.json`: an LFS file's SHA-256, a small file's git
blob id); a clean-up model downloaded into a folder keeps its ETag, which is its SHA-256, in
`.cache/huggingface/download/<file>.metadata`. A file that no longer matches was damaged, or
changed by something else - antivirus software quarantining part of it, typically.

Each result names its catalogue problem (`shared/problems.json`), so the shell words it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from localflow import problems
from localflow.config import MODELS_DIR

log = logging.getLogger(__name__)

#: CUDA 13, which the speech runtime and the clean-up server are built against, needs this.
MIN_DRIVER = 580
#: Below this much free space on the models' drive a model download or switch would fail.
MIN_FREE_GB = 5.0
HOSTS = {"huggingface.co": "https://huggingface.co", "github.com": "https://github.com"}


@dataclass
class Check:
    id: str
    name: str
    status: str  # ok | warn | fail | skip
    detail: str = ""  # a sentence: what was found (the {detail} of a problem)
    code: str | None = None  # the catalogue problem, when not ok
    vars: dict[str, str] = field(default_factory=dict)  # further placeholders for its message
    ms: int = 0
    #: Files found damaged, which "Download again" deletes so the next load fetches them afresh.
    paths: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _timed(fn):
    def run(*args, **kwargs) -> Check:
        t0 = time.perf_counter()
        try:
            check = fn(*args, **kwargs)
        except Exception as e:  # a check that breaks says so, rather than breaking the others
            log.exception("self-check %s failed", fn.__name__)
            check = Check(fn.__name__.removeprefix("check_"), fn.__name__, "skip", problems.detail(e))
        check.ms = round((time.perf_counter() - t0) * 1000)
        return check

    return run


# --- quick ------------------------------------------------------------------------------------
def driver_major(version: str | None) -> int | None:
    try:
        return int(str(version).split(".")[0])
    except (TypeError, ValueError):
        return None


@_timed
def check_driver() -> Check:
    from localflow import gpu

    mon = gpu.monitor()
    version = mon.driver_version() if hasattr(mon, "driver_version") else None
    if version is None:
        return Check("driver", "NVIDIA driver", "ok", "No NVIDIA graphics card, so no driver is needed.")
    major = driver_major(version)
    if major is not None and major < MIN_DRIVER:
        return Check("driver", "NVIDIA driver", "warn", f"Version {version} is installed.",
                     problems.DRIVER_TOO_OLD, {"version": version, "needed": str(MIN_DRIVER)})
    return Check("driver", "NVIDIA driver", "ok", f"Version {version}.")


@_timed
def check_models_folder(folder: Path = MODELS_DIR) -> Check:
    folder.mkdir(parents=True, exist_ok=True)
    probe = folder / f".write-test-{os.getpid()}"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as e:
        return Check("models_folder", "Models folder", "fail", problems.detail(e), problems.FOLDER_NOT_WRITABLE,
                     {"folder": str(folder)})
    return Check("models_folder", "Models folder", "ok", f"{folder} can be written.")


@_timed
def check_disk(folder: Path = MODELS_DIR, min_gb: float = MIN_FREE_GB) -> Check:
    folder.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(folder).free / 2**30
    drive = folder.drive or str(folder.anchor)
    if free_gb < min_gb:
        return Check("disk", "Free disk space", "warn", f"{free_gb:.1f} GB free on {drive}.", problems.DISK_LOW,
                     {"free": f"{free_gb:.1f} GB", "drive": drive, "needed": f"{min_gb:.0f} GB"})
    return Check("disk", "Free disk space", "ok", f"{free_gb:.0f} GB free on {drive}.")


@_timed
def check_cleanup_server(kind: str | None) -> Check:
    """`kind`: the llama.cpp build in use (cuda or cpu), or None when clean-up does not use one."""
    from localflow.llm import manifest as M

    if kind is None:
        return Check("cleanup_server", "Clean-up server", "skip", "AI clean-up doesn't use the bundled server.")
    exe = M.llama_server_exe(kind)
    if not exe.is_file():
        return Check("cleanup_server", "Clean-up server", "fail", f"{exe.name} isn't in {exe.parent}.",
                     problems.CLEANUP_SERVER_MISSING)
    return Check("cleanup_server", "Clean-up server", "ok", f"llama.cpp {M.LLAMA_BUILD} ({kind}).")


# --- full: model files ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def _git_blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_snapshot(snapshot: Path, names: list[str] | None = None) -> tuple[list[str], list[str], int]:
    """Check files of one Hugging Face cache snapshot against the listing it came from.

    Returns (damaged, unverifiable, bytes checked). `names` limits it to those files; by default,
    every file the listing knows. The listing is `<repo cache>/trees/<commit>.json`."""
    tree_path = snapshot.parent.parent / "trees" / f"{snapshot.name}.json"
    tree = json.loads(tree_path.read_text(encoding="utf-8"))["files"] if tree_path.is_file() else {}
    damaged, unverifiable, checked = [], [], 0
    for name in names if names is not None else sorted(tree):
        path = snapshot / name
        if not path.is_file():
            if names is not None:
                damaged.append(f"{name} (missing)")
            continue
        entry = tree.get(name)
        if entry is None:
            unverifiable.append(name)
            continue
        size = path.stat().st_size
        if "lfs_sha256" in entry:
            ok = size == entry.get("lfs_size", size) and _sha256(path) == entry["lfs_sha256"]
        else:
            ok = size == entry.get("size", size) and _git_blob_id(path) == entry.get("blob_id")
        checked += size
        if not ok:
            damaged.append(name)
    return damaged, unverifiable, checked


def _snapshot_of(repo: str, name: str) -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    try:
        found = try_to_load_from_cache(repo, name)
    except Exception:
        return None
    return Path(found).parent if isinstance(found, str) else None


@_timed
def check_speech_files(model, device: str) -> Check:
    """The speech model in use (`catalogue.SpeechModel`), and the voice-activity model."""
    names = list(model.files(device))
    snapshot = _snapshot_of(model.repo, names[0]) if names else None
    if snapshot is None:
        return Check("speech_files", "Speech model files", "skip", f"{model.label} isn't downloaded yet.")
    damaged, unverifiable, checked = verify_snapshot(snapshot, names)
    paths = [str(snapshot / n.removesuffix(" (missing)")) for n in damaged]
    vad = _snapshot_of("istupakov/silero-vad-onnx", "silero_vad.onnx")
    if vad is not None:
        d, _u, c = verify_snapshot(vad)
        paths += [str(vad / n) for n in d]
        damaged += [f"voice detector: {x}" for x in d]
        checked += c
    if damaged:
        return Check("speech_files", "Speech model files", "fail", f"Damaged: {', '.join(damaged)}.",
                     problems.SPEECH_FILES_DAMAGED, {"model": model.label}, paths=paths)
    if unverifiable:
        return Check("speech_files", "Speech model files", "ok",
                     f"{checked / 2**30:.1f} GB verified; no recorded hash for {', '.join(unverifiable)}.")
    return Check("speech_files", "Speech model files", "ok", f"{model.label}: {checked / 2**30:.1f} GB verified.")


def gguf_metadata(path: Path) -> dict[str, str] | None:
    """The commit and ETag `hf_hub_download(local_dir=...)` recorded for a downloaded file."""
    meta = path.parent / ".cache" / "huggingface" / "download" / f"{path.name}.metadata"
    try:
        lines = meta.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    return {"commit": lines[0].strip(), "etag": lines[1].strip().strip('"')} if len(lines) >= 2 else None


@_timed
def check_cleanup_files(key: str | None) -> Check:
    """The bundled clean-up model in use (a manifest key), or None when there is none."""
    from localflow.llm import manifest as M

    if key is None or key not in M.CLEANUP_MODELS:
        return Check("cleanup_files", "Clean-up model file", "skip", "AI clean-up doesn't use a bundled model.")
    path = M.gguf_path(key)
    label = M.CLEANUP_MODELS[key].label or key
    if not path.is_file():
        return Check("cleanup_files", "Clean-up model file", "skip", f"{label} isn't downloaded yet.")
    meta = gguf_metadata(path)
    expected = (meta or {}).get("etag", "")
    if len(expected) != 64:  # an ETag that is not a SHA-256 cannot verify the file
        return Check("cleanup_files", "Clean-up model file", "ok",
                     f"{label} is there; its download recorded no hash to check it against.")
    if _sha256(path) != expected:
        return Check("cleanup_files", "Clean-up model file", "fail", f"{path.name} doesn't match its download.",
                     problems.CLEANUP_FILES_DAMAGED, {"model": label}, paths=[str(path)])
    return Check("cleanup_files", "Clean-up model file", "ok",
                 f"{label}: {path.stat().st_size / 2**30:.1f} GB verified.")


# --- full: network --------------------------------------------------------------------------------
@_timed
def check_hosts(hosts: dict[str, str] | None = None, timeout: float = 8.0) -> Check:
    """Can a download start? One HEAD request per host, through the Windows proxy settings
    (urllib reads them from the registry). Nothing about the user is sent."""
    if hosts is None:
        # The mirror, when one is set, is where models come from; and through the proxy the
        # engine found (net.py), which urllib picks up from the environment.
        import os

        from localflow import net

        hosts = dict(HOSTS)
        if os.environ.get("HF_ENDPOINT"):
            hosts.pop("huggingface.co", None)
            hosts = {net.download_host(): os.environ["HF_ENDPOINT"], **hosts}
        net.wait_for_proxy(5.0)
    unreachable = []
    for name, url in hosts.items():
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "LocalFlow self-check"})
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                pass
        except urllib.error.HTTPError:
            pass  # it answered: reachable
        except Exception as e:
            unreachable.append(f"{name} ({problems.detail(e).rstrip('.')})")
    if unreachable:
        return Check("hosts", "Download sites", "warn", f"Can't reach {', '.join(unreachable)}.",
                     problems.DOWNLOAD_HOSTS_UNREACHABLE)
    return Check("hosts", "Download sites", "ok", f"{' and '.join(hosts)} can be reached.")


def repair(checks: list[Check]) -> list[str]:
    """Delete the damaged model files `checks` found, so the next load downloads them afresh
    (the caches count a file that is there as downloaded, however broken). Only files a check
    named, and only under the model caches."""
    from huggingface_hub.constants import HF_HUB_CACHE

    allowed = [Path(HF_HUB_CACHE).resolve(), MODELS_DIR.resolve()]
    removed = []
    for check in checks:
        for p in check.paths:
            path = Path(p).resolve()
            if not any(path.is_relative_to(root) for root in allowed):
                log.warning("not deleting %s: outside the model caches", path)
                continue
            try:
                path.unlink(missing_ok=True)
                removed.append(str(path))
            except OSError as e:
                log.warning("could not delete %s: %s", path, e)
    log.info("self-check repair removed %d file(s): %s", len(removed), removed)
    return removed


# --- together ----------------------------------------------------------------------------------------
def run(engine, full: bool) -> list[Check]:
    """Every check for `engine` (a running `Engine`); the slow ones only when `full`."""
    from localflow.stt import catalogue

    pp = engine.cfg.postprocess
    bundled = pp.llm_cleanup and pp.llm_provider == "bundled"
    kind = None
    if bundled:
        kind = "cuda" if engine.compute.placement.cleanup == "cuda" else "cpu"
    checks = [check_driver(), check_models_folder(), check_disk(), check_cleanup_server(kind)]
    if full:
        model = catalogue.current(engine.cfg.stt)
        device = getattr(engine.stt, "device", None) or "cpu"
        if model is not None:
            checks.append(check_speech_files(model, device))
        checks.append(check_cleanup_files(pp.llm_model if bundled else None))
        checks.append(check_hosts())
    return checks
