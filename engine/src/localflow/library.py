"""The models on this PC: how much disk each takes, and taking one off again.

Speech models live in the Hugging Face cache, where two catalogue entries can share a repository
(Parakeet v3 and v3 Compact share `istupakov/parakeet-tdt-0.6b-v3-onnx`, and with it the config,
the vocabulary and the preprocessor). Removing one keeps whatever another model still on disk
needs; once nothing of a repository is left, its whole folder goes. The clean-up models are one
GGUF file each.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

DEVICES = ("cuda", "cpu")  # Whisper's files differ by device (fp16 / int8); Parakeet's do not


def _cached(repo: str, name: str) -> Path | None:
    from huggingface_hub import try_to_load_from_cache

    try:
        found = try_to_load_from_cache(repo, name)
    except Exception:
        return None
    return Path(found) if isinstance(found, str) else None


def _speech_files(model) -> set[str]:
    return {f for device in DEVICES for f in model.files(device)}


def _size(path: Path) -> int:
    try:
        return path.resolve().stat().st_size
    except OSError:
        return 0


def speech_disk_bytes(model) -> int:
    """What `model`'s files take on disk, those it shares with another model included."""
    seen: set[Path] = set()
    total = 0
    for name in _speech_files(model):
        path = _cached(model.repo, name)
        if path is None:
            continue
        real = path.resolve()
        if real not in seen:
            seen.add(real)
            total += _size(path)
    return total


def cleanup_disk_bytes(key: str) -> int:
    from localflow.llm import manifest as M

    path = M.gguf_path(key)
    return _size(path) if path.exists() else 0


def _repo_dir(repo: str) -> Path:
    from localflow import net

    return net.hf_cache_dir() / ("models--" + repo.replace("/", "--"))


def remove_speech(model) -> int:
    """Delete `model`'s files but those another catalogue model still on disk needs. Returns the
    bytes freed. The caller makes sure it is not the model in use."""
    from localflow.stt import catalogue

    siblings = [m for m in catalogue.MODELS if m.repo == model.repo and m.key != model.key]
    keep = {f for m in siblings if any(catalogue.is_installed(m, d) for d in DEVICES) for f in _speech_files(m)}
    freed = 0
    for name in sorted(_speech_files(model) - keep):
        path = _cached(model.repo, name)
        if path is None:
            continue
        blob = path.resolve() if path.is_symlink() else None
        size = _size(path)
        try:
            path.unlink()
            if blob is not None and blob.exists():
                blob.unlink()
            freed += size
        except OSError as e:
            log.warning("could not delete %s: %s", path, e)
    folder = _repo_dir(model.repo)
    if folder.is_dir() and not any(_cached(m.repo, f) for m in [model, *siblings] for f in _speech_files(m)):
        rest = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file() and not f.is_symlink())
        shutil.rmtree(folder, ignore_errors=True)
        freed += rest
    log.info("removed speech model %s (%.2f GB)", model.label, freed / 1e9)
    return freed


def remove_cleanup(key: str) -> int:
    """Delete bundled clean-up model `key`. Returns the bytes freed."""
    from localflow.llm import manifest as M

    path = M.gguf_path(key)
    freed = _size(path) if path.exists() else 0
    path.unlink(missing_ok=True)
    # what hf_hub_download(local_dir=...) recorded about it, and anything a stopped download left
    meta = M.gguf_dir() / ".cache" / "huggingface" / "download"
    for extra in (meta / f"{path.name}.metadata", meta / f"{path.name}.lock"):
        extra.unlink(missing_ok=True)
    for partial in meta.glob(f"*{path.name}*.incomplete") if meta.is_dir() else []:
        freed += _size(partial)
        partial.unlink(missing_ok=True)
    log.info("removed clean-up model %s (%.2f GB)", key, freed / 1e9)
    return freed
