"""Pinned versions of everything the AI layer downloads.

Bump versions here, nowhere else. Checksums are verified after download; a mismatch is a
hard error (a partial or tampered file is worse than no file).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

from localflow.config import MODELS_DIR

BIN_DIR = MODELS_DIR.parent / "bin"  # %LOCALAPPDATA%/LocalFlow/bin


@dataclass(frozen=True)
class Asset:
    name: str
    url: str
    size: int | None = None
    sha256: str | None = None  # None = record on first download (dev), verified afterwards


LLAMA_BUILD = "b10819"
_LLAMA_BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}"

# llama.cpp Windows builds. The CUDA build needs the matching cudart package next to it.
LLAMA_ASSETS = {
    "cuda": [
        Asset(f"llama-{LLAMA_BUILD}-bin-win-cuda-13.3-x64.zip", f"{_LLAMA_BASE}/llama-{LLAMA_BUILD}-bin-win-cuda-13.3-x64.zip"),
        Asset("cudart-llama-bin-win-cuda-13.3-x64.zip", f"{_LLAMA_BASE}/cudart-llama-bin-win-cuda-13.3-x64.zip"),
    ],
    "cpu": [
        Asset(f"llama-{LLAMA_BUILD}-bin-win-cpu-x64.zip", f"{_LLAMA_BASE}/llama-{LLAMA_BUILD}-bin-win-cpu-x64.zip"),
    ],
}


@dataclass(frozen=True)
class GGUFModel:
    key: str
    repo: str
    filename: str
    context: int = 4096
    approx_gb: float = 0.0
    note: str = ""


# Clean-up models. Qwen3-4B-Instruct-2507 is the non-thinking instruct variant: strong at
# edit-style tasks, ~2.5 GB, fits beside the speech model on an 8 GB card.
CLEANUP_MODELS = {
    "qwen3-4b": GGUFModel("qwen3-4b", "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                          approx_gb=2.5, note="default: best quality that fits"),
    "qwen3-1.7b": GGUFModel("qwen3-1.7b", "unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf",
                            approx_gb=1.1, note="faster, for battery or small GPUs"),
}
DEFAULT_CLEANUP_MODEL = "qwen3-4b"


def llama_dir(kind: str = "cuda") -> Path:
    """Each build kind gets its own directory: the CUDA and CPU zips share file names."""
    return BIN_DIR / "llama" / LLAMA_BUILD / kind


def llama_archive_dir() -> Path:
    return BIN_DIR / "llama" / LLAMA_BUILD


def llama_server_exe(kind: str = "cuda") -> Path:
    return llama_dir(kind) / "llama-server.exe"


def gguf_dir() -> Path:
    return MODELS_DIR / "llm"


def gguf_path(key: str) -> Path:
    return gguf_dir() / CLEANUP_MODELS[key].filename


def is_windows_x64() -> bool:
    return platform.system() == "Windows" and platform.machine().lower() in ("amd64", "x86_64")
