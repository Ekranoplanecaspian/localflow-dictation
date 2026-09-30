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
    # AMD and Intel graphics, built in or not (B3). Size and checksum from the release page.
    "vulkan": [
        Asset(f"llama-{LLAMA_BUILD}-bin-win-vulkan-x64.zip", f"{_LLAMA_BASE}/llama-{LLAMA_BUILD}-bin-win-vulkan-x64.zip",
              size=35_227_586, sha256="4c5ff97b5440024906fc90f67809d84b92d9b77847c7d1a800701a36499e565e"),
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
    # How the Hub presents it. Only `offered` models are listed there; the rest stay usable by
    # key (config file, `bench cleanup --model`) so an evaluation never needs a code change.
    label: str = ""
    blurb: str = ""
    speed: int | None = None  # 1-5, from `bench cleanup`
    accuracy: int | None = None  # 1-5
    recommended: bool = False
    offered: bool = False


# Clean-up models. Measured with `localflow bench cleanup --model <key>` on 2026-09-22 (RTX 4060
# laptop, one run at a time with the GPU cooled below 60 C): the 30-case clean-up set plus the
# 12-case command-mode set. The prompts were tuned on Qwen3-4B, which favours it somewhat.
#
#                   exact  must-not  word err  clean-up p50  command  notes
#   qwen3-4b        22/30     0        3.3 %      381 ms      12/12
#   phi-4-mini      17/30     0        4.5 %      185 ms       8/12   never broke a correction
#   gemma-4-e2b     13/30     2        2.9 %      208 ms      10/12   violations: numbers left as words
#   qwen3.5-4b      19/30     2        6.4 %      559 ms       8/12   reverted a self-correction
#   llama-3.2-3b    13/30     2       11.5 %      171 ms       7/12
#   gemma-3-4b      10/30     4       15.1 %      377 ms       9/12
#   qwen3.5-2b       8/30     4       11.8 %      354 ms       5/12
#   qwen3-1.7b       8/30     5       15.6 %      175 ms       2/12   compresses instead of editing
#   ministral-3-3b   -                                                llama-server 500s on its template
# Only the first three are offered in the Hub. Qwen3.5-9B and Gemma 4 E4B (5+ GB) were not tested:
# neither fits beside the speech model on an 8 GB card.
CLEANUP_MODELS = {
    "qwen3-4b": GGUFModel(
        "qwen3-4b", "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        approx_gb=2.5, note="default: best quality that fits",
        label="Qwen3 4B",
        blurb="The most careful editor in our tests: the most exact clean-ups, never undid one of your "
              "corrections, and handled every command-mode rewrite.",
        speed=3, accuracy=5, recommended=True, offered=True,
    ),
    "phi-4-mini": GGUFModel(
        "phi-4-mini", "unsloth/Phi-4-mini-instruct-GGUF", "Phi-4-mini-instruct-Q4_K_M.gguf",
        approx_gb=2.5,
        label="Phi-4 mini",
        blurb="Twice as fast and just as careful with your corrections, but a plainer editor and weaker "
              "at command-mode rewrites.",
        speed=5, accuracy=4, offered=True,
    ),
    "gemma-4-e2b": GGUFModel(
        "gemma-4-e2b", "google/gemma-4-E2B-it-qat-q4_0-gguf", "gemma-4-E2B_q4_0-it.gguf",
        approx_gb=3.4,
        label="Gemma 4 E2B",
        blurb="Fast, and good at command-mode rewrites. Sometimes leaves numbers spelled out "
              "(\"two point four million\"), and the largest download.",
        speed=5, accuracy=3, offered=True,
    ),
    # Measured and not offered; kept so a configuration naming one still loads.
    "qwen3-1.7b": GGUFModel("qwen3-1.7b", "unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf", approx_gb=1.1),
    "qwen3.5-4b": GGUFModel("qwen3.5-4b", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", approx_gb=2.7),
    "qwen3.5-2b": GGUFModel("qwen3.5-2b", "unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", approx_gb=1.3),
    "gemma-3-4b": GGUFModel("gemma-3-4b", "unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf", approx_gb=2.5),
    "llama-3.2-3b": GGUFModel("llama-3.2-3b", "unsloth/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                              approx_gb=2.0),
}
DEFAULT_CLEANUP_MODEL = "qwen3-4b"


def offered() -> list[GGUFModel]:
    return [m for m in CLEANUP_MODELS.values() if m.offered]


def describe(current: str | None) -> list[dict]:
    """The clean-up models the Hub offers. `current` is the bundled model in use, or None when
    clean-up runs on another provider (then none of these is in use)."""
    return [
        {
            "key": m.key,
            "label": m.label or m.key,
            "blurb": m.blurb,
            "size_gb": m.approx_gb,
            "installed": gguf_path(m.key).exists(),
            "current": m.key == current,
            "recommended": m.recommended,
            "speed": m.speed,
            "accuracy": m.accuracy,
        }
        for m in offered()
    ]


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
