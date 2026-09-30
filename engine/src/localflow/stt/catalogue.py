"""The speech models a user can choose between, and whether each is on disk yet.

Every entry maps onto the existing STT settings (backend, model, precision), so a stored
config stays meaningful and a hand-edited one still works: `current()` recognises it.

Files are named exactly rather than by pattern, which is what lets `is_installed()` answer
from the Hugging Face cache without touching the network.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

from localflow.config import STTConfig

log = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PARAKEET_LANGUAGES = frozenset(
    "bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split()
)

_PARAKEET_FP32 = ("config.json", "vocab.txt", "nemo128.onnx", "encoder-model.onnx", "encoder-model.onnx.data",
                  "decoder_joint-model.onnx")
_PARAKEET_INT8 = ("config.json", "vocab.txt", "nemo128.onnx", "encoder-model.int8.onnx",
                  "decoder_joint-model.int8.onnx")


def _whisper_files(precision: str) -> tuple[str, ...]:
    return ("config.json", "vocab.json", "added_tokens.json",
            f"onnx/encoder_model_{precision}.onnx", f"onnx/decoder_model_merged_{precision}.onnx")


@dataclass(frozen=True)
class SpeechModel:
    key: str
    label: str
    blurb: str
    backend: str  # auto (this model, Whisper for languages it lacks) | whisper
    model: str  # the onnx-asr name or Hugging Face repo, as stored in stt.model
    repo: str
    languages: frozenset[str] | None  # None = Whisper's 99
    precision: str = "auto"  # the stt.precision this entry sets
    gb: dict[str, float] | None = None  # approximate download by device: {"cuda": .., "cpu": ..}
    recommended: bool = False
    speed: dict[str, int] | None = None  # 1-5 by device, from the benchmarks below
    accuracy: int | None = None  # 1-5

    def files(self, device: str) -> tuple[str, ...]:
        if self.backend == "whisper":
            return _whisper_files("fp16" if device == "cuda" else "int8")
        return _PARAKEET_INT8 if self.precision == "int8" else _PARAKEET_FP32

    def size_gb(self, device: str) -> float:
        gb = self.gb or {}
        return gb.get(device, gb.get("cpu", 0.0))

    def apply(self, cfg: STTConfig) -> None:
        cfg.backend, cfg.model, cfg.precision = self.backend, self.model, self.precision
        cfg.model_path = None


# Ratings are measured, not guessed: `localflow bench run --speech <key>` on 2026-09-22, RTX 4060
# laptop GPU and Ryzen AI 9 CPU, over 60 LibriSpeech files and 30 of the developer's own
# dictations. Accuracy weighs the own-voice set most, because dictation is what the app is for.
#
#                        word error  own / libri     ms per audio second   GPU / CPU
#   Parakeet v3          3.7 % / 2.1 %               41 / 163
#   Parakeet v2          5.5 % / 1.7 %               46 / (same encoder as v3)
#   Parakeet v3 Compact  5.8 % / 2.3 %               220 / 124 (int8 falls back to the CPU on CUDA)
#   Whisper Turbo        4.5 % / 2.4 %               92 / 1443
#
# Whisper Small was tried and dropped: it took the developer's English for Hindi and looped
# ("वो वो वो ..."), 68 % word error on the own-voice set, and was slower than Turbo anyway.
MODELS: tuple[SpeechModel, ...] = (
    SpeechModel(
        "parakeet-v3", "Parakeet v3",
        "The best all-rounder: the most accurate in our tests and the fastest on a graphics "
        "card. Knows 25 European languages and works out which one you are speaking.",
        "auto", "nemo-parakeet-tdt-0.6b-v3", "istupakov/parakeet-tdt-0.6b-v3-onnx",
        PARAKEET_LANGUAGES, gb={"cpu": 2.6}, recommended=True,
        speed={"cuda": 5, "cpu": 3}, accuracy=5,
    ),
    SpeechModel(
        "parakeet-v2", "Parakeet v2",
        "English only. Excellent on clear, read-aloud English, but less forgiving of accents "
        "than v3.",
        "auto", "nemo-parakeet-tdt-0.6b-v2", "istupakov/parakeet-tdt-0.6b-v2-onnx",
        frozenset({"en"}), gb={"cpu": 2.5},
        speed={"cuda": 5, "cpu": 3}, accuracy=4,
    ),
    SpeechModel(
        "parakeet-v3-compact", "Parakeet v3 Compact",
        "A quarter of the download and a third of the memory. For computers without an NVIDIA "
        "graphics card, where it is quicker than v3; a little less accurate.",
        "auto", "nemo-parakeet-tdt-0.6b-v3", "istupakov/parakeet-tdt-0.6b-v3-onnx",
        PARAKEET_LANGUAGES, precision="int8", gb={"cpu": 0.7},
        speed={"cuda": 1, "cpu": 4}, accuracy=3,
    ),
    SpeechModel(
        "whisper-turbo", "Whisper Large v3 Turbo",
        "OpenAI's model, for 99 languages. About half the speed of Parakeet on a graphics card, "
        "and far too slow without one.",
        "whisper", "onnx-community/whisper-large-v3-turbo", "onnx-community/whisper-large-v3-turbo",
        None, gb={"cuda": 1.6, "cpu": 1.1},
        speed={"cuda": 3, "cpu": 1}, accuracy=4,
    ),
)

BY_KEY = {m.key: m for m in MODELS}


def get(key: str) -> SpeechModel:
    try:
        return BY_KEY[key]
    except KeyError:
        raise ValueError(f"unknown speech model {key!r} (expected one of: {', '.join(BY_KEY)})") from None


def current(cfg: STTConfig) -> SpeechModel | None:
    """The catalogue entry the settings describe, or None for a hand-built configuration."""
    int8 = cfg.precision == "int8"
    for m in MODELS:
        backend_ok = cfg.backend == m.backend or (m.backend == "auto" and cfg.backend == "parakeet")
        if m.model == cfg.model and backend_ok and (m.precision == "int8") == int8:
            return m
    return None


def speech_device(cfg: STTConfig) -> str:
    """Where speech will run, for choosing which files a model needs. Mirrors the transcriber."""
    if cfg.device == "cpu":
        return "cpu"
    from localflow.stt.parakeet import cuda_available

    return "cuda" if cuda_available()[0] else "cpu"


def is_installed(model: SpeechModel, device: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    for name in model.files(device):
        try:
            found = try_to_load_from_cache(model.repo, name)
        except Exception:
            return False
        if not isinstance(found, str):
            return False
    return True


Progress = Callable[[int, int], None]  # (bytes done, bytes total)


def download(model: SpeechModel, device: str, progress: Progress | None = None) -> None:
    """Fetch the files `model` needs into the Hugging Face cache, where onnx-asr finds them."""
    from huggingface_hub import snapshot_download

    from localflow.hfprogress import reporter

    kwargs = {"tqdm_class": reporter(progress)} if progress is not None else {}
    snapshot_download(model.repo, allow_patterns=list(model.files(device)), **kwargs)


def describe(cfg: STTConfig, device: str | Callable[[SpeechModel], str]) -> list[dict]:
    """The choices, as the Hub shows them. `device` is where the models run, or a function
    giving where each one would: the placement puts some on the processor whatever the GPU is
    doing (Parakeet Compact), and a speed rating is only true of the device it was measured on."""
    chosen = current(cfg)
    where = device if callable(device) else (lambda _m: device)
    return [_row(m, where(m), chosen) for m in MODELS]


def _row(m: SpeechModel, device: str, chosen: SpeechModel | None) -> dict:
    return {
        "key": m.key,
        "label": m.label,
        "blurb": m.blurb,
        "languages": len(m.languages) if m.languages is not None else 99,
        "size_gb": m.size_gb(device),
        "installed": is_installed(m, device),
        "current": chosen is not None and chosen.key == m.key,
        "recommended": m.recommended,
        "speed": (m.speed or {}).get(device),
        "accuracy": m.accuracy,
        "device": device,
    }
