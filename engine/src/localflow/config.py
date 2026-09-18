"""User configuration: dataclasses + JSON persistence in %APPDATA%/LocalFlow."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

APP_NAME = "LocalFlow"
CONFIG_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / APP_NAME
CONFIG_PATH = CONFIG_DIR / "config.json"
LOG_PATH = CONFIG_DIR / "localflow.log"
MODELS_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / APP_NAME / "models"


@dataclass
class HotkeyConfig:
    # Chord that must be held to record. Names: ctrl, win, alt, shift, single chars, f1..f24, space.
    # Default matches Wispr Flow on Windows (Ctrl+Win).
    keys: list[str] = field(default_factory=lambda: ["ctrl", "win"])
    # Double-tap the chord to lock recording on (hands-free); press again to stop.
    double_tap_hands_free: bool = True
    double_tap_ms: int = 400
    # Recordings shorter than this (excluding pre-roll) are discarded as accidental taps.
    min_record_seconds: float = 0.3


@dataclass
class AudioConfig:
    device: int | str | None = None  # None = system default input; str = substring of device name
    sample_rate: int = 16000
    preroll_ms: int = 500  # audio kept from *before* the hotkey press so the first word is not clipped
    max_seconds: int = 180
    sounds: bool = True  # short start/stop tones


@dataclass
class STTConfig:
    # auto = Parakeet for its 25 European languages, Whisper turbo (loaded on demand) for the rest
    # parakeet | whisper (onnxruntime, all languages) | whisper-ct2 (faster-whisper, optional extra)
    backend: str = "auto"
    model: str = "nemo-parakeet-tdt-0.6b-v3"
    device: str = "auto"  # auto (CUDA if it works, else CPU) | cpu | cuda
    # auto = fp32 (fp16 export is not yet a net win, see stt/parakeet.py) | fp32 | fp16 | int8
    precision: str = "auto"
    model_path: str | None = None  # directory with the model files; None = download / convert automatically
    # None/"auto" = Parakeet auto-detects among its languages. A code outside them ("hi", "ja", "ar", ...)
    # routes dictation to Whisper. Sessions can override this per dictation.
    language: str | None = None
    longform_seconds: float = 25.0  # above this, split with VAD (Parakeet's per-chunk limit is ~30 s)
    # While the hotkey is held, keep the GPU clocked up with back-to-back inferences so the final
    # pass is ~2x faster (a laptop GPU idles at 210 MHz and needs continuous load to boost).
    # auto = only on mains power | always | never
    gpu_keep_warm: str = "auto"
    language: str | None = None  # None = auto-detect


@dataclass
class PostProcessConfig:
    remove_fillers: bool = True
    spoken_punctuation: bool = False  # "comma", "period" -> punctuation (Parakeet already punctuates)
    spoken_newlines: bool = True  # "new line" / "new paragraph"
    dictionary: dict[str, str] = field(default_factory=dict)  # exact replacements: "heard" -> "meant"
    dictionary_terms: list[str] = field(default_factory=list)  # correct spellings; matched by sound ("Arnub" -> "Arnab")
    snippets: dict[str, str] = field(default_factory=dict)  # "my email" -> "arnab@example.com"
    custom_instructions: str = ""  # your own style notes for the clean-up model
    # Wispr-style auto-edits (self-corrections, lists, numbers, tone) via a language model
    llm_cleanup: bool = True
    llm_provider: str = "bundled"  # bundled (llama-server, local) | ollama | openai | anthropic
    llm_model: str = "qwen3-4b"  # bundled: key in llm/manifest.py; others: the provider's model name
    llm_min_words: int = 6  # shorter utterances skip the model unless they contain correction/number cues
    llm_max_tokens: int = 400
    llm_timeout_s: float = 8.0
    # Warm the model's prompt cache mid-utterance. Off by default: measured on an RTX 4060,
    # it saves ~40 ms of prompt evaluation but steals GPU time from the live speech decoding
    # that is running at the same time, which cost ~150 ms on the final decode. Worth turning
    # on when speech and clean-up are not on the same device (cloud provider, or CPU speech).
    llm_prefill: bool = False
    # provider details
    llm_url: str = ""  # ollama: http://127.0.0.1:11434 ; openai-compatible: base URL (e.g. https://api.groq.com/openai)
    llm_api_key: str = ""  # openai / anthropic providers


@dataclass
class InjectConfig:
    method: str = "auto"  # auto | type | paste
    paste_threshold_chars: int = 200  # auto: type below, paste at/above
    restore_clipboard: bool = True
    trailing_space: bool = True


@dataclass
class UIConfig:
    overlay: bool = True  # floating "flow bar" with live waveform while recording
    tray: bool = True  # system tray icon with status, autostart toggle, quit
    accent: str = "#8b6cff"
    overlay_bottom_px: int = 28  # gap between the bar and the top of the taskbar


CONFIG_VERSION = 2  # 2: the AI layer (bundled llama-server) replaced the Ollama-only clean-up


@dataclass
class Config:
    hotkey: HotkeyConfig = field(default_factory=HotkeyConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    inject: InjectConfig = field(default_factory=InjectConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    log_level: str = "INFO"
    version: int = CONFIG_VERSION

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        if not path.exists():
            cfg = cls()
            cfg.save(path)
            log.info("Wrote default config to %s", path)
            return cfg
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        migrated = migrate(data)
        cfg = _from_dict(cls, migrated)
        if migrated is not data:
            cfg.save(path)
        return cfg

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Bring a stored config forward. Returns the same object when nothing changed.

    v1 -> v2: the clean-up model moved from an optional Ollama install to a bundled
    llama-server, so the old LLM settings (off by default, an Ollama URL, an Ollama model
    name) no longer mean anything. Drop them and let the new defaults apply; everything the
    user actually chose (hotkey, mic, dictionary, snippets) is kept.
    """
    if int(data.get("version", 1)) >= CONFIG_VERSION:
        return data
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in data.items()}
    pp = out.get("postprocess")
    if isinstance(pp, dict):
        for stale in ("llm_cleanup", "llm_model", "llm_url", "llm_provider"):
            pp.pop(stale, None)
    out["version"] = CONFIG_VERSION
    log.info("Config migrated to v%d: the clean-up model is now bundled (was Ollama-only)", CONFIG_VERSION)
    return out


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build nested dataclasses from a dict, ignoring unknown keys and filling defaults."""
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        ftype = f.default_factory if f.default_factory is not None else None  # type: ignore[misc]
        if callable(ftype) and is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(ftype, value)  # type: ignore[arg-type]
        else:
            kwargs[f.name] = value
    return cls(**kwargs)
