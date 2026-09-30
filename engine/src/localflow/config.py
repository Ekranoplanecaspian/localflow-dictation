"""User configuration: dataclasses + JSON persistence in %APPDATA%/LocalFlow."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

APP_NAME = "LocalFlow"
CONFIG_DIR = Path(os.environ.get("APPDATA", str(Path.home()))) / APP_NAME
CONFIG_PATH = CONFIG_DIR / "config.json"
LOG_PATH = CONFIG_DIR / "localflow.log"
# Where a hard crash (a fault in native code) writes its stack. Not the log: that file is
# renamed when it rolls over, which Windows refuses while another handle holds it open.
FAULT_PATH = CONFIG_DIR / "localflow-fault.log"
# Set by the shell after the engine has crashed three times in two minutes.
SAFE_MODE_ENV = "LOCALFLOW_SAFE_MODE"
MODELS_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / APP_NAME / "models"


@dataclass
class AudioConfig:
    # The microphone `localflow bench record` and `localflow devices` use; dictation uses the
    # one chosen in the Hub. None = system default input; str = substring of the device name.
    device: int | str | None = None


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
class ComputeConfig:
    """Where the models run. See localflow/placement.py for the policy."""

    # adaptive: LocalFlow moves models between the graphics card and the processor by the GPU's
    # temperature and load | gpu: always the graphics card | cpu: never the graphics card
    mode: str = "adaptive"
    temp_limit_c: int = 80  # adaptive: from here, clean-up moves to the processor
    idle_release_min: float = 10.0  # adaptive: free the graphics card after this long unused; 0 = never
    # Let LocalFlow choose the model: the most accurate one that is quick enough on the device it
    # runs on (localflow/modelchoice.py). Choosing a model in the Hub turns this off for that kind.
    auto_speech: bool = True
    auto_cleanup: bool = True


@dataclass
class NetworkConfig:
    """Where downloads come from (localflow/net.py). The proxy is Windows' own, not a setting."""

    # A Hugging Face mirror for networks where huggingface.co is blocked, e.g. https://hf-mirror.com;
    # "" for huggingface.co itself.
    hf_endpoint: str = ""


# 2: the AI layer (bundled llama-server) replaced the Ollama-only clean-up
# 3: automatic model choice; a model already chosen by hand stays chosen
CONFIG_VERSION = 3



def for_this_pc(cfg: "Config") -> "Config":
    """Defaults that depend on the machine, applied to a fresh config (first run, reset). On a
    PC with 8 GB of RAM or less, AI clean-up starts off: with speech it takes about 6 GB. The
    Hub says why, and it can be turned on."""
    from localflow import hwinfo

    if hwinfo.ram_gb() < hwinfo.LOW_MEMORY_GB:
        cfg.postprocess.llm_cleanup = False
    return cfg


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    log_level: str = "INFO"
    version: int = CONFIG_VERSION

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        if not path.exists():
            cfg = for_this_pc(cls())
            cfg.save(path)
            log.info("Wrote default config to %s", path)
            return cfg
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))  # -sig: a byte-order mark is not an error
            if not isinstance(data, dict):
                raise ValueError(f"expected a JSON object, found {type(data).__name__}")
            migrated = migrate(data)
            cfg = _from_dict(cls, migrated)
            written_by = data.get("version")
            if isinstance(written_by, int) and written_by > CONFIG_VERSION:
                # A newer LocalFlow wrote this (and this one is older: a downgrade). What is
                # understood is used; nothing is saved over it, so the newer version finds its
                # settings as it left them. Changes last until the engine stops.
                cfg.newer = written_by
                log.warning("%s was written by a newer LocalFlow (settings version %d; this one knows %d): "
                            "reading what is understood, and not saving changes over it",
                            path.name, written_by, CONFIG_VERSION)
            # A hand-edited file gets the same checks as a change from the Hub.
            from localflow.validate import check_postprocess

            cfg.postprocess, problems = check_postprocess(cfg.postprocess, PostProcessConfig())
            for p in problems:
                log.warning("%s: %s; using the default", path.name, p)
        except (ValueError, TypeError, AttributeError) as e:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
            # A settings file that cannot be read used to stop the engine at start-up, every
            # time, until someone repaired it by hand. Start on the defaults instead, and keep
            # the broken file beside it: the dictionary and snippets in it are the user's own
            # work, and may well be recoverable.
            aside = _set_aside(path)
            log.error("%s could not be read (%s); starting with default settings. The unreadable "
                      "file was kept as %s.", path, e, aside or "(could not be moved)")
            cfg = cls()
            cfg.save(path)
            return cfg
        if migrated is not data:
            cfg.save(path)
        return cfg

    # Not a setting: the settings version of the file, when a newer LocalFlow wrote it.
    newer: int | None = field(default=None, repr=False, compare=False)

    def save(self, path: Path = CONFIG_PATH) -> None:
        if self.newer:
            log.info("not saving settings: %s belongs to a newer LocalFlow", path.name)
            return
        data = asdict(self)
        data.pop("newer", None)
        _restore_pinned(self, data)
        write_atomic(path, json.dumps(data, indent=2, ensure_ascii=False))


# Safe mode: the plainest way to run, for an engine that keeps crashing. Every override is a
# (section, field) of Config and the value it takes.
SAFE_MODE: dict[tuple[str, str], Any] = {
    ("compute", "mode"): "cpu",
    ("compute", "auto_speech"): False,
    ("compute", "auto_cleanup"): False,
    ("stt", "backend"): STTConfig.backend,
    ("stt", "model"): STTConfig.model,
    ("stt", "precision"): STTConfig.precision,
    ("stt", "model_path"): None,
    ("stt", "device"): "cpu",
    ("postprocess", "llm_cleanup"): False,
}


def apply_safe_mode(cfg: Config) -> None:
    """Run on the processor, with the default speech model and no AI clean-up - in memory only.

    The engine saves its settings from several places (the Hub, model switches, placement
    moves), and none of them may write safe mode to disk, or it would outlive the problem that
    caused it. So each overridden setting is pinned to the user's own value, which is what
    `save` writes - unless the user changes that setting themselves while in safe mode, which
    is their choice to keep.
    """
    pinned = {}
    for (section, name), value in SAFE_MODE.items():
        part = getattr(cfg, section)
        pinned[(section, name)] = getattr(part, name)
        setattr(part, name, value)
    cfg._pinned = pinned  # type: ignore[attr-defined]


def in_safe_mode(cfg: Config) -> bool:
    return bool(getattr(cfg, "_pinned", None))


def _restore_pinned(cfg: Config, data: dict[str, Any]) -> None:
    pinned = getattr(cfg, "_pinned", None)
    if not pinned:
        return
    for (section, name), users in list(pinned.items()):
        if data[section][name] == SAFE_MODE[(section, name)]:
            data[section][name] = users
        else:
            # Changed by hand since safe mode began: the new value is the user's now.
            del pinned[(section, name)]


_WRITE_LOCK = threading.Lock()


def write_atomic(path: Path, text: str) -> None:
    """Replace `path` with `text` all at once: a reader sees the old file or the new one, never
    part of either.

    Writing in place truncated the file first, so a crash, a power cut or a second thread saving
    at the same moment could leave half a file - and a config file in that state stopped the
    engine from starting at all. The engine saves from several threads (the Hub's changes, model
    switches, placement moves), so writes are also taken one at a time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            # Windows refuses to replace a file another process has open this instant (the
            # shell reading it for the Hub, an antivirus scan); that passes in milliseconds.
            for attempt in range(20):
                try:
                    os.replace(tmp, path)
                    return
                except PermissionError:
                    if attempt == 19:
                        raise
                    time.sleep(0.025)
        finally:
            tmp.unlink(missing_ok=True)


def _set_aside(path: Path) -> Path | None:
    aside = path.with_name(f"{path.name}.broken-{time.strftime('%Y%m%d-%H%M%S')}")
    try:
        os.replace(path, aside)
        return aside
    except OSError:
        return None


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Bring a stored config forward. Returns the same object when nothing changed.

    v1 -> v2: the clean-up model moved from an optional Ollama install to a bundled
    llama-server, so the old LLM settings (off by default, an Ollama URL, an Ollama model
    name) no longer mean anything. Drop them and let the new defaults apply; everything the
    user actually chose (hotkey, mic, dictionary, snippets) is kept.
    """
    version = int(data.get("version", 1))
    if version >= CONFIG_VERSION:
        return data
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in data.items()}
    if version < 2:
        pp = out.get("postprocess")
        if isinstance(pp, dict):
            for stale in ("llm_cleanup", "llm_model", "llm_url", "llm_provider"):
                pp.pop(stale, None)
        log.info("Config migrated to v2: the clean-up model is now bundled (was Ollama-only)")
    if version < 3:
        # Automatic model choice is new. Somebody who had picked a model other than the default
        # picked it on purpose, so that kind stays as chosen; everyone else gets Automatic.
        stt, pp = out.get("stt") or {}, out.get("postprocess") or {}
        default_stt, default_pp = STTConfig(), PostProcessConfig()
        speech_default = all(stt.get(k, getattr(default_stt, k)) == getattr(default_stt, k)
                             for k in ("backend", "model", "precision"))
        cleanup_default = pp.get("llm_model", default_pp.llm_model) == default_pp.llm_model
        compute = dict(out.get("compute") or {})
        compute.setdefault("auto_speech", speech_default)
        compute.setdefault("auto_cleanup", cleanup_default)
        out["compute"] = compute
        log.info("Config migrated to v3: automatic model choice (speech %s, clean-up %s)",
                 "on" if speech_default else "off, a model was chosen", "on" if cleanup_default else "off, a model was chosen")
    out["version"] = CONFIG_VERSION
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
