"""Language routing: Parakeet for the languages it knows, Whisper for everything else.

Parakeet TDT 0.6B v3 auto-detects among 25 European languages and is the fast, accurate
default; v2 knows only English. A dictation whose language is set to anything else goes to
Whisper large-v3-turbo, which is loaded on first use so users who never need it never pay for it.
"""

from __future__ import annotations

import gc
import logging
import threading
import time

import numpy as np

from localflow.config import STTConfig
from localflow.stt import catalogue
from localflow.stt.catalogue import PARAKEET_LANGUAGES
from localflow.stt.parakeet import ParakeetTranscriber

log = logging.getLogger(__name__)

# Whisper is dropped after this long unused. It is loaded for the odd dictation in a language
# Parakeet does not know, and used to stay for good: 1.6 GB of graphics memory and more of RAM.
WHISPER_IDLE_S = 600.0


def normalize_language(language: str | None) -> str | None:
    if not language:
        return None
    code = language.strip().lower().replace("_", "-")
    if code in ("auto", "detect", ""):
        return None
    return code.split("-")[0]


class RoutedTranscriber:
    name = "auto"
    sample_rate = 16000

    def __init__(self, cfg: STTConfig):
        self.cfg = cfg
        entry = catalogue.current(cfg)
        self.languages = entry.languages if entry and entry.languages else PARAKEET_LANGUAGES
        self.primary = ParakeetTranscriber(cfg)
        self._whisper = None
        self._whisper_used = 0.0
        self._lock = threading.Lock()

    # the engine reports the primary model's placement
    @property
    def device(self) -> str:
        return self.primary.device

    @property
    def precision(self) -> str:
        return self.primary.precision

    def warmup(self) -> None:
        self.primary.warmup()
        if normalize_language(self.cfg.language) not in (None, *self.languages):
            self.whisper().warmup()

    def warm(self) -> None:
        self.primary.warm()

    def whisper(self):
        with self._lock:
            if self._whisper is None:
                from localflow.stt.whisper_onnx import WhisperOnnxTranscriber

                self._whisper = WhisperOnnxTranscriber(self.cfg)
            self._whisper_used = time.monotonic()
            return self._whisper

    def drop_idle_whisper(self, after_s: float = WHISPER_IDLE_S) -> bool:
        """Unload Whisper if it has not been used for `after_s`. Call it on the speech worker,
        so no decode can be using it. Returns whether it was unloaded."""
        with self._lock:
            if self._whisper is None or time.monotonic() - self._whisper_used < after_s:
                return False
            self._whisper = None
        gc.collect()
        log.info("Whisper unloaded: not used for %.0f minutes", after_s / 60)
        return True

    def pick(self, language: str | None):
        code = normalize_language(language) or normalize_language(self.cfg.language)
        if code is None or code in self.languages:
            return self.primary
        return self.whisper()

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        code = normalize_language(language) or normalize_language(self.cfg.language)
        backend = self.pick(code)
        return backend.transcribe(audio, language=code if backend is not self.primary else None)
