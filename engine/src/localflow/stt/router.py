"""Language routing: Parakeet for the languages it knows, Whisper for everything else.

Parakeet TDT 0.6B v3 auto-detects among 25 European languages and is the fast, accurate
default. A dictation whose language is set to anything else goes to Whisper large-v3-turbo,
which is loaded on first use so users who never need it never pay for it.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from localflow.config import STTConfig
from localflow.stt.parakeet import ParakeetTranscriber

log = logging.getLogger(__name__)

PARAKEET_LANGUAGES = frozenset(
    "bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split()
)


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
        self.primary = ParakeetTranscriber(cfg)
        self._whisper = None
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
        if normalize_language(self.cfg.language) not in (None, *PARAKEET_LANGUAGES):
            self.whisper().warmup()

    def warm(self) -> None:
        self.primary.warm()

    def whisper(self):
        with self._lock:
            if self._whisper is None:
                from localflow.stt.whisper_onnx import WhisperOnnxTranscriber

                self._whisper = WhisperOnnxTranscriber(self.cfg)
            return self._whisper

    def pick(self, language: str | None):
        code = normalize_language(language) or normalize_language(self.cfg.language)
        if code is None or code in PARAKEET_LANGUAGES:
            return self.primary
        return self.whisper()

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        code = normalize_language(language) or normalize_language(self.cfg.language)
        backend = self.pick(code)
        return backend.transcribe(audio, language=code if backend is not self.primary else None)
