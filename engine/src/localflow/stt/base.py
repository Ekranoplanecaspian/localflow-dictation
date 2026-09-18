"""Speech-to-text backend interface. Backends are swappable via config.stt.backend."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from localflow.config import STTConfig


@runtime_checkable
class Transcriber(Protocol):
    name: str
    sample_rate: int

    def warmup(self) -> None:
        """Run one dummy inference so the first real dictation is not slow."""

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        """audio: float32 mono in [-1, 1] at self.sample_rate. Returns plain text."""


def build_transcriber(cfg: STTConfig) -> Transcriber:
    if cfg.backend == "auto":  # Parakeet, with Whisper turbo loaded on demand for other languages
        from localflow.stt.router import RoutedTranscriber

        return RoutedTranscriber(cfg)
    if cfg.backend == "parakeet":
        from localflow.stt.parakeet import ParakeetTranscriber

        return ParakeetTranscriber(cfg)
    if cfg.backend == "whisper":  # Whisper turbo (onnxruntime) for everything
        from localflow.stt.whisper_onnx import WhisperOnnxTranscriber

        return WhisperOnnxTranscriber(cfg)
    if cfg.backend == "whisper-ct2":  # faster-whisper / CTranslate2 (optional extra)
        from localflow.stt.whisper import WhisperTranscriber

        return WhisperTranscriber(cfg)
    raise ValueError(f"unknown stt backend {cfg.backend!r} (expected auto, parakeet, whisper or whisper-ct2)")
