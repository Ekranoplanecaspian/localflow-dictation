"""OpenAI Whisper via faster-whisper (CTranslate2). Optional multilingual fallback (99 languages).

Install: pip install -e .[whisper]   then set stt.backend = "whisper", stt.model = "large-v3-turbo".
GPU needs CUDA 12 + cuDNN 9 DLLs on PATH (pip install nvidia-cublas-cu12 nvidia-cudnn-cu12).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from localflow.config import STTConfig

log = logging.getLogger(__name__)


class WhisperTranscriber:
    name = "whisper"
    sample_rate = 16000

    def __init__(self, cfg: STTConfig):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError("faster-whisper is not installed: pip install -e .[whisper]") from e

        self.cfg = cfg
        device = "cuda" if cfg.device in ("cuda", "auto") else "cpu"
        compute = {"fp32": "float32", "fp16": "float16", "int8": "int8"}.get(
            cfg.precision, "float16" if device == "cuda" else "int8")
        try:
            self.model = WhisperModel(cfg.model, device=device, compute_type=compute)
        except Exception as e:
            if device == "cuda" and cfg.device == "auto":
                log.warning("Whisper on CUDA failed (%s); falling back to CPU int8", e)
                device, compute = "cpu", "int8"
                self.model = WhisperModel(cfg.model, device=device, compute_type=compute)
            else:
                raise
        self.device, self.precision = device, compute
        log.info("Whisper %s loaded (%s, %s)", cfg.model, device, compute)

    def warmup(self) -> None:
        t0 = time.perf_counter()
        self.transcribe(np.zeros(self.sample_rate, dtype=np.float32))
        log.debug("whisper warmup %.0f ms", (time.perf_counter() - t0) * 1000)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        if audio.size == 0:
            return ""
        segments, _info = self.model.transcribe(
            np.ascontiguousarray(audio, dtype=np.float32),
            language=language or self.cfg.language,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()
