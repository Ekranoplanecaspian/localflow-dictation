"""OpenAI Whisper large-v3-turbo through onnx-asr, on the same onnxruntime path as Parakeet.

Used for the languages Parakeet does not cover (Parakeet: 25 European languages; Whisper: 99),
and loaded lazily the first time such a language is requested.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from localflow.config import STTConfig

log = logging.getLogger(__name__)

DEFAULT_MODEL = "onnx-community/whisper-large-v3-turbo"


class WhisperOnnxTranscriber:
    name = "whisper-onnx"
    sample_rate = 16000

    def __init__(self, cfg: STTConfig, model: str = DEFAULT_MODEL):
        import onnx_asr
        import onnxruntime as ort

        from localflow.stt.parakeet import ParakeetTranscriber

        self.cfg = cfg
        self.model_name = model
        self.device = ParakeetTranscriber._pick_device(cfg.device)
        # fp16 export is the sensible default on the GPU; int8 on the CPU
        self.precision = "fp16" if self.device == "cuda" else "int8"
        if self.device == "cuda":
            providers: list = [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested",
                                                          "cudnn_conv_algo_search": "HEURISTIC"}),
                               "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        t0 = time.perf_counter()
        self.model = onnx_asr.load_model(model, quantization=self.precision, providers=providers, sess_options=so)
        self._language_kwarg = True
        log.info("Whisper %s loaded (%s, %s) in %.1fs", model, self.device, self.precision, time.perf_counter() - t0)

    def warmup(self) -> None:
        self.transcribe(np.zeros(self.sample_rate * 2, dtype=np.float32))

    def warm(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        if audio.size == 0:
            return ""
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if language and self._language_kwarg:
            try:
                return str(self.model.recognize(audio, sample_rate=self.sample_rate, language=language)).strip()
            except TypeError:
                self._language_kwarg = False
                log.info("this onnx-asr build ignores `language`; Whisper will auto-detect")
        return str(self.model.recognize(audio, sample_rate=self.sample_rate)).strip()
