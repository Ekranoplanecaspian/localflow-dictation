"""NVIDIA Parakeet TDT 0.6B v3 via onnx-asr (onnxruntime).

Why this model: top of the Open ASR leaderboard for its size (WER ~6.3% vs ~7.4% for
Whisper large-v3), punctuates and capitalises on its own, 25 European languages, and it
is an order of magnitude faster than Whisper. Runs on CUDA (fp16) when the GPU works,
otherwise on the CPU (fp32).

GPU notes learned on the RTX 4060 laptop:
  * a 1 s tail costs ~30 ms warm; the transducer decoder's per-frame calls are not the
    bottleneck (decoder on CPU vs GPU is a wash), so everything stays on one device.
  * the GPU drops to 210 MHz after ~2 s idle and a cold call is 7x slower. Tiny kernels do
    not wake it; real inference does. `warm()` exists so the session can kick the GPU the
    moment the hotkey goes down.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from localflow.config import STTConfig

log = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_cuda_probe: tuple[bool, str] | None = None

# A one-node ONNX graph (c = a + b, float[1]), pre-serialised so the probe needs no `onnx` package.
_PROBE_MODEL = (
    b'\x08\t\x12\tlocalflow:J\n\x0e\n\x01a\n\x01b\x12\x01c"\x03Add\x12\x05probeZ\x0f\n\x01a\x12\n\n\x08'
    b'\x08\x01\x12\x04\n\x02\x08\x01Z\x0f\n\x01b\x12\n\n\x08\x08\x01\x12\x04\n\x02\x08\x01b\x0f\n\x01c'
    b'\x12\n\n\x08\x08\x01\x12\x04\n\x02\x08\x01B\x04\n\x00\x10\x11'
)


def cuda_available() -> tuple[bool, str]:
    """(usable, reason). Probes once per process by building a one-op session on the CUDA
    provider, which is the only reliable test that the driver, CUDA and cuDNN DLLs all agree."""
    global _cuda_probe
    if _cuda_probe is not None:
        return _cuda_probe
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        _cuda_probe = (False, "onnxruntime-gpu is not installed")
        return _cuda_probe
    try:
        ort.preload_dlls()  # loads CUDA/cuDNN from the nvidia-* pip packages when present
    except Exception as e:
        log.debug("preload_dlls: %s", e)
    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(_PROBE_MODEL, so, providers=["CUDAExecutionProvider"])
        if "CUDAExecutionProvider" not in sess.get_providers():
            _cuda_probe = (False, "CUDA provider did not load (missing CUDA/cuDNN DLLs?)")
        else:
            sess.run(None, {"a": np.ones(1, np.float32), "b": np.ones(1, np.float32)})
            _cuda_probe = (True, "ok")
    except Exception as e:
        _cuda_probe = (False, f"CUDA provider failed: {str(e).splitlines()[0][:160]}")
    return _cuda_probe


class ParakeetTranscriber:
    name = "parakeet"
    sample_rate = 16000

    def __init__(self, cfg: STTConfig):
        import onnx_asr
        import onnxruntime as ort

        self.cfg = cfg
        self.device = self._pick_device(cfg.device)
        self.precision = self._pick_precision(cfg.precision, self.device)
        if self.device == "cuda":
            # kSameAsRequested keeps the CUDA memory arena close to what the model needs
            # (the default doubles allocations and idled at ~3.5 GB for a 2.4 GB model).
            # cudnn_conv_algo_search=HEURISTIC: the default EXHAUSTIVE search re-benchmarks the
            # encoder's convolutions for every new input length, and every dictation has a new
            # length, which cost ~50-100 ms per utterance on this machine.
            cuda_opts = {
                "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": os.environ.get("LOCALFLOW_CUDNN_ALGO", "HEURISTIC"),
            }
            providers: list = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.log_severity_level = 3

        path = cfg.model_path
        quantization = "int8" if self.precision == "int8" else None
        if self.precision == "fp16" and path is None:
            try:
                from localflow.stt.models import ensure_parakeet_fp16

                path = str(ensure_parakeet_fp16(cfg.model))
            except Exception as e:
                log.warning("fp16 model unavailable (%s); using fp32", e)
                self.precision = "fp32"
        t0 = time.perf_counter()
        self.model = onnx_asr.load_model(cfg.model, path=path, quantization=quantization,
                                         providers=providers, sess_options=so)
        self._longform = None
        log.info("Parakeet %s loaded (%s, %s) in %.1fs", cfg.model, self.device, self.precision, time.perf_counter() - t0)

    # device / precision selection -----------------------------------------------------------
    @staticmethod
    def _pick_device(device: str) -> str:
        if device == "cpu":
            return "cpu"
        ok, reason = cuda_available()
        if ok:
            return "cuda"
        if device == "cuda":
            raise RuntimeError(f"CUDA requested but unavailable: {reason}. Install with: pip install -e ./engine[gpu]")
        log.info("Using CPU for speech recognition (%s)", reason)
        return "cpu"

    @staticmethod
    def _pick_precision(precision: str, device: str) -> str:
        """auto -> fp32 on both devices for now. fp16 is opt-in: the converted encoder that loads
        (onnxruntime's converter) keeps its big Constant tensors in fp32 and runs ~40 % slower
        through the extra casts, so it buys little until a proper NeMo fp16 export exists."""
        if precision in ("fp32", "fp16", "int8"):
            return precision
        return "fp32"

    # inference ---------------------------------------------------------------------------------
    def warmup(self) -> None:
        t0 = time.perf_counter()
        self.transcribe(np.zeros(self.sample_rate * 3, dtype=np.float32))
        log.debug("parakeet warmup %.0f ms", (time.perf_counter() - t0) * 1000)

    _warm_audio = np.zeros(2 * 16000, dtype=np.float32)  # 2 s of silence: enough GPU work to hold boost clocks

    def warm(self) -> None:
        """A short real inference (~30 ms on a busy GPU) that keeps the clocks up between real work."""
        if self.device == "cuda":
            self.model.recognize(self._warm_audio, sample_rate=self.sample_rate)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        if audio.size == 0:
            return ""
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.size / self.sample_rate > self.cfg.longform_seconds:
            segments = self._vad_model().recognize(audio, sample_rate=self.sample_rate)
            return " ".join(s.text.strip() for s in segments if getattr(s, "text", "").strip())
        return str(self.model.recognize(audio, sample_rate=self.sample_rate)).strip()

    def _vad_model(self):
        if self._longform is None:
            import onnx_asr

            self._longform = self.model.with_vad(onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"]))
        return self._longform
