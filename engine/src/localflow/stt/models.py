"""Model files on disk: where they live and how variants (fp16) are produced.

The upstream Parakeet ONNX export ships fp32 and int8 encoders. On the GPU we want fp16
(half the VRAM, tensor cores), so the fp32 encoder is converted once into
%LOCALAPPDATA%/LocalFlow/models/<name>-fp16/ and loaded from there afterwards.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from localflow.config import MODELS_DIR

log = logging.getLogger(__name__)

PARAKEET_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"
PARAKEET_SHARED_FILES = ["config.json", "vocab.txt", "nemo128.onnx", "decoder_joint-model.onnx"]


def parakeet_fp16_dir(model_name: str = "nemo-parakeet-tdt-0.6b-v3") -> Path:
    return MODELS_DIR / f"{model_name.removeprefix('nemo-')}-fp16"


def ensure_parakeet_fp16(model_name: str = "nemo-parakeet-tdt-0.6b-v3") -> Path:
    """Return the directory holding the fp16 Parakeet export, converting it on first use.
    Raises RuntimeError if the conversion tools are missing (caller falls back to fp32)."""
    d = parakeet_fp16_dir(model_name)
    if (d / "encoder-model.onnx").exists() and (d / "encoder-model.onnx.data").exists():
        return d
    try:
        import onnx
        from huggingface_hub import hf_hub_download
        from onnxconverter_common import float16
    except ImportError as e:
        raise RuntimeError(f"fp16 conversion needs onnx + onnxconverter-common ({e}); pip install -e ./engine[gpu]") from e

    log.info("Converting Parakeet encoder to fp16 (one-time, ~1 min) into %s", d)
    t0 = time.perf_counter()
    d.mkdir(parents=True, exist_ok=True)
    for f in PARAKEET_SHARED_FILES:
        shutil.copy(hf_hub_download(PARAKEET_REPO, f), d / f)
    enc = hf_hub_download(PARAKEET_REPO, "encoder-model.onnx")
    hf_hub_download(PARAKEET_REPO, "encoder-model.onnx.data")
    model = onnx.load(enc)
    model16 = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    tmp = d / "encoder-model.onnx.tmp"
    onnx.save_model(model16, str(tmp), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="encoder-model.onnx.data", size_threshold=1024)
    tmp.replace(d / "encoder-model.onnx")
    log.info("fp16 encoder ready in %.0fs", time.perf_counter() - t0)
    return d
