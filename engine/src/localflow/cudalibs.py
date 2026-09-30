"""The CUDA libraries speech needs on an NVIDIA card, downloaded on first use (B2).

onnxruntime-gpu runs Parakeet on the graphics card through NVIDIA's CUDA runtime, cuBLAS, cuFFT
and cuDNN. They are 1.5 GB unpacked - most of what the engine used to weigh - and a PC without
an NVIDIA card never loads them. So the installer leaves them out, and an NVIDIA PC fetches them
once, in the background, while speech runs on the processor:

  * from PyPI, as the very wheels the engine is developed and tested with (pinned below by
    version, size and SHA-256; a mismatch is a hard error), through the Windows proxy like every
    other download (net.py), resuming a partial file
  * about 1.0 GB to download; only the DLLs are kept, flat, in
    %LOCALAPPDATA%\\LocalFlow\\bin\\cuda\\<TAG>, and the wheels are deleted afterwards
  * `onnxruntime.preload_dlls(directory=...)` loads them from there

A development virtualenv, or an older build that still carries them, has them as the `nvidia`
packages beside onnxruntime; those are used as they are, and nothing is downloaded.
LOCALFLOW_CUDA_FROM_DOWNLOAD=1 ignores them, to try the downloaded path from a virtualenv.

Which wheels: those whose DLLs a Parakeet decode on the RTX 4060 loaded (2026-09-28: cuDNN,
cuBLAS/cuBLASLt, cuFFT, the CUDA runtime), plus NVRTC and nvJitLink, which cuDNN's runtime-
compiled engines and cuFFT can load for inputs that decode did not happen to need. cuRAND,
in the virtualenv too, was never loaded and is left out (53 MB).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from localflow.llm.manifest import BIN_DIR

log = logging.getLogger(__name__)

TAG = "cu13.3-cudnn9.25"  # change with the versions below, so an update unpacks beside the old
FROM_DOWNLOAD_ENV = "LOCALFLOW_CUDA_FROM_DOWNLOAD"
_PYPI = "https://files.pythonhosted.org/packages"


@dataclass(frozen=True)
class Wheel:
    name: str
    url: str
    size: int
    sha256: str


WHEELS: tuple[Wheel, ...] = (
    Wheel("nvidia_cublas-13.6.0.2-py3-none-win_amd64.whl",
          f"{_PYPI}/08/8f/890a96ea1ff615100296977cce23296052dcb8c114d4e451201ec39df9bf/nvidia_cublas-13.6.0.2-py3-none-win_amd64.whl",
          394568225, "3b5bcd6bfb6f65010ebf195851bcb9b2aa34b9fe08479432002991c1fe84b67d"),
    Wheel("nvidia_cuda_nvrtc-13.3.33-py3-none-win_amd64.whl",
          f"{_PYPI}/a1/42/edce72f2c5a0f587168109c867f25f4a9a6cd7289ecf0d68ed2b1070f273/nvidia_cuda_nvrtc-13.3.33-py3-none-win_amd64.whl",
          45319163, "7d2af818851c0c224d5f92221e9226e51ee23c236df4b51f9194563979c888be"),
    Wheel("nvidia_cuda_runtime-13.3.29-py3-none-win_amd64.whl",
          f"{_PYPI}/d2/27/b53a5e0397842a5c11f0e1a39d4e5b2f22638a4126e83b3c4e196f62c969/nvidia_cuda_runtime-13.3.29-py3-none-win_amd64.whl",
          2630354, "0667ec61c3d897388efa305ed4f7609ace88849a753ba9c6311d06dca55fff4f"),
    Wheel("nvidia_cudnn_cu13-9.25.1.1-py3-none-win_amd64.whl",
          f"{_PYPI}/fd/0f/d7e4141c1126899c7b8d202eb3085380164beefef32f94cc8967ed3a00ff/nvidia_cudnn_cu13-9.25.1.1-py3-none-win_amd64.whl",
          407762053, "e1de75bf1ad9040414f9b13cc87135d660d13dd3c859180dd6b63353e571f860"),
    Wheel("nvidia_cufft-12.3.0.29-py3-none-win_amd64.whl",
          f"{_PYPI}/94/64/8e9d808720559d3cbfcd1d1bc8a2e6f55deb29d692513d5a93c8d417b7e5/nvidia_cufft-12.3.0.29-py3-none-win_amd64.whl",
          183939745, "510036a2bbab5c83ae93dc5c907c3a49d3518e3066ac3a2052ff0f7f9b27dfc4"),
    Wheel("nvidia_nvjitlink-13.3.33-py3-none-win_amd64.whl",
          f"{_PYPI}/67/f2/ec9c05a108095828dfc58840978c627b3c313fdf2a567c6de9ffbbb46901/nvidia_nvjitlink-13.3.33-py3-none-win_amd64.whl",
          37766359, "4297ee49639b4f2e07255a1d69b3acc7ab2d011bb892b403e91ac98368962e3b"),
)
DOWNLOAD_BYTES = sum(w.size for w in WHEELS)
UNPACKED_BYTES = 1_600 << 20  # the DLLs, a little over what the virtualenv holds (1.53 GB)

Progress = Callable[[int, int], None]  # (bytes done, bytes in all), across every wheel


def lib_dir() -> Path:
    return BIN_DIR / "cuda" / TAG


def _archive_dir() -> Path:
    return BIN_DIR / "cuda" / "downloads"


def bundled() -> bool:
    """The `nvidia` packages sit beside onnxruntime (a virtualenv, or a build that carries them)."""
    if os.environ.get(FROM_DOWNLOAD_ENV) == "1":
        return False
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return False
    return any((Path(p) / "cu13").is_dir() for p in (spec.submodule_search_locations or [])) if spec else False


def installed() -> bool:
    return (lib_dir() / ".ok").exists()


def available() -> bool:
    return bundled() or installed()


def dll_dir() -> Path | None:
    """Where onnxruntime should load CUDA from: the downloaded folder, or None to let it find
    the bundled packages (or nothing) by itself."""
    return None if bundled() or not installed() else lib_dir()


def ensure(progress: Progress | None = None) -> Path:
    """Download, verify and unpack the libraries unless they are already there. Returns the
    folder. Raises on a network failure, a full disk or a checksum mismatch (net.py and the
    downloader say which), leaving any partial download to be resumed next time."""
    if installed():
        return lib_dir()
    from localflow import net
    from localflow.llm.downloader import download

    archives = _archive_dir()
    missing = [w for w in WHEELS if not (archives / w.name).exists()]
    if missing:
        net.wait_for_proxy()
        net.ensure_space(archives, sum(w.size for w in missing) + UNPACKED_BYTES, "the graphics card libraries")
    done_before = sum(w.size for w in WHEELS if w not in missing)
    for wheel in missing:
        log.info("Downloading %s (%.0f MB)", wheel.name, wheel.size / 2**20)

        def step(_name: str, done: int, _total: int | None, base: int = done_before) -> None:
            if progress:
                progress(base + done, DOWNLOAD_BYTES)

        download(wheel.url, archives / wheel.name, step, wheel.sha256)
        done_before += wheel.size
    _unpack(archives)
    return lib_dir()


def _unpack(archives: Path) -> None:
    """Every DLL from the wheels, flat in one folder, then the marker and the wheels gone. A
    folder half unpacked (no marker) is simply unpacked again."""
    target = lib_dir()
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for wheel in WHEELS:
        with zipfile.ZipFile(archives / wheel.name) as z:
            for info in z.infolist():
                if info.filename.lower().endswith(".dll"):
                    with z.open(info) as src, (target / Path(info.filename).name).open("wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    count += 1
    (target / ".ok").write_text(time.strftime("%Y-%m-%d"), encoding="utf-8")
    log.info("graphics card libraries ready: %d files in %s", count, target)
    shutil.rmtree(archives, ignore_errors=True)
