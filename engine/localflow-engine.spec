# -*- mode: python ; coding: utf-8 -*-
"""Freeze the engine into `localflow-engine.exe`.

The shell already looks for this file next to itself and falls back to the repository's
virtualenv when it is missing, so a packaged build is simply one where this exists.

Two things PyInstaller cannot work out by itself:

* **onnxruntime's native libraries.** The Python package is a thin wrapper around DLLs that are
  loaded by name at runtime, and the CUDA execution provider pulls in the nvidia-* wheels on top
  of that. None of it appears in an import graph.
* **onnx_asr's model registry.** Backends are looked up by name from a table, so nothing
  imports `onnx_asr.models.parakeet` directly and PyInstaller leaves it out - the engine then
  starts happily and fails the moment it is asked to load a model.

Everything else the engine needs at runtime is downloaded on first use and lives in
`%LOCALAPPDATA%\\LocalFlow`, not in here: the speech model, the clean-up model and llama-server
are gigabytes each and have their own resumable download path.

Build it with `python -m PyInstaller localflow-engine.spec --noconfirm` from `engine/`.
"""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

# --- native libraries -----------------------------------------------------------------------
binaries = collect_dynamic_libs("onnxruntime")
datas = collect_data_files("onnxruntime")

# The CUDA runtime, cuBLAS, cuDNN and friends ship as separate wheels whose DLLs onnxruntime
# loads by name. Missing one of these does not fail the build - it fails at the first attempt
# to use the GPU, which then silently falls back to the CPU and looks like a performance bug.
for package in (
    "nvidia",
    "onnxruntime_gpu",
):
    try:
        binaries += collect_dynamic_libs(package)
    except Exception:  # noqa: BLE001 - a CPU-only build simply has none of these
        pass

# sounddevice carries the PortAudio DLL the benchmark's recorder needs.
binaries += collect_dynamic_libs("sounddevice")
datas += collect_data_files("sounddevice")

# --- imports found only at runtime ----------------------------------------------------------
hiddenimports = [
    # Backends are chosen from a registry by name, so nothing imports them directly.
    *collect_submodules("onnx_asr"),
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    # Chosen from config at runtime.
    "localflow.stt.parakeet",
    "localflow.stt.whisper",
    "localflow.stt.whisper_onnx",
    "localflow.llm.providers",
    "localflow.cleanup.pipeline",
    "localflow.cleanup.command",
    "localflow.cleanup.joining",
    "localflow.cleanup.placeholders",
]
datas += collect_data_files("onnx_asr")
datas += collect_data_files("jellyfish")

# --- package metadata -----------------------------------------------------------------------
# `onnx_asr/__init__.py` asks `importlib.metadata` for its own version at import time, and a
# frozen build has the code but not the dist-info that answers that. The engine then starts,
# serves its handshake, connects - and throws PackageNotFoundError the moment it tries to load
# the speech model, which looks like a broken model rather than a missing file.
for package in ("onnx-asr", "onnxruntime-gpu", "onnxruntime", "numpy", "websockets"):
    try:
        datas += copy_metadata(package)
    except Exception:  # noqa: BLE001 - only one of the onnxruntime flavours is installed
        pass

# --- what to leave out ----------------------------------------------------------------------
# These are development-only and each drags in a great deal: matplotlib alone is tens of
# megabytes, and torch would be gigabytes.
excludes = [
    "torch",
    "matplotlib",
    "tkinter",
    "pytest",
    "IPython",
    "notebook",
    "PIL",
    "pandas",
    "scipy",
    "faster_whisper",
]

a = Analysis(
    [str(Path("src") / "localflow" / "__main__.py")],
    pathex=[str(Path("src"))],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="localflow-engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # A console window would flash up on every launch; the shell captures stdout for the
    # handshake either way, and the engine logs to a file.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# One directory, not one file: a single-file build unpacks a gigabyte of CUDA libraries to a
# temporary folder on every single launch, which would add seconds to every start-up.
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="localflow-engine",
)
