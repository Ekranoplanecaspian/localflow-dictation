"""An NVIDIA PC's first run without the CUDA libraries (B2): speech starts on the processor, the
libraries download in the background, and speech moves to the graphics card once they are
here. Nothing downloads for real."""

import urllib.error

import pytest

from localflow import cudalibs, hwinfo, modelchoice, problems
from tests.test_cleanup_models import FakeServer, engine  # noqa: F401 - the fixture
from tests.test_compute import rig  # noqa: F401 - the fixture
from tests.test_speech_models import wait


# placement ------------------------------------------------------------------------------------------

def test_without_the_libraries_speech_stays_on_the_processor_on_purpose(rig, monkeypatch):  # noqa: F811
    engine, _, _, ctl, tick = rig
    monkeypatch.setattr(cudalibs, "available", lambda: False)
    ctl.cuda_ready = False
    tick(2)
    assert engine.where == {"speech": "cpu", "cleanup": "cuda"}  # clean-up brings its own CUDA
    assert "speech" not in ctl.no_gpu, "not a failure: it was never asked onto the card"
    ctl.cuda_ready = True  # they arrived
    tick(1)
    assert engine.where["speech"] == "cuda"


# the download ---------------------------------------------------------------------------------------

@pytest.fixture
def first_run(engine, monkeypatch):  # noqa: F811
    """The engine as on an NVIDIA PC whose installer left the libraries out."""
    import localflow.service.engine as eng

    state = {"installed": False, "fail": [], "calls": 0}
    # The placement loop would ask the fake speech model onto the card, which always answers
    # from the processor: these tests are about the download, so it stops here.
    engine.compute.stop()

    def ensure(progress=None):
        state["calls"] += 1
        if state["fail"]:
            raise state["fail"].pop(0)
        if progress:
            progress(cudalibs.DOWNLOAD_BYTES // 2, cudalibs.DOWNLOAD_BYTES)
            progress(cudalibs.DOWNLOAD_BYTES, cudalibs.DOWNLOAD_BYTES)
        state["installed"] = True
        return cudalibs.lib_dir()

    monkeypatch.setattr(cudalibs, "available", lambda: state["installed"])
    monkeypatch.setattr(cudalibs, "ensure", ensure)
    monkeypatch.setattr(eng, "RETRY_FIRST_S", 0.1)
    report = hwinfo.Report(cpu=hwinfo.Cpu("x", 8, 16, None), ram_gb=32.0,
                           gpus=(hwinfo.Gpu("NVIDIA GeForce RTX 4060 Laptop GPU", "nvidia", 8188, 16000, False),))
    engine.compute.hardware = modelchoice.Hardware(cpu_cores=8, ram_gb=32.0, vram_mb=8188, report=report)
    engine.compute.cuda_ready = False
    engine.compute.no_gpu.add("speech")  # as if an earlier ask had come back on the processor
    return engine, state


def test_an_nvidia_pc_fetches_them_and_speech_may_then_use_the_card(first_run):
    engine, state = first_run
    engine.fetch_cuda_libs()
    assert wait(lambda: (engine.cuda_libs or {}).get("state") == "ready")
    assert engine.compute.cuda_ready and "speech" not in engine.compute.no_gpu
    engine.fetch_cuda_libs()  # nothing twice
    assert state["calls"] == 1


def test_no_connection_is_tried_again_by_itself(first_run):
    engine, state = first_run
    state["fail"] = [urllib.error.URLError("[Errno 11001] getaddrinfo failed")]
    engine.fetch_cuda_libs()
    assert wait(lambda: (engine.cuda_libs or {}).get("state") == "error")
    assert engine.compute.cuda_ready is False
    assert wait(lambda: (engine.cuda_libs or {}).get("state") == "ready", timeout=5)
    assert state["calls"] == 2


def test_a_damaged_download_is_said_and_not_retried_in_a_loop(first_run):
    engine, state = first_run
    state["fail"] = [RuntimeError("checksum mismatch for nvidia_cudnn_cu13.whl: expected e1de75bf1ad9..., got 000...")]
    engine.fetch_cuda_libs()
    assert wait(lambda: (engine.cuda_libs or {}).get("state") == "error")
    assert engine.cuda_libs["error"].startswith("checksum mismatch")
    assert "cuda" not in engine._retries
    assert engine.compute.status()["cuda_libs"]["state"] == "error"


def test_nothing_is_fetched_without_an_nvidia_card_or_with_processor_only(first_run):
    engine, state = first_run
    engine.cfg.compute.mode = "cpu"
    engine.fetch_cuda_libs()
    engine.cfg.compute.mode = "adaptive"
    report = hwinfo.Report(cpu=hwinfo.Cpu("x", 8, 16, None), ram_gb=32.0,
                           gpus=(hwinfo.Gpu("AMD Radeon(TM) 890M Graphics", "amd", 338, 16000, True),))
    engine.compute.hardware = modelchoice.Hardware(cpu_cores=8, ram_gb=32.0, vram_mb=None, report=report)
    engine.fetch_cuda_libs()
    assert state["calls"] == 0 and engine.cuda_libs is None


def test_the_failure_has_words():
    assert problems.GPU_LIBS_DOWNLOAD_FAILED in problems.CODES
