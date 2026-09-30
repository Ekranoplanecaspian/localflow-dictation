import pytest


class NoGpu:
    """A GPU monitor that has nothing to report, so no test depends on how warm the real
    graphics card happens to be while the suite runs."""

    name = None

    def sample(self):
        return None


@pytest.fixture(autouse=True)
def _no_real_gpu(monkeypatch):
    import localflow.gpu as gpu

    monkeypatch.setattr(gpu, "_monitor", NoGpu())


@pytest.fixture(autouse=True)
def _roomy_machine(monkeypatch):
    """A 32 GB machine with 16 GB free and an NVIDIA card, whatever this one has right now: model choice and the
    clean-up memory check read free RAM, and a busy machine must not change what a test sees.
    Tests about memory set their own."""
    import localflow.hwinfo as hwinfo

    monkeypatch.setattr(hwinfo, "ram_gb", lambda: 32.0)
    monkeypatch.setattr(hwinfo, "ram_free_gb", lambda: 16.0)
    # and an NVIDIA card, only: which graphics this machine has must not change placement either
    monkeypatch.setattr(hwinfo, "graphics_adapters", lambda: (
        hwinfo.Gpu("NVIDIA GeForce RTX 4060 Laptop GPU", "nvidia", 8188, 16000, integrated=False),))
    # and the development machine's 12 cores: model choice estimates speed from them, and a
    # 2-core CI runner otherwise finds nothing quick enough on its processor.
    # test_hwinfo.py puts the real one back where it is the subject.
    if not hasattr(hwinfo, "_real_physical_cores"):
        hwinfo._real_physical_cores = hwinfo.physical_cores
    monkeypatch.setattr(hwinfo, "physical_cores", lambda: 12)


@pytest.fixture(autouse=True)
def _cuda_ready(monkeypatch):
    """Speech's CUDA libraries are here, as on the development machine, whether or not this one
    has them (a CI runner has neither them nor onnxruntime-gpu): placement must not depend on
    it. Tests about the libraries' first download set their own."""
    from localflow import cudalibs

    monkeypatch.setattr(cudalibs, "available", lambda: True)


@pytest.fixture(autouse=True, scope="session")
def _timings_in_memory():
    """Model-choice timings stay in memory: a test run must not rewrite the user's perf.json.
    For the whole session, not per test: test_connection's engine is built once per module,
    before any per-test fixture runs, and its instant fake decodes went on filling perf.json
    with 0 ms timings for Parakeet v3 on the processor (found 2026-09-29)."""
    import localflow.modelchoice as modelchoice

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(modelchoice, "PERF_PATH", None)
        yield


@pytest.fixture(autouse=True)
def _no_tidying(monkeypatch):
    r"""An engine started by a test tidies the download folder, which is the user's own
    %LOCALAPPDATA%\LocalFlow: a test run once deleted archives there. Tests that want it
    call localflow.llm.server.tidy_downloads directly, on a folder of their own."""
    from localflow.service.engine import Engine

    monkeypatch.setattr(Engine, "_tidy_downloads", staticmethod(lambda: None))


@pytest.fixture(autouse=True, scope="session")
def _speech_in_process():
    """Tests build speech in the test process (most stub it out): a speech worker process per
    engine would make the suite slow and load real models. test_remote.py covers the worker.
    Session-wide, so module-scoped engine fixtures (set up before any per-test one) see it too."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOCALFLOW_SPEECH_IN_PROCESS", "1")
        # and downloads in the calling thread, where the functions tests replace are the ones
        # that run (fetch.py; test_fetch.py covers the child process)
        mp.setenv("LOCALFLOW_FETCH_IN_PROCESS", "1")
        yield


@pytest.fixture(autouse=True, scope="session")
def _no_first_run_download():
    """An engine started by a test never downloads a model because this machine lacks it (a CI
    runner lacks them all). test_net.py puts the real one back where it is the subject.
    Session-wide: test_connection's engine is built once per module, before any per-test
    fixture, and on a CI runner it sat downloading the 2.6 GB speech model until it timed out."""
    import localflow.service.engine as eng

    if not hasattr(eng.Engine, "_real_fetch_speech"):
        eng.Engine._real_fetch_speech = eng.Engine._fetch_speech
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(eng.Engine, "_fetch_speech", lambda self: None)
        # nor llama.cpp for clean-up: on a CI runner the tests that fake the clean-up server
        # downloaded the real runtime from GitHub, and timed out when that was slow
        mp.setattr(eng.Engine, "_fetch_cleanup_runtime", lambda self, where: None)
        yield
