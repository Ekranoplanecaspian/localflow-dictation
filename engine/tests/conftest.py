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
        yield


@pytest.fixture(autouse=True)
def _no_first_run_download(monkeypatch):
    """An engine started by a test never downloads a model because this machine lacks it (a CI
    runner lacks them all). test_net.py puts the real one back where it is the subject."""
    import localflow.service.engine as eng

    if not hasattr(eng.Engine, "_real_fetch_speech"):
        eng.Engine._real_fetch_speech = eng.Engine._fetch_speech
    monkeypatch.setattr(eng.Engine, "_fetch_speech", lambda self: None)
