"""Speech on the graphics card failing under a take (a driver reset, the speech worker gone):
the take is decoded again on the processor rather than lost."""

import time

import numpy as np

from localflow.config import Config
from test_service import SR, FakeSTT, blocks, fixture_audio


class BrokenGpu(FakeSTT):
    """A model on the graphics card whose device has just gone away."""

    device = "cuda"
    alive = True

    def transcribe(self, audio, language=None):
        raise RuntimeError("CUDA failure 999: unknown error ; GPU=0 ; hostname=x ; expr=cudaStreamSynchronize")


class BrokenCpu(FakeSTT):
    def transcribe(self, audio, language=None):
        raise RuntimeError("the model file is damaged")


def _engine(monkeypatch, gpu, cpu=FakeSTT):
    import localflow.service.engine as eng

    built: list[str] = []

    def build(cfg):
        built.append(cfg.device)
        return gpu() if cfg.device != "cpu" else cpu()

    monkeypatch.setattr(eng, "build_transcriber", build)
    cfg = Config()
    cfg.postprocess.llm_cleanup = False
    cfg.compute.mode = "off"  # no placement moves of its own during the test
    engine = eng.Engine(cfg)
    engine.compute.placement = engine.compute.placement.__class__(0, "test", "cuda", "cuda", False)
    monkeypatch.setattr(engine.compute, "decide", lambda: engine.compute.placement)
    engine.load()
    deadline = time.monotonic() + 30
    while engine.state == "loading" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert engine.state == "ready", engine.error
    return engine, built


def _dictate(engine) -> list[dict]:
    events: list[dict] = []
    session = engine.start_session("s1", {"app": "notepad.exe"}, events.append)
    for b in blocks(fixture_audio()):
        session.feed((np.clip(b * 32767, -32768, 32767)).astype("<i2").tobytes())
    session.end()
    deadline = time.monotonic() + 30
    while not session.done and time.monotonic() < deadline:
        time.sleep(0.05)
    return events


def test_a_take_the_graphics_card_failed_under_is_decoded_again_on_the_processor(monkeypatch):
    engine, built = _engine(monkeypatch, BrokenGpu)
    try:
        assert engine.stt.device == "cuda"
        events = _dictate(engine)
        finals = [e for e in events if e["type"] == "final"]
        assert finals and finals[0]["text"].startswith("<"), events
        assert not [e for e in events if e["type"] == "error"]
        assert engine.stt.device == "cpu" and engine.gpu_failures == 1
        assert built[-1] == "cpu"
        # Not "CUDA never works here": the placement may move speech back to the card.
        assert engine.compute.asked["speech"] == "cpu"
        assert "speech" not in engine.compute.no_gpu
    finally:
        engine.shutdown()


def test_a_failure_on_the_processor_is_not_retried(monkeypatch):
    engine, _ = _engine(monkeypatch, BrokenCpu, cpu=BrokenCpu)
    try:
        engine.stt._inner.device = "cpu"
        events = _dictate(engine)
        assert [e for e in events if e["type"] == "error"], events
        assert engine.gpu_failures == 0
    finally:
        engine.shutdown()


def test_a_dead_speech_worker_is_replaced_before_the_next_take(monkeypatch):
    engine, _ = _engine(monkeypatch, BrokenGpu)
    try:
        engine.stt._inner.alive = False
        engine.check_speech()
        deadline = time.monotonic() + 10
        while engine.stt.device != "cpu" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert engine.stt.device == "cpu" and engine.gpu_failures == 1
    finally:
        engine.shutdown()
