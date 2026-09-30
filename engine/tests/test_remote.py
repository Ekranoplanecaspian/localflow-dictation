"""The speech worker process (stt/remote.py): a real child process, real pipes, a real model
on the processor - the GPU is not needed to prove the channel."""

import wave
from pathlib import Path

import numpy as np
import pytest

from localflow.config import STTConfig
from localflow.stt import remote

FIXTURE = Path(__file__).parent / "fixtures" / "speech.wav"


@pytest.fixture(autouse=True)
def _workers_on(monkeypatch):
    monkeypatch.delenv(remote.IN_PROCESS_ENV, raising=False)  # conftest turns them off for the rest


def _wav() -> np.ndarray:
    with wave.open(str(FIXTURE)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


@pytest.fixture(scope="module")
def worker():
    stt = remote.RemoteTranscriber(STTConfig(device="cpu"))
    yield stt
    stt.close()


def test_a_worker_transcribes_like_the_model_in_process(worker):
    audio = _wav()
    text = worker.transcribe(audio)
    assert "fox" in text.lower()
    assert worker.device == "cpu"
    assert worker.transcribe(np.zeros(0, np.float32)) == ""
    worker.warm()  # a no-op on the processor, but it must answer
    assert worker.drop_idle_whisper() is False


def test_the_models_own_error_comes_back_as_itself():
    with pytest.raises(ValueError, match="unknown stt backend"):
        remote.RemoteTranscriber(STTConfig(backend="nonsense"))


def test_a_dead_worker_is_reported_not_waited_for():
    stt = remote.RemoteTranscriber(STTConfig(device="cpu"))
    stt._worker.proc.kill()
    stt._worker.proc.wait(5)
    with pytest.raises(remote.WorkerDied):
        stt.transcribe(np.zeros(16000, np.float32))


def test_closing_ends_the_process():
    stt = remote.RemoteTranscriber(STTConfig(device="cpu"))
    proc = stt._worker.proc
    stt.close()
    assert proc.poll() is not None


def test_an_early_load_is_used_when_the_settings_match(caplog):
    import logging

    caplog.set_level(logging.INFO)
    cfg = STTConfig(device="cpu")
    remote.prestart(cfg)
    stt = remote.RemoteTranscriber(cfg)
    try:
        assert "loading as asked" not in caplog.text
        assert "fox" in stt.transcribe(_wav()).lower()
    finally:
        stt.close()


def test_an_early_load_of_other_settings_is_replaced(caplog):
    import logging

    caplog.set_level(logging.INFO)
    remote.prestart(STTConfig(device="cpu", language="de"))
    stt = remote.RemoteTranscriber(STTConfig(device="cpu"))
    try:
        assert "speech settings changed since the early load" in caplog.text
        assert "fox" in stt.transcribe(_wav()).lower()
    finally:
        stt.close()


def test_a_discarded_spare_is_ended_without_waiting_for_its_load():
    import time

    remote.prestart(STTConfig(device="cpu"))
    proc = remote._spare.proc
    t0 = time.perf_counter()
    remote.discard_spare()
    assert time.perf_counter() - t0 < 1.0
    proc.wait(5)
