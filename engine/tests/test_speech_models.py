"""The speech model catalogue, and switching models on a running engine. Nothing here
downloads or loads a real model."""

import time

import numpy as np
import pytest

from localflow.config import Config, STTConfig
from localflow.stt import catalogue as C
from localflow.stt import router as R

from tests.test_router import FakeBackend


def wait(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.02)
    return predicate()


def test_default_settings_are_parakeet_v3():
    assert C.current(STTConfig()).key == "parakeet-v3"


@pytest.mark.parametrize("key", [m.key for m in C.MODELS])
def test_every_entry_round_trips_through_the_settings(key):
    cfg = STTConfig()
    C.get(key).apply(cfg)
    assert C.current(cfg).key == key


def test_compact_is_told_apart_from_full_precision_by_precision_alone():
    cfg = STTConfig(precision="int8")
    assert C.current(cfg).key == "parakeet-v3-compact"
    assert C.current(STTConfig(precision="fp32")).key == "parakeet-v3"


def test_older_parakeet_only_configs_are_still_recognised():
    assert C.current(STTConfig(backend="parakeet")).key == "parakeet-v3"


def test_hand_built_configuration_is_not_claimed_by_any_entry():
    assert C.current(STTConfig(model="some/other-model")) is None


def test_unknown_key_is_refused():
    with pytest.raises(ValueError):
        C.get("parakeet-v9")


def test_whisper_files_follow_the_device():
    turbo = C.get("whisper-turbo")
    assert "onnx/encoder_model_fp16.onnx" in turbo.files("cuda")
    assert "onnx/encoder_model_int8.onnx" in turbo.files("cpu")
    assert turbo.size_gb("cuda") > turbo.size_gb("cpu")


def test_describe_marks_the_current_model_and_what_is_on_disk(monkeypatch):
    monkeypatch.setattr(C, "is_installed", lambda m, device: m.key in ("parakeet-v3", "whisper-turbo"))
    cfg = STTConfig()
    C.get("whisper-turbo").apply(cfg)
    rows = {r["key"]: r for r in C.describe(cfg, "cuda")}
    assert rows["whisper-turbo"]["current"] and not rows["parakeet-v3"]["current"]
    assert rows["parakeet-v3"]["installed"] and not rows["parakeet-v2"]["installed"]
    assert rows["parakeet-v3"]["languages"] == 25 and rows["whisper-turbo"]["languages"] == 99
    assert sum(r["recommended"] for r in rows.values()) == 1


def test_speed_is_rated_for_the_device_it_will_run_on():
    cfg = STTConfig()
    gpu = {r["key"]: r["speed"] for r in C.describe(cfg, "cuda")}
    cpu = {r["key"]: r["speed"] for r in C.describe(cfg, "cpu")}
    assert gpu["parakeet-v3"] > gpu["parakeet-v3-compact"]  # int8 falls back to the CPU on CUDA
    assert cpu["parakeet-v3-compact"] > cpu["parakeet-v3"]
    assert all(1 <= v <= 5 for v in (*gpu.values(), *cpu.values()))


def test_english_only_parakeet_sends_other_languages_to_whisper(monkeypatch):
    monkeypatch.setattr(R, "ParakeetTranscriber", lambda cfg: FakeBackend("parakeet"))
    cfg = STTConfig()
    C.get("parakeet-v2").apply(cfg)
    r = R.RoutedTranscriber(cfg)
    r._whisper = FakeBackend("whisper")
    audio = np.zeros(16000, np.float32)
    assert r.transcribe(audio, language="en") == "parakeet"
    assert r.transcribe(audio, language="de") == "whisper"  # v3 would have kept German


def test_progress_reporter_forwards_only_the_bytes_written_bar():
    seen = []
    from localflow.hfprogress import reporter

    Reporter = reporter(lambda done, total: seen.append((done, total)))
    files = Reporter(total=3, desc="Fetching 3 files")
    written = Reporter(total=0, desc="Reconstructing (incomplete total...)", unit="B")
    written.total = 1000  # only the config files are counted so far
    written.update(1000)
    MB = 1 << 20
    written.total = 1000 + 100 * MB  # now the weights are
    files.update(1)
    written.update(50 * MB)
    written.update(50 * MB)
    assert seen == [(1000 + 50 * MB, 1000 + 100 * MB), (1000 + 100 * MB, 1000 + 100 * MB)]


def test_bytes_received_move_the_bar_but_only_bytes_written_finish_it():
    """Xet updates its bytes-received bar ten times a second and its bytes-written bar only per
    large chunk (7 times in 51 s, measured 2026-10-01): received bytes lead, written ones end."""
    from localflow.hfprogress import reporter

    seen = []
    Reporter = reporter(lambda done, total: seen.append(done))
    MB = 1 << 20
    received = Reporter(total=100 * MB, desc="model.gguf: downloading bytes", unit="B")
    written = Reporter(total=100 * MB, desc="model.gguf", unit="B")
    received.update(80 * MB)
    written.update(40 * MB)  # behind what was received: the bar does not go back
    received.update(20 * MB)  # all received, not all on disk
    assert seen == [80 * MB, 100 * MB - 1]
    written.update(60 * MB)
    assert seen[-1] == 100 * MB


# switching on a running engine ------------------------------------------------------------------

class FakeSTT(FakeBackend):
    sample_rate = 16000

    def __init__(self, cfg):
        super().__init__(cfg.model)
        self.cfg = cfg


def ready_engine(monkeypatch, tmp_path, build=FakeSTT):
    import localflow.service.engine as eng

    monkeypatch.setattr(eng, "build_transcriber", build)
    monkeypatch.setattr(Config, "save", lambda self, path=None: None)
    cfg = Config()
    cfg.postprocess.llm_cleanup = False
    engine = eng.Engine(cfg)
    engine.load()
    assert wait(lambda: engine.state == "ready"), engine.error
    return engine


def test_switch_downloads_then_swaps_and_keeps_the_new_choice(monkeypatch, tmp_path):
    engine = ready_engine(monkeypatch, tmp_path)
    fetched = []

    def fake_download(model, device, progress=None):
        fetched.append(model.key)
        progress(50, 100)
        progress(100, 100)

    monkeypatch.setattr(C, "is_installed", lambda m, device: False)
    monkeypatch.setattr(C, "download", fake_download)
    statuses = []
    engine.add_status_listener(statuses.append)

    engine.apply_settings({"stt": {"model": "parakeet-v2"}})
    assert wait(lambda: engine.speech_switch is None and engine.stt.name == "nemo-parakeet-tdt-0.6b-v2")

    assert fetched == ["parakeet-v2"]
    assert C.current(engine.cfg.stt).key == "parakeet-v2"
    states = [s["stt"]["switch"]["state"] for s in statuses if s["stt"]["switch"]]
    assert states[0] == "downloading" and "loading" in states
    # the switch clears just before its final broadcast goes out
    assert wait(lambda: statuses[-1]["stt"]["key"] == "parakeet-v2" and statuses[-1]["stt"]["switch"] is None)
    assert statuses[-1]["stt"]["state"] == "ready"
    engine.shutdown()


def test_failed_load_puts_the_previous_model_back(monkeypatch, tmp_path):
    def build(cfg):
        if "whisper" in cfg.model:
            raise RuntimeError("CUDA out of memory")
        return FakeSTT(cfg)

    engine = ready_engine(monkeypatch, tmp_path, build)
    monkeypatch.setattr(C, "is_installed", lambda m, device: True)
    engine.switch_speech("whisper-turbo")
    assert wait(lambda: (engine.speech_switch or {}).get("state") == "error")
    engine.shutdown()

    assert "out of memory" in engine.speech_switch["error"]
    assert engine.state == "ready" and engine.stt.name == "nemo-parakeet-tdt-0.6b-v3"
    assert C.current(engine.cfg.stt).key == "parakeet-v3"  # the setting never changed


def test_choosing_the_current_model_does_nothing(monkeypatch, tmp_path):
    engine = ready_engine(monkeypatch, tmp_path)
    before = engine.stt
    engine.switch_speech("parakeet-v3")
    engine.shutdown()
    assert engine.speech_switch is None and engine.stt is before


def test_each_model_is_rated_on_the_device_it_would_run_on():
    # the placement puts Compact on the processor even when the GPU is free
    where = {"parakeet-v3-compact": "cpu"}
    rows = {r["key"]: r for r in C.describe(STTConfig(), lambda m: where.get(m.key, "cuda"))}
    assert rows["parakeet-v3-compact"]["speed"] == 4 and rows["parakeet-v3-compact"]["device"] == "cpu"
    assert rows["parakeet-v3"]["speed"] == 5 and rows["parakeet-v3"]["device"] == "cuda"


def test_choosing_a_model_by_hand_turns_automatic_off_for_that_kind(monkeypatch, tmp_path):
    engine = ready_engine(monkeypatch, tmp_path)
    monkeypatch.setattr(C, "is_installed", lambda m, device: True)
    assert engine.cfg.compute.auto_speech is True
    engine.switch_speech("parakeet-v2")
    assert wait(lambda: engine.speech_switch is None)
    engine.shutdown()
    assert engine.cfg.compute.auto_speech is False and engine.cfg.compute.auto_cleanup is True


def test_real_dictations_are_timed_for_the_model_choice(monkeypatch, tmp_path):
    class TakesTime(FakeSTT):  # an instant decode is dropped as impossible (modelchoice.IMPOSSIBLY_QUICK)
        def transcribe(self, audio, language=None):
            time.sleep(0.02)
            return super().transcribe(audio, language)

    engine = ready_engine(monkeypatch, tmp_path, build=TakesTime)
    audio = np.zeros(16000 * 3, np.float32)
    for _ in range(5):
        engine.stt.transcribe(audio)
    engine.stt.transcribe(np.zeros(8000, np.float32))  # too short to say anything about speed
    engine.stt.transcribe(audio, language="hi")  # may have gone to Whisper: not this model's time
    engine.shutdown()
    key = "speech:parakeet-v3:cpu"
    assert len(engine.perf.samples[key]) == 5
