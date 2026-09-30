"""The model library as the engine offers it (M1): how each model suits this PC, downloading one
without switching, stopping a download, removing a model - and a first-run clean-up download
that is visible, and that the user can stop. Fake models and downloads; nothing goes online."""

import threading
import time

import pytest

from localflow import hwinfo, modelchoice
from localflow.config import Config

DEV = modelchoice.Hardware(cpu_cores=12, ram_gb=32.0, vram_mb=8188)


# how a model suits this PC --------------------------------------------------------------------------
def test_quick_models_suit_the_pc():
    perf = modelchoice.PerfLog(None)
    fit = modelchoice.fit("speech", "parakeet-v3", "cuda", DEV, perf, ram_gb=1.4, vram_mb=2900)
    assert fit.rating == "good" and fit.why.startswith("quick on the graphics card (0.04 s per second of speech")


def test_a_slow_one_works_but_says_so():
    perf = modelchoice.PerfLog(None)
    fit = modelchoice.fit("speech", "whisper-turbo", "cpu", DEV, perf, ram_gb=1.4)
    assert fit.rating == "slow" and fit.why == "1.44 s per second of speech on the processor, estimated"


def test_one_too_big_for_the_memory_says_how_much_it_needs():
    small = modelchoice.Hardware(cpu_cores=4, ram_gb=6.0, vram_mb=None)
    need = hwinfo.cleanup_ram_gb(3.4, "cpu")
    fit = modelchoice.fit("cleanup", "gemma-4-e2b", "cpu", small, modelchoice.PerfLog(None), ram_gb=need)
    assert fit == modelchoice.Fit("too-big", "needs about 3.9 GB of memory, and this PC has 6 GB")
    card = modelchoice.fit("speech", "parakeet-v3", "cuda", modelchoice.Hardware(12, 32.0, 2048),
                           modelchoice.PerfLog(None), ram_gb=1.4, vram_mb=2900)
    assert card.rating == "too-big" and "on the graphics card, which has 2 GB" in card.why


def test_measurements_outrank_the_estimate():
    perf = modelchoice.PerfLog(None)
    for _ in range(modelchoice.MIN_SAMPLES):
        perf.record("cleanup", "qwen3-4b", "cpu", 2400)
    fit = modelchoice.fit("cleanup", "qwen3-4b", "cpu", DEV, perf, ram_gb=3.0)
    assert fit.rating == "slow" and fit.why == "2.4 s per clean-up on the processor, measured"


# the engine ------------------------------------------------------------------------------------------
@pytest.fixture
def engine(monkeypatch, tmp_path):
    import localflow.service.engine as eng
    from localflow.llm import manifest as M
    from localflow.stt import catalogue
    from test_service import FakeSTT

    monkeypatch.setattr(eng, "build_transcriber", lambda cfg: FakeSTT())
    monkeypatch.setattr(Config, "save", lambda self, *a, **k: None)
    monkeypatch.setattr(M, "gguf_dir", lambda: tmp_path)
    # Parakeet v3 is on this PC; nothing else is
    monkeypatch.setattr(catalogue, "is_installed", lambda m, d: m.key == "parakeet-v3")
    cfg = Config()
    cfg.postprocess.llm_cleanup = False
    e = eng.Engine(cfg)
    e.load()
    deadline = time.monotonic() + 20
    while e.state != "ready" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert e.state == "ready", e.error
    yield e
    e.shutdown()


@pytest.fixture
def slow_download(monkeypatch):
    """A clean-up model download that reports progress until it is let through or stopped."""
    import localflow.llm.server as server

    go = threading.Event()

    def ensure_model(key, progress=None):
        done = 0
        while not go.is_set():
            done = min(done + (100 << 20), 2500 << 20)
            progress(key, done, 2500 << 20)  # a stop is seen here
            time.sleep(0.02)

    monkeypatch.setattr(server, "ensure_model", ensure_model)
    return go


def test_every_model_says_how_it_suits_this_pc_and_what_it_takes(engine):
    rows = {r["key"]: r for r in engine.status()["stt"]["choices"]}
    assert set(rows) == {"parakeet-v3", "parakeet-v2", "parakeet-v3-compact", "whisper-turbo"}
    assert rows["parakeet-v3"]["fit"]["rating"] == "good"
    assert rows["parakeet-v3"]["removable"] is False, "the model in use"
    assert rows["parakeet-v2"]["disk_gb"] == 0 and rows["parakeet-v2"]["removable"] is False
    cleanup = {r["key"]: r for r in engine.status()["llm"]["choices"]}
    assert set(cleanup) == {"qwen3-4b", "phi-4-mini", "gemma-4-e2b"}
    assert all(r["fit"]["rating"] in ("good", "slow", "too-big") and r["fit"]["why"] for r in cleanup.values())


def test_a_model_downloads_without_being_switched_to_and_can_be_stopped(engine, slow_download):
    engine.download_model("cleanup", "phi-4-mini")
    deadline = time.monotonic() + 5
    while not (engine.status()["downloads"] or [{}])[0].get("done") and time.monotonic() < deadline:
        time.sleep(0.02)
    job = engine.status()["downloads"][0]
    assert (job["kind"], job["key"], job["label"], job["reason"]) == ("cleanup", "phi-4-mini", "Phi-4 mini", "library")
    assert job["state"] == "downloading" and job["cancellable"] and job["done"] > 0
    row = next(r for r in engine.status()["llm"]["choices"] if r["key"] == "phi-4-mini")
    assert row["download"] == job["id"]
    assert engine.cfg.postprocess.llm_model != "phi-4-mini", "downloading is not choosing"
    assert engine.cancel_download(job["id"])
    deadline = time.monotonic() + 5
    while engine.status()["downloads"][0]["state"] != "cancelled" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert engine.status()["downloads"][0]["state"] == "cancelled"


def test_the_model_in_use_cannot_be_removed(engine):
    with pytest.raises(ValueError, match="the speech model in use"):
        engine.remove_model("speech", "parakeet-v3")


def test_a_removed_clean_up_model_is_gone(engine):
    from localflow.llm import manifest as M

    M.gguf_path("gemma-4-e2b").write_bytes(b"x" * 5000)
    row = next(r for r in engine.status()["llm"]["choices"] if r["key"] == "gemma-4-e2b")
    assert row["installed"] and row["removable"] and row["disk_gb"] == 0.0
    assert engine.remove_model("cleanup", "gemma-4-e2b") == 5000
    assert not M.gguf_path("gemma-4-e2b").exists()


# what LocalFlow recommends (M4) ---------------------------------------------------------------------
def recommended(engine):
    return {(r["kind"], r["key"]): r for r in engine.status()["recommended"]}


def test_nothing_is_recommended_on_a_pc_already_using_the_best(engine):
    engine.cfg.postprocess.llm_cleanup = True
    engine.cfg.postprocess.llm_provider, engine.cfg.postprocess.llm_model = "bundled", "qwen3-4b"
    assert recommended(engine) == {}


def test_auto_edits_that_are_off_are_recommended_where_they_suit_the_pc(engine):
    engine.cfg.postprocess.llm_provider, engine.cfg.postprocess.llm_model = "bundled", "qwen3-4b"
    r = recommended(engine)[("cleanup", "qwen3-4b")]
    assert r["action"] == "enable"
    assert r["why"] == ("Removes fillers, applies your self-corrections and formats lists and numbers. "
                        "Uses Qwen3 4B, a 2.5 GB download.")


def test_auto_edits_left_off_on_a_small_pc_are_not_pressed_on_it(engine):
    """Below 9 GB they start off on purpose (B5)."""
    from dataclasses import replace

    engine.cfg.postprocess.llm_provider, engine.cfg.postprocess.llm_model = "bundled", "qwen3-4b"
    engine.compute.hardware = replace(engine.compute.hardware, ram_gb=7.6)
    assert ("cleanup", "qwen3-4b") not in recommended(engine)


def test_a_more_accurate_speech_model_is_recommended_and_automatic_picks_it_up(engine, monkeypatch):
    from localflow.stt import catalogue

    catalogue.get("parakeet-v3-compact").apply(engine.cfg.stt)
    monkeypatch.setattr(catalogue, "is_installed", lambda m, d: m.key == "parakeet-v3-compact")
    engine._ratings.clear()
    r = recommended(engine)[("speech", "parakeet-v3")]
    assert r["why"] == "More accurate than Parakeet v3 Compact." and r["size_gb"] == 2.6
    assert r["action"] == "download", "Automatic chooses among the models on this PC: fetching it is enough"
    engine.cfg.compute.auto_speech = False
    assert recommended(engine)[("speech", "parakeet-v3")]["action"] == "use"


def test_automatic_is_not_argued_with_about_a_model_it_already_has(engine, monkeypatch):
    """On this PC and not chosen: Automatic passed it over (free memory right now, say)."""
    from localflow.stt import catalogue

    catalogue.get("parakeet-v3-compact").apply(engine.cfg.stt)  # v3 is on this PC too (the fixture)
    assert ("speech", "parakeet-v3") not in recommended(engine)
    engine.cfg.compute.auto_speech = False
    assert recommended(engine)[("speech", "parakeet-v3")]["action"] == "use", "picked by hand: say so"


def test_whisper_only_for_a_language_parakeet_does_not_know(engine):
    assert ("speech", "whisper-turbo") not in recommended(engine)
    engine.cfg.stt.language = "hi"
    r = recommended(engine)[("speech", "whisper-turbo")]
    assert r["action"] == "download" and "\"hi\", which Parakeet does not know" in r["why"]


def test_a_quicker_clean_up_model_when_the_one_in_use_is_slow_here(engine, monkeypatch):
    engine.cfg.postprocess.llm_cleanup = True
    engine.cfg.postprocess.llm_provider, engine.cfg.postprocess.llm_model = "bundled", "qwen3-4b"
    slow = modelchoice.Fit("slow", "2.1 s per clean-up on the processor, measured")
    quick = modelchoice.Fit("good", "quick on the processor (0.9 s per clean-up, estimated)")
    real = type(engine)._rating
    monkeypatch.setattr(type(engine), "_rating", lambda self, kind, key, installed: (
        ("cpu", slow if key == "qwen3-4b" else quick, 0) if kind == "cleanup" else real(self, kind, key, installed)))
    r = recommended(engine)
    assert set(r) == {("cleanup", "phi-4-mini")}, "the most accurate of the quick ones"
    assert r[("cleanup", "phi-4-mini")]["why"] == (
        "Quick on this PC, where Qwen3 4B is slow (2.1 s per clean-up on the processor, measured).")


def test_the_first_clean_up_download_shows_and_stopping_it_turns_auto_edits_off(engine, slow_download, monkeypatch):
    monkeypatch.setattr(type(engine), "_fetch_cleanup_runtime", lambda self, where: None)
    monkeypatch.setattr(type(engine), "_room_for_cleanup", lambda self, device=None: None)
    engine.cfg.postprocess.llm_cleanup = True
    engine.cfg.postprocess.llm_provider, engine.cfg.postprocess.llm_model = "bundled", "qwen3-4b"
    t = threading.Thread(target=engine._load_llm)
    t.start()
    deadline = time.monotonic() + 5
    while not (engine.status()["llm"]["download"] or {}).get("progress") and time.monotonic() < deadline:
        time.sleep(0.02)
    llm = engine.status()["llm"]
    assert llm["state"] == "loading"
    assert llm["download"]["label"] == "Qwen3 4B" and 0 < llm["download"]["progress"] < 1
    assert engine.cancel_download(llm["download"]["id"])
    t.join(5)
    assert engine.llm_state == "off" and engine.cfg.postprocess.llm_cleanup is False
