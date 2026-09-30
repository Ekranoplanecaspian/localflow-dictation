"""Switching the bundled clean-up model on a running engine, with a fake llama-server: nothing
here downloads or starts a real model."""

import threading
import time

import pytest

from localflow.config import Config
from localflow.llm import manifest as M

from tests.test_speech_models import FakeSTT, wait


class FakeServer:
    started: list[str] = []
    stopped: list[str] = []
    fail: set[str] = set()
    while_starting = None  # called during start(), as a user changing settings would be

    def __init__(self, model_key, device="auto", **_):
        self.model_key = model_key
        self.device = device
        self.kind = "cpu" if device == "cpu" else device
        self.spawned = threading.Event()

    def start(self, progress=None, timeout=120.0):
        if self.device in FakeServer.fail or self.model_key in FakeServer.fail:
            raise RuntimeError("CUDA error: out of memory")
        if FakeServer.while_starting:
            FakeServer.while_starting()
        FakeServer.started.append(self.model_key)

    def alive(self):
        return True

    def stop(self):
        FakeServer.stopped.append(self.model_key)


@pytest.fixture
def engine(monkeypatch):
    import localflow.llm.providers as providers
    import localflow.llm.server as server
    import localflow.service.engine as eng

    FakeServer.started, FakeServer.stopped, FakeServer.fail = [], [], set()
    FakeServer.while_starting = None
    monkeypatch.setattr(eng, "build_transcriber", FakeSTT)
    monkeypatch.setattr(Config, "save", lambda self, path=None: None)
    monkeypatch.setattr(server, "LlamaServer", FakeServer)
    monkeypatch.setattr(server, "ensure_model", lambda key, progress=None: progress and progress(key, 1, 1))
    monkeypatch.setattr(providers, "build_provider", lambda pp, server_factory=None: ("provider", server_factory()))
    cfg = Config()
    cfg.postprocess.llm_model = "qwen3-4b"
    e = eng.Engine(cfg)
    e.load()
    assert wait(lambda: e.state == "ready" and e.llm_state == "ready"), (e.error, e.llm_error)
    yield e
    e.shutdown()


def test_switch_starts_the_new_server_after_stopping_the_old_and_saves(engine):
    statuses = []
    engine.add_status_listener(statuses.append)
    engine.apply_settings({"llm": {"model": "qwen3-1.7b"}})
    assert wait(lambda: engine.cleanup_switch is None and engine.cfg.postprocess.llm_model == "qwen3-1.7b")

    assert FakeServer.started == ["qwen3-4b", "qwen3-1.7b"]
    assert FakeServer.stopped == ["qwen3-4b"]
    assert engine.llm_state == "ready" and engine.cleanup.provider[1].model_key == "qwen3-1.7b"
    seen = [s["llm"]["switch"]["state"] for s in statuses if s["llm"]["switch"]]
    assert "loading" in seen
    # the switch clears just before its final broadcast goes out
    assert wait(lambda: statuses[-1]["llm"]["model"] == "qwen3-1.7b" and statuses[-1]["llm"]["switch"] is None)


def test_failed_start_brings_the_previous_model_back(engine):
    FakeServer.fail = {"qwen3-1.7b"}
    engine.switch_cleanup("qwen3-1.7b")
    assert wait(lambda: (engine.cleanup_switch or {}).get("state") == "error")
    assert wait(lambda: engine.llm_state == "ready")

    assert "out of memory" in engine.cleanup_switch["error"]
    assert engine.cfg.postprocess.llm_model == "qwen3-4b"  # never changed
    assert FakeServer.started == ["qwen3-4b", "qwen3-4b"]  # started again after the failure


def test_picking_a_bundled_model_turns_auto_edits_back_on(engine):
    engine.cfg.postprocess.llm_provider = "ollama"
    engine.cfg.postprocess.llm_cleanup = False
    engine.switch_cleanup("qwen3-4b")
    assert wait(lambda: engine.cleanup_switch is None and engine.cfg.postprocess.llm_provider == "bundled")
    assert engine.cfg.postprocess.llm_cleanup is True


def _add_a_word_in_the_hub(engine):
    """What the Hub's Dictionary page does, while a model server is starting."""
    def add():
        FakeServer.while_starting = None
        engine.apply_settings({"postprocess": {"dictionary_terms": ["Priya"]}})
    FakeServer.while_starting = add


def test_a_hub_change_made_while_a_model_moves_is_kept(engine):
    """Starting the server takes seconds, and the move used to put the settings back the way
    they were when it began: a word added to the dictionary meanwhile was lost."""
    _add_a_word_in_the_hub(engine)
    assert engine.move_cleanup("cpu")
    assert engine.cfg.postprocess.dictionary_terms == ["Priya"]
    assert engine.cleanup.cfg.dictionary_terms == ["Priya"], "and the running clean-up uses it"


def test_a_hub_change_made_while_a_model_is_switched_is_kept(engine):
    _add_a_word_in_the_hub(engine)
    engine.switch_cleanup("qwen3-1.7b")
    assert wait(lambda: engine.cleanup_switch is None and engine.cfg.postprocess.llm_model == "qwen3-1.7b")
    assert engine.cfg.postprocess.dictionary_terms == ["Priya"]
    assert engine.cleanup.cfg.dictionary_terms == ["Priya"]


def test_choosing_the_current_model_does_nothing(engine):
    engine.switch_cleanup("qwen3-4b")
    assert engine.cleanup_switch is None and FakeServer.started == ["qwen3-4b"]


def test_unknown_model_is_refused(engine):
    with pytest.raises(ValueError):
        engine.switch_cleanup("gpt-7")


def test_only_offered_models_are_listed_and_the_current_one_is_marked(monkeypatch):
    rows = M.describe("qwen3-4b")
    assert all(M.CLEANUP_MODELS[r["key"]].offered for r in rows)
    assert [r["key"] for r in rows if r["current"]] in ([], ["qwen3-4b"])
    assert not any(r["current"] for r in M.describe(None))


def test_a_new_api_key_or_address_reaches_a_cloud_provider_at_once(engine, monkeypatch):
    """A corrected key went on failing with the old one until LocalFlow was restarted: only a
    change of provider, model or the on/off switch rebuilt the connection."""
    loads = []
    monkeypatch.setattr(engine, "_load_llm", lambda device=None: loads.append(engine.cfg.postprocess.llm_api_key))
    engine.cfg.postprocess.llm_provider = "openai"
    engine.apply_settings({"postprocess": {"llm_api_key": "sk-new"}})
    assert wait(lambda: loads == ["sk-new"])
    engine.apply_settings({"postprocess": {"llm_url": "https://api.example.com"}})
    assert wait(lambda: len(loads) == 2)

    # The bundled model has no address or key of its own: nothing to reload for those.
    engine.cfg.postprocess.llm_provider = "bundled"
    engine.apply_settings({"postprocess": {"llm_api_key": "irrelevant"}})
    time.sleep(0.2)
    assert len(loads) == 2


# --- too little free memory (B5) ------------------------------------------------------------------
def test_clean_up_waits_for_free_memory_and_starts_by_itself(engine, monkeypatch):
    import localflow.service.engine as eng
    from localflow import hwinfo, problems

    engine.sleep_cleanup()
    started = list(FakeServer.started)
    free = {"gb": 1.0}
    monkeypatch.setattr(hwinfo, "ram_free_gb", lambda: free["gb"])
    monkeypatch.setattr(eng, "RETRY_FIRST_S", 0.2)

    engine.wake_cleanup("cpu")
    assert wait(lambda: engine.llm_state == "error"), engine.llm_state
    assert engine.llm_error_code == problems.CLEANUP_LOW_MEMORY
    assert "Qwen3 4B needs about 3.8 GB of free memory, and 1.0 GB is free now." == engine.llm_error
    assert FakeServer.started == started  # nothing was started to find out

    free["gb"] = 12.0  # programs closed: it starts without anyone pressing a button
    assert wait(lambda: engine.llm_state == "ready", timeout=5), engine.llm_error
    assert FakeServer.started == started + ["qwen3-4b"]


def test_on_the_graphics_card_clean_up_needs_little_ram(engine, monkeypatch):
    from localflow import hwinfo

    engine.sleep_cleanup()
    monkeypatch.setattr(hwinfo, "ram_free_gb", lambda: 2.5)  # too little for it on the processor
    monkeypatch.setattr(engine, "_llama_device", lambda device=None: "auto")
    engine.wake_cleanup("cuda")
    assert wait(lambda: engine.llm_state == "ready"), engine.llm_error


def test_graphics_that_will_not_run_clean_up_leave_it_on_the_processor(engine):
    """B3: a driver without Vulkan, or one that crashes it, costs one try, not AI clean-up."""
    engine.sleep_cleanup()
    FakeServer.fail.add("vulkan")
    engine._load_llm("vulkan")
    assert engine.llm_state == "ready" and engine.llm_server.device == "cpu"
    assert engine.compute.no_vulkan

