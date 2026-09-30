"""A take spoken into a password field: typed exactly as heard - no clean-up rules, no
language model (which may be a cloud provider), no prompt pre-fill."""

import time

import numpy as np

from localflow.config import Config

from test_service import FakeSTT, blocks, fixture_audio


def run_take(monkeypatch, context: dict) -> tuple[dict, object]:
    import localflow.service.engine as eng
    from localflow.cleanup.pipeline import CleanupPipeline
    from localflow.config import PostProcessConfig

    from tests.test_cleanup import FakeProvider

    monkeypatch.setattr(eng, "build_transcriber", lambda cfg: FakeSTT())
    cfg = Config()
    cfg.postprocess.llm_cleanup = False
    engine = eng.Engine(cfg)
    engine.load()
    deadline = time.monotonic() + 30
    while engine.state == "loading" and time.monotonic() < deadline:
        time.sleep(0.05)
    provider = FakeProvider("Cleaned up.")
    engine.cleanup = CleanupPipeline(PostProcessConfig(llm_cleanup=True, llm_min_words=1, llm_prefill=True), provider)

    events: list[dict] = []
    session = engine.start_session("p1", context, events.append)
    speech = fixture_audio()
    for b in blocks(np.concatenate([speech, speech])):
        session.feed((np.clip(b * 32767, -32768, 32767)).astype("<i2").tobytes())
    session.end()
    deadline = time.monotonic() + 30
    while not session.done and time.monotonic() < deadline:
        time.sleep(0.05)
    engine.shutdown()
    return next(e for e in events if e["type"] == "final"), provider


def test_a_password_is_typed_exactly_as_heard(monkeypatch):
    # The context arrives from the shell with every value as text (protocol.checked).
    final, provider = run_take(monkeypatch, {"app": "chrome.exe", "password": "True"})
    assert final["text"] == final["raw"], "no clean-up of any kind"
    assert final["timings"]["private"] is True and final["timings"]["used_llm"] is False
    assert not provider.calls and not provider.prefills, "the language model never saw it"


def test_an_ordinary_take_is_still_cleaned_up(monkeypatch):
    final, provider = run_take(monkeypatch, {"app": "chrome.exe", "password": "False"})
    assert final["text"] == "Cleaned up." and "private" not in final["timings"]
