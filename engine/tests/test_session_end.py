"""A take ends once: a repeated or stale end never produces a second final."""

import time

import numpy as np

from localflow.config import Config
from tests.test_service import FakeSTT, blocks, fixture_audio


def _ready_engine(monkeypatch):
    import localflow.service.engine as eng

    monkeypatch.setattr(eng, "build_transcriber", lambda cfg: FakeSTT())
    cfg = Config()
    cfg.postprocess.llm_cleanup = False
    engine = eng.Engine(cfg)
    engine.load()
    deadline = time.monotonic() + 30
    while engine.state == "loading" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert engine.state == "ready", engine.error
    return engine


def _feed(session, audio):
    for b in blocks(audio):
        session.feed((np.clip(b * 32767, -32768, 32767)).astype("<i2").tobytes())


def test_ending_a_take_twice_gives_one_final(monkeypatch):
    """Stopping hands-free with a tap sent one end for the press and one for the release, and
    each produced a final - so the shell typed the same text twice."""
    engine = _ready_engine(monkeypatch)
    events: list[dict] = []
    session = engine.start_session("s1", {}, events.append)
    _feed(session, fixture_audio())
    session.end()
    session.end()
    deadline = time.monotonic() + 30
    while not session.done and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.3)  # room for a second final to arrive, if one were coming
    engine.shutdown()
    assert len([e for e in events if e["type"] == "final"]) == 1


def test_a_new_take_lets_the_one_still_finishing_deliver_its_text(monkeypatch):
    """Pressing the chord again straight after letting go used to cancel the take that was
    still being decoded, and the sentence just spoken was never typed."""
    import localflow.service.engine as eng
    from localflow.service.client import EngineClient
    from localflow.service.server import serve_in_thread

    class SlowSTT(FakeSTT):
        def transcribe(self, audio, language=None):
            time.sleep(0.3)  # the new take starts while this one is still decoding
            return super().transcribe(audio, language)

    monkeypatch.setattr(eng, "build_transcriber", lambda cfg: SlowSTT())
    cfg = Config()
    cfg.postprocess.llm_cleanup = False
    _thread, server, url = serve_in_thread(cfg)
    client = EngineClient("test")
    finals: list[dict] = []
    client.on_final = finals.append
    client.connect(int(url.rsplit(":", 1)[1]), server.token)
    deadline = time.monotonic() + 30
    while server.engine.state != "ready" and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        speech = fixture_audio()
        first = client.start_session({})
        client.send_audio(speech)
        client.end_session()
        second = client.start_session({})  # at once, while the first is decoding
        client.send_audio(speech)
        client.end_session()
        deadline = time.monotonic() + 30
        while len(finals) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert [f["id"] for f in finals] == [first, second]
    finally:
        client.close()
        server._stop.set()
        time.sleep(0.2)


def test_a_new_take_abandons_one_still_recording(monkeypatch):
    """Only a take that has ended is left to finish; one never ended is abandoned."""
    engine = _ready_engine(monkeypatch)
    events: list[dict] = []
    first = engine.start_session("s1", {}, events.append)
    _feed(first, fixture_audio())

    from localflow.service.server import ClientConn, EngineServer

    server = EngineServer.__new__(EngineServer)
    server.engine = engine
    client = ClientConn.__new__(ClientConn)
    client.session, client.emit = first, events.append
    import asyncio

    asyncio.run(server._dispatch(client, {"type": "session.start", "id": "s2", "context": {}}))
    engine.shutdown()
    assert first.cancelled and client.session.id == "s2"


def test_a_stale_end_or_cancel_leaves_the_current_take_alone():
    """An end or cancel naming an earlier take must not touch the one running now."""
    from localflow.service.server import EngineServer

    class Current:
        id = "s2"

    class Client:
        session = Current()

    assert EngineServer._names_current(Client, {"id": "s2"})
    assert not EngineServer._names_current(Client, {"id": "s1"})
    assert EngineServer._names_current(Client, {}), "no id still means the current take"
    Client.session = None
    assert not EngineServer._names_current(Client, {"id": "s2"})
