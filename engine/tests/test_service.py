"""Streaming segmenter and the engine service round trip, with a fake speech model so the
suite needs no GPU and no 2 GB download (Silero VAD is ~2 MB)."""

import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from localflow.config import Config

FIXTURE = Path(__file__).parent / "fixtures" / "speech.wav"
SR = 16000


def fixture_audio() -> np.ndarray:
    with wave.open(str(FIXTURE)) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768


class FakeSTT:
    name = "fake"
    sample_rate = SR
    device = "cpu"
    precision = "fp32"

    def __init__(self, *_):
        self.calls: list[float] = []

    def warmup(self):
        pass

    def warm(self):
        pass

    def transcribe(self, audio, language=None):
        self.calls.append(audio.size / SR)
        return f"<{audio.size / SR:.1f}s>"


@pytest.fixture(scope="module")
def vad():
    from localflow.service.vad import SileroVad

    return SileroVad()


def blocks(audio: np.ndarray, ms: int = 20):
    n = SR * ms // 1000
    for i in range(0, len(audio), n):
        yield audio[i:i + n]


def test_segmenter_closes_phrases_after_silence(vad):
    from localflow.service.vad import Segmenter

    speech = fixture_audio()  # one sentence with ~0.5 s of trailing silence
    silence = np.zeros(SR, dtype=np.float32)
    audio = np.concatenate([speech, silence, speech])
    seg = Segmenter(vad)
    closed = []
    for b in blocks(audio):
        closed += seg.feed(b)
    assert len(closed) == 2, closed
    assert all(2.0 < c.seconds < 4.5 for c in closed)
    assert closed[1].start >= closed[0].end
    # both phrases had closed before the "key release": nothing is left for the tail
    assert seg.finish(len(audio)) is None


def test_segmenter_returns_open_phrase_as_tail(vad):
    from localflow.service.vad import Segmenter

    speech = fixture_audio()[: int(2.5 * SR)]  # cut mid-sentence, as when the key is released early
    seg = Segmenter(vad)
    closed = []
    for b in blocks(speech):
        closed += seg.feed(b)
    assert closed == []
    tail = seg.finish(len(speech))
    assert tail is not None and tail.start == 0 and abs(tail.seconds - 2.5) < 0.05


def test_chunk_boundary_prefers_phrase_starts_and_hard_cuts_otherwise():
    from localflow.service.engine import CHUNK_HARD_S, CHUNK_MIN_S, CHUNK_TARGET_S, Session
    from localflow.service.vad import Segment

    s = Session.__new__(Session)
    s.chunk_start = 0
    s.closed = [Segment(int(a * SR), int(b * SR)) for a, b in ((0, 8), (9, 17), (18, 21))]
    s.samples = int(20 * SR)
    assert s._chunk_boundary() is None  # under the target: keep growing
    s.samples = int(23 * SR)
    assert s._chunk_boundary() == int(18 * SR)  # latest phrase start that leaves >= CHUNK_MIN_S
    s.closed = [Segment(0, int(5 * SR))]  # only an early phrase: too short a chunk, wait
    assert s._chunk_boundary() is None
    s.samples = int(CHUNK_HARD_S * SR)
    assert s._chunk_boundary() == int(CHUNK_HARD_S * SR)  # hard cut
    assert CHUNK_MIN_S < CHUNK_TARGET_S < CHUNK_HARD_S


def test_segmenter_silence_only_has_no_tail(vad):
    from localflow.service.vad import Segmenter

    seg = Segmenter(vad)
    audio = np.zeros(SR * 2, dtype=np.float32)
    closed = []
    for b in blocks(audio):
        closed += seg.feed(b)
    assert closed == []
    assert seg.finish(len(audio)) is None


def test_service_round_trip(monkeypatch):
    import localflow.service.engine as eng
    from localflow.service.client import EngineClient
    from localflow.service.server import serve_in_thread

    monkeypatch.setattr(eng, "build_transcriber", lambda cfg: FakeSTT())
    cfg = Config()
    cfg.postprocess.llm_cleanup = False
    _thread, server, url = serve_in_thread(cfg)
    port = int(url.rsplit(":", 1)[1])

    client = EngineClient("test")
    ready = threading.Event()
    finals: list[dict] = []
    partials: list[dict] = []
    client.on_status = lambda st: ready.set() if st.get("stt", {}).get("state") == "ready" else None
    client.on_partial = partials.append
    client.on_final = lambda ev: (finals.append(ev), done.set())
    done = threading.Event()
    client.connect(port, server.token)
    client.on_status(client.status)
    assert ready.wait(30), "engine did not become ready"

    # wrong token is refused
    bad = EngineClient("bad")
    with pytest.raises(Exception):
        bad.connect(port, "nope")

    speech = fixture_audio()
    audio = np.concatenate([speech, np.zeros(SR, dtype=np.float32), speech])
    client.start_session({"app": "pytest", "title": "round trip"})
    for b in blocks(audio):
        client.send_audio(b)
    client.end_session()
    assert done.wait(30), "no final"
    final = finals[0]
    t = final["timings"]
    assert t["mode"] == "periodic"  # fake model reports cpu
    assert t["phrases"] == 2  # both sentences closed while "speaking"
    assert t["live_decodes"] >= 1 and len(partials) >= 1
    # the final is ONE pass over the whole take (zero-padded up to the next 1 s bucket), never stitched phrases
    import math

    assert final["raw"] == f"<{math.ceil(len(audio) / SR):.1f}s>" and t["chunks"] == 1
    assert t["release_to_final_ms"] is not None and t["stt_final_ms"] >= 0
    assert abs(t["audio_s"] - len(audio) / SR) < 0.05

    # a second session on the same connection works and cancelled ones produce nothing
    client.start_session({})
    client.send_audio(speech)
    client.cancel_session()
    client.start_session({})
    done.clear()
    client.send_audio(speech)
    client.end_session()
    assert done.wait(30)
    assert len(finals) == 2

    client.close()
    server._stop.set()
    time.sleep(0.2)


def test_engine_cleans_up_the_final_text_and_prefills_mid_utterance(monkeypatch):
    """The whole phase-2 path with a fake speech model and a fake language model: live
    decoding warms the prompt, and the final text comes back auto-edited."""
    import localflow.service.engine as eng
    from localflow.cleanup.pipeline import CleanupPipeline
    from localflow.config import PostProcessConfig

    from tests.test_cleanup import FakeProvider

    monkeypatch.setattr(eng, "build_transcriber", lambda cfg: FakeSTT())
    cfg = Config()
    cfg.postprocess.llm_cleanup = False  # no real llama-server in the test
    engine = eng.Engine(cfg)
    engine.load()
    deadline = time.monotonic() + 30
    while engine.state == "loading" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert engine.state == "ready", engine.error

    provider = FakeProvider("Cleaned up.")
    pp = PostProcessConfig(llm_cleanup=True, llm_min_words=1, llm_prefill=True)
    engine.cleanup = CleanupPipeline(pp, provider)

    events: list[dict] = []
    session = engine.start_session("s1", {"app": "slack.exe", "title": "general - Slack"}, events.append)
    speech = fixture_audio()
    for b in blocks(np.concatenate([speech, speech])):  # ~7 s, enough for a live decode + prefill
        session.feed((np.clip(b * 32767, -32768, 32767)).astype("<i2").tobytes())
    session.end()
    deadline = time.monotonic() + 30
    while not session.done and time.monotonic() < deadline:
        time.sleep(0.05)
    engine.shutdown()

    final = [e for e in events if e["type"] == "final"]
    assert final, "no final event"
    t = final[0]["timings"]
    assert final[0]["text"] == "Cleaned up." and t["used_llm"] is True
    assert t["profile"] == "chat", "the target app picked the style profile"
    assert provider.prefills, "the prompt cache was never warmed while speaking"
    assert t["llm_ms"] is not None and t["release_to_final_ms"] is not None
