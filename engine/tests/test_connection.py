"""The engine's local connection: who may connect, and what malformed messages do (nothing).

A real server on a thread, with the fake speech model from test_service, so the suite needs no
GPU. The clean-up model, model switches and saving settings are stubbed out: a fuzzed
settings message must not start llama-server or rewrite the user's settings file.
"""

import asyncio
import base64
import json
import os
import random
import socket
import threading
import time

import numpy as np
import pytest
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

import localflow.service.server as srv
from localflow.config import Config
from localflow.service import protocol as P

from test_service import SR, FakeSTT, blocks, fixture_audio


@pytest.fixture(scope="module")
def engine():
    import localflow.service.engine as eng

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(eng, "build_transcriber", lambda cfg: FakeSTT())
        mp.setattr(Config, "save", lambda self, *a, **k: None)
        mp.setattr(eng.Engine, "_load_llm", lambda self, *a, **k: None)

        def no_switch(self, key):
            raise ValueError(f"no model {key!r} in a test")

        mp.setattr(eng.Engine, "switch_speech", no_switch)
        mp.setattr(eng.Engine, "switch_cleanup", no_switch)
        cfg = Config()
        cfg.postprocess.llm_cleanup = False
        _thread, server, url = srv.serve_in_thread(cfg)
        port = int(url.rsplit(":", 1)[1])
        deadline = time.monotonic() + 30
        while server.engine.state != "ready":
            assert time.monotonic() < deadline, "engine did not become ready"
            time.sleep(0.05)
        yield server, port
        server._stop.set()
        time.sleep(0.2)


def open_ws(port, **kw):
    # websockets 17 wants connect() entered as a context manager; the tests close by hand.
    return connect(f"ws://127.0.0.1:{port}", open_timeout=5, max_size=None, **kw).__enter__()


def greeted(server, port, name="test"):
    ws = open_ws(port)
    ws.send(json.dumps({"type": "hello", "token": server.token, "client": name}))
    assert json.loads(ws.recv(timeout=5))["type"] == "hello.ok"
    return ws


def close_code(ws, timeout=10.0) -> int | None:
    """Read until the server closes the connection; its close code."""
    try:
        while True:
            ws.recv(timeout=timeout)
    except ConnectionClosed as e:
        return e.rcvd.code if e.rcvd else None


def wait_for(ws, pred, timeout=10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        msg = json.loads(ws.recv(timeout=max(0.01, deadline - time.monotonic())))
        if pred(msg):
            return msg


def raw_handshake(port, **headers) -> int:
    """The HTTP status of an opening handshake with exactly these headers."""
    sent = {
        "Host": f"127.0.0.1:{port}",
        "Upgrade": "websocket",
        "Connection": "Upgrade",
        "Sec-WebSocket-Key": base64.b64encode(os.urandom(16)).decode(),
        "Sec-WebSocket-Version": "13",
        **headers,
    }
    request = "GET / HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in sent.items()) + "\r\n"
    with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
        s.sendall(request.encode())
        return int(s.recv(1024).split(b" ")[1])


def pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def dictate(ws, sid="take") -> dict:
    """A whole take on this connection: its final."""
    ws.send(json.dumps({"type": "session.start", "id": sid, "context": {"app": "pytest"}}))
    for b in blocks(fixture_audio()):
        ws.send(pcm(b))
    ws.send(json.dumps({"type": "session.end", "id": sid}))
    return wait_for(ws, lambda m: m["type"] == "final" and m["id"] == sid, 30)


# --- who may connect ----------------------------------------------------------------------------
def test_it_listens_on_the_loopback_address_only(engine):
    _server, port = engine
    try:
        outside = [a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError:
        outside = []
    outside = [a for a in outside if not a.startswith("127.")]
    if not outside:
        pytest.skip("this machine has no address other than loopback")
    with pytest.raises(OSError):
        socket.create_connection((outside[0], port), timeout=2).close()


def test_web_pages_are_refused(engine):
    _server, port = engine
    assert raw_handshake(port) == 101
    for origin in ("https://example.com", "null", "http://127.0.0.1:1420", "tauri://localhost"):
        assert raw_handshake(port, Origin=origin) == 403, origin
    with pytest.raises(InvalidStatus) as refused:
        open_ws(port, origin="https://example.com")
    assert refused.value.response.status_code == 403


def test_other_host_names_are_refused(engine):
    """A web page that points its own name at 127.0.0.1 (DNS rebinding) still sends that name."""
    _server, port = engine
    assert raw_handshake(port, Host=f"localhost:{port}") == 101
    for host in (f"evil.example:{port}", "evil.example", f"127.0.0.1.evil.example:{port}", ""):
        assert raw_handshake(port, Host=host) == 403, host


@pytest.mark.parametrize("hello", [
    {"type": "hello", "client": "no token"},
    {"type": "hello", "token": "0" * 32},
    {"type": "hello", "token": 12345},
    {"type": "hello", "token": None},
    {"type": "hello", "token": ["x"]},
])
def test_a_hello_without_the_token_is_refused(engine, hello):
    _server, port = engine
    ws = open_ws(port)
    ws.send(json.dumps(hello))
    assert close_code(ws) == P.CLOSE_UNAUTHORIZED


@pytest.mark.parametrize("first", [
    json.dumps({"type": "status.get"}),
    json.dumps([1, 2]),
    "not json",
    "[" * 3000,  # deeper than the JSON parser recurses, smaller than a hello may be
    b"\x00\x01binary",
], ids=["status", "array", "text", "deep", "binary"])
def test_anything_but_a_hello_first_is_refused(engine, first):
    server, port = engine
    ws = open_ws(port)
    ws.send(first)
    assert close_code(ws) == P.CLOSE_BAD_HELLO


def test_a_hello_larger_than_any_real_one_is_refused(engine):
    server, port = engine
    ws = open_ws(port)
    ws.send(json.dumps({"type": "hello", "token": server.token, "client": "x" * P.MAX_HELLO_BYTES}))
    assert close_code(ws) == 1009  # message too big


def test_a_silent_connection_is_closed(engine, monkeypatch):
    monkeypatch.setattr(srv, "HELLO_TIMEOUT_S", 0.3)
    _server, port = engine
    ws = open_ws(port)
    assert close_code(ws, timeout=5) == P.CLOSE_BAD_HELLO


def test_too_many_connections_are_refused(engine, monkeypatch):
    server, port = engine
    monkeypatch.setattr(srv, "MAX_CONNECTIONS", 3)
    time.sleep(0.2)  # connections from earlier tests wind down
    held = [greeted(server, port, f"held{i}") for i in range(3)]
    with pytest.raises(InvalidStatus) as refused:
        open_ws(port)
    assert refused.value.response.status_code == 503
    held.pop().close()
    time.sleep(0.2)
    held.append(greeted(server, port, "again"))  # room again
    for ws in held:
        ws.close()


# --- what a connected client may send -----------------------------------------------------------
def test_large_messages_are_fine_once_greeted(engine):
    """The hello-sized limit is lifted after the token: the Hub's settings can be megabytes."""
    server, port = engine
    ws = greeted(server, port)
    ws.send(json.dumps({"type": "settings.set", "postprocess": {"snippets": {"sig": "x" * 3_000_000}}}))
    wait_for(ws, lambda m: m["type"] == "status")
    ws.send(json.dumps({"type": "status.get"}))
    assert wait_for(ws, lambda m: m["type"] == "status")
    ws.close()


def test_malformed_messages_are_answered_and_change_nothing(engine):
    server, port = engine
    ws = greeted(server, port)
    replies: list[dict] = []

    def send(msg, expect):
        ws.send(msg if isinstance(msg, (str, bytes)) else json.dumps(msg))
        replies.append(wait_for(ws, expect))

    is_error = lambda m: m["type"] == "error"  # noqa: E731
    send("not json", is_error)
    send("[" * 50_000, is_error)
    send(json.dumps([1, 2, 3]), is_error)
    send({"no": "type"}, is_error)
    send({"type": 7}, is_error)
    send({"type": "x" * 500}, lambda m: m["type"] == "error" and m["code"] == "unknown-message")
    assert len(replies[-1]["message"]) <= 40
    send({"type": "session.start", "id": "s1", "context": ["a", "list"]},
         lambda m: is_error(m) and m.get("id") == "s1")
    send({"type": "session.start", "id": "s2", "context": {"app": {"nested": 1}}},
         lambda m: is_error(m) and m.get("id") == "s2")
    send({"type": "session.start", "id": "s3", "language": 42}, lambda m: is_error(m) and m.get("id") == "s3")
    send({"type": "session.start", "id": {"not": "an id"}}, lambda m: is_error(m) and "id" not in m)
    send({"type": "session.start", "id": "x" * 1000}, lambda m: is_error(m) and "id" not in m)
    send({"type": "command.run", "id": "c1", "selection": 12, "instruction": "shorter"},
         lambda m: m["type"] == "command.result" and m["id"] == "c1")
    assert replies[-1]["changed"] is False and replies[-1]["rejected"].startswith("bad request")
    send({"type": "command.run", "id": "c2", "selection": "a" * (P.MAX_SELECTION_CHARS + 1), "instruction": "x"},
         lambda m: m["type"] == "command.result" and m["id"] == "c2")
    assert replies[-1]["changed"] is False
    send({"type": "settings.set", "postprocess": "not an object", "compute": [1], "stt": 5, "llm": None},
         lambda m: m["type"] == "status")
    # an end or cancel for nothing, and audio with no take, change nothing
    for msg in ({"type": "session.end", "id": "none"}, {"type": "session.cancel"}, b"\x00\x00" * 320):
        ws.send(msg if isinstance(msg, bytes) else json.dumps(msg))
    ws.send(json.dumps({"type": "hello", "token": "wrong"}))  # a second hello is ignored

    assert server.engine.state == "ready"
    assert dictate(ws)["raw"].startswith("<")
    ws.close()


def test_bad_audio_frames_are_dropped_and_the_take_goes_on(engine):
    server, port = engine
    ws = greeted(server, port)
    ws.send(json.dumps({"type": "session.start", "id": "t1"}))
    speech = fixture_audio()
    for i, b in enumerate(blocks(speech)):
        ws.send(pcm(b))
        if i == 5:
            ws.send(b"\x01\x02\x03")  # odd: not 16-bit samples
            ws.send(b"\x00" * (P.MAX_AUDIO_FRAME_BYTES + 2))  # too large
    ws.send(json.dumps({"type": "session.end", "id": "t1"}))
    got = []
    final = wait_for(ws, lambda m: (got.append(m), m["type"] == "final")[1], 30)
    errors = [m for m in got if m["type"] == "error"]
    assert [e["code"] for e in errors] == ["bad-audio"], "one report per take"
    assert "id" not in errors[0], "naming the take would end it at the shell"
    assert final["id"] == "t1" and abs(final["timings"]["audio_s"] - len(speech) / SR) < 0.05
    ws.close()


def _junk(rng: random.Random, depth: int = 0):
    pick = rng.randrange(10 if depth < 3 else 7)
    if pick == 0:
        return None
    if pick == 1:
        return rng.choice([True, False])
    if pick == 2:
        return rng.choice([0, -1, 7, 2**70, -(2**63)])
    if pick == 3:
        return rng.choice([1.5, -0.0, 1e308, float("nan"), float("inf")])
    if pick in (4, 5, 6):
        n = rng.choice([0, 1, 5, 40, 3000])
        return "".join(chr(rng.choice([rng.randrange(32, 127), rng.randrange(0x80, 0xD7FF)])) for _ in range(n))
    if pick in (7, 8):
        return [_junk(rng, depth + 1) for _ in range(rng.randrange(4))]
    return {str(_junk(rng, 3))[:20]: _junk(rng, depth + 1) for _ in range(rng.randrange(4))}


def test_fuzzed_messages_never_stop_dictation(engine):
    """Hundreds of messages with the right types and random fields, and random frames; the
    engine answers what it answers, and a real take works afterwards."""
    server, port = engine
    ws = greeted(server, port)
    rng = random.Random(9)
    types = [P.SESSION_START, P.SESSION_END, P.SESSION_CANCEL, P.COMMAND_RUN, P.STATUS_GET,
             P.SETTINGS_SET, P.HELLO, "status", "final", "", "session"]
    fields = ["id", "context", "language", "selection", "instruction", "token", "client",
              "postprocess", "stt", "llm", "compute"]
    stop = threading.Event()
    received = []

    def drain():  # keep reading, as a live client does
        while not stop.is_set():
            try:
                received.append(ws.recv(timeout=0.1))
            except TimeoutError:
                continue
            except ConnectionClosed:
                return

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    for _ in range(600):
        if rng.random() < 0.2:
            ws.send(rng.randbytes(rng.choice([0, 1, 3, 640, 70_000])))
            continue
        msg = {"type": rng.choice(types)}
        for f in rng.sample(fields, rng.randrange(4)):
            msg[f] = _junk(rng)
        if msg["type"] == P.SETTINGS_SET and isinstance(msg.get("postprocess"), dict):
            # Keys the settings do not have are dropped; real ones are A8's to check
            # (test_validate). Here, only the connection is under test.
            msg["postprocess"] = {"zz_" + k: v for k, v in msg["postprocess"].items()}
        ws.send(json.dumps(msg))
    ws.send(json.dumps({"type": P.SESSION_CANCEL}))
    time.sleep(1.0)
    stop.set()
    reader.join()

    assert received, "the engine answered nothing"
    assert server.engine.state == "ready"
    assert dictate(ws, "after-fuzz")["raw"].startswith("<")
    ws.close()


def test_a_client_that_stops_reading_is_disconnected(engine, monkeypatch):
    """Replies pile up once the client's socket is full. How much the operating system buffers
    first varies, so its sender is held instead: the same backlog, at once."""
    server, port = engine
    monkeypatch.setattr(srv, "MAX_PENDING_REPLIES", 50)
    sender = srv.EngineServer._sender

    async def held(self, client):
        if client.name == "stalled":
            await asyncio.Event().wait()  # a socket that never drains
        await sender(self, client)

    monkeypatch.setattr(srv.EngineServer, "_sender", held)
    stalled = open_ws(port)
    stalled.send(json.dumps({"type": "hello", "token": server.token, "client": "stalled"}))
    request = json.dumps({"type": "status.get"})
    for _ in range(60):
        stalled.send(request)
    deadline = time.monotonic() + 10
    while any(c.name == "stalled" for c in list(server._clients)):
        assert time.monotonic() < deadline, "a client that never reads was kept"
        time.sleep(0.1)
    stalled.close()
    # everyone else is unaffected
    ws = greeted(server, port)
    ws.send(request)
    assert wait_for(ws, lambda m: m["type"] == "status")
    ws.close()


def test_one_client_s_backlog_does_not_hold_up_another(engine):
    """Messages already received used to be handled back to back without letting anything else
    run: a burst of status requests (5 ms each) held every other client for seconds."""
    server, port = engine
    flood = greeted(server, port, "flood")
    request = json.dumps({"type": "status.get"})
    for _ in range(1000):
        flood.send(request)
    time.sleep(0.3)  # the burst has arrived and is being worked through
    other = greeted(server, port, "other")
    started = time.monotonic()
    other.send(request)
    wait_for(other, lambda m: m["type"] == "status")
    assert time.monotonic() - started < 1.5, "stuck behind the other client's backlog"
    flood.close()
    other.close()
