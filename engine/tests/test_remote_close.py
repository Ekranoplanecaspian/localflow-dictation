"""Closing a speech worker that has stopped answering (no model needed: the child never replies)."""

import sys
import threading
import time

import pytest

from localflow.stt import remote


@pytest.fixture
def silent_worker(monkeypatch):
    # A child that takes requests and never answers one: a worker stuck mid-decode.
    monkeypatch.setattr(remote, "_command",
                        lambda: [sys.executable, "-c", "import time\nwhile True: time.sleep(1)"])
    monkeypatch.setattr(remote, "QUIT_WAIT_S", 0.5, raising=False)
    w = remote.Worker()
    yield w
    if w.alive:
        w.proc.kill()


def test_a_worker_that_never_replies_is_closed_and_its_caller_woken(silent_worker):
    """close() waited for the lock a stalled call held for ever, and never reached its kill
    (found by an outside review of 0.2.3)."""
    caught = {}

    def call():
        try:
            silent_worker.call("transcribe")
        except remote.WorkerDied as e:
            caught["died"] = e

    caller = threading.Thread(target=call, daemon=True)
    caller.start()
    time.sleep(0.3)  # the call is now waiting for a reply, holding the lock
    started = time.monotonic()
    closer = threading.Thread(target=silent_worker.close, daemon=True)
    closer.start()
    closer.join(15)
    assert not closer.is_alive(), "close() hung behind the stalled call"
    assert time.monotonic() - started < 10
    assert not silent_worker.alive
    caller.join(5)
    assert not caller.is_alive() and "died" in caught, "the stalled call was woken by the end of the pipe"
