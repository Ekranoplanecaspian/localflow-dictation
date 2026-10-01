"""The download queue (M1): one download at a time, most urgent first, each one visible and -
unless it is LocalFlow's own part - stoppable. Nothing here goes online."""

import threading
import time

import pytest

from localflow.fetch import Cancelled
from localflow.service.downloads import Downloads


def wait(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


class Gate:
    """A runner that reports progress and then waits to be let through (or stopped)."""

    def __init__(self, size=100):
        self.size = size
        self.go = threading.Event()
        self.started = threading.Event()

    def __call__(self, report, stop):
        self.started.set()
        report(self.size // 2, self.size)
        while not self.go.is_set():
            if stop.wait(0.01):
                raise Cancelled()
        report(self.size, self.size)


@pytest.fixture
def q():
    changes = []
    downloads = Downloads(lambda: changes.append(1))
    yield downloads
    downloads.close()


def test_one_at_a_time_and_the_most_urgent_first(q):
    order = []

    def runner(name, gate=None):
        def run(report, stop):
            order.append(name)
            if gate:
                gate(report, stop)
        return run

    first = Gate()
    q.request("cleanup", "phi-4-mini", "Phi-4 mini", 100, runner("library", first), "library")
    assert first.started.wait(2)
    q.request("cleanup", "gemma-4-e2b", "Gemma", 100, runner("library 2"), "library")
    q.request("speech", "parakeet-v3", "Parakeet v3", 100, runner("switch"), "switch")
    states = {v["label"]: v["state"] for v in q.views()}
    assert states == {"Phi-4 mini": "downloading", "Parakeet v3": "queued", "Gemma": "queued"}
    assert [v["label"] for v in q.views()][:2] == ["Phi-4 mini", "Parakeet v3"]  # running, then most urgent
    first.go.set()
    assert wait(lambda: len(order) == 3)
    assert order == ["library", "switch", "library 2"]


def test_asking_again_joins_the_download_and_can_make_it_more_urgent(q):
    gate = Gate()
    q.request("cleanup", "a", "A", 100, gate, "library")
    assert gate.started.wait(2)
    q.request("speech", "b", "B", 100, Gate(), "library")
    job = q.request("speech", "b", "B", 100, Gate(), "first-run")
    assert len(q.views()) == 2 and job.reason == "first-run"
    gate.go.set()


def test_a_library_download_joined_by_the_first_run_can_no_longer_be_stopped(q):
    gate = Gate()
    library = q.request("speech", "parakeet-v3", "Parakeet v3", 100, gate, "library")
    assert gate.started.wait(2) and library.cancellable
    joined = q.request("speech", "parakeet-v3", "Parakeet v3", 100, Gate(), "first-run", cancellable=False)
    assert joined is library
    assert not q.cancel(library.id), "the first run is waiting for it"
    assert [v["cancellable"] for v in q.views()] == [False]
    gate.go.set()
    assert wait(lambda: library.state == "done")


def test_a_running_download_is_stopped_and_a_queued_one_taken_out(q):
    running, queued = Gate(), Gate()
    a = q.request("speech", "a", "A", 100, running, "library")
    b = q.request("speech", "b", "B", 100, queued, "library")
    assert running.started.wait(2)
    assert q.cancel(b.id) and b.state == "cancelled"
    assert q.cancel(a.id)
    assert wait(lambda: a.state == "cancelled")
    assert not queued.started.is_set()
    assert not q.cancel(a.id), "finished: nothing to stop"


def test_localflows_own_parts_cannot_be_stopped(q):
    gate = Gate()
    job = q.request("gpu-libs", "cuda", "CUDA", 100, gate, "automatic", cancellable=False)
    assert gate.started.wait(2)
    assert not q.cancel(job.id)
    gate.go.set()
    assert wait(lambda: job.state == "done")


def test_fetch_waits_and_raises_what_the_download_raised(q):
    def broken(report, stop):
        raise OSError("[Errno 28] No space left on device")

    with pytest.raises(OSError, match="No space"):
        q.fetch("cleanup", "x", "X", 100, broken, "first-run")
    views = q.views()
    assert views[0]["state"] == "error" and "No space" in views[0]["error"]


def test_fetch_of_a_cancelled_download_raises_cancelled(q):
    gate = Gate()
    got = []

    def waiter():
        try:
            q.fetch("speech", "a", "A", 100, gate, "switch")
        except Cancelled:
            got.append("cancelled")

    t = threading.Thread(target=waiter)
    t.start()
    assert gate.started.wait(2)
    q.cancel(q.views()[0]["id"])
    t.join(5)
    assert got == ["cancelled"]


def test_a_download_says_how_far_how_fast_and_how_long(q):
    def run(report, stop):
        for i in range(1, 9):  # 10 MB every 0.15 s: about 65 MB/s
            report(i * 10 << 20, 1000 << 20)
            time.sleep(0.15)
        stop.wait(10)
        raise Cancelled()

    job = q.request("speech", "a", "A", 1000 << 20, run, "library")
    assert wait(lambda: job.done >= 80 << 20)
    v = job.view()
    assert v["state"] == "downloading" and v["total"] == 1000 << 20 and v["progress"] == 0.08
    assert 30 << 20 < v["speed_bps"] < 100 << 20 and 8 < v["eta_s"] < 35
    q.cancel(job.id)


def test_a_download_is_never_shown_complete_before_it_is(q):
    def run(report, stop):
        report(999_999, 1_000_000)  # the last bytes received; the file is still being written
        stop.wait(10)
        raise Cancelled()

    job = q.request("speech", "a", "A", 1_000_000, run, "library")
    assert wait(lambda: job.done == 999_999)
    assert job.view()["progress"] == 0.999
    q.cancel(job.id)


def test_listeners_hear_the_progress_of_the_download_they_asked_for(q):
    heard = []

    def run(report, stop):
        report(50, 100)
        report(100, 100)

    q.fetch("cleanup", "a", "A", 100, run, "switch", progress=lambda d, t: heard.append((d, t)))
    assert heard == [(50, 100), (100, 100)]
    assert q.views()[0]["state"] == "done" and q.views()[0]["progress"] == 1.0
