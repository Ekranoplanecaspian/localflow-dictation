"""A model download in a process of its own (fetch.py): what it says, and that it can be stopped
at any moment - the reason it is a process (Xet downloads cannot be interrupted from inside).
A small stand-in script plays the child; nothing goes online."""

import sys
import textwrap
import threading
import time
import urllib.error

import pytest

from localflow import fetch, problems


def test_the_child_reports_progress_then_ok(monkeypatch, capsys):
    def download(kind, key, device, progress):
        progress(50, 100)
        progress(100, 100)

    monkeypatch.setattr(fetch, "_download", download)
    assert fetch.child_main(["speech", "parakeet-v3", "cpu"]) == 0
    assert capsys.readouterr().out.splitlines() == ["P 50 100", "P 100 100", "OK"]


def test_the_child_classifies_its_own_failure(monkeypatch, capsys):
    def download(kind, key, device, progress):
        raise urllib.error.URLError(OSError("[Errno 11001] getaddrinfo failed"))

    monkeypatch.setattr(fetch, "_download", download)
    assert fetch.child_main(["cleanup", "qwen3-4b", ""]) == 1
    out = capsys.readouterr().out.strip()
    assert out == f"ERR {problems.CLEANUP_DOWNLOAD_FAILED}\tThis PC could not reach the internet."


@pytest.fixture
def child(tmp_path, monkeypatch):
    """Play the child with a script: `lines` it prints, then it sleeps `hang` seconds."""
    monkeypatch.delenv(fetch.IN_PROCESS_ENV, raising=False)
    removed = []
    monkeypatch.setattr(fetch, "remove_partial", lambda kind, key: removed.append((kind, key)) or 0)

    def make(lines, hang=0.0, code=0):
        script = tmp_path / "child.py"
        script.write_text(textwrap.dedent(f"""
            import sys, time
            for line in {lines!r}:
                print(line, flush=True)
                time.sleep(0.02)
            time.sleep({hang})
            sys.exit({code})
        """), encoding="utf-8")
        monkeypatch.setattr(fetch, "_command", lambda: [sys.executable, str(script)])

    make.removed = removed
    return make


def test_the_engine_side_follows_the_child(child):
    child(["P 10 100", "P 100 100", "OK"])
    seen = []
    fetch.run("speech", "parakeet-v3", "cpu", lambda d, t: seen.append((d, t)), threading.Event())
    assert seen == [(10, 100), (100, 100)]


def test_a_failure_comes_back_with_its_code(child):
    child([f"ERR {problems.SPEECH_NO_SPACE}\tThere is not enough space on drive C:."], code=1)
    with pytest.raises(problems.Classified) as e:
        fetch.run("speech", "parakeet-v3", "cpu", lambda d, t: None, threading.Event())
    assert problems.classify_speech(e.value) == problems.SPEECH_NO_SPACE
    assert problems.detail(e.value) == "There is not enough space on drive C:."


def test_a_child_that_dies_says_the_download_failed(child):
    child(["P 10 100"], code=3)
    with pytest.raises(problems.Classified) as e:
        fetch.run("cleanup", "qwen3-4b", "", lambda d, t: None, threading.Event())
    assert e.value.code == problems.CLEANUP_DOWNLOAD_FAILED


def test_stopping_ends_the_child_at_once_and_clears_what_it_left(child):
    child(["P 10 100"], hang=60)
    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    t0 = time.monotonic()
    with pytest.raises(fetch.Cancelled):
        fetch.run("speech", "parakeet-v3", "cpu", lambda d, t: None, stop)
    assert time.monotonic() - t0 < 5, "not left to run on"
    assert child.removed == [("speech", "parakeet-v3")]


def test_in_process_a_stop_is_seen_at_the_next_progress_report(monkeypatch):
    removed = []
    monkeypatch.setattr(fetch, "remove_partial", lambda kind, key: removed.append(key) or 0)
    stop = threading.Event()

    def download(kind, key, device, progress):
        progress(10, 100)
        stop.set()
        progress(20, 100)
        raise AssertionError("not reached")

    monkeypatch.setattr(fetch, "_download", download)
    with pytest.raises(fetch.Cancelled):
        fetch.run("cleanup", "phi-4-mini", "", lambda d, t: None, stop)
    assert removed == ["phi-4-mini"]


def test_what_a_stopped_download_leaves_is_deleted(tmp_path, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    blobs = tmp_path / "models--istupakov--parakeet-tdt-0.6b-v3-onnx" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc.incomplete").write_bytes(b"x" * 1000)
    (blobs / "def").write_bytes(b"kept")
    assert fetch.remove_partial("speech", "parakeet-v3") == 1000
    assert [p.name for p in blobs.iterdir()] == ["def"]
