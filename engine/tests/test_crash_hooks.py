"""The engine's last words: every way it can die leaves something in the log."""

import logging
import subprocess
import sys
import threading

import pytest

from localflow import cli


@pytest.fixture
def hooks(tmp_path):
    """Install the hooks for one test and put the interpreter's own back afterwards."""
    import faulthandler

    before = sys.excepthook, threading.excepthook
    fault = tmp_path / "localflow-fault.log"
    yield fault
    sys.excepthook, threading.excepthook = before
    faulthandler.disable()
    if cli._fault_file is not None:
        cli._fault_file.close()
        cli._fault_file = None


def test_an_exception_in_a_thread_is_logged_with_its_traceback(hooks, caplog):
    cli._install_crash_hooks(hooks)

    def boom():
        raise RuntimeError("model load went wrong")

    with caplog.at_level(logging.CRITICAL):
        t = threading.Thread(target=boom, name="llm-load")
        t.start()
        t.join()
    record = next(r for r in caplog.records if "unhandled exception in thread llm-load" in r.getMessage())
    assert record.exc_info and "model load went wrong" in str(record.exc_info[1])


def test_an_uncaught_exception_is_logged(hooks, caplog):
    cli._install_crash_hooks(hooks)
    try:
        raise ValueError("nobody caught this")
    except ValueError:
        with caplog.at_level(logging.CRITICAL):
            sys.excepthook(*sys.exc_info())
    assert any("unhandled exception" in r.getMessage() and r.exc_info for r in caplog.records)


def test_the_previous_engine_s_fatal_error_is_moved_into_the_log(hooks, caplog):
    hooks.write_text("Windows fatal exception: access violation\n\nCurrent thread 0x1 (most recent call first):\n",
                     encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        cli._install_crash_hooks(hooks)
    assert any("previous engine ended with a fatal error" in r.getMessage()
               and "access violation" in r.getMessage() for r in caplog.records)
    cli._fault_file.flush()
    assert hooks.read_text(encoding="utf-8") == "", "reported once, then the file starts afresh"


def test_a_native_crash_writes_its_stack(tmp_path):
    """A real segmentation fault, in a child process: the case that used to leave nothing at all."""
    fault = tmp_path / "localflow-fault.log"
    code = ("import sys, faulthandler\n"
            "from pathlib import Path\n"
            "from localflow import cli\n"
            "cli._install_crash_hooks(Path(sys.argv[1]))\n"
            "faulthandler._sigsegv()\n")
    done = subprocess.run([sys.executable, "-c", code, str(fault)], capture_output=True, timeout=60)
    assert done.returncode != 0
    text = fault.read_text(encoding="utf-8", errors="replace")
    assert "fatal" in text.lower() and "most recent call first" in text, text
