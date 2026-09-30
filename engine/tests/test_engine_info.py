"""engine.json names the engine other clients join; a second engine must not take it over."""

import json
import os
import subprocess
import sys

from localflow.service.server import claim_engine_info, release_engine_info


def test_an_engine_claims_a_free_or_stale_file(tmp_path):
    path = tmp_path / "engine.json"
    assert claim_engine_info({"port": 1, "token": "a", "pid": os.getpid()}, path)
    assert json.loads(path.read_text())["port"] == 1

    # A file naming a process that has gone is taken over.
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    path.write_text(json.dumps({"port": 2, "token": "b", "pid": gone.pid}))
    assert claim_engine_info({"port": 3, "token": "c", "pid": os.getpid()}, path)
    assert json.loads(path.read_text())["port"] == 3


def test_a_live_engine_keeps_its_file(tmp_path):
    path = tmp_path / "engine.json"
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        path.write_text(json.dumps({"port": 5, "token": "users", "pid": other.pid}))
        assert not claim_engine_info({"port": 6, "token": "second", "pid": os.getpid()}, path)
        assert json.loads(path.read_text())["token"] == "users"

        # Stopping, the second engine leaves the first one's file alone.
        release_engine_info(os.getpid(), path)
        assert path.exists()
        release_engine_info(other.pid, path)
        assert not path.exists()
    finally:
        other.kill()
        other.wait()
