"""The bundled llama-server's lifetime. A terminated engine used to leave its model server
behind holding ~2.7 GB of VRAM; two of those filled an 8 GB card and wrecked a benchmark."""

import os

from localflow.llm import manifest as M
from localflow.jobobject import kill_on_close_job
from localflow.llm.server import reap_orphans


def test_job_object_is_available_and_reused():
    job = kill_on_close_job()
    assert job, "no job object: llama-server would outlive a killed engine"
    assert kill_on_close_job() == job, "the job must be process-wide, not per server"


def test_reaper_leaves_everything_else_alone():
    """No orphan of ours is running in the test process, so nothing may be killed."""
    before = _llama_pids()
    assert reap_orphans(M.llama_server_exe("cuda")) == 0
    assert _llama_pids() == before


def _llama_pids() -> set[int]:
    import subprocess

    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                          "(Get-Process llama-server -ErrorAction SilentlyContinue).Id"],
                         capture_output=True, text=True, timeout=30).stdout
    return {int(x) for x in out.split() if x.strip().isdigit()}


def test_unknown_model_key_falls_back_to_the_default():
    from localflow.llm.server import LlamaServer

    assert LlamaServer(model_key="not-a-model").model_key == M.DEFAULT_CLEANUP_MODEL
    assert LlamaServer(model_key="qwen3-1.7b").model_key == "qwen3-1.7b"


def test_builds_live_in_separate_directories():
    """The CUDA and CPU zips carry the same file names; one directory each."""
    assert M.llama_dir("cuda") != M.llama_dir("cpu")
    assert M.llama_server_exe("cuda").parent.name == "cuda"
    assert os.path.join("localflow", "bin", "llama").lower() in str(M.llama_server_exe("cuda")).lower()
