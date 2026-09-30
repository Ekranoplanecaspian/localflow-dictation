"""The bundled llama-server's lifetime. A terminated engine used to leave its model server
behind holding ~2.7 GB of VRAM; two of those filled an 8 GB card and wrecked a benchmark."""

import os

from localflow.llm import manifest as M
from localflow.jobobject import kill_on_close_job
from localflow.llm import server
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


def test_the_server_log_is_rolled_once_it_grows_large(tmp_path):
    from localflow.llm import server

    log = tmp_path / "llama-server.log"
    log.write_bytes(b"x" * 100)
    server._roll_log(log)
    assert log.exists(), "a small log is left alone"
    log.write_bytes(b"x" * (server.LOG_MAX_BYTES + 1))
    server._roll_log(log)
    assert not log.exists() and (tmp_path / "llama-server.log.old").stat().st_size > server.LOG_MAX_BYTES
    server._roll_log(tmp_path / "missing.log")  # nothing there: no error


def test_unpacked_archives_and_old_builds_are_removed(tmp_path, monkeypatch):
    """The zips stayed for good after unpacking (~0.5 GB), and so did every build an update
    replaced."""
    from localflow.llm import manifest as M
    from localflow.llm import server as S

    monkeypatch.setattr(M, "BIN_DIR", tmp_path / "bin")
    current = M.llama_archive_dir()
    for kind in M.LLAMA_ASSETS:
        M.llama_dir(kind).mkdir(parents=True)
    (M.llama_dir("cuda") / ".ok").write_text("x")  # CUDA unpacked, CPU not (yet)
    for kind, assets in M.LLAMA_ASSETS.items():
        for a in assets:
            (current / a.name).write_bytes(b"z" * 100)
    old = tmp_path / "bin" / "llama" / "b1000"
    (old / "cuda").mkdir(parents=True)
    (old / "cuda" / "llama-server.exe").write_bytes(b"e" * 50)
    other = tmp_path / "bin" / "llama" / "notes"  # not a build: left alone
    other.mkdir()

    freed = S.tidy_downloads()

    assert not old.exists() and other.exists()
    for a in M.LLAMA_ASSETS["cuda"]:
        assert not (current / a.name).exists(), "unpacked: its archive goes"
    for a in M.LLAMA_ASSETS["cpu"]:
        assert (current / a.name).exists(), "not unpacked yet: its archive stays"
    assert freed == 100 * len(M.LLAMA_ASSETS["cuda"]) + 50
    assert S.tidy_downloads() == 0, "nothing more to do"


def test_a_build_in_use_is_left_for_later(tmp_path, monkeypatch):
    from localflow.llm import manifest as M
    from localflow.llm import server as S

    monkeypatch.setattr(M, "BIN_DIR", tmp_path / "bin")
    old = tmp_path / "bin" / "llama" / "b1000"
    old.mkdir(parents=True)
    held = (old / "llama-server.exe").open("wb")  # Windows will not rename a folder with an open file
    try:
        assert S.tidy_downloads() == 0
        assert old.exists()
    finally:
        held.close()
    assert S.tidy_downloads() == 0 and not old.exists(), "gone once nothing holds it"


# the Vulkan build, for AMD and Intel graphics (B3) --------------------------------------------------
LISTED = """ggml_vulkan: Found 2 Vulkan devices:
Available devices:
  Vulkan0: NVIDIA GeForce RTX 4060 Laptop GPU (7956 MiB, 7188 MiB free)
  Vulkan1: AMD Radeon(TM) 890M Graphics (16444 MiB, 15622 MiB free)
"""


def _listing(monkeypatch, text, calls):
    import subprocess

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")

    monkeypatch.setattr(server.subprocess, "run", run)


def test_the_vulkan_build_uses_the_graphics_that_are_not_nvidia_and_asks_once(tmp_path, monkeypatch):
    exe = tmp_path / "llama-server.exe"
    calls = []
    _listing(monkeypatch, LISTED, calls)
    assert server.vulkan_device(exe) == (1, "AMD Radeon(TM) 890M Graphics")
    assert server.vulkan_device(exe) == (1, "AMD Radeon(TM) 890M Graphics")
    assert len(calls) == 1, "listing touches every adapter, the NVIDIA card too: once per install"


def test_nothing_but_nvidia_means_no_vulkan(tmp_path, monkeypatch):
    _listing(monkeypatch, "Available devices:\n  Vulkan0: NVIDIA GeForce RTX 4060 Laptop GPU (7956 MiB, 7188 MiB free)\n", [])
    assert server.vulkan_device(tmp_path / "llama-server.exe") is None


def test_the_vulkan_build_is_pinned_and_verified():
    (asset,) = M.LLAMA_ASSETS["vulkan"]
    assert asset.name == f"llama-{M.LLAMA_BUILD}-bin-win-vulkan-x64.zip"
    assert asset.sha256 and len(asset.sha256) == 64 and asset.size
