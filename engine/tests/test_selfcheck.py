"""The engine's self-check (selfcheck.py): verifying model files against the hashes their
downloads recorded, the quick checks, "Download again", and the protocol messages."""

import hashlib
import json
from pathlib import Path

import pytest

from localflow import problems
from localflow import selfcheck as S


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fake_snapshot(root: Path, files: dict[str, bytes], lfs: set[str]) -> Path:
    """A Hugging Face cache repo folder with one snapshot and its tree listing."""
    commit = "a" * 40
    snap = root / "snapshots" / commit
    snap.mkdir(parents=True)
    tree = {}
    for name, data in files.items():
        (snap / name).write_bytes(data)
        if name in lfs:
            tree[name] = {"size": len(data), "lfs_sha256": sha(data), "lfs_size": len(data)}
        else:
            tree[name] = {"size": len(data), "blob_id": blob_id(data)}
    (root / "trees").mkdir()
    (root / "trees" / f"{commit}.json").write_text(json.dumps({"format_version": 1, "files": tree}))
    return snap


def test_intact_files_verify(tmp_path):
    snap = fake_snapshot(tmp_path, {"model.onnx": b"weights" * 1000, "vocab.txt": b"a\nb\n"}, {"model.onnx"})
    damaged, unverifiable, checked = S.verify_snapshot(snap)
    assert (damaged, unverifiable) == ([], []) and checked == 7004


def test_a_changed_file_is_damaged_and_a_missing_one_too(tmp_path):
    snap = fake_snapshot(tmp_path, {"model.onnx": b"weights" * 1000, "vocab.txt": b"a\nb\n"}, {"model.onnx"})
    (snap / "model.onnx").write_bytes(b"weights" * 999 + b"WEIGHTS")  # same size, different bytes
    (snap / "vocab.txt").unlink()
    damaged, _u, _c = S.verify_snapshot(snap, ["model.onnx", "vocab.txt"])
    assert damaged == ["model.onnx", "vocab.txt (missing)"]


def test_a_file_the_listing_does_not_know_is_unverifiable_not_damaged(tmp_path):
    snap = fake_snapshot(tmp_path, {"model.onnx": b"x"}, {"model.onnx"})
    (snap / "extra.bin").write_bytes(b"y")
    damaged, unverifiable, _ = S.verify_snapshot(snap, ["model.onnx", "extra.bin"])
    assert (damaged, unverifiable) == ([], ["extra.bin"])


def gguf(folder: Path, name: str, data: bytes, etag: str) -> Path:
    path = folder / name
    path.write_bytes(data)
    meta = folder / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / f"{name}.metadata").write_text(f"{'c' * 40}\n{etag}\n1790000000.0\n")
    return path


def test_the_clean_up_model_is_checked_against_its_etag(tmp_path, monkeypatch):
    from localflow.llm import manifest as M

    monkeypatch.setattr(M, "gguf_dir", lambda: tmp_path)
    key = "qwen3-4b"
    name = M.CLEANUP_MODELS[key].filename
    gguf(tmp_path, name, b"model" * 100, sha(b"model" * 100))
    assert S.check_cleanup_files(key).status == "ok"
    gguf(tmp_path, name, b"broken" * 100, sha(b"model" * 100))
    check = S.check_cleanup_files(key)
    assert (check.status, check.code) == ("fail", problems.CLEANUP_FILES_DAMAGED)
    assert check.paths == [str(tmp_path / name)]


def test_quick_checks_report_low_disk_and_an_unwritable_folder(tmp_path, monkeypatch):
    check = S.check_disk(tmp_path, min_gb=10**9)
    assert (check.status, check.code) == ("warn", problems.DISK_LOW)
    assert check.vars["needed"] == "1000000000 GB"
    assert S.check_disk(tmp_path, min_gb=0).status == "ok"
    assert S.check_models_folder(tmp_path).status == "ok"

    def refuse(self, data):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(Path, "write_bytes", refuse)
    check = S.check_models_folder(tmp_path)
    assert (check.status, check.code) == ("fail", problems.FOLDER_NOT_WRITABLE)


def test_an_old_driver_is_reported_and_no_card_is_fine(monkeypatch):
    from localflow import gpu

    class Card:
        def __init__(self, version):
            self.version = version

        def driver_version(self):
            return self.version

    monkeypatch.setattr(gpu, "_monitor", Card("561.09"))
    check = S.check_driver()
    assert (check.status, check.code, check.vars["needed"]) == ("warn", problems.DRIVER_TOO_OLD, "580")
    monkeypatch.setattr(gpu, "_monitor", Card("610.62"))
    assert S.check_driver().status == "ok"
    monkeypatch.setattr(gpu, "_monitor", Card(None))
    assert S.check_driver().status == "ok"


def test_a_check_that_breaks_is_skipped_not_fatal(monkeypatch):
    from localflow import gpu

    class Broken:
        def driver_version(self):
            raise RuntimeError("NVML fell over")

    monkeypatch.setattr(gpu, "_monitor", Broken())
    check = S.check_driver()
    assert check.status == "skip" and "NVML fell over" in check.detail


def test_unreachable_hosts_are_reported():
    check = S.check_hosts({"nowhere": "http://127.0.0.1:9"}, timeout=2)
    assert (check.status, check.code) == ("warn", problems.DOWNLOAD_HOSTS_UNREACHABLE)
    assert check.detail.startswith("Can't reach nowhere")


def test_repair_deletes_only_named_files_inside_the_model_caches(tmp_path, monkeypatch):
    import huggingface_hub.constants as C

    cache = tmp_path / "hub"
    models = tmp_path / "models"
    cache.mkdir()
    models.mkdir()
    monkeypatch.setattr(C, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(S, "MODELS_DIR", models)
    inside = cache / "damaged.onnx"
    inside.write_bytes(b"x")
    gguf_file = models / "model.gguf"
    gguf_file.write_bytes(b"x")
    outside = tmp_path / "precious.txt"
    outside.write_text("keep me")
    checks = [S.Check("speech_files", "", "fail", paths=[str(inside), str(outside)]),
              S.Check("cleanup_files", "", "fail", paths=[str(gguf_file)])]
    removed = S.repair(checks)
    assert not inside.exists() and not gguf_file.exists()
    assert outside.read_text() == "keep me", "never outside the caches"
    assert len(removed) == 2


def test_every_code_a_check_can_report_is_in_the_catalogue():
    repo = Path(__file__).resolve().parents[2]
    codes = {p["code"] for p in json.loads((repo / "shared" / "problems.json").read_text(encoding="utf-8"))["problems"]}
    for code in (problems.DRIVER_TOO_OLD, problems.FOLDER_NOT_WRITABLE, problems.DISK_LOW,
                 problems.CLEANUP_FILES_DAMAGED, problems.DOWNLOAD_HOSTS_UNREACHABLE,
                 problems.CLEANUP_SERVER_MISSING, problems.SPEECH_FILES_DAMAGED):
        assert code in codes


def test_the_engine_reports_quick_checks_and_answers_a_full_check(monkeypatch):
    """Over the real protocol: quick results in the status, a full check on request."""
    import time

    from test_connection import greeted, wait_for
    import localflow.service.engine as eng
    import localflow.service.server as srv
    from localflow.config import Config
    from test_service import FakeSTT

    monkeypatch.setattr(eng, "build_transcriber", lambda cfg: FakeSTT())
    monkeypatch.setattr(Config, "save", lambda self, *a, **k: None)
    monkeypatch.setattr(S, "check_hosts", lambda *a, **k: S.Check("hosts", "Download sites", "ok", "offline test"))
    cfg = Config()
    cfg.postprocess.llm_cleanup = False
    _t, server, url = srv.serve_in_thread(cfg)
    port = int(url.rsplit(":", 1)[1])
    deadline = time.monotonic() + 30
    while server.engine.state != "ready" or not server.engine.checks:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    ws = greeted(server, port)
    ws.send(json.dumps({"type": "status.get"}))
    status = wait_for(ws, lambda m: m["type"] == "status")
    assert {c["id"] for c in status["checks"]} == {"driver", "models_folder", "disk", "cleanup_server"}
    ws.send(json.dumps({"type": "selfcheck.run", "id": "c1", "full": True}))
    result = wait_for(ws, lambda m: m["type"] == "selfcheck.result", 120)
    ids = [c["id"] for c in result["checks"]]
    assert result["id"] == "c1" and "hosts" in ids and "cleanup_files" in ids
    ws.send(json.dumps({"type": "selfcheck.repair", "id": "r1"}))
    assert wait_for(ws, lambda m: m["type"] == "selfcheck.repaired")["removed"] == []
    ws.close()
    server._stop.set()
