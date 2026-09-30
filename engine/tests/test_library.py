"""The models on this PC (library.py): the disk each takes, and removing one without breaking
another that shares its files. A made-up Hugging Face cache; nothing goes online."""

import pytest

from localflow import library
from localflow.stt import catalogue

COMMIT = "0" * 40
REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / ("models--" + REPO.replace("/", "--"))
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(COMMIT)
    snap = repo / "snapshots" / COMMIT
    snap.mkdir(parents=True)

    def put(*names, size=1000):
        for n in names:
            (snap / n).write_bytes(b"x" * size)

    return repo, snap, put


V3, COMPACT = catalogue.get("parakeet-v3"), catalogue.get("parakeet-v3-compact")
SHARED = ("config.json", "vocab.txt", "nemo128.onnx")


def test_disk_is_what_the_models_files_take(cache):
    _, _, put = cache
    put(*SHARED, size=10)
    put("encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx", size=1000)
    assert library.speech_disk_bytes(V3) == 30 + 3000
    assert library.speech_disk_bytes(COMPACT) == 30  # only the shared files are there


def test_removing_one_keeps_what_a_sibling_on_disk_still_needs(cache):
    repo, snap, put = cache
    put(*SHARED, size=10)
    put("encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx", size=1000)
    put("encoder-model.int8.onnx", "decoder_joint-model.int8.onnx", size=500)
    assert catalogue.is_installed(V3, "cpu") and catalogue.is_installed(COMPACT, "cpu")
    freed = library.remove_speech(COMPACT)
    assert freed == 1000
    assert catalogue.is_installed(V3, "cpu") and not catalogue.is_installed(COMPACT, "cpu")
    assert sorted(p.name for p in snap.iterdir()) == sorted(
        [*SHARED, "encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx"])


def test_the_last_model_of_a_repository_takes_its_folder_with_it(cache):
    repo, _, put = cache
    put(*SHARED, size=10)
    put("encoder-model.onnx", "encoder-model.onnx.data", "decoder_joint-model.onnx", size=1000)
    assert library.remove_speech(V3) == 3030 + len(COMMIT)  # the files, and refs/main
    assert not repo.exists()


def test_a_clean_up_model_is_one_file(tmp_path, monkeypatch):
    from localflow.llm import manifest as M

    monkeypatch.setattr(M, "gguf_dir", lambda: tmp_path)
    path = M.gguf_path("phi-4-mini")
    path.write_bytes(b"x" * 2000)
    meta = tmp_path / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / f"{path.name}.metadata").write_text("commit\netag\n")
    assert library.cleanup_disk_bytes("phi-4-mini") == 2000
    assert library.remove_cleanup("phi-4-mini") == 2000
    assert not path.exists() and not (meta / f"{path.name}.metadata").exists()
    assert library.cleanup_disk_bytes("phi-4-mini") == 0
