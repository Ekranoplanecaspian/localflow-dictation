"""The CUDA libraries an NVIDIA PC downloads on first run (B2). Nothing here goes online: the
wheels are made up on the spot."""

import importlib
import re
import sys
import zipfile

import pytest

from localflow import cudalibs


def test_every_wheel_is_pinned_and_verified():
    for w in cudalibs.WHEELS:
        assert re.fullmatch(r"[0-9a-f]{64}", w.sha256), w.name
        assert w.url.endswith("/" + w.name) and w.url.startswith("https://files.pythonhosted.org/")
        assert w.name.endswith("-win_amd64.whl") and w.size > 0
    assert not any("curand" in w.name for w in cudalibs.WHEELS), "never loaded: left out"
    assert 900 << 20 < cudalibs.DOWNLOAD_BYTES < 1200 << 20


@pytest.fixture
def fake_wheels(tmp_path, monkeypatch):
    """Two small wheels laid out like NVIDIA's, served by a `download` that copies them."""
    source = tmp_path / "pypi"
    source.mkdir()
    wheels = []
    for name, dlls in (("a.whl", ["nvidia/cu13/bin/x86_64/cudart64_13.dll", "nvidia/cu13/include/x.h"]),
                       ("b.whl", ["nvidia/cudnn/bin/cudnn64_9.dll", "nvidia/cudnn/bin/cudnn_ops64_9.dll"])):
        with zipfile.ZipFile(source / name, "w") as z:
            for d in dlls:
                z.writestr(d, b"MZ" + d.encode())
        wheels.append(cudalibs.Wheel(name, f"https://files.pythonhosted.org/x/{name}", (source / name).stat().st_size, "0" * 64))
    monkeypatch.setattr(cudalibs, "WHEELS", tuple(wheels))
    monkeypatch.setattr(cudalibs, "DOWNLOAD_BYTES", sum(w.size for w in wheels))
    monkeypatch.setattr(cudalibs, "lib_dir", lambda: tmp_path / "bin" / "cuda" / "tag")
    monkeypatch.setattr(cudalibs, "_archive_dir", lambda: tmp_path / "bin" / "cuda" / "downloads")
    fetched = []

    def download(url, dest, progress=None, sha256=None):
        fetched.append(dest.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = (source / dest.name).read_bytes()
        dest.write_bytes(data)
        if progress:
            progress(dest.name, len(data), len(data))
        return dest

    import localflow.llm.downloader as downloader
    import localflow.net as net

    monkeypatch.setattr(downloader, "download", download)
    monkeypatch.setattr(net, "wait_for_proxy", lambda timeout=15.0: None)
    monkeypatch.setattr(net, "ensure_space", lambda folder, needed, what: None)
    return fetched


def test_the_dlls_are_unpacked_flat_and_the_wheels_removed(fake_wheels):
    seen = []
    folder = cudalibs.ensure(lambda done, total: seen.append((done, total)))
    assert sorted(p.name for p in folder.iterdir()) == [".ok", "cudart64_13.dll", "cudnn64_9.dll", "cudnn_ops64_9.dll"]
    assert cudalibs.installed()
    assert not cudalibs._archive_dir().exists()
    assert seen[-1][0] == seen[-1][1] == cudalibs.DOWNLOAD_BYTES  # progress runs across all wheels
    assert fake_wheels == ["a.whl", "b.whl"]


def test_installed_libraries_are_not_fetched_again(fake_wheels):
    cudalibs.ensure()
    cudalibs.ensure()
    assert fake_wheels == ["a.whl", "b.whl"]


def test_a_wheel_already_downloaded_is_not_fetched_again(fake_wheels):
    """An interrupted first run: what arrived is kept, and the rest is fetched."""
    archives = cudalibs._archive_dir()
    archives.mkdir(parents=True)
    (archives / "a.whl").write_bytes((archives.parent.parent.parent / "pypi" / "a.whl").read_bytes())
    cudalibs.ensure()
    assert fake_wheels == ["b.whl"]


def test_a_virtualenv_uses_its_own_packages_unless_told_not_to(tmp_path, monkeypatch):
    # The `nvidia` packages as pip lays them out in a virtualenv with ".[gpu]" - made up here,
    # so the test does not depend on this one having them (a CI runner does not).
    (tmp_path / "nvidia" / "cu13").mkdir(parents=True)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "nvidia", raising=False)
    importlib.invalidate_caches()
    assert cudalibs.bundled()
    monkeypatch.setenv(cudalibs.FROM_DOWNLOAD_ENV, "1")
    assert not cudalibs.bundled()


def test_without_the_libraries_speech_says_why_it_is_on_the_processor(monkeypatch):
    import onnxruntime as ort

    import localflow.stt.parakeet as parakeet

    monkeypatch.setattr(parakeet, "_cuda_probe", None)
    # the engine as it ships: onnxruntime-gpu, whether or not this virtualenv has it
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(cudalibs, "available", lambda: False)
    ok, reason = parakeet.cuda_available()
    assert not ok and reason == "the graphics card libraries are not downloaded yet"
