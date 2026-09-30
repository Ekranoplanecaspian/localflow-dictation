"""Downloads on real networks (localflow/net.py): the Windows proxy, a mirror, room on the disk,
and a first run that has no connection yet."""

import errno
import http.server
import os
import ssl
import threading
import time
import urllib.error
from collections import namedtuple

import pytest

from localflow import net, problems


def test_proxy_strings_from_windows_settings():
    assert net.parse_proxy_list("proxy.corp:8080") == "http://proxy.corp:8080"
    assert net.parse_proxy_list("http=a:1;https=b:2") == "http://b:2"
    assert net.parse_proxy_list("http=a:1") == "http://a:1"
    assert net.parse_proxy_list("socks=s:1080") is None, "not something an HTTP client can use"
    assert net.parse_proxy_list("") is None
    # Local addresses always bypass it: the clean-up server is on 127.0.0.1.
    bypass = net.parse_bypass("<local>;*.corp.example;10.*")
    assert bypass.split(",")[:3] == ["localhost", "127.0.0.1", "::1"]
    assert ".corp.example" in bypass and "<local>" not in bypass


@pytest.mark.skipif(os.name != "nt", reason="WinHTTP")
def test_an_automatic_configuration_script_is_followed():
    """What company networks use, and what Python's own proxy lookup ignores."""
    pac = b'function FindProxyForURL(url, host) { return "PROXY 10.1.2.3:3128; DIRECT"; }'

    class Pac(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ns-proxy-autoconfig")
            self.send_header("Content-Length", str(len(pac)))
            self.end_headers()
            self.wfile.write(pac)

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Pac)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        proxy, bypass = net.windows_proxy(pac_url=f"http://127.0.0.1:{server.server_address[1]}/p.pac")
    finally:
        server.shutdown()
    assert proxy == "http://10.1.2.3:3128"
    assert "127.0.0.1" in bypass


def test_a_mirror_is_used_by_hugging_face_downloads(monkeypatch):
    from huggingface_hub import constants, hf_hub_url

    monkeypatch.setattr(constants, "ENDPOINT", constants.ENDPOINT)
    monkeypatch.setattr(constants, "HUGGINGFACE_CO_URL_TEMPLATE", constants.HUGGINGFACE_CO_URL_TEMPLATE)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    net.apply_endpoint("https://hf-mirror.example/")
    try:
        assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.example"
        assert hf_hub_url("org/model", "config.json").startswith("https://hf-mirror.example/org/model/")
        assert net.download_host() == "hf-mirror.example"
    finally:
        net.apply_endpoint("")
    assert "HF_ENDPOINT" not in os.environ
    assert hf_hub_url("org/model", "config.json").startswith("https://huggingface.co/")


def test_the_mirror_setting_takes_web_addresses_only():
    from localflow.validate import check_mirror

    assert check_mirror(" https://hf-mirror.com/ ") == ("https://hf-mirror.com", None)
    assert check_mirror("") == ("", None)
    assert check_mirror("ftp://x")[0] is None
    assert check_mirror("hf-mirror.com")[0] is None


def test_a_download_that_would_not_fit_says_how_much_it_needs(monkeypatch, tmp_path):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(net.shutil, "disk_usage", lambda p: usage(100 * 10**9, 99 * 10**9, 1 * 10**9))
    with pytest.raises(net.NotEnoughSpace) as e:
        net.ensure_space(tmp_path / "not" / "there" / "yet", 2 * 10**9, "Parakeet v3")
    assert e.value.errno == errno.ENOSPC
    assert "needs about 2.0 GB" in str(e.value) and "there is 1.0 GB" in str(e.value)
    assert problems.classify_speech(e.value) == problems.SPEECH_NO_SPACE
    assert problems.classify_cleanup(e.value, "bundled") == problems.CLEANUP_NO_SPACE
    net.ensure_space(tmp_path, 100 * 2**20, "small")  # fits


def test_a_wrong_clock_is_named_not_blamed_on_the_network():
    err = urllib.error.URLError(ssl.SSLCertVerificationError(
        1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1010)"))
    assert problems.classify_speech(err) == problems.SPEECH_CLOCK_WRONG
    assert problems.classify_cleanup(err, "bundled") == problems.CLEANUP_CLOCK_WRONG
    assert problems.detail(err).startswith("This PC's clock says ")
    # An ordinary network failure is still one.
    offline = urllib.error.URLError(OSError("getaddrinfo failed"))
    assert problems.classify_speech(offline) == problems.SPEECH_DOWNLOAD_FAILED


def test_a_first_run_without_a_connection_downloads_once_it_is_back(monkeypatch):
    """No connection on the first run: said, and tried again by itself - dictation starts once
    the download goes through, with nobody pressing anything."""
    import localflow.service.engine as eng
    from localflow.config import Config
    from localflow.stt import catalogue
    from test_service import FakeSTT

    monkeypatch.setattr(eng.Engine, "_fetch_speech", eng.Engine._real_fetch_speech)
    monkeypatch.setattr(eng, "build_transcriber", lambda cfg: FakeSTT())
    monkeypatch.setattr(eng, "RETRY_FIRST_S", 0.2)
    monkeypatch.setattr(catalogue, "is_installed", lambda m, d: False)
    monkeypatch.setattr(net, "ensure_space", lambda *a: None)
    attempts: list[float] = []

    def download(model, device, progress=None):
        attempts.append(time.monotonic())
        if len(attempts) < 3:
            raise urllib.error.URLError(OSError("getaddrinfo failed"))
        progress(50, 100)
        progress(100, 100)

    monkeypatch.setattr(catalogue, "download", download)
    cfg = Config()
    cfg.postprocess.llm_cleanup = False
    engine = eng.Engine(cfg)
    seen: list[dict] = []
    engine.add_status_listener(lambda m: seen.append(m))
    engine.load()
    try:
        deadline = time.monotonic() + 20
        while engine.state != "ready" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert engine.state == "ready", engine.error
        assert len(attempts) == 3
        codes = [m["stt"]["error_code"] for m in seen if m.get("stt", {}).get("error_code")]
        assert codes and set(codes) == {problems.SPEECH_DOWNLOAD_FAILED}
        progress = [m["stt"]["download"]["progress"] for m in seen if m.get("stt", {}).get("download")]
        assert 1.0 in progress, "the download's progress was reported"
        assert engine.status()["stt"]["download"] is None, "and cleared once done"
    finally:
        engine.shutdown()


def test_a_wrapped_network_error_reads_as_words():
    e = RuntimeError("Got: ConnectError: [WinError 10061] No connection could be made because the target "
                     "machine actively refused it")
    assert problems.detail(e) == "No connection could be made because the target machine actively refused it."
    assert problems.detail(ValueError("plain words")) == "plain words."


def test_a_clean_up_server_windows_refuses_to_run_is_named_as_blocked():
    blocked = OSError(None, "This program is blocked by group policy", None, 1260)
    assert blocked.winerror == 1260
    assert problems.classify_cleanup(RuntimeError("could not start llama-server"), "bundled") != problems.CLEANUP_BLOCKED
    try:
        raise RuntimeError("could not start llama-server") from blocked
    except RuntimeError as e:
        assert problems.classify_cleanup(e, "bundled") == problems.CLEANUP_BLOCKED
