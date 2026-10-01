"""The resumable download: a partial file is finished, and a whole one is not asked for again."""

from __future__ import annotations

import hashlib
import http.server
import threading

import pytest

from localflow.llm.downloader import download

BODY = bytes(range(256)) * 400  # 100 KB


class _Ranged(http.server.BaseHTTPRequestHandler):
    """Serves BODY, honouring `Range: bytes=N-` the way GitHub's release storage does: 416 when
    nothing is left after N."""

    requests: list[str | None] = []

    def do_GET(self):  # noqa: N802 (the handler's own name)
        wanted = self.headers.get("Range")
        type(self).requests.append(wanted)
        start = int(wanted.split("=")[1].rstrip("-")) if wanted else 0
        if start >= len(BODY):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{len(BODY)}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        rest = BODY[start:]
        self.send_response(206 if wanted else 200)
        self.send_header("Content-Length", str(len(rest)))
        self.end_headers()
        self.wfile.write(rest)

    def log_message(self, *args):  # quiet
        pass


@pytest.fixture
def server():
    _Ranged.requests = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Ranged)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/llama.zip"
    httpd.shutdown()


SHA = hashlib.sha256(BODY).hexdigest()


def test_a_partial_download_is_finished(server, tmp_path):
    dest = tmp_path / "llama.zip"
    (tmp_path / "llama.zip.part").write_bytes(BODY[:1000])
    download(server, dest, sha256=SHA)
    assert dest.read_bytes() == BODY
    assert _Ranged.requests == ["bytes=1000-"]


def test_a_whole_partial_file_is_used_not_asked_for_again_for_ever(server, tmp_path):
    # Stopped after the last chunk, before the move into place: every start after used to get
    # 416 for the bytes after the end, and the clean-up runtime never arrived.
    dest = tmp_path / "llama.zip"
    (tmp_path / "llama.zip.part").write_bytes(BODY)
    download(server, dest, sha256=SHA)
    assert dest.read_bytes() == BODY
    assert not (tmp_path / "llama.zip.part").exists()


def test_a_partial_file_that_is_too_long_or_wrong_is_downloaded_again(server, tmp_path):
    dest = tmp_path / "llama.zip"
    (tmp_path / "llama.zip.part").write_bytes(b"x" * (len(BODY) + 10))
    download(server, dest, sha256=SHA)
    assert dest.read_bytes() == BODY
    assert _Ranged.requests == [f"bytes={len(BODY) + 10}-", None]
