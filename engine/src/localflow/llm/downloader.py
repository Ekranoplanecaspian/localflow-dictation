"""Resumable, verified downloads with progress callbacks."""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

Progress = Callable[[str, int, int | None], None]  # (name, bytes_done, bytes_total)


def download(url: str, dest: Path, progress: Progress | None = None, sha256: str | None = None) -> Path:
    """Download `url` to `dest`, resuming a partial file, verifying the checksum if given."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "LocalFlow"})
    if done:
        req.add_header("Range", f"bytes={done}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if not (done and e.code == 416):
            raise
        # Nothing left after the bytes already here: the partial file is whole, or longer than
        # the file. LocalFlow stopped between the last chunk and the move into place - while
        # hashing, say - and asking for the rest got this answer at every start after, for good.
        e.close()
        if sha256 and file_sha256(part).lower() == sha256.lower():
            part.replace(dest)
            return dest
        part.unlink(missing_ok=True)
        return download(url, dest, progress, sha256)
    with resp:
        if done and resp.status != 206:  # server ignored the range: start over
            done = 0
            part.unlink(missing_ok=True)
        total = resp.headers.get("Content-Length")
        total_bytes = (int(total) + done) if total else None
        with part.open("ab" if done else "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(dest.name, done, total_bytes)
    if sha256:
        actual = file_sha256(part)
        if actual.lower() != sha256.lower():
            part.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {dest.name}: expected {sha256[:12]}..., got {actual[:12]}...")
    part.replace(dest)
    return dest


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_zip(archive: Path, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = into / Path(member.filename).name  # flatten: all DLLs and exes side by side
            if member.is_dir():
                continue
            with z.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
