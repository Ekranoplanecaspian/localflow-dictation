"""Byte-level progress out of huggingface_hub downloads, for the Hub's progress bars.

huggingface_hub reports progress only through tqdm bars, and drives several at once: one
counting files, one for bytes received and one for bytes written. `reporter()` returns a tqdm
stand-in that forwards the bytes-written bar and ignores the rest.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)

Progress = Callable[[int, int], None]  # (bytes done, bytes total)

# The total grows as snapshot_download registers each file, and the small ones (config, vocab)
# finish first: reporting before the weights are counted would flash 100 %.
FLOOR = 16 << 20


def _counts_bytes_written(kwargs: dict) -> bool:
    desc = str(kwargs.get("desc") or "")
    if desc.startswith("Reconstructing") or desc.endswith("reconstructing file"):
        return True  # snapshot_download's aggregate bar, or a single Xet file being written
    if "downloading bytes" in desc.lower():
        return False  # bytes received, which run ahead of what is on disk
    return kwargs.get("unit") == "B" and not desc.startswith("Fetching")  # a single plain HTTP file


def reporter(report: Progress):
    from tqdm.auto import tqdm

    class Reporter(tqdm):
        def __init__(self, *args, **kwargs):
            self._bytes = _counts_bytes_written(kwargs)
            self._done = int(kwargs.get("initial") or 0)
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            self._done += int(n or 0)
            if self._bytes and self.total and self.total >= FLOOR:
                try:
                    report(self._done, int(self.total))
                except Exception:
                    log.debug("progress report failed", exc_info=True)
            return True

    return Reporter
