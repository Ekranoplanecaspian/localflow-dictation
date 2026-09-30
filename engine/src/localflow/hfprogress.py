"""Byte-level progress out of huggingface_hub downloads, for the Hub's progress bars.

huggingface_hub reports progress only through tqdm bars, and drives several at once: one
counting files, one for bytes received and one for bytes written. `reporter()` returns a tqdm
stand-in that forwards the two byte bars as one steady progress.

Bytes received move the bar: a Xet download updates that bar about ten times a second, and its
bytes-written bar only as each large chunk is reconstructed - 7 updates in 51 s for a 0.6 GB
model (measured 2026-10-01), a bar that jumped a sixth at a time. Only bytes written can finish
it: what has arrived is not yet on disk, so received bytes stop one short of the total. The
progress reported never goes backwards.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)

Progress = Callable[[int, int], None]  # (bytes done, bytes total)

# The total grows as snapshot_download registers each file, and the small ones (config, vocab)
# finish first: reporting before the weights are counted would flash 100 %.
FLOOR = 16 << 20


def _role(kwargs: dict) -> str | None:
    """Which bar this is: "written" (bytes on disk), "received" (bytes off the network), or None."""
    desc = str(kwargs.get("desc") or "")
    if desc.startswith("Reconstructing") or desc.endswith("reconstructing file"):
        return "written"  # snapshot_download's aggregate bar, or a single Xet file being written
    if "downloading bytes" in desc.lower():
        return "received"
    if kwargs.get("unit") == "B" and not desc.startswith("Fetching"):
        return "written"  # a single plain HTTP file
    return None


def reporter(report: Progress):
    from tqdm.auto import tqdm

    latest = {"written": (0, 0), "received": (0, 0)}
    shown = [0.0]  # the fraction last reported

    def combined() -> tuple[int, int] | None:
        wd, wt = latest["written"]
        rd, rt = latest["received"]
        written = wd / wt if wt else 0.0
        received = min(rd, rt - 1) / rt if rt else 0.0
        if written >= received:
            done, total, fraction = wd, wt, written
        else:
            done, total, fraction = min(rd, rt - 1), rt, received
        if not total or fraction <= shown[0]:
            return None  # nothing new, or behind what was already shown
        shown[0] = fraction
        return done, total

    class Reporter(tqdm):
        def __init__(self, *args, **kwargs):
            self._role = _role(kwargs)
            self._done = int(kwargs.get("initial") or 0)
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            self._done += int(n or 0)
            if self._role and self.total and self.total >= FLOOR:
                latest[self._role] = (self._done, int(self.total))
                now = combined()
                if now is not None:
                    try:
                        report(*now)
                    except Exception:
                        log.debug("progress report failed", exc_info=True)
            return True

    return Reporter
