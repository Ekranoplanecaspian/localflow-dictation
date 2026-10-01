"""Every download the engine makes, in one queue the Hub can show and the user can stop.

Speech and clean-up models, the clean-up runtime (llama.cpp) and the graphics card's CUDA
libraries all used to download on their own threads, each reporting - if at all - in its own
place: the clean-up model's first download of 2.5 GB reported to the log only, so the Hub said
"loading" for minutes (found 2026-09-30 on a wiped PC). Now each download is a job here, run one
at a time, most urgent first:

    first-run   the model this PC needs to dictate at all
    switch      a model the user just chose
    automatic   what LocalFlow fetches by itself (the CUDA libraries, the clean-up runtime)
    library     a model downloaded from the Hub to have ready

A job that is already queued or running is not queued twice: asking again joins it (and a more
urgent reason moves it forward). Each job says what it is, how much of it has arrived, how
quickly, and about how long is left; finished jobs stay in the list for a while so the Hub can
say "done".
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from localflow.fetch import Cancelled

log = logging.getLogger(__name__)

PRIORITY = {"first-run": 0, "switch": 1, "automatic": 2, "library": 3}
KEEP_FINISHED = 6  # finished jobs kept in the list
KEEP_FINISHED_S = 15 * 60
REPORT_EVERY_S = 0.25  # progress reports, at most this often

# A job's work: (progress(done, total), stop) -> None. Raises Cancelled once stop is set.
Runner = Callable[[Callable[[int, int], None], threading.Event], None]


@dataclass
class Job:
    id: str
    kind: str  # speech | cleanup | runtime | gpu-libs
    key: str
    label: str
    size: int  # bytes, as the catalogue gives it, until the download itself says
    reason: str
    run: Runner
    cancellable: bool = True
    state: str = "queued"  # queued | downloading | done | error | cancelled
    done: int = 0
    total: int = 0
    speed: float = 0.0  # bytes per second, smoothed
    error: str | None = None
    exc: BaseException | None = None
    queued_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    listeners: list[Callable[[int, int], None]] = field(default_factory=list)
    stop: threading.Event = field(default_factory=threading.Event)
    ended: threading.Event = field(default_factory=threading.Event)
    _sample: tuple[float, int] | None = None

    @property
    def active(self) -> bool:
        return self.state in ("queued", "downloading")

    def view(self) -> dict[str, Any]:
        total = self.total or self.size
        left = max(0, total - self.done)
        return {
            "id": self.id, "kind": self.kind, "key": self.key, "label": self.label,
            "state": self.state, "reason": self.reason, "cancellable": self.cancellable,
            "done": self.done, "total": total,
            # never 100 % before it is on disk: the last bytes received round up to it while
            # Xet is still writing the file, which took 20 s on a real first run
            "progress": (round(min(1.0 if self.state == "done" else 0.999, self.done / total), 3)
                         if total else 0.0),
            "speed_bps": round(self.speed) if self.state == "downloading" else 0,
            "eta_s": round(left / self.speed) if self.state == "downloading" and self.speed > 1 else None,
            "error": self.error,
            # how long ago it finished, so the Hub can say "done" for a while and then stop
            "ended_s_ago": round(time.monotonic() - self.finished_at) if self.finished_at is not None else None,
        }


class Downloads:
    def __init__(self, on_change: Callable[[], None]):
        self._on_change = on_change
        self._jobs: list[Job] = []
        self._cv = threading.Condition()
        self._ids = itertools.count(1)
        self._thread: threading.Thread | None = None
        self._closing = False
        self._reported = 0.0

    # asking ------------------------------------------------------------------------------------
    def request(self, kind: str, key: str, label: str, size: int, run: Runner, reason: str = "library",
                *, cancellable: bool = True, progress: Callable[[int, int], None] | None = None) -> Job:
        """Queue a download, or join the one already queued or running for the same thing."""
        with self._cv:
            job = next((j for j in self._jobs if j.active and j.kind == kind and j.key == key), None)
            if job is None:
                job = Job(f"d{next(self._ids)}", kind, key, label, int(size), reason, run, cancellable)
                self._jobs.append(job)
                log.info("download queued: %s (%s, %.1f GB)", label, reason, size / 1e9)
            else:
                if PRIORITY.get(reason, 9) < PRIORITY.get(job.reason, 9):
                    job.reason = reason  # more urgent now: moves ahead of what is queued
                # Joined by something LocalFlow cannot do without (the first-run speech model),
                # it can no longer be stopped: cancelling the download the user started from
                # the library cancelled the first run's wait too, and dictation never came.
                job.cancellable = job.cancellable and cancellable
            if progress is not None:
                job.listeners.append(progress)
            if self._thread is None and not self._closing:
                self._thread = threading.Thread(target=self._worker, name="downloads", daemon=True)
                self._thread.start()
            self._cv.notify_all()
        self._changed(force=True)
        return job

    def fetch(self, kind: str, key: str, label: str, size: int, run: Runner, reason: str,
              *, cancellable: bool = True, progress: Callable[[int, int], None] | None = None) -> None:
        """Queue a download and wait for it. Raises what it raised, or Cancelled."""
        job = self.request(kind, key, label, size, run, reason, cancellable=cancellable, progress=progress)
        job.ended.wait()
        if job.state == "cancelled":
            raise Cancelled()
        if job.state == "error":
            raise job.exc if job.exc is not None else RuntimeError(job.error or "the download failed")

    def cancel(self, job_id: str) -> bool:
        """Stop a download, or take it out of the queue. False for one that cannot be stopped
        (LocalFlow's own parts) or is not there."""
        with self._cv:
            job = next((j for j in self._jobs if j.id == job_id and j.active), None)
            if job is None or not job.cancellable:
                return False
            job.stop.set()
            if job.state == "queued":
                self._finish(job, "cancelled")
            self._cv.notify_all()
        log.info("download cancelled: %s", job.label)
        self._changed(force=True)
        return True

    def active(self, kind: str, key: str | None = None) -> Job | None:
        with self._cv:
            return next((j for j in self._jobs if j.active and j.kind == kind and (key is None or j.key == key)), None)

    def views(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._cv:
            self._jobs = [j for j in self._jobs if j.active or now - (j.finished_at or now) < KEEP_FINISHED_S]
            finished = [j for j in self._jobs if not j.active]
            del finished[:-KEEP_FINISHED]
            running = [j for j in self._jobs if j.state == "downloading"]
            queued = sorted((j for j in self._jobs if j.state == "queued"), key=self._order)
            return [j.view() for j in running + queued + finished[::-1]]

    def close(self) -> None:
        with self._cv:
            self._closing = True
            for j in self._jobs:
                j.stop.set()
            self._cv.notify_all()

    # the worker --------------------------------------------------------------------------------------
    @staticmethod
    def _order(job: Job) -> tuple[int, float]:
        return PRIORITY.get(job.reason, 9), job.queued_at

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._closing and not any(j.state == "queued" for j in self._jobs):
                    self._cv.wait()
                if self._closing:
                    for j in self._jobs:
                        if j.active:
                            self._finish(j, "cancelled")
                    return
                job = min((j for j in self._jobs if j.state == "queued"), key=self._order)
                job.state = "downloading"
            self._changed(force=True)
            log.info("downloading %s", job.label)
            try:
                job.run(lambda done, total: self._progress(job, done, total), job.stop)
            except Cancelled:
                outcome, error, exc = "cancelled", None, None
            except Exception as e:
                from localflow import problems

                outcome, error, exc = "error", problems.detail(e), e
                log.warning("download of %s failed: %s", job.label, e)
            else:
                outcome, error, exc = "done", None, None
                log.info("downloaded %s", job.label)
            with self._cv:
                job.error, job.exc = error, exc
                if outcome == "done" and job.total:
                    job.done = job.total
                self._finish(job, outcome)
            self._changed(force=True)

    def _finish(self, job: Job, state: str) -> None:
        job.state, job.finished_at = state, time.monotonic()
        job.ended.set()

    def _progress(self, job: Job, done: int, total: int) -> None:
        now = time.monotonic()
        job.done, job.total = int(done), int(total or job.total)
        if job._sample is not None:
            t0, d0 = job._sample
            if now - t0 >= 0.5:
                rate = max(0.0, (done - d0) / (now - t0))
                job.speed = rate if job.speed == 0 else 0.7 * job.speed + 0.3 * rate
                job._sample = (now, done)
        else:
            job._sample = (now, done)
        for listen in list(job.listeners):
            try:
                listen(done, total)
            except Cancelled:
                raise
            except Exception:
                log.debug("download listener failed", exc_info=True)
        self._changed(force=bool(total) and done >= total)  # the last report always goes out

    def _changed(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._reported < REPORT_EVERY_S:
            return
        self._reported = now
        try:
            self._on_change()
        except Exception:
            log.debug("download status report failed", exc_info=True)
