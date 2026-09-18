"""Engine core: owns the models and the worker threads; sessions submit work in order.

Two single-threaded workers, because neither model is thread-safe and both share one GPU:
  stt  - voice activity detection and every speech decode, strictly in order
  llm  - clean-up and prompt pre-fill, so a 200 ms language-model call never delays the
         next dictation's live decoding

How a session produces text (learned the hard way on an RTX 4060 laptop)
  * The GPU idles at 210 MHz and only reaches full clocks under continuous load: 5 s of
    audio decodes in 45 ms when the GPU is busy, ~90 ms when it is merely warm, 350 ms cold.
  * onnxruntime pays ~35 ms whenever the encoder's internal sequence length differs from the
    previous call; a difference under one 80 ms frame is free.
  * Stitching per-phrase transcriptions reads badly: each phrase gets sentence punctuation
    ("Doctor. Okonko to the calendar. Invite.").
  So while the key is held the worker re-decodes the whole current chunk of audio back to
  back ("live" decodes). That is the live preview text, it keeps the GPU clocked up, and it
  leaves the final decode within a frame of the last live one, so the final costs ~45 ms.
  Takes longer than ~22 s are split at phrase boundaries (from the VAD) into chunks that are
  finalised as they close. On battery, or on the CPU, live decodes run every ~1.5 s instead
  of continuously.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from localflow import __version__
from localflow.cleanup.command import CommandResult, CommandRunner
from localflow.cleanup.joining import join
from localflow.cleanup.pipeline import CleanupPipeline
from localflow.config import Config, PostProcessConfig
from localflow.service import protocol as P
from localflow.service.vad import SAMPLE_RATE, Segment, Segmenter, SileroVad
from localflow.stt import Transcriber, build_transcriber

log = logging.getLogger(__name__)

EventSink = Callable[[dict[str, Any]], None]

LIVE_MIN_S = 0.3  # do not decode less than this
LIVE_PERIOD_S = 1.5  # periodic mode: re-decode the current chunk this often
BUCKET_S = 1.0  # decode inputs are zero-padded to a multiple of this so consecutive decodes share a shape
REUSE_LIVE_MS = 100  # the last live decode stands as the final if no more than this much (silent) audio followed it
CHUNK_TARGET_S = 22.0  # start looking for a phrase boundary to close the chunk after this
CHUNK_MIN_S = 10.0  # ... but never make a chunk shorter than this
CHUNK_HARD_S = 26.0  # cut here if no phrase boundary showed up (model limit is ~30 s)
PREFILL_AFTER_S = 1.5  # warm the language model's prompt cache once the take is this long

QUIET_PEAK = 10 ** (-30 / 20)  # below this peak (-30 dBFS) the take counts as whispered
TARGET_PEAK = 10 ** (-6 / 20)  # ... and is brought up to -6 dBFS
MAX_GAIN = 10 ** (24 / 20)  # never more than +24 dB (noise floor)
DC_THRESHOLD = 0.005  # only remove a DC offset that is actually there


def condition(audio: np.ndarray) -> np.ndarray:
    """Whisper mode: lift genuinely quiet takes so the model sees a normal level.

    Deliberately conservative: Parakeet is level-robust, and touching normal laptop-mic audio
    (peaks around -19 to -30 dBFS) measurably *hurt* accuracy (4 extra errors in 381 words),
    because a growing live-decode slice got a slightly different transform every time."""
    if audio.size == 0:
        return audio
    x = audio
    mean = float(x.mean())
    if abs(mean) > DC_THRESHOLD:
        x = x - mean
    peak = float(np.max(np.abs(x)))
    if 0.0 < peak < QUIET_PEAK:
        x = x * min(TARGET_PEAK / peak, MAX_GAIN)
    return x.astype(np.float32, copy=False)


@dataclass
class Timings:
    started: float = field(default_factory=time.perf_counter)
    ended: float | None = None
    stt_live_ms: float = 0.0
    stt_final_ms: float = 0.0
    post_ms: float = 0.0
    finalised: float | None = None


class Session:
    """One dictation: audio in, partial text while it lasts, final text at the end."""

    def __init__(self, session_id: str, engine: "Engine", context: dict[str, Any], emit: EventSink,
                 language: str | None = None):
        self.id = session_id
        self.engine = engine
        self.context = context
        self.emit = emit
        self.language = language
        self.mode = engine.live_mode()  # continuous | periodic
        self.segmenter = Segmenter(engine.vad)
        self._audio = np.zeros(0, dtype=np.float32)
        self.samples = 0
        self.closed: list[Segment] = []
        self.chunk_start = 0
        self.finalized: list[str | None] = []  # closed chunks, in order (None until decoded)
        self.live_text = ""
        self.live_decodes = 0
        self._live_pending = False
        self._live_at = 0  # samples at the last live submission
        self._live_done_at = 0  # samples covered by the last *completed* live decode
        self._prefilled = False
        self.reused_live = False
        self.timings = Timings()
        self.cancelled = False
        self.ending = False
        self.done = False
        self._jobs: list[Future] = []
        if self.mode == "continuous":
            self._jobs.append(engine.submit(self._live))  # cold-start cost hides inside the hold

    @property
    def app(self) -> str:
        return str((self.context or {}).get("app") or "")

    @property
    def title(self) -> str:
        return str((self.context or {}).get("title") or "")

    @property
    def profile_override(self) -> str:
        """A per-app style rule chosen by the user, or "" to let the engine guess."""
        return str((self.context or {}).get("profile") or "")

    @property
    def before_caret(self) -> str:
        """Text immediately to the left of the caret, when UI Automation could read it."""
        return str((self.context or {}).get("before_caret") or "")

    # audio ---------------------------------------------------------------------------------
    def feed(self, pcm16: bytes) -> None:
        if self.cancelled or self.done or self.ending:
            return
        block = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        self._audio = np.concatenate([self._audio, block]) if self._audio.size else block
        self.samples = self._audio.size
        self.closed += self.segmenter.feed(block)
        boundary = self._chunk_boundary()
        if boundary is not None:
            self._finalize_chunk(boundary)
        if self.mode == "periodic" and not self._live_pending and self.samples - self._live_at >= LIVE_PERIOD_S * SAMPLE_RATE:
            self._live_pending = True
            self._jobs.append(self.engine.submit(self._live))

    def _slice(self, start: int, end: int) -> np.ndarray:
        """Audio [start, end), conditioned and zero-padded up to the next BUCKET_S multiple.
        onnxruntime's CUDA path pays ~35 ms whenever the encoder's input length differs from the
        previous call, so keeping consecutive live decodes (and the final) on one shape is worth
        a little silence."""
        n = end - start
        bucket = int(BUCKET_S * SAMPLE_RATE)
        padded = ((n + bucket - 1) // bucket) * bucket
        out = np.zeros(max(padded, bucket), dtype=np.float32)
        out[:n] = condition(self._audio[start:end])
        return out

    # chunking (long takes) ---------------------------------------------------------------------
    def _chunk_boundary(self) -> int | None:
        """Where to close the current chunk, if it has grown past the target: the start of the
        latest closed phrase that still leaves a chunk of at least CHUNK_MIN_S; or a hard cut."""
        length = self.samples - self.chunk_start
        if length < CHUNK_TARGET_S * SAMPLE_RATE:
            return None
        for seg in reversed(self.closed):
            if seg.start - self.chunk_start >= CHUNK_MIN_S * SAMPLE_RATE and seg.start < self.samples:
                return seg.start
        if length >= CHUNK_HARD_S * SAMPLE_RATE:
            return self.chunk_start + int(CHUNK_HARD_S * SAMPLE_RATE)
        return None

    def _finalize_chunk(self, boundary: int) -> None:
        index = len(self.finalized)
        self.finalized.append(None)
        audio = self._slice(self.chunk_start, boundary)
        self.chunk_start = boundary
        self._live_at = boundary
        self.live_text = ""

        def job() -> None:
            if self.cancelled:
                return
            t0 = time.perf_counter()
            self.finalized[index] = self.engine.stt.transcribe(audio, language=self.language)
            self.timings.stt_live_ms += (time.perf_counter() - t0) * 1000
            self._emit_partial()

        self._jobs.append(self.engine.submit(job))

    # live decode -----------------------------------------------------------------------------------
    def _live(self) -> None:
        try:
            if self.cancelled or self.done or self.ending:
                return
            a, b = self.chunk_start, self.samples
            if b - a >= LIVE_MIN_S * SAMPLE_RATE:
                self._live_at = b
                t0 = time.perf_counter()
                self.live_text = self.engine.stt.transcribe(self._slice(a, b), language=self.language)
                self.timings.stt_live_ms += (time.perf_counter() - t0) * 1000
                self.live_decodes += 1
                self._live_done_at = b
                self._emit_partial()
                self._maybe_prefill(b - a)
            elif self.mode == "continuous":
                self.engine.stt.warm()  # nothing to decode yet: keep the clocks up anyway
        finally:
            self._live_pending = False
            if self.mode == "continuous" and not (self.cancelled or self.done or self.ending):
                self.engine.submit(self._live)

    def _maybe_prefill(self, chunk_samples: int) -> None:
        """Once, mid-utterance, hand the language model what we have so far so its prompt is
        already evaluated when the key is released. Runs on the llm worker: it must never
        delay a speech decode."""
        if self._prefilled or not self.engine.cleanup.cfg.llm_prefill:
            return
        if chunk_samples < PREFILL_AFTER_S * SAMPLE_RATE:
            return
        text = self._text_so_far(self.live_text)
        if not text:
            return
        self._prefilled = True
        self.engine.submit_llm(lambda: self.engine.cleanup.prefill(text, self.app, self.title))

    def _text_so_far(self, tail: str) -> str:
        return " ".join(t.strip() for t in [*self.finalized, tail] if t and t.strip())

    def _emit_partial(self) -> None:
        self.emit({"type": P.PARTIAL, "id": self.id, "text": self._text_so_far(self.live_text),
                   "chunks": len(self.finalized), "phrases": len(self.closed)})

    # end ----------------------------------------------------------------------------------------------
    def end(self) -> None:
        if self.cancelled or self.done:
            return
        self.ending = True
        self.timings.ended = time.perf_counter()
        a, b = self.chunk_start, self.samples
        tail_audio = self._slice(a, b) if b - a >= LIVE_MIN_S * SAMPLE_RATE else None
        # If the live decode that is finishing right now covered all but the last few silent
        # frames, its text is the final text: skip the extra decode.
        quiet_tail = not self.segmenter.speech_within_last(REUSE_LIVE_MS + 40)

        def decode_tail() -> None:
            """Speech worker: finish the transcript, then hand clean-up to the llm worker."""
            if self.cancelled:
                return
            try:
                t0 = time.perf_counter()
                if tail_audio is None:
                    tail = ""
                elif quiet_tail and b - self._live_done_at <= REUSE_LIVE_MS * SAMPLE_RATE // 1000 and self.live_text:
                    tail = self.live_text
                    self.reused_live = True
                else:
                    tail = self.engine.stt.transcribe(tail_audio, language=self.language)
                self.timings.stt_final_ms = (time.perf_counter() - t0) * 1000
                raw = self._text_so_far(tail)
                self.engine.submit_llm(lambda: self._clean_and_emit(raw))
            except Exception as e:
                log.exception("final decode failed")
                self.emit(P.error("finalize_failed", str(e), self.id))
                self.done = True

        self._jobs.append(self.engine.submit(decode_tail))

    def _clean_and_emit(self, raw: str) -> None:
        try:
            t0 = time.perf_counter()
            result = self.engine.cleanup.process(raw, self.app, self.title, self.before_caret,
                                                 self.profile_override)
            # Clean-up writes every take as a whole sentence. Fit it to where it is actually
            # going: no capital in the middle of a clause, and a space if one is missing.
            text = join(self.before_caret, result.text)
            tm = self.timings
            tm.post_ms = (time.perf_counter() - t0) * 1000
            tm.finalised = time.perf_counter()
            self.emit({"type": P.FINAL, "id": self.id, "raw": raw, "text": text, "timings": {
                "audio_s": round(self.samples / SAMPLE_RATE, 2),
                "mode": self.mode,
                "chunks": len(self.finalized) + 1,
                "phrases": len(self.closed),
                "live_decodes": self.live_decodes,
                "reused_live": self.reused_live,
                "stt_live_ms": round(tm.stt_live_ms),
                "stt_final_ms": round(tm.stt_final_ms),
                "post_ms": round(tm.post_ms, 1),
                "release_to_final_ms": round((tm.finalised - tm.ended) * 1000) if tm.ended else None,
                **result.as_dict(),
            }})
        except Exception as e:
            log.exception("clean-up failed")
            self.emit(P.error("finalize_failed", str(e), self.id))
        finally:
            self.done = True

    def cancel(self) -> None:
        self.cancelled = True
        self.done = True


class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")
        self._llm_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm")
        self.stt: Transcriber | None = None
        self.vad: SileroVad | None = None
        self.cleanup = CleanupPipeline(cfg.postprocess, None)  # rules only until the model is up
        self.state = "loading"
        self.error: str | None = None
        self.llm_state = "off"
        self.llm_error: str | None = None
        self.llm_server = None
        self._status_listeners: list[EventSink] = []
        self._lock = threading.Lock()

    # lifecycle -------------------------------------------------------------------------------
    def load(self) -> None:
        """Load models. Speech first, on the speech worker; the clean-up model loads in
        parallel on its own thread so dictation works (rules only) while it downloads."""
        def job() -> None:
            try:
                t0 = time.perf_counter()
                self.vad = SileroVad()
                self.stt = build_transcriber(self.cfg.stt)
                self.stt.warmup()
                self.state = "ready"
                log.info("Engine ready in %.1fs", time.perf_counter() - t0)
            except Exception as e:
                self.state = "error"
                self.error = str(e)
                log.exception("engine failed to load")
            self._broadcast_status()

        self._pool.submit(job)
        if self.cfg.postprocess.llm_cleanup:
            threading.Thread(target=self._load_llm, name="llm-load", daemon=True).start()

    def _load_llm(self) -> None:
        from localflow.llm.providers import build_provider

        pp = self.cfg.postprocess
        self.llm_state, self.llm_error = "loading", None
        self._broadcast_status()
        try:
            t0 = time.perf_counter()

            def server_factory():
                from localflow.llm.server import LlamaServer

                if self.llm_server is None or not self.llm_server.alive():
                    self.llm_server = LlamaServer(model_key=pp.llm_model, device=self.cfg.stt.device)
                    self.llm_server.start(progress=self._download_progress)
                return self.llm_server

            provider = build_provider(pp, server_factory=server_factory)
            self.cleanup = CleanupPipeline(pp, provider)
            self.llm_state = "ready"
            log.info("Clean-up model ready in %.1fs (%s:%s)", time.perf_counter() - t0, pp.llm_provider, pp.llm_model)
        except Exception as e:
            self.llm_state, self.llm_error = "error", str(e)
            log.warning("Clean-up model unavailable (%s); dictation will use the rule-based clean-up", e)
        self._broadcast_status()

    def _download_progress(self, name: str, done: int, total: int | None) -> None:
        if total and done and (done == total or done % (64 << 20) < (1 << 20)):
            log.info("downloading %s: %.0f%% of %.1f GB", name, 100 * done / total, total / 2**30)

    def submit(self, fn: Callable[[], Any]) -> Future:
        return self._pool.submit(fn)

    def run_command(self, selection: str, instruction: str) -> CommandResult:
        """Apply a spoken instruction to selected text (command mode).

        The runner is built per call rather than held, so that changing the provider from the
        Hub takes effect on the next command instead of the next restart.
        """
        return CommandRunner(self.cleanup.provider).run(selection, instruction)

    def submit_llm(self, fn: Callable[[], Any]) -> Future:
        return self._llm_pool.submit(fn)

    def live_mode(self) -> str:
        """continuous (GPU on mains, or forced) or periodic."""
        if getattr(self.stt, "device", "cpu") != "cuda":
            return "periodic"
        mode = self.cfg.stt.gpu_keep_warm
        if mode == "never":
            return "periodic"
        if mode == "always":
            return "continuous"
        from localflow.power import on_ac_power

        return "continuous" if on_ac_power() else "periodic"

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._llm_pool.shutdown(wait=False, cancel_futures=True)
        if self.llm_server is not None:
            self.llm_server.stop()

    # status -------------------------------------------------------------------------------------
    def add_status_listener(self, sink: EventSink) -> None:
        with self._lock:
            self._status_listeners.append(sink)

    def remove_status_listener(self, sink: EventSink) -> None:
        with self._lock:
            if sink in self._status_listeners:
                self._status_listeners.remove(sink)

    def _broadcast_status(self) -> None:
        msg = self.status()
        with self._lock:
            sinks = list(self._status_listeners)
        for s in sinks:
            try:
                s(msg)
            except Exception:
                log.exception("status listener failed")

    def status(self) -> dict[str, Any]:
        stt = self.stt
        pp = self.cfg.postprocess
        return {
            "type": P.STATUS,
            "version": __version__,
            "pid": os.getpid(),
            "stt": {
                "state": self.state,
                "error": self.error,
                "backend": self.cfg.stt.backend,
                "model": self.cfg.stt.model,
                "device": getattr(stt, "device", None),
                "precision": getattr(stt, "precision", None),
            },
            "llm": {
                "state": self.llm_state,
                "error": self.llm_error,
                "enabled": pp.llm_cleanup,
                "provider": pp.llm_provider,
                "model": pp.llm_model,
            },
        }

    # sessions -------------------------------------------------------------------------------------
    def start_session(self, session_id: str, context: dict[str, Any], emit: EventSink,
                      language: str | None = None) -> Session:
        if self.state != "ready" or self.stt is None or self.vad is None:
            raise RuntimeError(f"engine not ready ({self.state}{': ' + self.error if self.error else ''})")
        return Session(session_id, self, context, emit, language)

    def apply_settings(self, settings: dict[str, Any]) -> None:
        pp = settings.get("postprocess")
        if not isinstance(pp, dict):
            return
        before = self.cfg.postprocess
        merged = {**before.__dict__, **pp}
        after = PostProcessConfig(**{k: v for k, v in merged.items() if k in PostProcessConfig.__dataclass_fields__})
        self.cfg.postprocess = after
        # Settings changed from the Hub have to survive a restart, or the user changes them
        # once and wonders why they came back.
        try:
            self.cfg.save()
        except Exception as e:
            log.warning("could not save settings: %s", e)
        # keep the current provider unless the model or provider itself changed
        restart = (after.llm_cleanup != before.llm_cleanup or after.llm_provider != before.llm_provider
                   or after.llm_model != before.llm_model)
        self.cleanup = CleanupPipeline(after, None if restart else self.cleanup.provider)
        log.info("clean-up settings updated%s", " (reloading the model)" if restart else "")
        if restart:
            if self.llm_server is not None:
                self.llm_server.stop()
                self.llm_server = None
            if after.llm_cleanup:
                threading.Thread(target=self._load_llm, name="llm-load", daemon=True).start()
            else:
                self.llm_state, self.llm_error = "off", None
                self._broadcast_status()
