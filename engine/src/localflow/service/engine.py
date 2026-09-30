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

import gc
import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from localflow import __version__, cudalibs, hwinfo, modelchoice, problems
from localflow.cleanup.command import CommandResult, CommandRunner
from localflow.cleanup.joining import join
from localflow.cleanup.pipeline import CleanupPipeline
from localflow.config import ComputeConfig, Config, PostProcessConfig, STTConfig, for_this_pc, in_safe_mode
from localflow.service import protocol as P
from localflow.service.compute import ComputeController
from localflow.service.vad import SAMPLE_RATE, Segment, Segmenter, SileroVad
from localflow.stt import Transcriber, build_transcriber, catalogue, remote

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
# A take that ends while the clean-up model is waking from an idle unload waits this long for
# it before going out with the rules alone. The model is woken when the take starts, so usually
# it is ready by the time the take ends.
WAKE_WAIT_S = 4.0
# At start-up the clean-up model waits for speech to load, at most this long.
SPEECH_FIRST_WAIT_S = 20.0
# A load that failed for a cause that goes away by itself (no connection, no room, a wrong clock)
# is tried again after this long, doubling each time up to the maximum.
RETRY_FIRST_S = 15.0
RETRY_MAX_S = 300.0

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
    def private(self) -> bool:
        """Spoken into a password field (the shell asks UI Automation). Typed exactly as heard:
        no clean-up, no model, nothing kept."""
        return str((self.context or {}).get("password") or "").lower() in ("true", "1")

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
            self.finalized[index] = self.engine.transcribe(audio, language=self.language)
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
                self.live_text = self.engine.transcribe(self._slice(a, b), language=self.language)
                self.timings.stt_live_ms += (time.perf_counter() - t0) * 1000
                self.live_decodes += 1
                # Our own load on the graphics card. Recorded only at the start and the end of a
                # take, a long take's live decoding looked like another app using the card after
                # eight seconds, and clean-up was moved to the processor mid-take - the take then
                # waited 16 s for its text.
                self.engine.compute.worked()
                self._live_done_at = b
                self._emit_partial()
                self._maybe_prefill(b - a)
            elif self.mode == "continuous":
                self.engine.warm()  # nothing to decode yet: keep the clocks up anyway
        finally:
            self._live_pending = False
            if self.mode == "continuous" and not (self.cancelled or self.done or self.ending):
                self.engine.submit(self._live)

    def _maybe_prefill(self, chunk_samples: int) -> None:
        """Once, mid-utterance, hand the language model what we have so far so its prompt is
        already evaluated when the key is released. Runs on the llm worker: it must never
        delay a speech decode."""
        if self._prefilled or self.private or not self.engine.cleanup.cfg.llm_prefill:
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
        # Once only. A second end used to decode the tail again and emit a second final, and
        # the shell typed that one too: stopping hands-free with the chord sends one end for
        # the press and another for the release.
        if self.cancelled or self.done or self.ending:
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
                    tail = self.engine.transcribe(tail_audio, language=self.language)
                self.timings.stt_final_ms = (time.perf_counter() - t0) * 1000
                raw = self._text_so_far(tail)
                self.engine.submit_llm(lambda: self._clean_and_emit(raw))
            except Exception as e:
                log.exception("final decode failed")
                self.emit(P.error(problems.TAKE_FAILED, str(e), self.id))
                self.done = True

        self._jobs.append(self.engine.submit(decode_tail))

    def _clean_and_emit(self, raw: str) -> None:
        try:
            t0 = time.perf_counter()
            if self.private:
                # A password: exactly what was heard. Filler removal, spoken punctuation, the
                # dictionary and the model could all change it, and it must not leave for a
                # cloud clean-up provider.
                text = raw.strip()
                extra = {"private": True, "used_llm": False, "llm_ms": None, "llm_rejected": None,
                         "profile": None, "dictionary_hits": 0}
            else:
                self.engine.wait_for_waking_cleanup(WAKE_WAIT_S)
                result = self.engine.cleanup.process(raw, self.app, self.title, self.before_caret,
                                                     self.profile_override)
                self.engine.record_cleanup(result)
                # Clean-up writes every take as a whole sentence. Fit it to where it is actually
                # going: no capital in the middle of a clause, and a space if one is missing.
                text = join(self.before_caret, result.text, from_model=result.used_llm)
                extra = result.as_dict()
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
                **extra,
            }})
        except Exception as e:
            log.exception("clean-up failed")
            self.emit(P.error(problems.TAKE_FAILED, str(e), self.id))
        finally:
            self.done = True
            self.engine.compute.worked()

    def cancel(self) -> None:
        self.cancelled = True
        self.done = True


def _net_proxy() -> str | None:
    from localflow import net

    return net.proxy_in_use


class TimedTranscriber:
    """A transcriber that times its real decodes, so the model choice can see how fast speech
    actually is on this machine. Everything else is passed straight through."""

    MIN_AUDIO_S = 1.5  # shorter decodes are dominated by fixed overhead, not the model

    def __init__(self, inner: Transcriber, key: str | None, perf: modelchoice.PerfLog):
        self._inner, self._key, self._perf = inner, key, perf

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        t0 = time.perf_counter()
        text = self._inner.transcribe(audio, language=language)
        seconds = audio.size / SAMPLE_RATE
        # A named language may have gone to Whisper instead: only time the model itself.
        if self._key and language is None and seconds >= self.MIN_AUDIO_S:
            ms = (time.perf_counter() - t0) * 1000
            self._perf.record("speech", self._key, getattr(self._inner, "device", "cpu"), ms / seconds)
        return text


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
        self.error_code: str | None = None  # problems.py; read only while state is "error"
        self.llm_state = "off"
        self.llm_error: str | None = None
        self.llm_error_code: str | None = None  # likewise, while llm_state is "error"
        self.checks: list[dict[str, Any]] = []  # the quick self-checks (selfcheck.py), in status
        self.last_full_check: list = []  # the last full check's results, for "Download again"
        self.llm_server = None
        self.gpu_failures = 0  # speech on the graphics card failed and moved to the processor
        self.first_download: dict[str, Any] | None = None  # the speech model's first download, for the status
        # The CUDA libraries' first download on an NVIDIA PC (B2): {"state": downloading|ready|error,
        # "done", "total"} or {"state": "error", "error"}; None when there is nothing to fetch.
        self.cuda_libs: dict[str, Any] | None = None
        self._retries: dict[str, int] = {}  # tries so far of a load that retries by itself, by what
        self._timers: list[threading.Timer] = []  # those tries still to come
        self._closing = False
        # A speech model change in progress: {"to", "state": downloading|loading|error, "progress", "error"}
        self.speech_switch: dict[str, Any] | None = None
        # The same for the bundled clean-up model.
        self.cleanup_switch: dict[str, Any] | None = None
        self._switch_reported = 0.0
        self._status_listeners: list[EventSink] = []
        self._lock = threading.Lock()
        # Held by anything that replaces a model: the Hub's pickers and the placement
        # controller's moves between the graphics card and the processor.
        self._model_lock = threading.Lock()
        self._waking = False  # the clean-up model is loading again after an idle unload
        self.perf = modelchoice.PerfLog(modelchoice.PERF_PATH)
        self.compute = ComputeController(self)

    # lifecycle -------------------------------------------------------------------------------
    def load(self) -> None:
        """Load models. Speech first, on the speech worker; then the clean-up model, on its own
        thread so dictation works (rules only) meanwhile. Not both at once: loading onto the
        graphics card side by side, speech took 6.7 s instead of 2.5 (2026-09-25)."""
        # Decide where things run before loading anything, so a hot or busy GPU is never
        # loaded up only to be emptied again five seconds later.
        try:
            self.compute.placement = self.compute.decide()
            self.compute.asked = {"speech": self.compute.placement.speech, "cleanup": self.compute.placement.cleanup}
            self._apply_initial_choices()
            log.info("placement: speech on %s, clean-up on %s (%s)", self.compute.placement.speech,
                     self.compute.placement.cleanup, self.compute.placement.reason)
        except Exception:
            log.exception("could not decide where to run the models; using the defaults")

        def job() -> None:
            self._load_speech()
            speech_settled.set()
            self.fetch_cuda_libs()

        def load_llm() -> None:
            # After speech - but a speech model still downloading on a first run does not hold
            # the clean-up model's own download up for long.
            speech_settled.wait(SPEECH_FIRST_WAIT_S)
            self._load_llm()

        speech_settled = threading.Event()
        self._pool.submit(job)
        threading.Thread(target=self._tidy_downloads, name="tidy", daemon=True).start()
        if self.cfg.postprocess.llm_cleanup:
            threading.Thread(target=load_llm, name="llm-load", daemon=True).start()
        self.compute.start()

    def _load_speech(self) -> None:
        """On the speech worker: the voice detector and the speech model, downloading the model
        first if this is its first run. A failure that goes away by itself - no connection yet,
        a full disk, a wrong clock - is tried again by itself, sooner at first, then every five
        minutes, so dictation starts once the cause is gone without anyone pressing a button."""
        try:
            t0 = time.perf_counter()
            self.vad = SileroVad()
            self._fetch_speech()
            # Done downloading: "ready" below must not go out beside a download at 100 %, which
            # stayed in the status until the warm-up decode had finished too.
            self.first_download = None
            self.stt = self._build_stt(self._speech_cfg())
            # Ready before the warm-up decode, not after: this job still holds the speech
            # worker, so a take started meanwhile has its first decode wait for the warm-up,
            # inside the key hold, rather than the whole engine waiting for it (0.3-0.8 s).
            self.state, self.error = "ready", None
            self._retries.pop("speech", None)
            log.info("Engine ready in %.1fs", time.perf_counter() - t0)
            self._broadcast_status()
            self.stt.warmup()
            self.quick_check()
        except Exception as e:
            self.state = "error"
            self.error, self.error_code = problems.detail(e), problems.classify_speech(e)
            log.exception("engine failed to load (%s)", self.error_code)
            if self.error_code in problems.RETRY_BY_ITSELF:
                self._retry(lambda: self._pool.submit(self._load_speech), "speech")
        finally:
            self.first_download = None
        self._broadcast_status()

    def _retry(self, again: Callable[[], Any], what: str) -> None:
        attempt = self._retries.get(what, 0)
        self._retries[what] = attempt + 1
        delay = min(RETRY_MAX_S, RETRY_FIRST_S * 2 ** attempt)
        log.info("trying %s again in %.0f s", what, delay)

        def fire() -> None:
            if not self._closing:
                again()

        timer = threading.Timer(delay, fire)
        timer.daemon = True
        self._timers.append(timer)
        timer.start()

    def _fetch_speech(self) -> None:
        """A first run: download the speech model here, with a check for room first and progress
        in the status, rather than inside the model loader where neither is possible."""
        from localflow import net

        entry = catalogue.current(self.cfg.stt)
        device = self.compute.placement.speech
        if entry is None or catalogue.is_installed(entry, device):
            return
        net.wait_for_proxy()
        size = int(entry.size_gb(device) * 1e9)
        net.ensure_space(net.hf_cache_dir(), size, entry.label)
        log.info("downloading %s (%.1f GB) from %s", entry.label, size / 1e9, net.download_host())
        self.first_download = {"label": entry.label, "progress": 0.0, "size_gb": round(size / 1e9, 1)}
        self._broadcast_status()
        last = [0.0]

        def progress(done: int, total: int) -> None:
            now = time.monotonic()
            if total and (done >= total or now - last[0] > 0.5):
                last[0] = now
                self.first_download = {**(self.first_download or {}), "progress": round(done / total, 3)}
                self._broadcast_status()

        catalogue.download(entry, device, progress=progress)

    # where the models run, and which ------------------------------------------------------------------
    def _build_stt(self, cfg: STTConfig) -> Transcriber:
        """Speech that may use the graphics card is built in a speech worker process (stt/remote.py),
        so that when it leaves the card everything CUDA held leaves with it. Speech on the
        processor is built here."""
        entry = catalogue.current(cfg)
        if cfg.device != "cpu" and remote.enabled():
            try:
                inner: Transcriber = remote.RemoteTranscriber(cfg)
            except remote.WorkerDied as e:
                # The graphics card took the worker down while loading (a driver in trouble):
                # speech on the processor rather than none at all.
                log.warning("the speech worker died loading speech (%s); loading it on the processor", e)
                return TimedTranscriber(build_transcriber(replace(cfg, device="cpu")), entry.key if entry else None,
                                        self.perf)
            if inner.device == "cpu":
                # CUDA does not work here after all: no reason to keep a second process for it.
                inner.close()
                inner = build_transcriber(replace(cfg, device="cpu"))
        else:
            remote.discard_spare()
            inner = build_transcriber(cfg)
        return TimedTranscriber(inner, entry.key if entry else None, self.perf)

    @staticmethod
    def _close_stt(stt: Transcriber | None) -> None:
        """Let go of a speech model now: a speech worker's process ends, and its memory with it."""
        close = getattr(stt, "close", None)
        if close is not None:
            try:
                close()
            except Exception as e:
                log.debug("closing the old speech model: %s", e)

    def _apply_initial_choices(self) -> None:
        """Start on the models Automatic wants, rather than loading one and swapping it."""
        p = self.compute.placement
        picks = self.compute.choices(p.speech, p.cleanup)
        speech, cleanup = picks["speech"], picks["cleanup"]
        current = catalogue.current(self.cfg.stt)
        if speech is not None and (current is None or current.key != speech.key):
            catalogue.get(speech.key).apply(self.cfg.stt)
            log.info("automatic: speech model %s (%s)", speech.key, speech.why)
        if cleanup is not None and cleanup.key != self.cfg.postprocess.llm_model:
            self.cfg.postprocess.llm_model = cleanup.key
            log.info("automatic: clean-up model %s (%s)", cleanup.key, cleanup.why)
        self.compute.chosen = picks

    def _speech_cfg(self, cfg: STTConfig | None = None, device: str | None = None) -> STTConfig:
        """The speech settings with the device the placement chose. "cuda" becomes "auto", so a
        GPU that turns out not to work still falls back to the processor."""
        cfg = cfg or self.cfg.stt
        device = device or self.compute.placement.speech
        if device == "cpu":
            return replace(cfg, device="cpu")
        return replace(cfg, device="cuda" if cfg.device == "cuda" else "auto")

    def _llama_device(self, device: str | None = None) -> str:
        device = device or self.compute.placement.cleanup
        if device == "cpu" or self.cfg.stt.device == "cpu":  # stt.device=cpu always meant both
            return "cpu"
        if device == "vulkan":
            return "vulkan"
        return "cuda" if self.cfg.stt.device == "cuda" else "auto"

    def devices(self) -> dict[str, str | None]:
        """Where each model actually is right now, read from the models rather than remembered:
        None when there is nothing to place (loading, or clean-up not on this computer)."""
        pp = self.cfg.postprocess
        server = self.llm_server if (pp.llm_cleanup and pp.llm_provider == "bundled") else None
        cleanup = getattr(server, "kind", None) if server is not None and self.llm_state == "ready" else None
        entry = catalogue.current(self.cfg.stt)
        return {"speech": getattr(self.stt, "device", None) if self.state == "ready" else None,
                "cleanup": cleanup,
                "speech_model": entry.key if entry else None,
                "cleanup_model": pp.llm_model if cleanup else None}

    def move_speech(self, device: str, key: str | None = None) -> bool:
        """Put speech on `device`, as model `key` if given (Automatic's choice) or the current
        one. Built beside the running one and swapped in on the speech worker, so dictation
        carries on meanwhile."""
        if self.stt is None or self.state != "ready":
            return False
        if key is not None:
            self._swap_speech(catalogue.get(key), device)
            return True
        stt = self._build_stt(self._speech_cfg(device=device))
        stt.warmup()

        def swap():
            old, self.stt = self.stt, stt
            return old

        old = self._pool.submit(swap).result()
        self._close_stt(old)  # a speech worker's process ends here: all its VRAM and CUDA memory go
        del old
        gc.collect()
        return True

    def move_cleanup(self, device: str, key: str | None = None) -> bool:
        """Restart the bundled clean-up model on `device`, as model `key` if given: the new
        server starts beside the old one, takes over on the clean-up worker, and only then is
        the old one stopped. Changing model while staying on the GPU is the exception - two
        language models and speech do not fit in 8 GB - so that one stops the old server first."""
        from localflow.llm.providers import build_provider
        from localflow.llm.server import LlamaServer

        pp = self.cfg.postprocess
        if not (pp.llm_cleanup and pp.llm_provider == "bundled") or self.llm_server is None:
            return False
        if key is not None and key != pp.llm_model:
            if device == "cuda" and getattr(self.llm_server, "kind", None) == "cuda":
                self._llm_pool.submit(self._swap_cleanup, key, device).result()
                return True
            pp = replace(pp, llm_model=key)
        server = LlamaServer(model_key=pp.llm_model, device=self._llama_device(device))
        server.start(progress=self._download_progress)
        provider = build_provider(pp, server_factory=lambda: server)
        CleanupPipeline(pp, provider).warm()  # before it takes over, so no dictation pays for a cold prompt cache
        model = pp.llm_model

        def swap():
            # The settings as they are now, not as they were when the move began: starting the
            # server takes seconds, and a dictionary entry or style change made in the Hub
            # meanwhile used to be put back the way it was. Only the model is this move's to set.
            after = replace(self.cfg.postprocess, llm_model=model)
            old, self.llm_server = self.llm_server, server
            self.cleanup = CleanupPipeline(after, provider)
            self.cfg.postprocess = after
            return old

        old = self._llm_pool.submit(swap).result()
        if old is not None:
            old.stop()
        if key is not None:
            self._save_quietly()
        return True

    def _save_quietly(self) -> None:
        try:
            self.cfg.save()
        except Exception as e:
            log.warning("could not save settings: %s", e)

    def _load_llm(self, device: str | None = None) -> None:
        from localflow.llm.providers import build_provider

        pp = self.cfg.postprocess
        self.llm_state, self.llm_error = "loading", None
        self._broadcast_status()
        try:
            t0 = time.perf_counter()

            def server_factory():
                from localflow.llm.server import LlamaServer

                if self.llm_server is None or not self.llm_server.alive():
                    self._room_for_cleanup(device)
                    where = self._llama_device(device)
                    self.llm_server = LlamaServer(model_key=pp.llm_model, device=where)
                    try:
                        self.llm_server.start(progress=self._download_progress)
                    except Exception as e:
                        if where != "vulkan":
                            raise
                        # AMD/Intel graphics that will not run it (a driver, no Vulkan): the
                        # processor, and not the graphics again this session (B3)
                        log.warning("clean-up could not start on the built-in graphics (%s); using the processor", e)
                        self.compute.no_vulkan = True
                        self.llm_server = LlamaServer(model_key=pp.llm_model, device="cpu")
                        self.llm_server.start(progress=self._download_progress)
                return self.llm_server

            provider = build_provider(pp, server_factory=server_factory)
            pipeline = CleanupPipeline(pp, provider)
            pipeline.warm()
            self.cleanup = pipeline
            self.llm_state = "ready"
            log.info("Clean-up model ready in %.1fs (%s:%s)", time.perf_counter() - t0, pp.llm_provider, pp.llm_model)
        except Exception as e:
            self.llm_state, self.llm_error = "error", problems.detail(e)
            self.llm_error_code = problems.classify_cleanup(e, pp.llm_provider)
            log.warning("Clean-up model unavailable (%s: %s); dictation will use the rule-based clean-up",
                        self.llm_error_code, e)
            if self.llm_error_code in problems.RETRY_BY_ITSELF:
                def again() -> None:
                    # Only if nothing else has loaded or changed it meanwhile.
                    if self.llm_state == "error" and self.cfg.postprocess.llm_cleanup:
                        self._load_llm(device)
                self._retry(again, "clean-up")
        else:
            self._retries.pop("clean-up", None)
        self._broadcast_status()

    def _room_for_cleanup(self, device: str | None = None) -> None:
        """Refuse to start the bundled clean-up model when free RAM is short of what it takes
        plus a margin (B5): loaded anyway, Windows pages everything else out, and the app being
        dictated into slows down with it. Raises problems.NotEnoughMemory, which retries by
        itself."""
        from localflow.llm import manifest as M

        # built-in graphics use RAM like the processor does
        where = "cuda" if self._llama_device(device) in ("cuda", "auto") else "cpu"
        model = M.CLEANUP_MODELS.get(self.cfg.postprocess.llm_model)
        need = hwinfo.cleanup_ram_gb(model.approx_gb if model else 2.5, where) + hwinfo.HEADROOM_GB
        free = hwinfo.ram_free_gb()
        if free is not None and free < need:
            raise problems.NotEnoughMemory(
                f"{self._cleanup_label()} needs about {need:.1f} GB of free memory, and {free:.1f} GB is free now")

    # the graphics card's libraries (B2) -------------------------------------------------------------
    def fetch_cuda_libs(self) -> None:
        """On an NVIDIA PC without the CUDA libraries (the installer leaves them out), download
        them in the background; speech stays on the processor meanwhile, and moves to the card
        once they are here. Nothing to do anywhere else - a PC without an NVIDIA card, a driver
        too old for them, "Processor only" - and nothing twice."""
        with self._lock:
            if (self.cuda_libs or {}).get("state") == "downloading" or not self._wants_cuda_libs():
                return
            self.cuda_libs = {"state": "downloading", "done": 0, "total": cudalibs.DOWNLOAD_BYTES}
        threading.Thread(target=self._fetch_cuda_libs, name="cuda-libs", daemon=True).start()

    def _wants_cuda_libs(self) -> bool:
        if self._closing or cudalibs.available() or self.cfg.compute.mode == "cpu" or self.cfg.stt.device == "cpu":
            return False
        hw = self.compute.hardware
        if hw is None or not hw.nvidia:
            return False
        from localflow import gpu, selfcheck

        mon = gpu.monitor()
        major = selfcheck.driver_major(mon.driver_version() if hasattr(mon, "driver_version") else None)
        if major is not None and major < selfcheck.MIN_DRIVER:
            return False  # these need a newer driver; the self-check already says so
        try:
            import onnxruntime as ort
        except ImportError:
            return False
        return "CUDAExecutionProvider" in ort.get_available_providers()

    def _fetch_cuda_libs(self) -> None:
        self._broadcast_status()
        last = [0.0]

        def progress(done: int, total: int) -> None:
            self.cuda_libs = {"state": "downloading", "done": done, "total": total}
            if time.monotonic() - last[0] >= 1.0:
                last[0] = time.monotonic()
                self._broadcast_status()

        try:
            cudalibs.ensure(progress)
        except Exception as e:
            self.cuda_libs = {"state": "error", "error": problems.detail(e)}
            log.warning("could not download the graphics card libraries: %s", e)
            if problems.is_network(e) or problems.is_no_space(e) or problems.is_clock_wrong(e):
                self._retry(self.fetch_cuda_libs, "cuda")
            self._broadcast_status()
            return
        self._retries.pop("cuda", None)
        self.cuda_libs = {"state": "ready"}
        # Everything that asked before now asks again: the probe here, a spare speech worker
        # started without them, and the placement that kept speech off the card.
        from localflow.stt import parakeet

        parakeet.reset_cuda_probe()
        remote.discard_spare()
        self.compute.no_gpu.discard("speech")
        self.compute.cuda_ready = True
        self.compute.poke()
        log.info("graphics card libraries ready; speech can move to the graphics card")
        self._broadcast_status()

    # the clean-up model asleep while LocalFlow is idle ----------------------------------------------
    def sleep_cleanup(self) -> bool:
        """Stop the bundled clean-up server while LocalFlow is idle; dictation meanwhile uses the
        rules. It used to be moved to the processor instead, where it went on holding 2 to 3 GB
        of memory for nothing. Woken by the next take (`wake_cleanup`)."""
        pp = self.cfg.postprocess
        if not (pp.llm_cleanup and pp.llm_provider == "bundled") or self.llm_server is None or self.llm_state != "ready":
            return False

        def swap():
            old, self.llm_server = self.llm_server, None
            self.cleanup = CleanupPipeline(self.cfg.postprocess, None)
            self.llm_state, self.llm_error = "asleep", None
            return old

        old = self._llm_pool.submit(swap).result()
        if old is not None:
            old.stop()
        log.info("clean-up model unloaded while LocalFlow is idle; the next dictation wakes it")
        self._broadcast_status()
        return True

    def wake_cleanup(self, device: str) -> None:
        """Load the clean-up model again after an idle unload, in the background, straight onto
        `device` - where the placement wants it now, not where the idle placement had it.

        Returns once the server process is launched. The speech model usually moves back to
        the graphics card at the same moment, and creating its session holds Python's lock for
        seconds: launched after that began, the server started five seconds late, and the first
        take after a rest waited for it. Launched first, it loads alongside."""
        from localflow.llm.server import LlamaServer

        if self.llm_state != "asleep":
            return
        self._waking = True
        self.llm_state = "loading"
        server = LlamaServer(model_key=self.cfg.postprocess.llm_model, device=self._llama_device(device))

        def load():
            try:
                try:
                    self._room_for_cleanup(device)
                    server.start(progress=self._download_progress)
                    self.llm_server = server  # _load_llm takes a running server as it is
                except Exception as e:
                    log.warning("clean-up model did not wake on %s (%s); trying again", device, e)
                finally:
                    server.spawned.set()
                self._load_llm(device)
            finally:
                self._waking = False

        threading.Thread(target=load, name="llm-wake", daemon=True).start()
        server.spawned.wait(3.0)

    def unload_unused(self) -> None:
        """Let go of models nobody has used for a while (Whisper). Queued on the speech worker,
        behind any decode, so nothing in use can be unloaded."""
        stt = self.stt
        if stt is not None and self.state == "ready" and hasattr(stt, "drop_idle_whisper"):
            self._pool.submit(stt.drop_idle_whisper)

    def wait_for_waking_cleanup(self, timeout: float) -> None:
        """If the clean-up model is waking from an idle unload, give it up to `timeout` to be
        ready. Only then: a first load, or a download, can take minutes and is not waited for."""
        deadline = time.perf_counter() + timeout
        while self._waking and time.perf_counter() < deadline:
            time.sleep(0.05)

    @staticmethod
    def _tidy_downloads() -> None:
        from localflow.llm.server import tidy_downloads

        try:
            tidy_downloads()
        except Exception:
            log.exception("could not tidy old downloads")

    def _download_progress(self, name: str, done: int, total: int | None) -> None:
        if total and done and (done == total or done % (64 << 20) < (1 << 20)):
            log.info("downloading %s: %.0f%% of %.1f GB", name, 100 * done / total, total / 2**30)

    def submit(self, fn: Callable[[], Any]) -> Future:
        return self._pool.submit(fn)

    # the graphics card failing under speech ------------------------------------------------------
    def transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        """Every decode goes through here, on the speech worker. If speech on the graphics card
        fails - a driver reset, the speech worker process gone - speech moves to the processor
        and the same audio is decoded again: the take is slower, not lost."""
        stt = self.stt
        try:
            return stt.transcribe(audio, language=language)
        except Exception as e:
            if getattr(stt, "device", "cpu") != "cuda":
                raise
            log.warning("speech on the graphics card failed (%s: %s); decoding again on the processor",
                        type(e).__name__, str(e)[:200])
            self._speech_to_cpu(stt)
            return self.stt.transcribe(audio, language=language)

    def warm(self) -> None:
        stt = self.stt
        try:
            stt.warm()
        except Exception as e:
            if getattr(stt, "device", "cpu") != "cuda":
                raise
            log.warning("speech on the graphics card failed while warming (%s); moving to the processor", e)
            self._speech_to_cpu(stt)

    def check_speech(self) -> None:
        """From the placement's loop: a speech worker that has died (a crash, the driver reset
        under it) is replaced now, not when the next take finds it gone."""
        stt = self.stt
        if self.state == "ready" and getattr(stt, "alive", True) is False:
            log.warning("the speech worker process has stopped; moving speech to the processor")
            self._pool.submit(self._speech_to_cpu, stt)

    def _speech_to_cpu(self, broken: Transcriber) -> None:
        """On the speech worker: replace a failed model on the graphics card with one on the
        processor. Not recorded as "CUDA does not work here": the placement moves speech back
        at its next reading, and if the card still fails, tries again two minutes later."""
        if self.stt is not broken:
            return  # replaced meanwhile
        t0 = time.perf_counter()
        cpu = self._build_stt(self._speech_cfg(device="cpu"))
        self.stt = cpu
        self._close_stt(broken)
        self.compute.asked["speech"] = "cpu"
        self.gpu_failures += 1
        log.info("speech is on the processor after the graphics card failed (%.1fs)", time.perf_counter() - t0)
        self._broadcast_status()

    # changing the speech model ------------------------------------------------------------------
    def switch_speech(self, key: str) -> None:
        """Move to another speech model from the catalogue.

        The download (the slow part, often gigabytes) happens off the speech worker, so the
        current model keeps taking dictation meanwhile. Only the swap itself runs on the
        worker, where no decode can be in flight, and the settings are saved only once the
        new model has loaded: a failed download or load leaves everything as it was.
        """
        model = catalogue.get(key)
        self._pin("auto_speech")
        # Checked up front so a model already on disk never flashes "Downloading 0 %".
        first = "loading" if catalogue.is_installed(model, self.compute.speech_device_for(model)) else "downloading"
        with self._lock:
            busy = self.speech_switch is not None and self.speech_switch.get("state") in ("downloading", "loading")
            if busy:
                raise RuntimeError("a speech model change is already in progress")
            now = catalogue.current(self.cfg.stt)
            if now is not None and now.key == key and self.state == "ready":
                self.speech_switch = None
                return
            self.speech_switch = {"to": key, "state": first, "progress": 0.0, "error": None}
        self._broadcast_status()
        threading.Thread(target=self._switch_speech, args=(model,), name="speech-switch", daemon=True).start()

    def _switch_speech(self, model: catalogue.SpeechModel) -> None:
        try:
            device = self.compute.speech_device_for(model)
            if not catalogue.is_installed(model, device):
                from localflow import net

                net.wait_for_proxy()
                net.ensure_space(net.hf_cache_dir(), int(model.size_gb(device) * 1e9), model.label)
                log.info("Downloading speech model %s (~%.1f GB)", model.label, model.size_gb(device))
                catalogue.download(model, device, progress=self._speech_progress)
            self._set_switch(state="loading", progress=1.0)
            with self._model_lock:
                self._swap_speech(model, device)
        except Exception as e:
            log.exception("could not switch to %s", model.key)
            self._set_switch(state="error", error=problems.detail(e), error_code=problems.SPEECH_SWITCH_FAILED,
                             cause=problems.classify_speech(e))
            return
        with self._lock:
            self.speech_switch = None
        self._broadcast_status()

    def _swap_speech(self, model: catalogue.SpeechModel, device: str) -> None:
        """Replace the speech model. Two speech models plus the clean-up model do not fit in
        8 GB of VRAM, so a GPU-to-GPU change releases the old one first and loads on the speech
        worker (dictation waits, and says it is warming up). Any change involving the processor
        builds beside the running model instead, and only the swap waits for the worker."""
        new_cfg = replace(self.cfg.stt)
        model.apply(new_cfg)
        build_cfg = self._speech_cfg(new_cfg, device)
        t0 = time.perf_counter()
        if device == "cuda" and getattr(self.stt, "device", None) == "cuda":
            self._pool.submit(self._reload_speech_in_place, build_cfg, model.label).result()
        else:
            stt = self._build_stt(build_cfg)
            stt.warmup()

            def swap():
                old, self.stt = self.stt, stt
                self.state, self.error = "ready", None
                return old

            old = self._pool.submit(swap).result()
            self._close_stt(old)
            del old
            gc.collect()
        self.cfg.stt = new_cfg
        try:
            self.cfg.save()
        except Exception as e:
            log.warning("could not save settings: %s", e)
        self.compute.asked["speech"] = device
        self.compute.poke()  # keep-warm and the like follow the new model's device
        log.info("Speech model is now %s on %s (%.1fs)", model.label, device, time.perf_counter() - t0)

    def _reload_speech_in_place(self, build_cfg: STTConfig, label: str) -> None:
        """On the speech worker: release the old model, load the new one, and put the old one
        back if the new one fails."""
        previous = self._speech_cfg()
        self._close_stt(self.stt)
        self.stt, self.state = None, "loading"
        gc.collect()
        self._broadcast_status()
        try:
            stt = self._build_stt(build_cfg)
            stt.warmup()
        except Exception:
            log.warning("%s failed to load; going back to the previous model", label)
            self.stt = self._build_stt(previous)
            self.stt.warmup()
            self.state = "ready"
            raise
        self.stt, self.state, self.error = stt, "ready", None

    # changing the clean-up model ---------------------------------------------------------------------
    def switch_cleanup(self, key: str) -> None:
        """Move the bundled clean-up model to another from the manifest.

        Same shape as a speech change: download while the current model keeps editing, swap
        on the clean-up worker so no edit is in flight, save only once the new server answers,
        and on any failure bring the previous model back. Choosing a bundled model also turns
        auto-edits on and the provider to bundled - that is what picking one means.
        """
        from localflow.llm import manifest as M

        if key not in M.CLEANUP_MODELS:
            raise ValueError(f"unknown clean-up model {key!r}")
        self._pin("auto_cleanup")
        pp = self.cfg.postprocess
        with self._lock:
            state = (self.cleanup_switch or {}).get("state")
            if state in ("downloading", "loading"):
                raise RuntimeError("a clean-up model change is already in progress")
            if pp.llm_provider == "bundled" and pp.llm_model == key and pp.llm_cleanup and self.llm_state == "ready":
                self.cleanup_switch = None
                return
            first = "loading" if M.gguf_path(key).exists() else "downloading"
            self.cleanup_switch = {"to": key, "state": first, "progress": 0.0, "error": None}
        self._broadcast_status()
        threading.Thread(target=self._switch_cleanup, args=(key,), name="cleanup-switch", daemon=True).start()

    def _pin(self, flag: str) -> None:
        """Choosing a model by hand means it stays chosen: Automatic is off for that kind."""
        if getattr(self.cfg.compute, flag):
            self.cfg.compute = replace(self.cfg.compute, **{flag: False})
            self._save_quietly()

    def _switch_cleanup(self, key: str) -> None:
        from localflow.llm.server import ensure_model

        try:
            ensure_model(key, progress=lambda _name, done, total: total and self._cleanup_progress(done, total))
            self._set_cleanup_switch(state="loading", progress=1.0)
            with self._model_lock:
                self._llm_pool.submit(self._swap_cleanup, key).result()
        except Exception as e:
            log.exception("could not switch the clean-up model to %s", key)
            self._set_cleanup_switch(state="error", error=problems.detail(e),
                                     error_code=problems.CLEANUP_SWITCH_FAILED,
                                     cause=problems.classify_cleanup(e, self.cfg.postprocess.llm_provider))
            return
        with self._lock:
            self.cleanup_switch = None
        self._broadcast_status()

    def _swap_cleanup(self, key: str, device: str | None = None) -> None:
        """On the clean-up worker. The old server goes first: two language models beside the
        speech model do not fit in 8 GB of VRAM."""
        from localflow.llm.providers import build_provider
        from localflow.llm.server import LlamaServer

        before = self.cfg.postprocess
        old_server = self.llm_server
        if old_server is not None:
            old_server.stop()
        self.llm_server = None
        self.cleanup = CleanupPipeline(before, None)  # rules only while the new model loads
        self.llm_state, self.llm_error = "loading", None
        self._broadcast_status()
        t0 = time.perf_counter()
        try:
            server = LlamaServer(model_key=key, device=self._llama_device(device))
            server.start()
        except Exception:
            log.warning("clean-up model %s failed to start; going back to the previous one", key)
            self.llm_state = "off"
            if before.llm_cleanup:
                self._load_llm()  # the previous settings, unchanged
            else:
                self._broadcast_status()
            raise
        self.llm_server = server
        # Taken again now that the server is up, so Hub changes made while it started survive.
        after = replace(self.cfg.postprocess, llm_cleanup=True, llm_provider="bundled", llm_model=key)
        self.cfg.postprocess = after
        pipeline = CleanupPipeline(after, build_provider(after, server_factory=lambda: self.llm_server))
        pipeline.warm()
        self.cleanup = pipeline
        self.llm_state, self.llm_error = "ready", None
        try:
            self.cfg.save()
        except Exception as e:
            log.warning("could not save settings: %s", e)
        log.info("Clean-up model is now %s (%.1fs)", key, time.perf_counter() - t0)

    def _cleanup_progress(self, done: int, total: int) -> None:
        now = time.monotonic()
        if done < total and now - self._switch_reported < 0.25:
            return
        self._switch_reported = now
        self._set_cleanup_switch(progress=round(min(1.0, done / total), 3))

    def _set_cleanup_switch(self, **fields: Any) -> None:
        with self._lock:
            if self.cleanup_switch is None:
                return
            self.cleanup_switch = {**self.cleanup_switch, **fields}
        self._broadcast_status()

    def _speech_progress(self, done: int, total: int) -> None:
        now = time.monotonic()
        if done < total and now - self._switch_reported < 0.25:
            return
        self._switch_reported = now
        self._set_switch(progress=round(min(1.0, done / total), 3) if total else 0.0)

    def _set_switch(self, **fields: Any) -> None:
        with self._lock:
            if self.speech_switch is None:
                return
            self.speech_switch = {**self.speech_switch, **fields}
        self._broadcast_status()

    def record_cleanup(self, result: Any) -> None:
        """Time a real clean-up for the model choice (bundled models only; the rest are not ours
        to choose)."""
        pp = self.cfg.postprocess
        server = self.llm_server
        if result.used_llm and result.llm_ms and pp.llm_provider == "bundled" and server is not None:
            self.perf.record("cleanup", pp.llm_model, getattr(server, "kind", "cuda"), result.llm_ms)

    def run_command(self, selection: str, instruction: str) -> CommandResult:
        """Apply a spoken instruction to selected text (command mode).

        The runner is built per call rather than held, so that changing the provider from the
        Hub takes effect on the next command instead of the next restart.
        """
        self.compute.activity()
        try:
            return CommandRunner(self.cleanup.provider).run(selection, instruction)
        finally:
            self.compute.worked()

    def submit_llm(self, fn: Callable[[], Any]) -> Future:
        return self._llm_pool.submit(fn)

    def live_mode(self) -> str:
        """continuous or periodic. Continuous re-decodes back to back while the key is held,
        which holds the GPU at full clocks - so only when the placement allows it (a cool GPU,
        on mains, stt.gpu_keep_warm not "never")."""
        if getattr(self.stt, "device", "cpu") != "cuda":
            return "periodic"
        return "continuous" if self.compute.placement.keep_warm else "periodic"

    def shutdown(self) -> None:
        self._closing = True
        for timer in self._timers:
            timer.cancel()
        self.compute.stop()
        self.perf.flush()
        self._close_stt(self.stt)
        remote.discard_spare()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._llm_pool.shutdown(wait=False, cancel_futures=True)
        if self.llm_server is not None:
            self.llm_server.stop()

    # status -------------------------------------------------------------------------------------
    def add_status_listener(self, sink: EventSink) -> None:
        with self._lock:
            self._status_listeners.append(sink)

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
        entry = catalogue.current(self.cfg.stt)
        try:
            choices = catalogue.describe(self.cfg.stt, self.compute.speech_device_for)
        except Exception:
            log.debug("could not describe the speech models", exc_info=True)
            choices = []
        return {
            "type": P.STATUS,
            "version": __version__,
            "pid": os.getpid(),
            "safe_mode": in_safe_mode(self.cfg),
            "stt": {
                "state": self.state,
                "error": self.error,
                "error_code": self.error_code if self.state == "error" else None,
                "backend": self.cfg.stt.backend,
                "model": self.cfg.stt.model,
                "key": entry.key if entry else None,
                "label": entry.label if entry else self.cfg.stt.model,
                "device": getattr(stt, "device", None),
                "precision": getattr(stt, "precision", None),
                "choices": choices,
                "switch": self.speech_switch,
                # A first run's download of the speech model: {label, progress 0-1, size_gb}.
                "download": self.first_download,
            },
            "llm": {
                "state": self.llm_state,
                "error": self.llm_error,
                "error_code": self.llm_error_code if self.llm_state == "error" else None,
                "enabled": pp.llm_cleanup,
                "provider": pp.llm_provider,
                "model": pp.llm_model,
                "label": self._cleanup_label(),
                "choices": self._cleanup_choices(),
                "switch": self.cleanup_switch,
            },
            "compute": self.compute.status(),
            # Settings written by a newer LocalFlow: read, never saved over (their version).
            "settings_newer": self.cfg.newer,
            # Where downloads come from: the mirror set in the Hub, and the proxy Windows gave.
            "network": {"hf_endpoint": self.cfg.network.hf_endpoint, "proxy": _net_proxy()},
            "checks": self.checks,
        }

    def quick_check(self) -> None:
        """The cheap self-checks, whose results go out with every status."""
        from localflow import selfcheck

        try:
            self.checks = [c.as_dict() for c in selfcheck.run(self, full=False)]
        except Exception:
            log.exception("quick self-check failed")

    def full_check(self) -> list[dict[str, Any]]:
        """Every self-check; slow (hashes the model files). Refreshes the quick results too."""
        from localflow import selfcheck

        checks = selfcheck.run(self, full=True)
        self.last_full_check = checks
        quick = {"driver", "models_folder", "disk", "cleanup_server"}
        self.checks = [c.as_dict() for c in checks if c.id in quick]
        self._broadcast_status()
        return [c.as_dict() for c in checks]

    def repair_models(self) -> list[str]:
        """Delete what the last full check found damaged; the caller restarts the engine."""
        from localflow import selfcheck

        return selfcheck.repair(self.last_full_check)

    def _apply_compute(self, patch: dict[str, Any]) -> None:
        c = self.cfg.compute
        mode = patch.get("mode", c.mode)
        try:
            limit = int(patch.get("temp_limit_c", c.temp_limit_c))
            idle = float(patch.get("idle_release_min", c.idle_release_min))
        except (TypeError, ValueError):
            log.warning("ignoring malformed compute settings: %r", patch)
            return
        if mode not in ("adaptive", "gpu", "cpu"):
            log.warning("unknown compute mode %r", mode)
            return
        self.cfg.compute = ComputeConfig(
            mode=mode, temp_limit_c=max(60, min(limit, 90)), idle_release_min=max(0.0, min(idle, 240.0)),
            auto_speech=bool(patch.get("auto_speech", c.auto_speech)),
            auto_cleanup=bool(patch.get("auto_cleanup", c.auto_cleanup)))
        try:
            self.cfg.save()
        except Exception as e:
            log.warning("could not save settings: %s", e)
        log.info("compute settings: %s", self.cfg.compute)
        self.compute.configure()
        self._broadcast_status()
        self.fetch_cuda_libs()  # Processor only turned off: the graphics card may want its libraries now

    # What "Reset preferences" keeps: the user's own words, their API key (a credential), the
    # dictation language and the download mirror (who they are and where, not a preference).
    KEEP_ON_RESET = ("dictionary", "dictionary_terms", "snippets", "custom_instructions", "llm_api_key")

    def reset_preferences(self) -> None:
        """Every engine setting back to its default, except what KEEP_ON_RESET names (and the
        language and mirror). The speech model is left loaded: with Automatic model choice back
        on, the placement moves to its choice as usual, rather than dictation stopping for one."""
        fresh = for_this_pc(Config())
        self.cfg.stt = replace(self.cfg.stt, device=fresh.stt.device, longform_seconds=fresh.stt.longform_seconds,
                               gpu_keep_warm=fresh.stt.gpu_keep_warm)
        self._apply_compute(asdict(fresh.compute))
        keep = {k: getattr(self.cfg.postprocess, k) for k in self.KEEP_ON_RESET}
        self.apply_settings({"postprocess": asdict(replace(fresh.postprocess, **keep))})
        log.info("preferences reset to their defaults; the dictionary, snippets and house style kept")

    def _apply_network(self, endpoint: Any) -> list[str]:
        """The download mirror, from the Hub. Used from the next download on."""
        from localflow import net
        from localflow.validate import check_mirror

        tidy, why = check_mirror(endpoint)
        if tidy is None:
            log.warning("setting not applied: %s", why)
            return [why or "the download mirror was refused"]
        self.cfg.network.hf_endpoint = tidy
        net.apply_endpoint(tidy)
        self._save_quietly()
        self._broadcast_status()
        # A download waiting to try again tries the new address at once.
        if self.state == "error" and self.error_code in problems.RETRY_BY_ITSELF:
            self._pool.submit(self._load_speech)
        return []

    def _cleanup_label(self) -> str:
        from localflow.llm import manifest as M

        pp = self.cfg.postprocess
        entry = M.CLEANUP_MODELS.get(pp.llm_model) if pp.llm_provider == "bundled" else None
        return (entry.label or entry.key) if entry else pp.llm_model

    def _cleanup_choices(self) -> list[dict]:
        from localflow.llm import manifest as M

        pp = self.cfg.postprocess
        try:
            return M.describe(pp.llm_model if pp.llm_provider == "bundled" and pp.llm_cleanup else None)
        except Exception:
            log.debug("could not describe the clean-up models", exc_info=True)
            return []

    # sessions -------------------------------------------------------------------------------------
    def start_session(self, session_id: str, context: dict[str, Any], emit: EventSink,
                      language: str | None = None) -> Session:
        if self.state != "ready" or self.stt is None or self.vad is None:
            raise RuntimeError(f"engine not ready ({self.state}{': ' + self.error if self.error else ''})")
        self.compute.activity()
        return Session(session_id, self, context, emit, language)

    def apply_settings(self, settings: dict[str, Any]) -> list[str]:
        """Apply what the Hub changed. Returns the problems with it, in plain words: anything
        refused keeps its previous value."""
        from localflow.validate import check_postprocess

        compute = settings.get("compute")
        if isinstance(compute, dict):
            self._apply_compute(compute)
        network = settings.get("network")
        refused: list[str] = []
        if isinstance(network, dict) and "hf_endpoint" in network:
            refused += self._apply_network(network["hf_endpoint"])
        stt = settings.get("stt")
        if isinstance(stt, dict) and isinstance(stt.get("model"), str):
            try:
                self.switch_speech(stt["model"])
            except (ValueError, RuntimeError) as e:
                log.warning("speech model not changed: %s", e)
        llm = settings.get("llm")
        if isinstance(llm, dict) and isinstance(llm.get("model"), str):
            try:
                self.switch_cleanup(llm["model"])
            except (ValueError, RuntimeError) as e:
                log.warning("clean-up model not changed: %s", e)
        pp = settings.get("postprocess")
        if not isinstance(pp, dict):
            return refused
        before = self.cfg.postprocess
        merged = {**before.__dict__, **pp}
        after = PostProcessConfig(**{k: v for k, v in merged.items() if k in PostProcessConfig.__dataclass_fields__})
        after, problems = check_postprocess(after, before)
        for p in problems:
            log.warning("setting not applied: %s", p)
        self.cfg.postprocess = after
        # Settings changed from the Hub have to survive a restart, or the user changes them
        # once and wonders why they came back.
        try:
            self.cfg.save()
        except Exception as e:
            log.warning("could not save settings: %s", e)
        # keep the current provider unless the model or provider itself changed - or, for a provider
        # reached over the network, its address or key: a corrected API key used to go on failing
        # with the old one until LocalFlow was restarted
        restart = (after.llm_cleanup != before.llm_cleanup or after.llm_provider != before.llm_provider
                   or after.llm_model != before.llm_model
                   or (after.llm_provider != "bundled"
                       and (after.llm_url, after.llm_api_key) != (before.llm_url, before.llm_api_key)))
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
        return refused + problems
