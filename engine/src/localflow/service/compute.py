"""The engine's side of placement: watch the GPU, and move or change models when the policy says so.

Two policies meet here. localflow/placement.py decides *where* (graphics card or processor) from
the GPU's temperature and load; localflow/modelchoice.py decides *which* model, when the user
left that to LocalFlow: the most accurate one that is quick enough on that device. This module
polls the GPU every few seconds, asks both, and has the engine carry out any difference:

  * a move builds the model on its new device first and swaps it in afterwards, so dictation
    never stops for one - the old copy keeps working until the new one is warm
  * speech and clean-up move in parallel, because waking from an idle release is on the path
    of a dictation that has already started
  * a model change the user asked for (the Hub's pickers) holds the same lock, so the two
    never interleave; a move that finds the lock taken simply tries again at the next reading
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from localflow import gpu, hwinfo, modelchoice
from localflow.config import MODELS_DIR
from localflow.llm import manifest as llm_manifest
from localflow.placement import SPEECH_VRAM_MB, Governor, Placement, VramGuard, cleanup_vram_mb, placement_for
from localflow.stt import catalogue

if TYPE_CHECKING:
    from localflow.service.engine import Engine

log = logging.getLogger(__name__)

POLL_S = 5.0
RETRY_AFTER_S = 120.0  # a move that failed is not tried again for this long
WORDS = {"speech": "speech", "cleanup": "clean-up"}


def _on_ac() -> bool:
    try:
        from localflow.power import on_ac_power

        return on_ac_power()
    except Exception:
        return True


class ComputeController:
    def __init__(self, engine: Engine, monitor: gpu.GpuMonitor | None = None, clock=time.monotonic):
        self.engine = engine
        self.clock = clock
        c = engine.cfg.compute
        self.governor = Governor(c.mode, c.temp_limit_c, c.idle_release_min)
        self.vram = VramGuard()
        self.level: tuple[int, str] = (self.governor.level, self.governor.reason)  # with the VRAM guard's say
        self.monitor = monitor or gpu.monitor()
        self.sample: gpu.GpuSample | None = None
        self.placement: Placement = placement_for(0, "starting", on_ac=True)
        self.moving: str | None = None  # what is being moved right now, for the Hub
        self.recent: deque[dict[str, Any]] = deque(maxlen=6)
        now = clock()
        self.last_activity = now  # a dictation or command started
        self.last_work = now  # LocalFlow last had work on the GPU (or could have)
        self.asked: dict[str, str] = {}  # the device each model was last loaded or moved for
        self.no_gpu: set[str] = set()  # models whose GPU request came back on the processor
        self.cuda_ready = False  # speech's CUDA libraries are here (B2); checked until they are
        self.no_vulkan = False  # the Vulkan build failed here: clean-up off the card stays on the processor
        self._failed: dict[tuple[str, str, str | None], float] = {}
        self.hardware: modelchoice.Hardware | None = None
        self.chosen: dict[str, modelchoice.Choice | None] = {"speech": None, "cleanup": None}
        self._reported: gpu.GpuSample | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # the policy ----------------------------------------------------------------------------------
    def decide(self) -> Placement:
        """Take a reading and return where things should be now. Also used once at start-up, so
        the models are loaded where they belong rather than loaded and then moved."""
        now = self.clock()
        self.sample = self.monitor.sample()
        if self.hardware is None:
            self.hardware = modelchoice.detect_hardware(self.sample.mem_total_mb if self.sample else None)
            log.info("hardware: %s", self.hardware)
        on_ac = _on_ac()
        level, reason = self.governor.observe(self.sample, now=now, on_ac=on_ac,
                                              idle_s=now - self.last_activity,
                                              own_work_age_s=now - self.last_work)
        room = self._vram_level(now)
        if room is not None and room[0] > level and self.engine.cfg.compute.mode != "cpu":
            level, reason = room
        self.level = (level, reason)
        return self.placement_at(level, reason, on_ac=on_ac)

    def _vram_level(self, now: float) -> tuple[int, str] | None:
        """What the graphics memory other apps have taken allows (B5), or None with no reading,
        or while a model is loading: its memory is on the card before the engine reports it
        there, and would be counted as someone else's."""
        s = self.sample
        if s is None:
            return None
        e = self.engine
        if (getattr(e, "llm_state", None) == "loading" or getattr(e, "speech_switch", None)
                or getattr(e, "cleanup_switch", None)):
            return self.vram.level, self.vram.reason
        loaded = e.devices()
        own = 0
        if loaded["speech"] == "cuda":
            own += SPEECH_VRAM_MB.get(loaded["speech_model"] or "", 2900)
        if loaded["cleanup"] == "cuda":
            own += self._cleanup_vram(loaded["cleanup_model"])
        stt = self.engine.cfg.stt
        entry = catalogue.get(modelchoice.SPEECH_ORDER[0]) if self.engine.cfg.compute.auto_speech else catalogue.current(stt)
        speech_mb = SPEECH_VRAM_MB.get(entry.key if entry else "", 2900)
        return self.vram.observe(total_mb=s.mem_total_mb, used_mb=s.mem_used_mb, own_mb=own, speech_mb=speech_mb,
                                 cleanup_mb=self._cleanup_vram(self.engine.cfg.postprocess.llm_model), now=now)

    @staticmethod
    def _cleanup_vram(key: str | None) -> int:
        model = llm_manifest.CLEANUP_MODELS.get(key or "")
        return cleanup_vram_mb(model.approx_gb if model else 2.5)

    def placement_at(self, level: int, reason: str, *, on_ac: bool | None = None,
                     speech: catalogue.SpeechModel | None = None) -> Placement:
        stt = self.engine.cfg.stt
        # With the model left to LocalFlow, where speech goes is decided for the best model; the
        # choice of model then follows the device, not the other way round.
        auto = self.engine.cfg.compute.auto_speech and speech is None
        entry = catalogue.get(modelchoice.SPEECH_ORDER[0]) if auto else (speech or catalogue.current(stt))
        speed = (entry.speed or {}) if entry else {}
        return placement_for(
            level, reason, on_ac=_on_ac() if on_ac is None else on_ac,
            keep_warm_setting=stt.gpu_keep_warm,
            # a model that is simply faster on the processor lives there (Parakeet Compact)
            speech_prefers_cpu=(bool(speed) and speed.get("cpu", 0) > speed.get("cuda", 0)) or not self.speech_can_use_card(),
            # one too slow there to dictate with stays on the GPU until nothing may use it
            speech_cpu_usable=speed.get("cpu", 3) >= 2,
            speech_device_setting=stt.device,
            vram_mb=self.hardware.vram_mb if self.hardware else None,
            nvidia=self.hardware.nvidia if self.hardware else True,
            cleanup_off_card=self.cleanup_off_card(),
        )

    def speech_can_use_card(self) -> bool:
        """False while speech's CUDA libraries are still to be downloaded (B2): speech then stays
        on the processor on purpose, rather than being asked onto the card and falling back."""
        if not self.cuda_ready:
            from localflow import cudalibs

            self.cuda_ready = cudalibs.available()
        return self.cuda_ready

    def cleanup_off_card(self) -> str:
        """Where clean-up goes when it is not on the NVIDIA card: AMD/Intel graphics when there
        are some with room for the model and they are quicker there than the processor -
        estimated at first, then as measured on this machine - otherwise the processor (B3)."""
        hw = self.hardware
        gpu = hw.other_gpu if hw else None
        pp = self.engine.cfg.postprocess
        if gpu is None or self.no_vulkan or self.engine.cfg.compute.mode == "cpu" or pp.llm_provider != "bundled":
            return "cpu"
        model = llm_manifest.CLEANUP_MODELS.get(pp.llm_model)
        if gpu.usable_mb < self._cleanup_vram(pp.llm_model) + 512:
            return "cpu"
        on_gpu, _ = self.engine.perf.estimate("cleanup", model.key if model else pp.llm_model, "vulkan", hw)
        on_cpu, _ = self.engine.perf.estimate("cleanup", model.key if model else pp.llm_model, "cpu", hw)
        return "vulkan" if on_gpu is not None and (on_cpu is None or on_gpu < on_cpu) else "cpu"

    def device_words(self, device: str) -> str:
        return modelchoice.device_words(device, self.hardware)

    def choices(self, speech_device: str, cleanup_device: str) -> dict[str, modelchoice.Choice | None]:
        """The models Automatic wants on those devices; None where the user chose the model, or
        where clean-up is not a bundled model at all."""
        c, pp = self.engine.cfg.compute, self.engine.cfg.postprocess
        hw = self.hardware or modelchoice.detect_hardware(None)
        perf = self.engine.perf
        current_speech = catalogue.current(self.engine.cfg.stt)

        def speech_ok(key: str) -> bool:
            m = catalogue.get(key)
            return ((current_speech is not None and current_speech.key == key)
                    or catalogue.is_installed(m, speech_device)
                    or m.size_gb(speech_device) < modelchoice.QUIET_DOWNLOAD_GB)

        loaded = self.engine.devices()

        def speech_short_of_memory(key: str) -> str | None:
            """Why `key` would not fit in free RAM on `speech_device` right now. A model already
            loaded there fits by definition; any other is built beside the one running, so it
            needs its whole size free (B5)."""
            if loaded["speech_model"] == key and loaded["speech"] == speech_device:
                return None
            free = hwinfo.ram_free_gb()
            if free is None:
                return None
            m = catalogue.get(key)
            need = hwinfo.speech_ram_gb(key, speech_device, m.size_gb("cpu")) + hwinfo.HEADROOM_GB
            if free >= need:
                return None
            return f"{m.label} needs about {need:.1f} GB of free memory, and {free:.1f} GB is free"

        def cleanup_ok(key: str) -> bool:
            return key == pp.llm_model or llm_manifest.gguf_path(key).exists()

        bundled = pp.llm_cleanup and pp.llm_provider == "bundled"
        return {
            "speech": (modelchoice.choose_speech(speech_device, hw, perf, speech_ok, speech_short_of_memory)
                       if c.auto_speech else None),
            "cleanup": (modelchoice.choose_cleanup(cleanup_device, hw, perf, cleanup_ok)
                        if c.auto_cleanup and bundled else None),
        }

    def speech_device_for(self, model: catalogue.SpeechModel) -> str:
        """Where `model` would run right now, were it chosen."""
        if "speech" in self.no_gpu:
            return "cpu"
        return self.placement_at(*self.level, speech=model).speech

    # the loop ------------------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="compute", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def poke(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait(POLL_S)
            self._wake.clear()
            if self._stop.is_set():
                return
            try:
                self.engine.check_speech()
                self.apply(self.decide())
                self.engine.unload_unused()
                self._report_reading()
            except Exception:
                log.exception("placement check failed")

    def _report_reading(self) -> None:
        """The Hub shows the temperature live, but the shell only hears the engine's status when
        it is broadcast: send one when the reading has moved enough to be worth redrawing."""
        s, last = self.sample, self._reported
        if s is None:
            return
        if (last is None or abs(s.temp_c - last.temp_c) >= 2 or abs(s.util_pct - last.util_pct) >= 10
                or abs(s.mem_used_mb - last.mem_used_mb) >= 256):
            self._reported = s
            self.engine._broadcast_status()

    def apply(self, target: Placement) -> None:
        """Move or change whatever is not as `target` and the model choice want it. Compares
        against the models as they actually are, so a move that failed, or a GPU that quietly
        fell back to the processor, is seen as it is rather than as it was meant to be."""
        now = self.clock()
        # Idle: the clean-up model is not moved to the processor but unloaded, and the next take
        # wakes it. Speech still moves, so dictation stays instant.
        if self.governor.idle_released:
            if self.engine.llm_state == "ready" and self.engine._model_lock.acquire(blocking=False):
                try:
                    self.engine.sleep_cleanup()
                finally:
                    self.engine._model_lock.release()
        elif self.engine.llm_state == "asleep":
            self.engine.wake_cleanup("cpu" if "cleanup" in self.no_gpu else target.cleanup)
        actual = self.engine.devices()
        want = {"speech": target.speech, "cleanup": target.cleanup}
        for what in ("speech", "cleanup"):
            # Asked for the GPU and got the processor: CUDA does not work for this model here.
            # Remember that, rather than rebuilding it every five seconds.
            if actual[what] == "cpu" and self.asked.get(what) == "cuda" and what not in self.no_gpu:
                self.no_gpu.add(what)
                log.info("%s cannot use the graphics card on this machine; keeping it on the processor", what)
            if what in self.no_gpu and want[what] == "cuda":
                want[what] = "cpu"
        self.chosen = self.choices(want["speech"], want["cleanup"])
        moves: list[tuple[str, str, str | None]] = []
        for what in ("speech", "cleanup"):
            if actual[what] is None:
                continue
            pick = self.chosen[what]
            model = pick.key if pick is not None and pick.key != actual[what + "_model"] else None
            recently_failed = now - self._failed.get((what, want[what], model), -1e9) < RETRY_AFTER_S
            if (actual[what] != want[what] or model) and not recently_failed:
                moves.append((what, want[what], model))
        if not moves or self.engine.state != "ready":
            self._settle(target, want, actual)
            return
        lock = self.engine._model_lock
        if not lock.acquire(blocking=False):
            return  # a model change the user asked for is running; look again next time
        try:
            self.moving = ", ".join(self._describe(what, where, model) for what, where, model in moves)
            self.engine._broadcast_status()
            log.info("moving %s (%s)", self.moving, target.reason)
            results: dict[str, bool] = {}

            def move(what: str, where: str, model: str | None) -> None:
                t0 = time.perf_counter()
                fn = self.engine.move_speech if what == "speech" else self.engine.move_cleanup
                self.asked[what] = where
                try:
                    results[what] = bool(fn(where, model))
                except Exception:
                    log.exception("moving %s to %s (%s) failed", what, where, model or "same model")
                    results[what] = False
                log.info("%s on %s%s: %s in %.1fs", what, where, f" as {model}" if model else "",
                         "ok" if results[what] else "failed", time.perf_counter() - t0)

            threads = [threading.Thread(target=move, args=m, name=f"move-{m[0]}") for m in moves]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for what, where, model in moves:
                if results.get(what):
                    pick = self.chosen[what]
                    self.recent.appendleft({
                        "at": time.time(), "what": WORDS[what], "to": where,
                        "model": _label(what, model) if model else None,
                        "reason": pick.why if model and pick else target.reason,
                    })
                else:
                    self._failed[(what, where, model)] = self.clock()
                    if where == "vulkan" and not self.no_vulkan:
                        self.no_vulkan = True  # never tried again this session; the processor it is
                        log.info("clean-up could not run on %s; keeping it on the processor", self.device_words(where))
        finally:
            self.moving = None
            lock.release()
        self._settle(target, want, self.engine.devices())

    def _describe(self, what: str, where: str, model: str | None) -> str:
        if model:
            return f"{WORDS[what]} to {_label(what, model)} on {self.device_words(where)}"
        return f"{WORDS[what]} to {self.device_words(where)}"

    def _settle(self, target: Placement, want: dict[str, str], actual: dict[str, str | None]) -> None:
        """Record the placement: where things are, or for a model not loaded yet, where it will
        be loaded. Keep-warm only ever applies to speech that is really on the GPU."""
        speech = actual["speech"] or want["speech"]
        placed = Placement(target.level, target.reason, speech, actual["cleanup"] or want["cleanup"],
                           target.keep_warm and speech == "cuda")
        if placed != self.placement:
            self.placement = placed
            self.engine._broadcast_status()

    # signals from the engine ---------------------------------------------------------------------
    def activity(self) -> None:
        """A dictation or command started. Wakes the GPU at once if it had been freed."""
        now = self.clock()
        self.last_activity = self.last_work = now
        if self.governor.idle_released:
            self.poke()

    def worked(self) -> None:
        """LocalFlow just finished some work, so the GPU load of the last few seconds was ours."""
        self.last_work = self.clock()

    def configure(self) -> None:
        c = self.engine.cfg.compute
        self.governor.configure(c.mode, c.temp_limit_c, c.idle_release_min)
        self.poke()

    def status(self) -> dict[str, Any]:
        c = self.engine.cfg.compute
        chosen = {what: ({"key": pick.key, "label": _label(what, pick.key), "why": pick.why} if pick else None)
                  for what, pick in self.chosen.items()}
        return {
            "mode": c.mode,
            "auto": {"speech": c.auto_speech, "cleanup": c.auto_cleanup},
            "chosen": chosen,
            "hardware": self._hardware_now(),
            # what "vulkan" means on this machine, for the Hub: "Built-in graphics" or a card's name
            "cleanup_off_card": self.cleanup_off_card(),
            "cuda_libs": getattr(self.engine, "cuda_libs", None),  # their first download (B2)  # where clean-up goes when it leaves the NVIDIA card
            "vulkan": _sentence_case(self.device_words("vulkan")[4:]) if self.hardware and self.hardware.other_gpu else None,
            "temp_limit_c": c.temp_limit_c,
            "idle_release_min": c.idle_release_min,
            **self.placement.as_dict(),
            "moving": self.moving,
            "gpu": {**self.sample.as_dict(), "name": self.monitor.name} if self.sample else None,
            "recent": list(self.recent),
        }

    def _hardware_now(self) -> dict[str, Any] | None:
        """The capability report, with what changes (free RAM and disk) read fresh."""
        if not self.hardware:
            return None
        free_disk = hwinfo.disk_free_gb(MODELS_DIR)
        free_ram = hwinfo.ram_free_gb()
        return {**self.hardware.as_dict(),
                "ram_free_gb": round(free_ram, 1) if free_ram is not None else None,
                "disk_free_gb": round(free_disk) if free_disk is not None else None}


def _sentence_case(text: str) -> str:
    return text[:1].upper() + text[1:]


def _label(what: str, key: str) -> str:
    if what == "speech":
        try:
            return catalogue.get(key).label
        except ValueError:
            return key
    entry = llm_manifest.CLEANUP_MODELS.get(key)
    return (entry.label or entry.key) if entry else key
