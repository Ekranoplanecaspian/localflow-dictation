"""Where each model runs - graphics card or processor - and when that changes.

The governor watches the GPU every few seconds and settles on one of four levels. Each step
takes a little more off the graphics card, in the order that costs the least speed for the
most relief:

  0 full     everything on the GPU; live text keeps it clocked up while the key is held
  1 gentle   everything on the GPU, but no keep-warm: the GPU idles between live decodes
  2 light    clean-up moves to the processor (frees ~2.5 GB of VRAM and the heaviest burst
             of GPU work per dictation; costs ~0.4 s on the final text)
  3 off      speech moves too: nothing of LocalFlow's is on the GPU

Why clean-up goes first: measured on the RTX 4060 laptop, Qwen3 4B takes 381 ms on the GPU and
767 ms on the CPU (same quality), while Parakeet is 4x slower on the CPU (41 vs 163 ms per
second of audio) and runs on every dictation.

What moves the level:
  * temperature, against a limit the user sets (default 80 C, the driver throttles at ~87-91 C):
    limit-8 -> gentle, limit -> light, limit+6 -> off
  * another application keeping the GPU busy while LocalFlow is idle (a game, a render) -> light
  * no dictation for a while -> off, which frees the VRAM entirely; the next dictation wakes it
  * on battery -> at least gentle
  * graphics memory other apps have taken (VramGuard): too little room for both models -> light,
    too little for speech -> off. Judged against what LocalFlow's own models take, so its own
    use never counts against it
Levels rise after two readings in a row (about 10 s) and fall one step at a time, only after
the GPU has been 5 C under the threshold for a minute and a half, so a model never ping-pongs
between devices. Wake-ups from idle are immediate: somebody is waiting.

Everything here is pure (time is passed in), so the policy is tested without a GPU.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from localflow.gpu import GpuSample

FULL, GENTLE, LIGHT, OFF = 0, 1, 2, 3
LEVEL_NAMES = {FULL: "full", GENTLE: "gentle", LIGHT: "light", OFF: "off"}

RISE_READINGS = 2  # consecutive readings above a threshold before stepping up
FALL_AFTER_S = 90.0  # how long the GPU must stay under (threshold - FALL_MARGIN_C) to step down
FALL_MARGIN_C = 5
BUSY_UTIL_PCT = 50  # another app's load, judged only while LocalFlow itself is idle
BUSY_READINGS = 4  # ... in this many readings in a row
OWN_WORK_GRACE_S = 8.0  # utilisation this soon after our own work is ours, not someone else's


@dataclass(frozen=True)
class Placement:
    level: int
    reason: str
    speech: str  # "cuda" | "cpu"
    cleanup: str  # "cuda" | "cpu"
    keep_warm: bool

    def as_dict(self) -> dict:
        return {"level": LEVEL_NAMES[self.level], "reason": self.reason, "speech": self.speech,
                "cleanup": self.cleanup, "keep_warm": self.keep_warm}


# Graphics memory needed beside the desktop (~1 GB): Parakeet fp32 ~3 GB, a clean-up model ~2.7 GB.
VRAM_FOR_BOTH_MB = 6144
VRAM_FOR_SPEECH_MB = 4096


def placement_for(level: int, reason: str, *, on_ac: bool, keep_warm_setting: str = "auto",
                  speech_prefers_cpu: bool = False, speech_cpu_usable: bool = True,
                  speech_device_setting: str = "auto", vram_mb: int | None = None,
                  nvidia: bool = True, cleanup_off_card: str = "cpu") -> Placement:
    """Turn a level into devices. `speech_prefers_cpu` is for a model that is simply faster on
    the processor (Parakeet Compact: int8 kernels fall back to the CPU under CUDA anyway);
    `speech_cpu_usable` is False for one too slow there to dictate with (Whisper Turbo), which
    then stays on the GPU at every level short of "off". `vram_mb` is the card's total memory:
    a card too small for both models never gets both, whatever its temperature. Without an
    NVIDIA card (`nvidia` False) nothing goes on one. `cleanup_off_card` is where clean-up goes
    when it is not on the NVIDIA card: the processor, or AMD/Intel graphics ("vulkan", B3)."""
    if not nvidia and level < OFF:
        level, reason = OFF, "there is no NVIDIA graphics card"
    elif vram_mb is not None and vram_mb < VRAM_FOR_SPEECH_MB:
        level, reason = OFF, "the graphics card has too little memory for these models"
    elif vram_mb is not None and vram_mb < VRAM_FOR_BOTH_MB and level < LIGHT:
        level, reason = LIGHT, "the graphics card has room for speech but not clean-up as well"
    speech = "cpu" if level >= OFF or speech_prefers_cpu else "cuda"
    if speech == "cpu" and not speech_cpu_usable and level < OFF:
        speech = "cuda"
    if speech_device_setting in ("cpu", "cuda"):  # an explicit stt.device still wins
        speech = speech_device_setting
    cleanup = cleanup_off_card if level >= LIGHT else "cuda"
    if keep_warm_setting == "always":
        warm = level == FULL
    elif keep_warm_setting == "never":
        warm = False
    else:
        warm = level == FULL and on_ac
    return Placement(level, reason, speech, cleanup, warm and speech == "cuda")


# Graphics memory each model takes, measured 2026-09-28 on the RTX 4060 (NVML's used memory
# before and after): Parakeet v3 2856 MB after decodes of 2-10 s, Qwen3 4B 2706 MB. Compact runs
# on the processor; Whisper Turbo is its download plus the same working memory as Parakeet.
SPEECH_VRAM_MB = {"parakeet-v3": 2900, "parakeet-v2": 2900, "whisper-turbo": 1900, "parakeet-v3-compact": 0}
VRAM_MARGIN_MB = 512  # left for the desktop's own growth, a new window, a video starting
VRAM_BACK_MB = 768  # extra room wanted before moving back, so an estimate a little off never ping-pongs


def cleanup_vram_mb(model_gb: float) -> int:
    """A clean-up model's graphics memory: its file plus about 150 MB of context."""
    return int(model_gb * 1024) + 150


class VramGuard:
    """Keeps models off a graphics card other apps have filled (a game, a video editor, another
    AI app), and brings them back once there is room.

    `observe()` takes the card's total and used memory and LocalFlow's own share of it (the
    models it has there, by the numbers above), so what is left over is everyone else's. Rising
    takes two readings, or one when nothing has been decided yet (start-up: never load onto a
    full card); coming back wants VRAM_BACK_MB more than the bare need, for FALL_AFTER_S."""

    def __init__(self):
        self.level = FULL
        self.reason = ""
        self._first = True
        self._rise_count = 0
        self._roomy_since: float | None = None

    def observe(self, *, total_mb: int, used_mb: int, own_mb: int, speech_mb: int, cleanup_mb: int,
                now: float) -> tuple[int, str]:
        others = max(0, used_mb - own_mb)
        room = total_mb - others - VRAM_MARGIN_MB

        def level_for(extra: int) -> int:
            if room < speech_mb + extra:
                return OFF
            if room < speech_mb + cleanup_mb + extra:
                return LIGHT
            return FULL

        target = level_for(0)
        why = f"the graphics card is full: other apps are using {others / 1024:.1f} of its {total_mb / 1024:.0f} GB"
        if target > self.level:
            self._roomy_since = None
            self._rise_count += 1
            if self._first or self._rise_count >= RISE_READINGS:
                self._first, self._rise_count = False, 0
                self.level, self.reason = target, why
            return self.level, self.reason
        self._first, self._rise_count = False, 0
        if target < self.level:
            comfortable = level_for(VRAM_BACK_MB)
            if comfortable < self.level:
                if self._roomy_since is None:
                    self._roomy_since = now
                elif now - self._roomy_since >= FALL_AFTER_S:
                    self._roomy_since = None
                    self.level = comfortable
                    self.reason = why if comfortable > FULL else ""
                return self.level, self.reason
        self._roomy_since = None
        if self.level > FULL:
            self.reason = why  # the numbers as they are now
        return self.level, self.reason


class Governor:
    """Decides the level from a stream of GPU readings. Call `observe()` every few seconds."""

    def __init__(self, mode: str = "adaptive", temp_limit_c: int = 80, idle_release_min: float = 10.0):
        self.mode = mode
        self.temp_limit_c = temp_limit_c
        self.idle_release_min = idle_release_min
        self.level = FULL
        self.reason = "starting"
        self._temps: deque[int] = deque(maxlen=3)
        self._rise_count = 0
        self._cool_since: float | None = None
        self._busy_count = 0
        self._idle_released = False
        self._jump = True  # the first reading, and the first after a settings change, apply at once

    @property
    def idle_released(self) -> bool:
        """The GPU was freed because LocalFlow sat unused, rather than because of heat or load."""
        return self._idle_released

    def configure(self, mode: str, temp_limit_c: int, idle_release_min: float) -> None:
        changed = (mode, temp_limit_c, idle_release_min) != (self.mode, self.temp_limit_c, self.idle_release_min)
        self.mode, self.temp_limit_c, self.idle_release_min = mode, temp_limit_c, idle_release_min
        if changed:
            self._jump = True

    def thresholds(self) -> tuple[int, int, int]:
        t = self.temp_limit_c
        return t - 8, t, t + 6  # gentle, light, off

    def _level_for_temp(self, temp: float, margin: float = 0) -> int:
        gentle, light, off = (x - margin for x in self.thresholds())
        if temp >= off:
            return OFF
        if temp >= light:
            return LIGHT
        if temp >= gentle:
            return GENTLE
        return FULL

    def observe(self, sample: GpuSample | None, *, now: float, on_ac: bool, idle_s: float,
                own_work_age_s: float) -> tuple[int, str]:
        """Feed one reading. `idle_s`: time since the last dictation or command started.
        `own_work_age_s`: time since LocalFlow last had work on the GPU. Returns (level, reason)."""
        # Every reading goes into the history, whatever the mode and however idle: the median
        # below must describe the last fifteen seconds, not the last time it happened to be
        # consulted. (It once woke from an idle release still believing a 90 C reading from
        # minutes earlier, and kept the models on the processor.)
        if sample is not None:
            self._temps.append(sample.temp_c)
        if self.mode == "cpu":
            self._idle_released = False
            return self._set(OFF, "the graphics card is turned off in settings")
        if self.mode == "gpu":
            self._idle_released = False
            return self._set(FULL if on_ac else GENTLE, "always using the graphics card")
        if sample is None:
            return self._set(FULL if on_ac else GENTLE, "no temperature reading from the graphics card")

        if self.idle_release_min > 0 and idle_s >= self.idle_release_min * 60:
            self._rise_count, self._cool_since, self._idle_released = 0, None, True
            return self._set(OFF, f"not used for {self.idle_release_min:g} minutes, so the graphics card was freed")

        temp = sorted(self._temps)[len(self._temps) // 2]  # median: one spike is not a trend

        # Someone else's load. Our own dictation also shows up as utilisation, so only a GPU
        # that is busy while LocalFlow has been idle for a while counts.
        if own_work_age_s >= OWN_WORK_GRACE_S and sample.util_pct >= BUSY_UTIL_PCT:
            self._busy_count += 1
        else:
            self._busy_count = 0
        external = self._busy_count >= BUSY_READINGS

        target, why = self._level_for_temp(temp), f"graphics card at {temp} °C"
        if external and target < LIGHT:
            target, why = LIGHT, "another app is using the graphics card"
        if not on_ac and target < GENTLE:
            target, why = GENTLE, "on battery"
        if target == FULL:
            why = "graphics card is cool"

        # Waking from an idle release is immediate: someone just started dictating. So is the
        # first reading, and the first after the user changed the settings.
        if self._idle_released or self._jump:
            self._idle_released = self._jump = False
            self._rise_count, self._cool_since = 0, None
            return self._set(target, why)

        if target > self.level:
            self._cool_since = None
            self._rise_count += 1
            if self._rise_count >= RISE_READINGS or target == OFF and temp >= self.thresholds()[2] + 3:
                self._rise_count = 0
                return self._set(target, why)
            return self.level, self.reason
        self._rise_count = 0

        if target < self.level:
            # Step down only once it has been properly cool - below the threshold minus a margin.
            relaxed = self._level_for_temp(temp, margin=FALL_MARGIN_C)
            if external:
                relaxed = max(relaxed, LIGHT)
            if not on_ac:
                relaxed = max(relaxed, GENTLE)
            if relaxed < self.level:
                if self._cool_since is None:
                    self._cool_since = now
                elif now - self._cool_since >= FALL_AFTER_S:
                    self._cool_since = now
                    return self._set(self.level - 1, why if self.level - 1 == target else f"cooling down ({temp} °C)")
                return self.level, self.reason
            self._cool_since = None
            return self.level, self.reason

        self._cool_since = None
        if self.level == target:
            self.reason = why
        return self.level, self.reason

    def _set(self, level: int, reason: str) -> tuple[int, str]:
        self.level, self.reason = level, reason
        return level, reason
