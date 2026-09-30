"""The placement controller against a fake engine and a scripted GPU: which moves it makes,
when, and what it refuses to do."""

import threading

import pytest

from localflow import modelchoice
from localflow.config import Config
from localflow.gpu import GpuSample
from localflow.placement import FULL, LIGHT, OFF
from localflow.service import compute as C


class ScriptedGpu:
    name = "Test GPU"

    def __init__(self):
        self.temp = 50
        self.util = 5
        self.used = 3000  # MB of the card's 8188 in use, everyone's together

    def sample(self):
        return GpuSample(self.temp, self.util, self.used, 8188, 91)


class FakeEngine:
    def __init__(self, speech="cuda", cleanup="cuda"):
        self.cfg = Config()
        self.state = "ready"
        self._model_lock = threading.Lock()
        self.where = {"speech": speech, "cleanup": cleanup}
        self.moves: list[tuple[str, str]] = []
        self.fail: set[tuple[str, str]] = set()
        self.fallback: set[str] = set()  # asked for cuda, ends up on cpu
        self.perf = modelchoice.PerfLog(None)
        self.models = {"speech": "parakeet-v3", "cleanup": "qwen3-4b"}
        self.model_moves: list[tuple[str, str, str]] = []
        self.llm_state = "ready"
        self.slept, self.woke = 0, []

    def devices(self):
        return {**self.where, "speech_model": self.models["speech"],
                "cleanup_model": self.models["cleanup"] if self.where["cleanup"] else None}

    def _move(self, what, where, model=None):
        self.moves.append((what, where))
        if (what, where) in self.fail:
            raise RuntimeError("no room")
        self.where[what] = "cpu" if (where == "cuda" and what in self.fallback) else where
        if model:
            self.models[what] = model
            self.model_moves.append((what, where, model))
        return True

    def move_speech(self, where, model=None):
        return self._move("speech", where, model)

    def move_cleanup(self, where, model=None):
        return self._move("cleanup", where, model)

    def sleep_cleanup(self):
        self.where["cleanup"], self.llm_state = None, "asleep"
        self.slept += 1
        return True

    def wake_cleanup(self, device):
        self.where["cleanup"], self.llm_state = device, "ready"
        self.woke.append(device)

    def unload_unused(self):
        pass

    def _broadcast_status(self):
        pass


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setattr(C, "_on_ac", lambda: True)
    engine, gpu, clock = FakeEngine(), ScriptedGpu(), Clock()
    ctl = C.ComputeController(engine, monitor=gpu, clock=clock)
    ctl.placement = ctl.decide()
    ctl.asked = {"speech": "cuda", "cleanup": "cuda"}

    def tick(n=1, every=5.0, **gpu_state):
        for k, v in gpu_state.items():
            setattr(gpu, k, v)
        for _ in range(n):
            clock.t += every
            ctl.apply(ctl.decide())
        return ctl.placement

    return engine, gpu, clock, ctl, tick


def test_a_cool_gpu_keeps_everything_on_it_and_keeps_it_warm(rig):
    engine, _, _, ctl, tick = rig
    p = tick(3)
    assert (p.speech, p.cleanup, p.keep_warm) == ("cuda", "cuda", True)
    assert engine.moves == []


def test_heat_moves_clean_up_first_then_speech_and_back_when_cool(rig):
    engine, _, clock, ctl, tick = rig
    engine.cfg.compute.idle_release_min = 0  # ten quiet minutes follow; this test is about heat
    ctl.configure()
    tick(3, temp=82)
    assert engine.moves == [("cleanup", "cpu")]
    assert ctl.placement.level == LIGHT and ctl.placement.keep_warm is False
    tick(3, temp=88)
    assert engine.moves[-1] == ("speech", "cpu") and ctl.placement.level == OFF
    tick(120, temp=50)  # ten minutes of cool readings
    assert engine.where == {"speech": "cuda", "cleanup": "cuda"} and ctl.placement.level == FULL
    assert [m["what"] for m in ctl.recent][:2] == ["clean-up", "speech"]  # newest first: speech came back first


def test_idle_frees_the_gpu_and_a_dictation_brings_it_back_immediately(rig):
    engine, _, clock, ctl, tick = rig
    tick(1)
    clock.t += 600
    tick(1)
    # Speech moves to the processor so dictation stays instant; clean-up is unloaded, not moved:
    # on the processor it held 2-3 GB of memory for nothing.
    assert engine.where == {"speech": "cpu", "cleanup": None}
    assert engine.slept == 1 and ("cleanup", "cpu") not in engine.moves
    assert "freed" in ctl.placement.reason
    ctl.activity()  # someone presses the hotkey
    assert ctl._wake.is_set()
    tick(1, every=0.1)
    # Woken straight onto the graphics card, not loaded on the processor and moved after.
    assert engine.woke == ["cuda"]
    assert engine.where == {"speech": "cuda", "cleanup": "cuda"}
    assert ("cleanup", "cuda") not in engine.moves


def test_a_gpu_that_falls_back_to_the_processor_is_not_retried(rig):
    engine, _, _, ctl, tick = rig
    engine.where["speech"] = "cpu"  # loaded for cuda, came up on cpu (no working CUDA)
    tick(10)
    assert engine.moves == [] and "speech" in ctl.no_gpu
    assert ctl.placement.speech == "cpu" and ctl.placement.keep_warm is False


def test_a_failed_move_waits_two_minutes_before_trying_again(rig):
    engine, _, clock, ctl, tick = rig
    engine.fail.add(("cleanup", "cpu"))
    tick(3, temp=82)
    assert engine.moves == [("cleanup", "cpu")]
    tick(10, temp=82)  # 50 s later: not yet
    assert engine.moves == [("cleanup", "cpu")]
    engine.fail.clear()
    tick(16, temp=82)  # past two minutes
    assert engine.moves == [("cleanup", "cpu")] * 2 and engine.where["cleanup"] == "cpu"


def test_a_model_change_in_progress_is_never_interrupted(rig):
    engine, _, _, ctl, tick = rig
    engine._model_lock.acquire()
    tick(4, temp=82)
    assert engine.moves == []
    engine._model_lock.release()
    tick(1, temp=82)
    assert engine.moves == [("cleanup", "cpu")]


def test_nothing_moves_while_the_engine_is_loading(rig):
    engine, _, _, ctl, tick = rig
    engine.state = "loading"
    tick(4, temp=90)
    assert engine.moves == []


def test_a_model_that_is_not_loaded_is_placed_rather_than_moved(rig):
    engine, _, _, ctl, tick = rig
    engine.where["cleanup"] = None  # clean-up still starting, or not on this computer
    tick(3, temp=82)
    assert engine.moves == [] and ctl.placement.cleanup == "cpu"  # it will load there


def test_graphics_card_off_in_settings(rig):
    engine, _, _, ctl, tick = rig
    engine.cfg.compute.mode = "cpu"
    ctl.configure()
    tick(1)
    assert engine.where == {"speech": "cpu", "cleanup": "cpu"}
    assert "settings" in ctl.placement.reason


def test_status_reports_the_gpu_and_what_moved(rig):
    engine, _, _, ctl, tick = rig
    tick(3, temp=82)
    s = ctl.status()
    assert s["gpu"]["temp_c"] == 82 and s["gpu"]["name"] == "Test GPU"
    assert s["cleanup"] == "cpu" and s["level"] == "light" and s["moving"] is None
    assert s["recent"][0]["what"] == "clean-up" and s["mode"] == "adaptive"
    # the capability report, with free memory and disk read fresh on each status
    hw = s["hardware"]
    assert hw["cpu"]["cores"] == hw["cpu_cores"] and isinstance(hw["gpus"], list)
    assert hw["disk_free_gb"] > 0 and "ram_free_gb" in hw


# which model ------------------------------------------------------------------------------------

def slow_machine(ctl, cores=4):
    ctl.hardware = modelchoice.Hardware(cpu_cores=cores, ram_gb=8, vram_mb=8188)


def test_on_this_machine_automatic_keeps_the_best_models_everywhere(rig):
    engine, _, _, ctl, tick = rig
    ctl.hardware = modelchoice.Hardware(cpu_cores=12, ram_gb=32, vram_mb=8188)
    engine.cfg.compute.mode = "cpu"
    ctl.configure()
    tick(1)
    assert engine.where == {"speech": "cpu", "cleanup": "cpu"}
    assert engine.model_moves == []  # moved, but not changed: v3 and Qwen3 4B keep up here
    assert ctl.chosen["cleanup"].key == "qwen3-4b" and "most accurate" in ctl.chosen["cleanup"].why


def test_a_slow_processor_gets_a_lighter_clean_up_model_only_when_clean_up_runs_there(rig, monkeypatch):
    engine, _, _, ctl, tick = rig
    monkeypatch.setattr(C.llm_manifest, "gguf_path", lambda key: type("P", (), {"exists": lambda self: True})())
    slow_machine(ctl, cores=6)  # Qwen3 4B ~1.5 s here, Phi-4 mini ~1.2 s
    tick(1)
    assert engine.model_moves == []  # on the graphics card Qwen3 4B is quick enough
    tick(3, temp=82)  # clean-up moves to the processor, which is too slow for Qwen3 4B there
    assert ("cleanup", "cpu", "phi-4-mini") in engine.model_moves
    assert "keeps up" in ctl.chosen["cleanup"].why
    assert ctl.recent[0]["model"] == "Phi-4 mini"


def test_measurements_on_this_machine_outrank_the_estimate(rig, monkeypatch):
    engine, _, _, ctl, tick = rig
    monkeypatch.setattr(C.llm_manifest, "gguf_path", lambda key: type("P", (), {"exists": lambda self: True})())
    ctl.hardware = modelchoice.Hardware(cpu_cores=12, ram_gb=32, vram_mb=8188)
    for _ in range(6):  # Qwen3 4B turned out slow on this processor after all
        engine.perf.record("cleanup", "qwen3-4b", "cpu", 2400)
    tick(3, temp=82)
    assert engine.models["cleanup"] == "phi-4-mini"
    assert "estimated" in ctl.chosen["cleanup"].why  # Phi's own number is still the estimate


def test_a_model_chosen_by_hand_is_never_replaced(rig, monkeypatch):
    engine, _, _, ctl, tick = rig
    monkeypatch.setattr(C.llm_manifest, "gguf_path", lambda key: type("P", (), {"exists": lambda self: True})())
    slow_machine(ctl)
    engine.cfg.compute.auto_cleanup = False
    tick(3, temp=82)
    assert engine.where["cleanup"] == "cpu" and engine.model_moves == []
    assert ctl.chosen["cleanup"] is None


def test_automatic_does_not_download_a_big_model_unasked(rig, monkeypatch):
    engine, _, _, ctl, tick = rig
    monkeypatch.setattr(C.llm_manifest, "gguf_path", lambda key: type("P", (), {"exists": lambda self: False})())
    slow_machine(ctl)
    tick(3, temp=82)
    assert engine.model_moves == []  # Phi-4 mini would be better here, but it is not on disk
    assert ctl.chosen["cleanup"].key == "qwen3-4b"


def test_a_slow_processor_runs_compact_speech_but_only_on_the_processor(rig):
    """B4: a 4-core processor is quick enough for v3 by the estimate; it takes this one's own
    timings - v3 over the budget, and Compact, by the same ratio, under it - to step down."""
    engine, _, _, ctl, tick = rig
    slow_machine(ctl, cores=4)
    for _ in range(6):
        engine.perf.record("speech", "parakeet-v3", "cpu", 280)
    engine.cfg.compute.idle_release_min = 0
    ctl.configure()
    tick(4, temp=90)  # everything leaves the graphics card
    assert ("speech", "cpu", "parakeet-v3-compact") in engine.model_moves
    engine.cfg.compute.mode = "gpu"  # and comes back: on the GPU, v3 is best again
    ctl.configure()
    tick(1, temp=50)
    assert engine.models["speech"] == "parakeet-v3" and engine.where["speech"] == "cuda"


def test_when_nothing_keeps_up_accuracy_still_wins_among_the_quickest():
    very_slow = modelchoice.Hardware(cpu_cores=4, ram_gb=8, vram_mb=None)
    pick = modelchoice.choose_cleanup("cpu", very_slow, modelchoice.PerfLog(None), lambda k: True)
    # Gemma is quickest by a few tens of milliseconds; Phi-4 mini is more accurate
    assert pick.key == "phi-4-mini" and "nothing is quick enough" in pick.why


# too little memory (B5) -------------------------------------------------------------------------

def test_short_of_memory_automatic_takes_compact_speech_and_says_why(rig, monkeypatch):
    from localflow import hwinfo

    engine, _, _, ctl, tick = rig
    engine.cfg.compute.mode = "cpu"
    ctl.configure()
    engine.where["speech"], engine.models["speech"] = None, None  # nothing loaded yet: the first choice
    monkeypatch.setattr(hwinfo, "ram_free_gb", lambda: 2.0)
    picks = ctl.choices("cpu", "cpu")
    assert picks["speech"].key == "parakeet-v3-compact"
    assert picks["speech"].why.startswith("Parakeet v3 needs about 3.0 GB of free memory, and 2.0 GB is free; "
                                          "this is the most accurate one that fits")


def test_a_model_already_loaded_fits_by_definition(rig, monkeypatch):
    """Loaded, v3 is part of what is in use: free RAM is what is left beside it, and switching
    away would not bring anything back."""
    from localflow import hwinfo

    engine, _, _, ctl, tick = rig
    engine.cfg.compute.mode = "cpu"
    ctl.configure()
    tick(1)
    assert engine.where["speech"] == "cpu" and engine.models["speech"] == "parakeet-v3"
    monkeypatch.setattr(hwinfo, "ram_free_gb", lambda: 1.0)
    tick(2)
    assert engine.models["speech"] == "parakeet-v3"


def test_on_the_graphics_card_the_full_model_needs_little_ram(rig, monkeypatch):
    from localflow import hwinfo

    engine, _, _, ctl, tick = rig
    engine.where["speech"], engine.models["speech"] = None, None
    monkeypatch.setattr(hwinfo, "ram_free_gb", lambda: 2.5)  # too little for v3 on the processor
    assert ctl.choices("cuda", "cuda")["speech"].key == "parakeet-v3"
    assert ctl.choices("cpu", "cpu")["speech"].key == "parakeet-v3-compact"


def test_when_memory_frees_up_automatic_goes_back_to_the_full_model(rig, monkeypatch):
    from localflow import hwinfo

    engine, _, _, ctl, tick = rig
    engine.cfg.compute.mode = "cpu"
    ctl.configure()
    free = {"gb": 2.0}
    monkeypatch.setattr(hwinfo, "ram_free_gb", lambda: free["gb"])
    monkeypatch.setattr(C.catalogue, "is_installed", lambda m, device: True)
    tick(1)
    assert engine.models["speech"] == "parakeet-v3-compact"
    free["gb"] = 6.0  # programs closed
    tick(1)
    assert engine.models["speech"] == "parakeet-v3"


def test_a_speech_model_chosen_by_hand_stays_however_little_memory_there_is(rig, monkeypatch):
    from localflow import hwinfo

    engine, _, _, ctl, tick = rig
    engine.cfg.compute.auto_speech = False
    monkeypatch.setattr(hwinfo, "ram_free_gb", lambda: 0.5)
    tick(2)
    assert engine.models["speech"] == "parakeet-v3" and ctl.chosen["speech"] is None


def test_when_nothing_fits_the_smallest_is_chosen():
    hw = modelchoice.Hardware(cpu_cores=8, ram_gb=8, vram_mb=None)
    pick = modelchoice.choose_speech("cpu", hw, modelchoice.PerfLog(None), lambda k: True,
                                     lambda k: f"{k} does not fit")
    assert pick.key == "parakeet-v3-compact" and pick.why == "parakeet-v3 does not fit; nothing fits, so the smallest"


# a graphics card other apps have filled (B5) ------------------------------------------------------

def test_a_card_another_app_filled_keeps_the_models_off_it_until_there_is_room(rig):
    engine, gpu, clock, ctl, tick = rig
    engine.cfg.compute.idle_release_min = 0
    ctl.configure()
    tick(2)
    assert engine.where == {"speech": "cuda", "cleanup": "cuda"}
    # A game takes 1.5 GB beside the desktop's 1 and our 5.5: the card is nearly full, with
    # room for speech (2.9) but not clean-up (2.7) as well
    gpu.used = 1000 + 1500 + 2900 + 2710
    tick(2)
    assert engine.where == {"speech": "cuda", "cleanup": "cpu"}
    assert ctl.placement.reason == "the graphics card is full: other apps are using 2.4 of its 8 GB"
    gpu.used = 1000 + 1500 + 2900  # clean-up's memory is freed; the game is still there
    tick(3)
    assert engine.where == {"speech": "cuda", "cleanup": "cpu"}
    # The game grows to 3.9 GB: no room for speech either
    gpu.used = 1000 + 3900 + 2900
    tick(2)
    assert engine.where == {"speech": "cpu", "cleanup": "cpu"}
    # The game quits. Nothing comes back at once...
    gpu.used = 1000
    tick(3)
    assert engine.where == {"speech": "cpu", "cleanup": "cpu"}
    # ...but after a minute and a half of room, it does
    tick(20)
    assert engine.where == {"speech": "cuda", "cleanup": "cuda"}


def test_its_own_models_never_count_against_it(rig):
    engine, gpu, clock, ctl, tick = rig
    gpu.used = 1100 + 2900 + 2710  # the desktop and LocalFlow's two models: 6.7 of 8 GB
    tick(5)
    assert engine.where == {"speech": "cuda", "cleanup": "cuda"} and engine.moves == []


def test_at_start_up_nothing_is_loaded_onto_a_full_card(monkeypatch):
    monkeypatch.setattr(C, "_on_ac", lambda: True)
    engine, gpu = FakeEngine(speech=None, cleanup=None), ScriptedGpu()
    engine.models = {"speech": None, "cleanup": None}
    gpu.used = 6500  # before anything of LocalFlow's is there
    ctl = C.ComputeController(engine, monitor=gpu, clock=Clock())
    p = ctl.decide()
    assert (p.speech, p.cleanup) == ("cpu", "cpu") and "the graphics card is full" in p.reason


def test_processor_only_is_left_alone(rig):
    engine, gpu, clock, ctl, tick = rig
    engine.cfg.compute.mode = "cpu"
    ctl.configure()
    gpu.used = 8000
    tick(2)
    assert ctl.placement.reason == "the graphics card is turned off in settings"



# AMD and Intel graphics for clean-up (B3) ----------------------------------------------------------

def _machine(ctl, *gpus, cores=12):
    from localflow import hwinfo

    report = hwinfo.Report(cpu=hwinfo.Cpu("Ryzen", cores, cores * 2, ("avx2",)), ram_gb=32.0, gpus=tuple(gpus))
    nvidia = any(g.vendor == "nvidia" for g in gpus)
    ctl.hardware = modelchoice.Hardware(cpu_cores=cores, ram_gb=32.0, vram_mb=8188 if nvidia else None, report=report)


def _rtx():
    from localflow import hwinfo
    return hwinfo.Gpu("NVIDIA GeForce RTX 4060 Laptop GPU", "nvidia", 7956, 15932, integrated=False)


def _radeon_890m():
    from localflow import hwinfo
    return hwinfo.Gpu("AMD Radeon(TM) 890M Graphics", "amd", 338, 15932, integrated=True)


def test_a_hot_nvidia_card_sends_clean_up_to_the_built_in_graphics_when_quicker(rig):
    engine, _, _, ctl, tick = rig
    _machine(ctl, _rtx(), _radeon_890m())
    tick(3, temp=82)
    assert engine.where == {"speech": "cuda", "cleanup": "vulkan"}
    assert ctl.moving is None and ctl.recent[0]["to"] == "vulkan"
    assert ctl.status()["vulkan"] == "Built-in graphics"


def test_built_in_graphics_measured_slower_than_the_processor_are_left_alone(rig):
    engine, _, _, ctl, tick = rig
    _machine(ctl, _rtx(), _radeon_890m())
    for _ in range(6):
        engine.perf.record("cleanup", "qwen3-4b", "vulkan", 1500)  # slower here than the processor's 767
    tick(3, temp=82)
    assert engine.where["cleanup"] == "cpu"


def test_without_an_nvidia_card_speech_is_on_the_processor_and_clean_up_on_the_graphics(rig):
    engine, _, _, ctl, tick = rig
    _machine(ctl, _radeon_890m())
    tick(2)
    assert engine.where == {"speech": "cpu", "cleanup": "vulkan"}
    assert ctl.placement.reason == "there is no NVIDIA graphics card"


def test_processor_only_means_no_graphics_of_any_kind(rig):
    engine, _, _, ctl, tick = rig
    _machine(ctl, _rtx(), _radeon_890m())
    engine.cfg.compute.mode = "cpu"
    ctl.configure()
    tick(2)
    assert engine.where == {"speech": "cpu", "cleanup": "cpu"}


def test_graphics_that_fail_are_not_tried_again(rig):
    engine, _, _, ctl, tick = rig
    _machine(ctl, _rtx(), _radeon_890m())
    engine.fail.add(("cleanup", "vulkan"))
    tick(6, temp=82)
    assert ctl.no_vulkan and engine.where["cleanup"] == "cpu"  # the next reading after the failure
    assert engine.moves.count(("cleanup", "vulkan")) == 1


def test_built_in_graphics_with_too_little_memory_are_skipped(rig):
    from localflow import hwinfo

    engine, _, _, ctl, tick = rig
    _machine(ctl, _rtx(), hwinfo.Gpu("Intel(R) UHD Graphics 620", "intel", 128, 1900, integrated=True))
    tick(3, temp=82)
    assert engine.where["cleanup"] == "cpu"


def test_a_discrete_amd_card_is_called_by_its_name(rig):
    from localflow import hwinfo

    engine, _, _, ctl, tick = rig
    _machine(ctl, hwinfo.Gpu("AMD Radeon RX 7600", "amd", 8176, 15932, integrated=False))
    assert ctl.device_words("vulkan") == "the AMD Radeon RX 7600"
    assert ctl.status()["vulkan"] == "AMD Radeon RX 7600"


def test_impossible_timings_are_ignored_even_in_an_old_file(tmp_path):
    """A test's instant fake speech model once filled perf.json with zeros."""
    import json

    path = tmp_path / "perf.json"
    path.write_text(json.dumps({"speech:parakeet-v3:cpu": [0.0] * 30 + [70.0]}), encoding="utf-8")
    perf = modelchoice.PerfLog(path)
    assert perf.samples["speech:parakeet-v3:cpu"] == [70.0]
    perf.record("speech", "parakeet-v3", "cpu", 0.0)
    assert perf.samples["speech:parakeet-v3:cpu"] == [70.0]


def test_a_four_core_processor_keeps_the_most_accurate_speech_model():
    """B4: the estimate used to scale by cores in a straight line, which made a 4-core PC look six
    times slower than measured, and Automatic settled on Compact there for good."""
    hw = modelchoice.Hardware(cpu_cores=4, ram_gb=16, vram_mb=None)
    pick = modelchoice.choose_speech("cpu", hw, modelchoice.PerfLog(None), lambda k: True)
    assert pick.key == "parakeet-v3" and "quick enough on the processor" in pick.why


def test_a_model_never_used_here_is_judged_by_one_that_was():
    hw = modelchoice.Hardware(cpu_cores=12, ram_gb=32, vram_mb=None)
    perf = modelchoice.PerfLog(None)
    for _ in range(5):
        perf.record("speech", "parakeet-v3-compact", "cpu", 114)  # this PC: twice the estimate
    assert perf.estimate("speech", "parakeet-v3", "cpu", hw) == (134.0, False)  # 67 x 114 / 57

