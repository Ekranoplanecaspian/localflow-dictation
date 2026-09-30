"""The CPU/GPU placement policy, driven by made-up GPU readings: no GPU needed."""

from localflow.gpu import GpuSample
from localflow.placement import FULL, GENTLE, LIGHT, OFF, Governor, placement_for


def reading(temp, util=5):
    return GpuSample(temp_c=temp, util_pct=util, mem_used_mb=3000, mem_total_mb=8188, slowdown_c=91)


class Clock:
    def __init__(self):
        self.t = 0.0

    def feed(self, gov, temp, util=5, *, on_ac=True, idle_s=0.0, own_age=0.0, steps=1, every=5.0):
        level = None
        for _ in range(steps):
            self.t += every
            level, _ = gov.observe(reading(temp, util), now=self.t, on_ac=on_ac, idle_s=idle_s,
                                   own_work_age_s=own_age)
        return level


def test_cool_gpu_runs_everything_on_it():
    gov, clk = Governor(), Clock()
    assert clk.feed(gov, 50) == FULL
    p = placement_for(FULL, gov.reason, on_ac=True)
    assert (p.speech, p.cleanup, p.keep_warm) == ("cuda", "cuda", True)


def test_heat_steps_up_through_the_levels_after_two_readings():
    gov, clk = Governor(temp_limit_c=80), Clock()
    clk.feed(gov, 50)
    assert clk.feed(gov, 74) == FULL  # one reading is not enough...
    assert clk.feed(gov, 74) == GENTLE  # ...two are (median of 50, 74, 74)
    assert clk.feed(gov, 82, steps=3) == LIGHT
    assert "82" in gov.reason
    assert clk.feed(gov, 87, steps=3) == OFF


def test_a_single_spike_does_not_move_anything():
    gov, clk = Governor(temp_limit_c=80), Clock()
    clk.feed(gov, 55, steps=3)
    assert clk.feed(gov, 84) == FULL  # the median of 55, 55, 84 is still 55
    assert clk.feed(gov, 56) == FULL


def test_cooling_down_is_slow_and_one_step_at_a_time():
    gov, clk = Governor(temp_limit_c=80), Clock()
    clk.feed(gov, 50)
    clk.feed(gov, 88, steps=4)
    assert gov.level == OFF
    # Under the "off" threshold (86) but not 5 C under it: stays off, however long.
    assert clk.feed(gov, 84, steps=60) == OFF
    # Properly cool: one level per 90 s, never a jump straight to full.
    seen = [clk.feed(gov, 55) for _ in range(80)]  # 400 s
    drops = [(a, b) for a, b in zip(seen, seen[1:]) if b != a]
    assert drops == [(OFF, LIGHT), (LIGHT, GENTLE), (GENTLE, FULL)]
    assert seen.index(LIGHT) >= 17  # ~90 s of readings before the first step


def test_hovering_near_a_threshold_does_not_ping_pong():
    gov, clk = Governor(temp_limit_c=80), Clock()
    clk.feed(gov, 50)
    levels = [clk.feed(gov, t) for t in [81, 81, 79, 81, 78, 80, 79, 81, 78, 80] * 6]
    changes = sum(1 for a, b in zip(levels, levels[1:]) if a != b)
    assert changes <= 1 and levels[-1] == LIGHT


def test_another_app_using_the_gpu_moves_clean_up_off_it():
    gov, clk = Governor(), Clock()
    clk.feed(gov, 60)
    assert clk.feed(gov, 60, util=90, own_age=60, steps=3) == FULL  # not yet: four in a row
    assert clk.feed(gov, 60, util=90, own_age=60) == FULL  # the rise itself needs two readings
    assert clk.feed(gov, 60, util=90, own_age=60) == LIGHT
    assert "another app" in gov.reason


def test_our_own_dictation_is_not_mistaken_for_another_app():
    gov, clk = Governor(), Clock()
    clk.feed(gov, 60)
    assert clk.feed(gov, 62, util=100, own_age=1.0, steps=10) == FULL


def test_idle_frees_the_gpu_and_the_next_dictation_wakes_it_at_once():
    gov, clk = Governor(idle_release_min=10), Clock()
    clk.feed(gov, 50)
    assert clk.feed(gov, 50, idle_s=599) == FULL
    assert clk.feed(gov, 48, idle_s=600) == OFF
    assert "freed" in gov.reason
    assert placement_for(OFF, gov.reason, on_ac=True).speech == "cpu"
    assert clk.feed(gov, 48, idle_s=0) == FULL  # no 90 s wait: somebody is dictating


def test_idle_release_can_be_turned_off():
    gov, clk = Governor(idle_release_min=0), Clock()
    assert clk.feed(gov, 50, idle_s=10_000) == FULL


def test_battery_means_no_keep_warm():
    gov, clk = Governor(), Clock()
    assert clk.feed(gov, 50, on_ac=False) == GENTLE
    p = placement_for(GENTLE, gov.reason, on_ac=False)
    assert (p.speech, p.cleanup, p.keep_warm) == ("cuda", "cuda", False)


def test_modes_override_the_readings():
    gov, clk = Governor(mode="cpu"), Clock()
    assert clk.feed(gov, 40) == OFF
    gov.configure("gpu", 80, 10)
    assert clk.feed(gov, 95) == FULL


def test_changing_the_settings_applies_at_once():
    gov, clk = Governor(mode="cpu"), Clock()
    clk.feed(gov, 50)
    gov.configure("adaptive", 80, 10)
    assert clk.feed(gov, 50) == FULL  # not four 90-second steps down from "off"


def test_no_reading_behaves_like_before():
    gov = Governor()
    assert gov.observe(None, now=0, on_ac=True, idle_s=0, own_work_age_s=0)[0] == FULL


def test_compact_parakeet_always_runs_on_the_processor():
    assert placement_for(FULL, "", on_ac=True, speech_prefers_cpu=True).speech == "cpu"
    assert placement_for(FULL, "", on_ac=True, speech_prefers_cpu=True).keep_warm is False


def test_whisper_stays_on_the_gpu_until_everything_has_to_leave():
    assert placement_for(LIGHT, "", on_ac=True, speech_cpu_usable=False).speech == "cuda"
    assert placement_for(OFF, "", on_ac=True, speech_cpu_usable=False).speech == "cpu"


def test_an_explicit_speech_device_setting_still_wins():
    assert placement_for(OFF, "", on_ac=True, speech_device_setting="cuda").speech == "cuda"
    assert placement_for(FULL, "", on_ac=True, speech_device_setting="cpu").speech == "cpu"


def test_waking_from_idle_uses_todays_temperature_not_the_last_one_it_looked_at():
    gov, clk = Governor(temp_limit_c=80, idle_release_min=10), Clock()
    clk.feed(gov, 50)
    clk.feed(gov, 90, steps=4)
    assert gov.level == OFF
    # it cools down while nobody dictates, and the GPU is released for idleness
    assert clk.feed(gov, 50, idle_s=900, steps=10) == OFF
    assert clk.feed(gov, 50, idle_s=0) == FULL


def test_a_spell_in_always_gpu_mode_does_not_leave_stale_readings():
    gov, clk = Governor(temp_limit_c=80), Clock()
    clk.feed(gov, 90, steps=4)
    gov.configure("gpu", 80, 10)
    clk.feed(gov, 50, steps=5)
    gov.configure("adaptive", 80, 10)
    assert clk.feed(gov, 50) == FULL


def test_vram_guard_comes_back_only_with_room_to_spare():
    """An estimate of LocalFlow's own share a little off must not bounce a model between the
    graphics card and the processor: going back wants 768 MB more than the bare need."""
    from localflow.placement import VramGuard

    g = VramGuard()
    need = dict(speech_mb=2900, cleanup_mb=2710)
    # at start-up, 3 GB of the card already taken by others: speech fits, clean-up does not
    assert g.observe(total_mb=8188, used_mb=3072, own_mb=0, now=0, **need)[0] == LIGHT
    # others shrink to leave exactly enough for both, but not the extra: it stays put for good
    for t in range(0, 600, 5):
        level, _ = g.observe(total_mb=8188, used_mb=2900 + 2000, own_mb=2900, now=t, **need)
    assert level == LIGHT
    # with room to spare, it comes back after a minute and a half
    levels = [g.observe(total_mb=8188, used_mb=2900 + 1000, own_mb=2900, now=600 + t, **need)[0]
              for t in range(0, 100, 5)]
    assert levels[0] == LIGHT and levels[-1] == FULL
