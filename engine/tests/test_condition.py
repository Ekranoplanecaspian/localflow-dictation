import numpy as np

from localflow.service.engine import DC_THRESHOLD, MAX_GAIN, QUIET_PEAK, TARGET_PEAK, condition


def tone(amplitude: float, dc: float = 0.0) -> np.ndarray:
    return (np.sin(np.linspace(0, 200, 16000)) * amplitude + dc).astype(np.float32)


def test_whispered_audio_is_lifted_up_to_the_gain_cap():
    # Anything under the whisper threshold gets at most +24 dB, never past the target peak.
    for amp in (0.01, 0.002):  # -40 and -54 dBFS
        out = condition(tone(amp))
        expected = min(amp * MAX_GAIN, TARGET_PEAK)
        assert abs(float(np.max(np.abs(out))) - expected) < 1e-4
        assert float(np.max(np.abs(out))) > amp * 10  # clearly louder than it was


def test_normal_laptop_mic_levels_are_untouched():
    for amp in (0.05, 0.1, 0.3):  # -26, -20, -10 dBFS: all above the whisper threshold
        x = tone(amp)
        assert np.array_equal(condition(x), x)
    assert QUIET_PEAK < 0.05 < TARGET_PEAK


def test_real_dc_offset_is_removed_but_tiny_ones_are_left():
    with_dc = tone(0.3, dc=0.1)
    out = condition(with_dc)
    assert abs(float(out.mean())) < 1e-3
    tiny = tone(0.3, dc=DC_THRESHOLD / 2)
    assert np.array_equal(condition(tiny), tiny)


def test_empty_audio():
    assert condition(np.zeros(0, np.float32)).size == 0
