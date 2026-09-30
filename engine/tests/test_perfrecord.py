import json

from localflow import perfrecord as P


def record(**over):
    r = {"version": "0.2.0", "machine": {"gpu": "RTX 4060", "cpu": "Ryzen"},
         "startup": {"speech_ready_s": 3.0, "cleanup_ready_s": 4.0},
         "latency": {"keyup_to_text_p50_ms": 300, "keyup_to_text_p95_ms": 500},
         "accuracy": {"wer_pct": 5.0},
         "memory": {"engine_private_mb": 6000, "cleanup_private_mb": 3000, "vram_mb": 6500},
         "cleanup": {"exact": 22, "violations": 0, "llm_ms_p50": 250}}
    for path, value in over.items():
        section, key = path.split("__")
        r[section][key] = value
    return r


def flagged(new, old):
    return {path for path, _, _, bad in P.compare(new, old) if bad}


def test_small_noise_is_not_a_regression():
    assert flagged(record(latency__keyup_to_text_p50_ms=320, memory__engine_private_mb=6100), record()) == set()


def test_real_regressions_are_flagged_in_the_right_direction():
    worse = record(latency__keyup_to_text_p50_ms=420, accuracy__wer_pct=6.0, cleanup__exact=19,
                   memory__vram_mb=7400)
    assert flagged(worse, record()) == {"latency.keyup_to_text_p50_ms", "accuracy.wer_pct", "cleanup.exact",
                                        "memory.vram_mb"}
    # Better is never a regression, however large the change.
    better = record(latency__keyup_to_text_p50_ms=100, cleanup__exact=29, memory__engine_private_mb=3000)
    assert flagged(better, record()) == set()


def test_a_measure_missing_from_either_record_is_skipped():
    old = record()
    del old["cleanup"]["llm_ms_p50"]
    assert "cleanup.llm_ms_p50" not in {p for p, *_ in P.compare(record(), old)}


def test_the_previous_record_is_the_newest_other_version(tmp_path):
    for v in ("0.1.0", "0.2.0", "0.10.0", "0.2.1"):
        (tmp_path / f"v{v}.json").write_text(json.dumps({"version": v}), encoding="utf-8")
    assert P.previous("0.10.0", tmp_path)["version"] == "0.2.1", "versions sort as numbers"
    assert P.previous("0.2.1", tmp_path)["version"] == "0.10.0"
    assert P.previous("9.9.9", tmp_path / "none") is None
