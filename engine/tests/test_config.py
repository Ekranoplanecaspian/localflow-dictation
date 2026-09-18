import json

from localflow.config import CONFIG_VERSION, Config, migrate


def test_v1_config_drops_the_stale_ollama_settings_and_keeps_the_user_s_own():
    v1 = {
        "hotkey": {"keys": ["f8"], "double_tap_ms": 300},
        "audio": {"device": "NVIDIA Broadcast"},
        "postprocess": {
            "llm_cleanup": False,            # the old default: off, because Ollama was optional
            "llm_model": "qwen3:4b",         # an Ollama tag, meaningless to the bundled server
            "llm_url": "http://127.0.0.1:11434",
            "remove_fillers": True,
            "dictionary": {"arnub": "Arnab"},
            "snippets": {"my email": "a@example.com"},
        },
    }
    out = migrate(v1)
    assert out is not v1 and v1["postprocess"]["llm_model"] == "qwen3:4b", "input must not be mutated"
    assert out["version"] == CONFIG_VERSION
    assert "llm_cleanup" not in out["postprocess"] and "llm_url" not in out["postprocess"]
    assert out["postprocess"]["dictionary"] == {"arnub": "Arnab"}
    assert out["hotkey"]["keys"] == ["f8"] and out["audio"]["device"] == "NVIDIA Broadcast"


def test_migrated_config_gets_the_new_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"postprocess": {"llm_cleanup": False, "llm_model": "qwen3:4b"}}), encoding="utf-8")
    cfg = Config.load(path)
    assert cfg.version == CONFIG_VERSION
    assert cfg.postprocess.llm_cleanup is True, "the AI layer is on by default now"
    assert cfg.postprocess.llm_provider == "bundled" and cfg.postprocess.llm_model == "qwen3-4b"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["version"] == CONFIG_VERSION, "the migration should be written back to disk"


def test_current_config_is_left_alone():
    current = {"version": CONFIG_VERSION, "postprocess": {"llm_cleanup": False, "llm_model": "qwen3-1.7b"}}
    assert migrate(current) is current
    assert Config().version == CONFIG_VERSION


def test_round_trip(tmp_path):
    path = tmp_path / "config.json"
    cfg = Config()
    cfg.postprocess.dictionary_terms = ["Arnab", "Okonkwo"]
    cfg.save(path)
    again = Config.load(path)
    assert again.postprocess.dictionary_terms == ["Arnab", "Okonkwo"]
    assert again.stt.backend == cfg.stt.backend and again.hotkey.keys == cfg.hotkey.keys
