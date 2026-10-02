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
    assert again.stt.backend == cfg.stt.backend and again.audio.device == cfg.audio.device


def test_sections_nothing_reads_any_more_are_dropped(tmp_path):
    """The Python tray app's hotkey, injection and overlay settings: the shell has its own."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"version": CONFIG_VERSION, "hotkey": {"keys": ["f8"]},
                                "inject": {"method": "paste"}, "ui": {"overlay": False},
                                "audio": {"device": "USB", "preroll_ms": 300}}), encoding="utf-8")
    cfg = Config.load(path)
    assert cfg.audio.device == "USB"
    cfg.save(path)
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {"audio", "stt", "postprocess", "compute", "network",
                                                                   "log_level", "version"}


def test_an_unreadable_config_starts_on_defaults_and_keeps_the_broken_file(tmp_path):
    """Half a file (from a crash mid-save) used to stop the engine at start-up, every time."""
    path = tmp_path / "config.json"
    path.write_text('{"postprocess": {"dictionary_terms": ["Arnab"', encoding="utf-8")
    cfg = Config.load(path)
    assert cfg.postprocess.dictionary_terms == [], "defaults"
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == CONFIG_VERSION, "a good file again"
    kept = list(tmp_path.glob("config.json.broken-*"))
    assert len(kept) == 1 and "Arnab" in kept[0].read_text(encoding="utf-8"), "the user's words are kept"


def test_a_config_that_is_not_an_object_is_treated_as_unreadable(tmp_path):
    path = tmp_path / "config.json"
    for junk in ("null", "[1, 2]", "￾\x00\x81"):
        path.write_text(junk, encoding="utf-8")
        assert Config.load(path).version == CONFIG_VERSION


def test_a_save_that_fails_halfway_leaves_the_old_file_whole(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    cfg = Config()
    cfg.postprocess.dictionary_terms = ["Arnab"]
    cfg.save(path)

    import localflow.config as C

    def crash(*_a, **_k):
        raise OSError("disk went away")

    monkeypatch.setattr(C.os, "replace", crash)
    cfg.postprocess.dictionary_terms = ["Okonkwo"]
    try:
        cfg.save(path)
    except OSError:
        pass
    monkeypatch.undo()
    assert Config.load(path).postprocess.dictionary_terms == ["Arnab"]
    assert not list(tmp_path.glob("*.tmp")), "no temporary file left behind"


def test_saves_from_many_threads_at_once_leave_a_readable_file(tmp_path):
    import threading

    path = tmp_path / "config.json"
    errors: list[Exception] = []

    def save(n: int) -> None:
        try:
            for _ in range(20):
                c = Config()
                c.postprocess.dictionary_terms = [f"word{n}x{i}" for i in range(200)]
                c.save(path)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=save, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    terms = Config.load(path).postprocess.dictionary_terms
    assert len(terms) == 200 and len({t.split("x")[0] for t in terms}) == 1, "one whole save, not a mix"
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob("*.broken-*"))


def test_v2_configs_get_automatic_model_choice_unless_a_model_was_picked():
    from localflow.config import migrate

    plain = migrate({"version": 2, "stt": {"model": "nemo-parakeet-tdt-0.6b-v3"}, "postprocess": {"llm_model": "qwen3-4b"}})
    assert plain["compute"] == {"auto_speech": True, "auto_cleanup": True} and plain["version"] == 3
    picked = migrate({"version": 2, "stt": {"model": "nemo-parakeet-tdt-0.6b-v2"}, "postprocess": {"llm_model": "phi-4-mini"}})
    assert picked["compute"] == {"auto_speech": False, "auto_cleanup": False}
    compact = migrate({"version": 2, "stt": {"precision": "int8"}})
    assert compact["compute"]["auto_speech"] is False


def test_safe_mode_runs_plainly_but_never_reaches_the_settings_file(tmp_path):
    """Safe mode is for one run of a crashing engine. Written to disk, it would outlive the fault."""
    from dataclasses import replace

    from localflow.config import apply_safe_mode, in_safe_mode

    path = tmp_path / "config.json"
    mine = Config()
    mine.compute.mode = "gpu"
    mine.compute.auto_speech = False
    mine.stt.model = "whisper-large-v3-turbo"
    mine.postprocess.dictionary_terms = ["Arnab"]
    mine.save(path)

    cfg = Config.load(path)
    apply_safe_mode(cfg)
    assert in_safe_mode(cfg) and not in_safe_mode(Config.load(path))
    assert cfg.compute.mode == "cpu" and cfg.stt.device == "cpu" and cfg.postprocess.llm_cleanup is False
    assert cfg.stt.model == Config().stt.model, "the default speech model"

    # The engine saves after unrelated changes, sometimes replacing a whole section.
    cfg.postprocess = replace(cfg.postprocess, dictionary_terms=["Arnab", "Okonkwo"])
    cfg.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["postprocess"]["dictionary_terms"] == ["Arnab", "Okonkwo"], "real changes are kept"
    assert saved["compute"]["mode"] == "gpu" and saved["compute"]["auto_speech"] is False
    assert saved["stt"]["model"] == "whisper-large-v3-turbo" and saved["stt"]["device"] == "auto"
    assert saved["postprocess"]["llm_cleanup"] is True

    # Turning clean-up back on by hand, in safe mode, is the user's decision and is kept.
    cfg.postprocess = replace(cfg.postprocess, llm_cleanup=True)
    cfg.save(path)
    cfg.postprocess = replace(cfg.postprocess, llm_cleanup=False)  # and later off again, also kept
    cfg.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["postprocess"]["llm_cleanup"] is False


def test_a_config_saved_with_a_byte_order_mark_is_read(tmp_path):
    """Older Notepad and Windows PowerShell write UTF-8 with a BOM, and the whole file used to be
    treated as unreadable: set aside, and the settings started again from the defaults."""
    path = tmp_path / "config.json"
    cfg = Config()
    cfg.postprocess.dictionary_terms = ["Arnab"]
    cfg.save(path)
    path.write_text("\ufeff" + path.read_text(encoding="utf-8"), encoding="utf-8")
    assert Config.load(path).postprocess.dictionary_terms == ["Arnab"]
    assert not list(tmp_path.glob("*.broken-*"))


def test_settings_from_a_newer_version_are_read_and_never_saved_over(tmp_path):
    """After a downgrade: what is understood is used; the file stays as the newer version left it."""
    import json

    path = tmp_path / "config.json"
    original = {"version": 9, "stt": {"language": "de"}, "something_new": {"x": 1}}
    path.write_text(json.dumps(original), encoding="utf-8")
    cfg = Config.load(path)
    assert cfg.newer == 9 and cfg.stt.language == "de"
    cfg.stt.language = "fr"
    cfg.save(path)
    assert json.loads(path.read_text(encoding="utf-8")) == original
    # An ordinary file is saved as always, without the marker.
    path.write_text(json.dumps({"version": 3}), encoding="utf-8")
    cfg = Config.load(path)
    assert cfg.newer is None
    cfg.save(path)
    assert "newer" not in json.loads(path.read_text(encoding="utf-8"))


def test_an_8_gb_pc_starts_with_clean_up_off(tmp_path, monkeypatch):
    """B5: with speech, clean-up takes about 6 GB, too much beside Windows and a browser."""
    from localflow import hwinfo

    monkeypatch.setattr(hwinfo, "ram_gb", lambda: 7.6)  # what an "8 GB" PC reports
    cfg = Config.load(tmp_path / "config.json")
    assert cfg.postprocess.llm_cleanup is False
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["postprocess"]["llm_cleanup"] is False

    # Only the default: once someone turns it on, it stays on.
    cfg.postprocess.llm_cleanup = True
    cfg.save(tmp_path / "config.json")
    assert Config.load(tmp_path / "config.json").postprocess.llm_cleanup is True


def test_a_16_gb_pc_starts_with_clean_up_on(tmp_path, monkeypatch):
    from localflow import hwinfo

    monkeypatch.setattr(hwinfo, "ram_gb", lambda: 15.4)
    assert Config.load(tmp_path / "config.json").postprocess.llm_cleanup is True


def test_an_unreadable_config_that_cannot_be_moved_aside_is_not_saved_over(tmp_path, monkeypatch):
    """The defaults were saved over it whether or not it had been moved aside, and with it the
    user's dictionary (found reviewing 0.2.5)."""
    from localflow import config as C

    path = tmp_path / "config.json"
    broken = '{"postprocess": {"dictionary_terms": ["Arnab"'
    path.write_text(broken, encoding="utf-8")
    monkeypatch.setattr(C, "_set_aside", lambda p: None)  # open elsewhere: it cannot be moved
    cfg = Config.load(path)
    assert cfg.postprocess.dictionary_terms == [], "defaults, for now"
    cfg.save(path)  # a change from the Hub, say
    assert path.read_text(encoding="utf-8") == broken, "the user's file is left as it was"


def test_a_config_that_cannot_be_read_starts_the_engine_on_defaults(tmp_path, monkeypatch):
    """A read error (a file another program holds) stopped the engine at start-up."""
    import pathlib

    path = tmp_path / "config.json"
    path.write_text('{"postprocess": {"dictionary_terms": ["Arnab"]}}', encoding="utf-8")
    real = pathlib.Path.read_text

    def locked(self, *a, **k):
        if self == path:
            raise PermissionError(13, "The process cannot access the file")
        return real(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", locked)
    cfg = Config.load(path)
    assert cfg.postprocess.dictionary_terms == []
    cfg.save(path)
    monkeypatch.setattr(pathlib.Path, "read_text", real)
    assert "Arnab" in path.read_text(encoding="utf-8"), "nothing was saved over it"
