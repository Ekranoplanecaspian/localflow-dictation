"""Hub > Help > Reset preferences: back to the defaults, but never the user's own words."""

import time

from localflow.config import Config
from test_service import FakeSTT


def test_reset_keeps_the_users_words_and_nothing_else(monkeypatch):
    import localflow.service.engine as eng

    monkeypatch.setattr(eng, "build_transcriber", lambda cfg: FakeSTT())
    monkeypatch.setattr(Config, "save", lambda self, *a, **k: None)
    cfg = Config()
    cfg.postprocess.llm_cleanup = False
    engine = eng.Engine(cfg)
    engine.load()
    deadline = time.monotonic() + 30
    while engine.state == "loading" and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        pp = engine.cfg.postprocess
        pp.dictionary = {"kwen": "Qwen"}
        pp.dictionary_terms = ["Okonkwo"]
        pp.snippets = {"my email": "a@example.com"}
        pp.custom_instructions = "British spelling."
        pp.llm_api_key = "sk-test"
        pp.remove_fillers = False
        pp.llm_min_words = 15
        pp.llm_provider, pp.llm_url = "openai", "https://api.example.com"
        engine.cfg.stt.language = "de"
        engine.cfg.stt.gpu_keep_warm = "never"
        engine.cfg.network.hf_endpoint = "https://hf-mirror.example"
        engine._apply_compute({"mode": "cpu", "temp_limit_c": 70, "idle_release_min": 30,
                               "auto_speech": False, "auto_cleanup": False})

        engine.reset_preferences()

        defaults = Config()
        pp = engine.cfg.postprocess
        assert (pp.dictionary, pp.dictionary_terms, pp.snippets) == (
            {"kwen": "Qwen"}, ["Okonkwo"], {"my email": "a@example.com"})
        assert pp.custom_instructions == "British spelling." and pp.llm_api_key == "sk-test"
        assert pp.remove_fillers is True and pp.llm_min_words == defaults.postprocess.llm_min_words
        assert (pp.llm_provider, pp.llm_url) == ("bundled", "")
        assert pp.llm_cleanup == defaults.postprocess.llm_cleanup
        assert engine.cfg.compute == defaults.compute, "placement and Automatic model choice back"
        assert engine.cfg.stt.gpu_keep_warm == "auto"
        assert engine.cfg.stt.language == "de", "the dictation language is who you are, not a preference"
        assert engine.cfg.network.hf_endpoint == "https://hf-mirror.example", "and the mirror is where"
    finally:
        engine.shutdown()
