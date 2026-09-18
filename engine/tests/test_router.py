import numpy as np

from localflow.config import STTConfig
from localflow.stt import router as R


class FakeBackend:
    def __init__(self, name):
        self.name = name
        self.device = "cpu"
        self.precision = "fp32"
        self.calls = []

    def transcribe(self, audio, language=None):
        self.calls.append(language)
        return self.name

    def warmup(self):
        pass

    def warm(self):
        pass


def make(monkeypatch, language=None):
    monkeypatch.setattr(R, "ParakeetTranscriber", lambda cfg: FakeBackend("parakeet"))
    r = R.RoutedTranscriber(STTConfig(language=language))
    whisper = FakeBackend("whisper")
    r._whisper = whisper  # pre-seed so the test never loads a real model
    return r, r.primary, whisper


def test_normalize_language():
    assert R.normalize_language(None) is None
    assert R.normalize_language("auto") is None
    assert R.normalize_language("en-US") == "en"
    assert R.normalize_language("hi_IN") == "hi"


def test_routes_parakeet_languages_to_parakeet(monkeypatch):
    r, parakeet, whisper = make(monkeypatch)
    audio = np.zeros(16000, np.float32)
    assert r.transcribe(audio) == "parakeet"
    assert r.transcribe(audio, language="de") == "parakeet"
    assert r.transcribe(audio, language="en-GB") == "parakeet"
    assert parakeet.calls == [None, None, None]  # Parakeet is never told a language; it detects
    assert whisper.calls == []


def test_routes_other_languages_to_whisper(monkeypatch):
    r, parakeet, whisper = make(monkeypatch)
    audio = np.zeros(16000, np.float32)
    assert r.transcribe(audio, language="hi") == "whisper"
    assert r.transcribe(audio, language="ja") == "whisper"
    assert whisper.calls == ["hi", "ja"]
    assert parakeet.calls == []


def test_config_default_language_applies_when_session_has_none(monkeypatch):
    r, parakeet, whisper = make(monkeypatch, language="ar")
    audio = np.zeros(16000, np.float32)
    assert r.transcribe(audio) == "whisper"
    assert r.transcribe(audio, language="fr") == "parakeet"  # a session override still wins
