"""The problem catalogue (shared/problems.json) and the engine's side of it: every code the
engine can send has an entry, and model-load failures are sorted into the right one."""

import json
import re
import socket
import urllib.error
from pathlib import Path

import pytest

from localflow import problems

REPO = Path(__file__).resolve().parents[2]
CATALOGUE = json.loads((REPO / "shared" / "problems.json").read_text(encoding="utf-8"))["problems"]
BY_CODE = {p["code"]: p for p in CATALOGUE}
LEVELS = {"failed", "degraded", "info", "handled", "internal"}
PARTS = {"engine", "speech", "cleanup", "gpu", "microphone", "hotkey", "storage", "network"}
PLACEHOLDERS = {"detail", "model", "to", "device", "chosen", "app", "version", "needed", "folder", "free", "drive",
                "progress"}
ACTIONS = {"restart_engine", "open_models", "open_voice", "open_apps", "privacy_microphone", "sound",
           "storage", "date_time", "redownload_models"}


# --- the catalogue itself ------------------------------------------------------------------------
def test_every_entry_is_complete_and_unique():
    assert len(BY_CODE) == len(CATALOGUE), "duplicate codes"
    for p in CATALOGUE:
        assert re.fullmatch(r"[a-z]+(-[a-z]+)+", p["code"]), p["code"]
        assert p["level"] in LEVELS, p
        assert p["part"] in PARTS, p
        assert p["source"] in {"engine", "shell", "both"}, p
        for field in ("detect", "title", "message"):
            assert p[field].strip(), (p["code"], field)
        assert not p["title"].endswith("."), (p["code"], "a title is not a sentence")
        assert p["message"].rstrip().endswith((".", "}")), (p["code"], "a message ends a sentence")
        if "action" in p:
            assert p["action"]["id"] in ACTIONS and p["action"]["label"], p["code"]
        if p["level"] in ("failed", "degraded"):
            # a part in this state is a row on the Status card, which needs its word or two
            assert p.get("summary", "").strip(), (p["code"], "summary")


def test_placeholders_are_known_and_details_are_whole_sentences():
    for p in CATALOGUE:
        for field in ("title", "message", "bar"):
            text = p.get(field, "")
            assert set(re.findall(r"\{(\w+)\}", text)) <= PLACEHOLDERS, (p["code"], field)
            # {detail} is always a full sentence (problems.detail adds the stop): never inside
            # brackets, never mid-sentence
            assert "({detail})" not in text, (p["code"], field)
        assert "{detail}" not in p["title"], p["code"]


def test_what_the_user_sees_is_plain():
    for p in CATALOGUE:
        if p["level"] in ("handled", "internal"):
            continue
        for field in ("title", "message", "bar"):
            words = p.get(field, "").lower()
            assert "error" not in words.replace("{detail}", ""), (p["code"], field, "no 'error'")
            assert "exception" not in words, (p["code"], field)


def test_only_problems_that_stop_or_hamper_dictation_are_failed_or_degraded():
    for p in CATALOGUE:
        if p["level"] == "failed":
            assert p["part"] in {"engine", "speech", "microphone", "hotkey"}, p["code"]


# --- the engine's codes ----------------------------------------------------------------------------
def test_every_engine_code_is_in_the_catalogue():
    missing = sorted(problems.CODES - set(BY_CODE))
    assert not missing, f"codes the engine uses with no catalogue entry: {missing}"
    for code in problems.CODES:
        assert BY_CODE[code]["source"] in {"engine", "both"}, code


def test_every_problem_the_engine_detects_has_a_constant():
    """And the other way: an engine problem in the catalogue that nothing reports is a promise
    the code does not keep."""
    for p in CATALOGUE:
        if p["source"] in ("engine", "both"):
            assert p["code"] in problems.CODES, p["code"]


def test_timeouts_are_told_from_other_failures():
    assert problems.is_timeout(socket.timeout("timed out"))
    assert problems.is_timeout(raised_from(RuntimeError("clean-up failed"), TimeoutError()))
    assert not problems.is_timeout(ConnectionRefusedError("refused"))


def test_the_engine_sends_no_code_that_is_not_a_constant():
    """`P.error("some_code", ...)` with a literal would slip past the check above."""
    src = REPO / "engine" / "src" / "localflow"
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        literal = re.findall(r"P\.error\(\s*[\"']", text)
        assert not literal, f"{path.name}: P.error with a literal code; use localflow.problems"


# --- sorting failures -------------------------------------------------------------------------------
def raised_from(outer: BaseException, inner: BaseException) -> BaseException:
    try:
        try:
            raise inner
        except BaseException as e:
            raise outer from e
    except BaseException as e:
        return e


class ConnectionErrorFromRequests(IOError):
    """Shaped like requests.exceptions.ConnectionError: an IOError, not the builtin."""


ConnectionErrorFromRequests.__name__ = "ConnectionError"


@pytest.mark.parametrize("exc, code", [
    (urllib.error.URLError("[Errno 11001] getaddrinfo failed"), problems.SPEECH_DOWNLOAD_FAILED),
    (socket.timeout("timed out"), problems.SPEECH_DOWNLOAD_FAILED),
    (ConnectionErrorFromRequests("HTTPSConnectionPool: Max retries exceeded"), problems.SPEECH_DOWNLOAD_FAILED),
    (raised_from(RuntimeError("could not fetch the model"), ConnectionResetError("reset")), problems.SPEECH_DOWNLOAD_FAILED),
    (RuntimeError("[ONNXRuntimeError] : 7 : INVALID_PROTOBUF : Load model from x failed:Protobuf parsing failed."),
     problems.SPEECH_FILES_DAMAGED),
    (FileNotFoundError("encoder-model.onnx"), problems.SPEECH_FILES_DAMAGED),
    (MemoryError(), problems.SPEECH_OUT_OF_MEMORY),
    (RuntimeError("CUDA failure 2: out of memory"), problems.SPEECH_OUT_OF_MEMORY),
    (RuntimeError("CUDA requested but unavailable: CUDA provider failed. Install with: pip install"),
     problems.SPEECH_GPU_UNAVAILABLE),
    (ValueError("unknown stt backend 'x'"), problems.SPEECH_LOAD_FAILED),
])
def test_speech_failures_are_sorted_by_cause(exc, code):
    assert problems.classify_speech(exc) == code


def http(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.example.com/v1/chat/completions", code, "status", {}, None)


@pytest.mark.parametrize("exc, provider, code", [
    (http(401), "openai", problems.CLEANUP_CLOUD_KEY_REJECTED),
    (http(403), "anthropic", problems.CLEANUP_CLOUD_KEY_REJECTED),
    (http(500), "openai", problems.CLEANUP_LOAD_FAILED),
    (urllib.error.URLError("[Errno 11001] getaddrinfo failed"), "openai", problems.CLEANUP_CLOUD_UNREACHABLE),
    (urllib.error.URLError("[Errno 11001] getaddrinfo failed"), "bundled", problems.CLEANUP_DOWNLOAD_FAILED),
    (RuntimeError("openai provider needs llm_url (e.g. https://api.openai.com)"), "openai",
     problems.CLEANUP_CLOUD_SETTINGS_MISSING),
    (RuntimeError("anthropic provider needs llm_api_key"), "anthropic", problems.CLEANUP_CLOUD_SETTINGS_MISSING),
    (RuntimeError("llama-server.exe not found after extracting into C:/x"), "bundled", problems.CLEANUP_SERVER_MISSING),
    (RuntimeError("bundled llama-server is only provided for Windows x64"), "bundled", problems.CLEANUP_SERVER_MISSING),
    (RuntimeError("llama-server exited with code 3; see C:/x/llama.log"), "bundled", problems.CLEANUP_SERVER_CRASHED),
    (RuntimeError("llama-server did not become healthy in time"), "bundled", problems.CLEANUP_SERVER_CRASHED),
    (MemoryError(), "bundled", problems.CLEANUP_OUT_OF_MEMORY),
    # refused before starting (B5): nothing ran out, and it tries again by itself
    (problems.NotEnoughMemory("Qwen3 4B needs about 3.8 GB of free memory, and 1.0 GB is free now"), "bundled",
     problems.CLEANUP_LOW_MEMORY),
    (ValueError("something else"), "bundled", problems.CLEANUP_LOAD_FAILED),
])
def test_clean_up_failures_are_sorted_by_cause(exc, provider, code):
    assert problems.classify_cleanup(exc, provider) == code


def test_waiting_for_memory_retries_by_itself():
    assert problems.CLEANUP_LOW_MEMORY in problems.RETRY_BY_ITSELF
    assert problems.CLEANUP_OUT_OF_MEMORY not in problems.RETRY_BY_ITSELF  # that one did run out


def test_a_detail_is_one_sentence():
    assert problems.detail(RuntimeError("first line\nsecond line")) == "first line."
    assert problems.detail(RuntimeError("Already a sentence.")) == "Already a sentence."
    assert problems.detail(MemoryError()) == "MemoryError."
    # offline, in words a person can use (B6)
    offline = urllib.error.URLError("[Errno 11001] getaddrinfo failed")
    assert problems.detail(offline) == "This PC could not reach the internet."


@pytest.mark.parametrize("rejected, code", [
    (None, None),
    ("no-model", problems.COMMAND_NO_MODEL),
    ("empty-selection", problems.COMMAND_NO_SELECTION),
    ("empty-instruction", problems.COMMAND_NOTHING_SAID),
    ("truncated", problems.COMMAND_TRUNCATED),
    ("no-change", problems.COMMAND_NO_CHANGE),
    ("answered", problems.COMMAND_NO_CHANGE),
    ("error: connection refused", problems.COMMAND_FAILED),
    ("bad request: 'selection' must be a string", problems.BAD_MESSAGE),
])
def test_command_refusals_have_codes(rejected, code):
    assert problems.command_code(rejected) == code


def test_a_speech_model_that_fails_to_load_reports_its_code(monkeypatch):
    import localflow.service.engine as eng
    from localflow.config import Config

    def broken(cfg):
        raise MemoryError()

    monkeypatch.setattr(eng, "build_transcriber", broken)
    engine = eng.Engine(Config())
    engine.load()
    engine._pool.submit(lambda: None).result(timeout=30)  # the load job has finished
    stt = engine.status()["stt"]
    assert (stt["state"], stt["error_code"]) == ("error", problems.SPEECH_OUT_OF_MEMORY)
    engine.shutdown()


# The Help page (app/src/hub/Help.tsx, `general`) says each problem without a live one's details.
_GENERAL = {"model": "the model", "device": "the microphone", "chosen": "the microphone you chose",
            "to": "the new model", "needed": "the space it needs", "app": "this app",
            "folder": "LocalFlow's folder", "free": "too little", "drive": "the drive",
            "version": "a newer version"}


def _general(text: str) -> str:
    text = re.sub(r"\s*\{detail\}\s*", " ", text)
    text = re.sub(r"\{(\w+)\}", lambda m: _GENERAL.get(m[1], m[1]), text)
    text = re.sub(r":\s*(?=[A-Z]|$)", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m[1] + m[2].upper(), text)


def test_every_problem_reads_well_on_the_help_page():
    """Without a live problem's details, a message can read "Dictation goes on with the model."
    or "kept as .": those need a `help` text of their own."""
    for p in CATALOGUE:
        if p["level"] == "internal":
            continue
        title = p.get("help_title") or _general(p["title"])
        text = p.get("help") or _general(p["message"])
        for s in (title, text):
            assert "{" not in s and "}" not in s, (p["code"], s)
        assert len(text) >= 20, (p["code"], "too little to help", text)
        assert not re.search(r"\s[.,;:]|\.\s*\.|the model\.$|with the model\b", text), (p["code"], text)
        for key in _GENERAL:
            assert f"{{{key}}}" not in p.get("help", ""), (p["code"], "help is read without details")
