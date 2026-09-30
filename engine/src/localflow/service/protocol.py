"""Session protocol between a shell (tray app, Tauri) and the engine.

Transport: WebSocket on 127.0.0.1. Text frames are JSON objects with a "type" field.
Binary frames are audio: int16 little-endian PCM, 16 kHz, mono, belonging to the sender's
active session. The first message must be `hello` with the launch token.

Only local programs are let in: the server listens on 127.0.0.1, refuses any request that
carries an Origin header (every browser sends one, so no web page can reach it) or names a host
other than the loopback address, and holds each connection to one small hello until the token
checks out. Malformed messages after that are answered with an `error` and change nothing.

shell -> engine
  hello           { token, client }
  session.start   { id, context: { app, title, url?, selection?, before_caret? }, language? }
  <binary>        20 ms audio frames while the hotkey is held
  session.end     { id }             the hotkey was released: finish and return text
  session.cancel  { id }
  status.get      {}
  settings.set    { postprocess?: {...}, stt?: { model: <catalogue key> }, network?: { hf_endpoint },
                    llm?: { model: <bundled model key> },
                    compute?: { mode, temp_limit_c, idle_release_min, auto_speech, auto_cleanup } }
                  (choosing a stt/llm model pins it: auto_speech/auto_cleanup turn off)
  models.download { kind: speech|cleanup, key }   fetch a model without switching to it
  models.cancel   { id }                          stop a download (status.downloads[].id)
  models.remove   { kind, key }                   delete a downloaded model not in use
  shutdown        {}

engine -> shell
  hello.ok        { version, status }
  status          { pid, version,
                    stt: { state, error, backend, model, key, label, device, precision,
                           choices: [{ key, label, blurb, languages, size_gb, installed,
                                       current, recommended }],
                           switch: { to, state: downloading|loading|error, progress, error } | null,
                           download: { label, progress, size_gb } | null },   # a first run's download
                    llm: { state, error, enabled, provider, model, label,
                           choices: [...as stt], switch: {...as stt} | null,
                           download: {...as stt} | null },              # what loading waits for
                    (choices also carry fit: { rating: good|slow|too-big, why }, rated_on,
                     disk_gb, removable, download: <job id> | null)
                    downloads: [{ id, kind: speech|cleanup|runtime|gpu-libs, key, label,
                                  state: queued|downloading|done|error|cancelled, reason, cancellable,
                                  done, total, progress, speed_bps, eta_s, error, ended_s_ago }],
                    recommended: [{ kind, key, label, why, size_gb, installed,
                                    action: use|download|enable }],   # models that would help here
                    compute: { mode, temp_limit_c, idle_release_min, level, reason, speech, cleanup,
                               keep_warm, moving, gpu: { name, temp_c, util_pct, mem_used_mb,
                               mem_total_mb, slowdown_c } | null, recent: [{ at, what, to, model,
                               reason }], auto: { speech, cleanup }, chosen: { speech, cleanup:
                               { key, label, why } | null }, hardware } }
                  stt.state: loading | ready | error   llm.state: off | loading | ready | error
                  (dictation works as soon as stt is ready; the clean-up model loads in
                   parallel and only adds auto-edits once its state is ready)
  partial         { id, text, chunks, phrases }  live text of the whole take so far
  final           { id, raw, text, timings }     timings: audio_s, mode, chunks, phrases,
                                                 live_decodes, reused_live, stt_live_ms,
                                                 stt_final_ms, post_ms, release_to_final_ms,
                                                 used_llm, llm_ms, llm_rejected, profile,
                                                 dictionary_hits
  error           { id?, code, message }
"""

from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = 1
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# message types
HELLO = "hello"
HELLO_OK = "hello.ok"
SESSION_START = "session.start"
SESSION_END = "session.end"
SESSION_CANCEL = "session.cancel"
STATUS_GET = "status.get"
STATUS = "status"
SETTINGS_SET = "settings.set"
# Hub > Help > Reset preferences: every setting to its default but the user's own words,
# their API key, the dictation language and the download mirror. Answered with a status.
SETTINGS_RESET = "settings.reset"
SHUTDOWN = "shutdown"
PARTIAL = "partial"
FINAL = "final"
ERROR = "error"
# Command mode: the shell sends the user's selection and the instruction it heard, and gets
# back replacement text - or, far more often than with dictation, a refusal to change anything.
COMMAND_RUN = "command.run"
COMMAND_RESULT = "command.result"
# Check LocalFlow: the engine's checks (selfcheck.py), quick or full; and deleting the damaged
# model files a full check found, so they are downloaded again.
SELFCHECK_RUN = "selfcheck.run"  # { id, full } -> selfcheck.result { id, checks: [...] }
SELFCHECK_RESULT = "selfcheck.result"
SELFCHECK_REPAIR = "selfcheck.repair"  # { id } -> selfcheck.repaired { id, removed: [path] }
SELFCHECK_REPAIRED = "selfcheck.repaired"
# The model library (Hub > Models, M1): download a model without switching to it, stop a
# download, take a model off the PC. Each is answered with a status; a refusal with an error.
MODELS_DOWNLOAD = "models.download"  # { kind: speech|cleanup, key }
MODELS_CANCEL = "models.cancel"  # { id }: a download's id from status.downloads
MODELS_REMOVE = "models.remove"  # { kind, key }
MODEL_KINDS = ("speech", "cleanup")

# close codes
CLOSE_UNAUTHORIZED = 4001
CLOSE_BAD_HELLO = 4002

# Size limits. Until the token checks out a connection may send one small hello and nothing
# else; afterwards the largest legitimate message is a settings.set carrying the whole
# dictionary and every snippet (a few MB at the Hub's own limits).
MAX_HELLO_BYTES = 4 * 2**10
MAX_MESSAGE_BYTES = 8 * 2**20
# Audio arrives in 20 ms frames (640 bytes) while the key is held, and in one-second frames
# when a take is replayed to a new engine.
MAX_AUDIO_FRAME_BYTES = 64 * 2**10
MAX_ID_CHARS = 64
MAX_CONTEXT_CHARS = 4096  # per context field: the shell sends at most 4096 characters of selection
MAX_LANGUAGE_CHARS = 16
# A command's selection goes to the language model whole. The bundled model's whole context is
# 2048 tokens; a cloud model can take much more, but a selection this long is a mistake.
MAX_SELECTION_CHARS = 100_000
MAX_INSTRUCTION_CHARS = 2000
MAX_CLIENT_NAME_CHARS = 32


def encode(msg: dict[str, Any]) -> str:
    return json.dumps(msg, ensure_ascii=False, separators=(",", ":"))


def decode(text: str) -> dict[str, Any]:
    msg = json.loads(text)
    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
        raise ValueError("message must be an object with a string 'type'")
    return msg


def _optional_str(msg: dict[str, Any], key: str, limit: int) -> str | None:
    value = msg.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string")
    if len(value) > limit:
        raise ValueError(f"'{key}' is longer than {limit} characters")
    return value


def checked(msg: dict[str, Any]) -> dict[str, Any]:
    """The fields of a decoded message that the engine acts on, with their types and sizes
    checked. Raises ValueError, in words fit for a log, for anything that does not fit.

    Every field used to be passed on as it came: a list for a context failed on a worker
    thread, so the take never ended, and a number for a command's selection failed on the
    clean-up worker, so the command was never answered.
    """
    t = msg["type"]
    if t in (SESSION_START, SESSION_END, SESSION_CANCEL, COMMAND_RUN):
        sid = msg.get("id")
        if sid is not None and not isinstance(sid, (str, int)):
            raise ValueError("'id' must be a string")
        if sid is not None and len(str(sid)) > MAX_ID_CHARS:
            raise ValueError(f"'id' is longer than {MAX_ID_CHARS} characters")
    if t == SESSION_START:
        context = msg.get("context")
        if context is None:
            context = {}
        if not isinstance(context, dict):
            raise ValueError("'context' must be an object")
        clean: dict[str, str] = {}
        for key, value in context.items():
            if value is None:
                continue
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"context '{str(key)[:20]}' must be text")
            clean[str(key)[:MAX_ID_CHARS]] = str(value)[:MAX_CONTEXT_CHARS]
        return {**msg, "context": clean, "language": _optional_str(msg, "language", MAX_LANGUAGE_CHARS) or None}
    if t == COMMAND_RUN:
        return {**msg,
                "selection": _optional_str(msg, "selection", MAX_SELECTION_CHARS) or "",
                "instruction": _optional_str(msg, "instruction", MAX_INSTRUCTION_CHARS) or ""}
    if t in (MODELS_DOWNLOAD, MODELS_REMOVE):
        kind = _optional_str(msg, "kind", MAX_ID_CHARS)
        if kind not in MODEL_KINDS:
            raise ValueError(f"'kind' must be one of {', '.join(MODEL_KINDS)}")
        key = _optional_str(msg, "key", MAX_ID_CHARS)
        if not key:
            raise ValueError("'key' is required")
        return {**msg, "kind": kind, "key": key}
    if t == MODELS_CANCEL:
        job = _optional_str(msg, "id", MAX_ID_CHARS)
        if not job:
            raise ValueError("'id' is required")
        return {**msg, "id": job}
    return msg


def error(code: str, message: str, session_id: str | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"type": ERROR, "code": code, "message": message}
    if session_id is not None:
        msg["id"] = session_id
    return msg
