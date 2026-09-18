"""Session protocol between a shell (tray app, Tauri) and the engine.

Transport: WebSocket on 127.0.0.1. Text frames are JSON objects with a "type" field.
Binary frames are audio: int16 little-endian PCM, 16 kHz, mono, belonging to the sender's
active session. The first message must be `hello` with the launch token.

shell -> engine
  hello           { token, client }
  session.start   { id, context: { app, title, url?, selection?, before_caret? }, language? }
  <binary>        20 ms audio frames while the hotkey is held
  session.end     { id }             the hotkey was released: finish and return text
  session.cancel  { id }
  status.get      {}
  settings.set    { postprocess?: {...} }
  shutdown        {}

engine -> shell
  hello.ok        { version, status }
  status          { pid, version,
                    stt: { state, error, backend, model, device, precision },
                    llm: { state, error, enabled, provider, model } }
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
SHUTDOWN = "shutdown"
PARTIAL = "partial"
FINAL = "final"
ERROR = "error"
# Command mode: the shell sends the user's selection and the instruction it heard, and gets
# back replacement text - or, far more often than with dictation, a refusal to change anything.
COMMAND_RUN = "command.run"
COMMAND_RESULT = "command.result"

# close codes
CLOSE_UNAUTHORIZED = 4001
CLOSE_BAD_HELLO = 4002


def encode(msg: dict[str, Any]) -> str:
    return json.dumps(msg, ensure_ascii=False, separators=(",", ":"))


def decode(text: str) -> dict[str, Any]:
    msg = json.loads(text)
    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
        raise ValueError("message must be an object with a string 'type'")
    return msg


def error(code: str, message: str, session_id: str | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"type": ERROR, "code": code, "message": message}
    if session_id is not None:
        msg["id"] = session_id
    return msg
