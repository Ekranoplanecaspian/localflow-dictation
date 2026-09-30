"""The problem codes the engine reports, and how a failure is sorted into one.

Their messages and fixes live in `shared/problems.json`, the catalogue the shell compiles in;
the engine only says which problem it hit and passes the underlying error along as the detail.
`tests/test_problems.py` checks that every code here, and every code the engine sends, is in
that catalogue - a new failure cannot ship without a message.

The failures that matter most, a model that will not load, used to reach the user as whatever
the exception said ("[ONNXRuntimeError] : 7 : INVALID_PROTOBUF ..."). Sorting them by cause is
what lets the message say what to do.
"""

from __future__ import annotations

import re
import socket
import urllib.error
from collections.abc import Iterator

# a take
TAKE_FAILED = "take-failed"
TAKE_REFUSED = "take-refused"
# speech model
SPEECH_DOWNLOAD_FAILED = "speech-download-failed"
SPEECH_NO_SPACE = "speech-no-space"
SPEECH_CLOCK_WRONG = "speech-clock-wrong"
SPEECH_FILES_DAMAGED = "speech-files-damaged"
SPEECH_OUT_OF_MEMORY = "speech-out-of-memory"
SPEECH_GPU_UNAVAILABLE = "speech-gpu-unavailable"
SPEECH_LOAD_FAILED = "speech-load-failed"
SPEECH_SWITCH_FAILED = "speech-switch-failed"
# clean-up
CLEANUP_DOWNLOAD_FAILED = "cleanup-download-failed"
CLEANUP_NO_SPACE = "cleanup-no-space"
CLEANUP_BLOCKED = "cleanup-blocked"
CLEANUP_CLOCK_WRONG = "cleanup-clock-wrong"
CLEANUP_SERVER_MISSING = "cleanup-server-missing"
CLEANUP_SERVER_CRASHED = "cleanup-server-crashed"
CLEANUP_OUT_OF_MEMORY = "cleanup-out-of-memory"
CLEANUP_LOW_MEMORY = "cleanup-low-memory"
CLEANUP_CLOUD_KEY_REJECTED = "cleanup-cloud-key-rejected"
CLEANUP_CLOUD_UNREACHABLE = "cleanup-cloud-unreachable"
CLEANUP_CLOUD_SETTINGS_MISSING = "cleanup-cloud-settings-missing"
CLEANUP_LOAD_FAILED = "cleanup-load-failed"
CLEANUP_SWITCH_FAILED = "cleanup-switch-failed"
CLEANUP_TIMEOUT = "cleanup-timeout"
# command mode
COMMAND_NO_MODEL = "command-no-model"
COMMAND_NO_SELECTION = "command-no-selection"
COMMAND_NOTHING_SAID = "command-nothing-said"
COMMAND_NO_CHANGE = "command-no-change"
COMMAND_TRUNCATED = "command-truncated"
COMMAND_FAILED = "command-failed"
# the self-check (selfcheck.py)
DRIVER_TOO_OLD = "driver-too-old"
GPU_LIBS_DOWNLOAD_FAILED = "gpu-libs-download-failed"
FOLDER_NOT_WRITABLE = "folder-not-writable"
DISK_LOW = "disk-low"
CLEANUP_FILES_DAMAGED = "cleanup-files-damaged"
DOWNLOAD_HOSTS_UNREACHABLE = "download-hosts-unreachable"
# the connection
SETTING_REFUSED = "setting-refused"
BAD_MESSAGE = "bad-message"
BAD_AUDIO = "bad-audio"
UNKNOWN_MESSAGE = "unknown-message"

CODES = frozenset(v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, str))

_NETWORK_WORDS = ("getaddrinfo", "name or service not known", "temporary failure in name resolution",
                  "nodename nor servname", "max retries exceeded", "connection refused", "connection reset",
                  "timed out", "no connection", "network is unreachable", "failed to establish a new connection",
                  "proxyerror", "ssl", "couldn't connect", "offline mode")
_NETWORK_TYPES = ("ConnectionError", "Timeout", "ConnectTimeout", "ReadTimeout", "ProxyError", "SSLError",
                  "LocalEntryNotFoundError", "OfflineModeIsEnabled")
_MEMORY_WORDS = ("out of memory", "out_of_memory", "bad allocation", "failed to allocate", "cudamalloc",
                 "cannot allocate memory", "std::bad_alloc")
_DAMAGED_WORDS = ("invalid_protobuf", "protobuf parsing failed", "no_suchfile", "no such file", "checksum",
                  "sha256", "corrupt", "unexpected end of file", "invalid model", "not a valid", "truncated")


class NotEnoughMemory(Exception):
    """A load refused before it began: there is not enough free memory for it right now. Not a
    MemoryError - nothing ran out - so it is told apart from a load that did."""


class Classified(Exception):
    """An error already told apart somewhere else - a download's own process (fetch.py) - that
    arrives as its problem code and its sentence. The classifiers pass the code on as it is."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


def _classified(exc: BaseException) -> str | None:
    return next((e.code for e in _chain(exc) if isinstance(e, Classified)), None)


_NO_INTERNET_WORDS = ("getaddrinfo failed", "name or service not known", "temporary failure in name resolution",
                      "nodename nor servname", "no address associated with hostname")


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """The exception and everything it was raised from or during, outermost first."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _text(exc: BaseException) -> str:
    return " | ".join(f"{type(e).__name__}: {e}" for e in _chain(exc)).lower()


def http_status(exc: BaseException) -> int | None:
    for e in _chain(exc):
        if isinstance(e, urllib.error.HTTPError):
            return e.code
        status = getattr(getattr(e, "response", None), "status_code", None)
        if isinstance(status, int):
            return status
    return None


def is_network(exc: BaseException) -> bool:
    for e in _chain(exc):
        if isinstance(e, urllib.error.HTTPError):
            continue  # the server answered: not a connection problem
        if isinstance(e, (urllib.error.URLError, socket.timeout, socket.gaierror, TimeoutError, ConnectionError)):
            return True
        if any(name in type(e).__name__ for name in _NETWORK_TYPES):
            return True
    text = _text(exc)
    return any(w in text for w in _NETWORK_WORDS)


def is_timeout(exc: BaseException) -> bool:
    """A time limit ran out (a socket timeout, or a library's own), rather than a refusal."""
    for e in _chain(exc):
        if isinstance(e, (socket.timeout, TimeoutError)) or "Timeout" in type(e).__name__:
            return True
    return "timed out" in _text(exc)


def is_out_of_memory(exc: BaseException) -> bool:
    return any(isinstance(e, MemoryError) for e in _chain(exc)) or any(w in _text(exc) for w in _MEMORY_WORDS)


# Windows refusing to run a program: a policy (AppLocker, Software Restriction Policies), Smart
# App Control or WDAC, antivirus (the file infected, or already removed).
_BLOCKED_WINERRORS = frozenset({1260, 4551, 225, 226})


def is_blocked(exc: BaseException) -> bool:
    return any(isinstance(e, OSError) and getattr(e, "winerror", None) in _BLOCKED_WINERRORS for e in _chain(exc))


def is_no_space(exc: BaseException) -> bool:
    import errno

    return any(isinstance(e, OSError) and e.errno == errno.ENOSPC for e in _chain(exc)) or         "no space left on device" in _text(exc) or "not enough space on the disk" in _text(exc)


_CLOCK_WORDS = ("certificate has expired", "certificate is not yet valid", "certificate_expired",
                "cert_not_yet_valid")


def is_clock_wrong(exc: BaseException) -> bool:
    """A secure connection refused for dates: with nearly every site's certificate current,
    that is this PC's clock, not the site."""
    return any(w in _text(exc) for w in _CLOCK_WORDS)


def is_damaged(exc: BaseException) -> bool:
    return any(isinstance(e, FileNotFoundError) for e in _chain(exc)) or any(w in _text(exc) for w in _DAMAGED_WORDS)


def classify_speech(exc: BaseException) -> str:
    """Why the speech model would not load."""
    if (code := _classified(exc)) is not None:
        return code
    text = _text(exc)
    if "cuda requested but unavailable" in text:
        return SPEECH_GPU_UNAVAILABLE
    if is_no_space(exc):
        return SPEECH_NO_SPACE
    if is_clock_wrong(exc):
        return SPEECH_CLOCK_WRONG
    if is_out_of_memory(exc):
        return SPEECH_OUT_OF_MEMORY
    if is_network(exc):
        return SPEECH_DOWNLOAD_FAILED
    if is_damaged(exc):
        return SPEECH_FILES_DAMAGED
    return SPEECH_LOAD_FAILED


def classify_cleanup(exc: BaseException, provider: str) -> str:
    """Why AI clean-up would not start. `provider` is `bundled` or a cloud provider's name."""
    if (code := _classified(exc)) is not None:
        return code
    text = _text(exc)
    cloud = provider != "bundled"
    if any(isinstance(e, NotEnoughMemory) for e in _chain(exc)):
        return CLEANUP_LOW_MEMORY
    if cloud and ("needs llm_url" in text or "needs llm_api_key" in text or "needs the sdk" in text):
        return CLEANUP_CLOUD_SETTINGS_MISSING
    status = http_status(exc)
    if cloud and status in (401, 403):
        return CLEANUP_CLOUD_KEY_REJECTED
    if is_blocked(exc):
        return CLEANUP_BLOCKED
    if is_no_space(exc):
        return CLEANUP_NO_SPACE
    if is_clock_wrong(exc):
        return CLEANUP_CLOCK_WRONG
    if is_network(exc):
        return CLEANUP_CLOUD_UNREACHABLE if cloud else CLEANUP_DOWNLOAD_FAILED
    if "llama-server.exe not found" in text or "only provided for windows" in text:
        return CLEANUP_SERVER_MISSING
    if is_out_of_memory(exc):
        return CLEANUP_OUT_OF_MEMORY
    if "llama-server exited" in text or "did not become healthy" in text:
        return CLEANUP_SERVER_CRASHED
    return CLEANUP_LOAD_FAILED


# Failures that go away by themselves - the connection comes back, space is freed, the clock is
# put right - so the engine tries again on its own rather than waiting for "Try again".
RETRY_BY_ITSELF = frozenset({SPEECH_DOWNLOAD_FAILED, SPEECH_NO_SPACE, SPEECH_CLOCK_WRONG,
                             CLEANUP_DOWNLOAD_FAILED, CLEANUP_NO_SPACE, CLEANUP_CLOCK_WRONG,
                             CLEANUP_LOW_MEMORY})


def detail(exc: BaseException) -> str:
    """The first line of an error, as a sentence to put in a message."""
    if is_clock_wrong(exc):
        import datetime

        return f"This PC's clock says {datetime.datetime.now():%A %d %B %Y, %H:%M}."
    if any(w in _text(exc) for w in _NO_INTERNET_WORDS):
        # "[Errno 11001] getaddrinfo failed": the name of the server could not be looked up,
        # which nearly always means no internet at all (seen on a clean PC offline, B6)
        return "This PC could not reach the internet."
    first = str(exc).strip().splitlines()[0][:300] if str(exc).strip() else type(exc).__name__
    # Libraries wrap the cause as "Got: ConnectError: [WinError 10061] No connection ...": the
    # words after the labels are the part a person can use.
    first = re.sub(r"^(?:Got:\s*)?(?:[A-Za-z_.]*(?:Error|Exception):\s*)?(?:\[WinError \d+\]\s*)?", "", first) or first
    return first if first.endswith((".", "!", "?")) else first + "."


# What command mode's refusals are called in the catalogue.
_COMMAND = {
    "no-model": COMMAND_NO_MODEL,
    "empty-selection": COMMAND_NO_SELECTION,
    "empty-instruction": COMMAND_NOTHING_SAID,
    "no-change": COMMAND_NO_CHANGE,
    "truncated": COMMAND_TRUNCATED,
}


def command_code(rejected: str | None) -> str | None:
    """The catalogue code for a command's `rejected` reason (None when the edit was applied)."""
    if not rejected:
        return None
    if rejected.startswith("bad request"):
        return BAD_MESSAGE
    if rejected.startswith("error"):
        return COMMAND_FAILED
    # the output guard's reasons ("added-quotes", "answered", ...) are all a refusal to change
    return _COMMAND.get(rejected, COMMAND_NO_CHANGE)
