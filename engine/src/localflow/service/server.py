"""WebSocket server exposing the engine on 127.0.0.1 with a per-launch token.

Run with `localflow serve`. Prints one JSON line ({"port", "token", "pid"}) to stdout when
ready (for a parent process that spawned it) and writes the same to
%APPDATA%/LocalFlow/engine.json so a standalone client can find it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import signal
import sys
import threading
import uuid
from http import HTTPStatus
from typing import Any

from localflow import problems
from localflow.cleanup.command import CommandResult
from localflow.config import CONFIG_DIR, Config, write_atomic  # noqa: F401 (CONFIG_DIR: tests)
from localflow.service import protocol as P
from localflow.service.engine import Engine, Session

log = logging.getLogger(__name__)

ENGINE_INFO_PATH = CONFIG_DIR / "engine.json"

#: The shell, the tray app, and a benchmark or the harness now and then: a handful. More than
#: this at once is something hammering the port.
MAX_CONNECTIONS = 16
HELLO_TIMEOUT_S = 5.0
#: Replies waiting for a client that has stopped reading. A healthy client drains them within
#: milliseconds; beyond this it is hung, and holding on would grow the engine's memory forever.
MAX_PENDING_REPLIES = 5000
_LOOPBACK_HOST = re.compile(r"^(127\.0\.0\.1|localhost)(:\d{1,5})?$", re.IGNORECASE)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        k32.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == 259  # STILL_ACTIVE
    finally:
        k32.CloseHandle(handle)


def _listed_pid(path=None) -> int | None:
    try:
        return int(json.loads((path or ENGINE_INFO_PATH).read_text(encoding="utf-8"))["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def claim_engine_info(info: dict[str, Any], path=None) -> bool:
    """Name this engine in engine.json, the file other clients find it by - unless a live
    engine is named there already.

    It was overwritten by every engine that started: a second one (a self-test, the end-to-end
    harness) took the file from the user's engine and, killed without a chance to tidy up, left
    it naming a dead process. The next client then found nothing to join and started a third.
    """
    path = path or ENGINE_INFO_PATH
    held = _listed_pid(path)
    if held is not None and held != info["pid"] and _pid_alive(held):
        log.info("engine.json already names a running engine (pid %d); not claiming it", held)
        return False
    # All at once: the shell reads this to find us, and half a file sends it off to start an
    # engine of its own.
    write_atomic(path, json.dumps(info))
    return True


def release_engine_info(pid: int, path=None) -> None:
    """Remove engine.json if it names `pid` - never another engine's."""
    path = path or ENGINE_INFO_PATH
    if _listed_pid(path) == pid:
        try:
            path.unlink()
        except OSError:
            pass


class ClientConn:
    def __init__(self, ws, loop: asyncio.AbstractEventLoop):
        self.ws = ws
        self.loop = loop
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.session: Session | None = None
        self.name = "?"
        self.too_slow = False
        self.bad_audio_in: str | None = None  # the take a bad audio frame was reported for

    def emit(self, msg: dict[str, Any]) -> None:
        """Thread-safe: called from the engine worker thread."""
        self.loop.call_soon_threadsafe(self._put, msg)

    def _put(self, msg: dict[str, Any]) -> None:
        if self.too_slow:
            return
        if self.queue.qsize() >= MAX_PENDING_REPLIES:
            self.too_slow = True
            log.warning("client %s stopped reading (%d replies waiting); disconnecting it",
                        self.name, self.queue.qsize())
            # Not a closing handshake: its close frame would queue behind the replies the
            # client is not reading, and wait there for ever.
            self.ws.transport.abort()
            return
        self.queue.put_nowait(msg)


class EngineServer:
    def __init__(self, cfg: Config, port: int = 0, token: str | None = None, handshake: bool = False):
        self.cfg = cfg
        self.port = port
        self.token = token or secrets.token_hex(16)
        self.handshake = handshake
        self.engine = Engine(cfg)
        self._stop = asyncio.Event()
        self._clients: set[ClientConn] = set()
        self._open = 0  # connections past the opening handshake, greeted or not
        self._refused = 0

    def serve_options(self) -> dict[str, Any]:
        """How the WebSocket server is set up, wherever it runs. Messages are held to the size
        of a hello until the token checks out (see `_serve_client`)."""
        return {"max_size": P.MAX_HELLO_BYTES, "max_queue": 512,
                "process_request": self._check_request, "server_header": None}

    def _refuse(self, why: str) -> None:
        self._refused += 1
        if self._refused <= 20 or self._refused % 100 == 0:
            log.warning("refused a connection (%d so far): %s", self._refused, why)

    def _check_request(self, connection, request):
        """Before the WebSocket opens: only local programs, and not too many of them.

        A browser lets any web page open a WebSocket to 127.0.0.1, and always names the page in
        the Origin header; LocalFlow's own clients never send one. A page that points its own
        domain name at 127.0.0.1 still sends that name as the Host."""
        origin = request.headers.get("Origin")
        if origin is not None:
            self._refuse(f"from a web page ({origin[:80]})")
            return connection.respond(HTTPStatus.FORBIDDEN, "The engine does not take connections from web pages.\n")
        host = request.headers.get("Host", "")
        if not _LOOPBACK_HOST.match(host):
            self._refuse(f"for another host ({host[:80]!r})")
            return connection.respond(HTTPStatus.FORBIDDEN, "Wrong host.\n")
        if self._open >= MAX_CONNECTIONS:
            self._refuse(f"{self._open} connections already open")
            return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "Too many connections.\n")
        return None

    # --------------------------------------------------------------------------------------------
    async def run(self) -> None:
        from websockets.asyncio.server import serve

        loop = asyncio.get_running_loop()
        self.engine.add_status_listener(self._broadcast)
        self.engine.load()
        async with serve(self._handler, "127.0.0.1", self.port, **self.serve_options()) as server:
            port = server.sockets[0].getsockname()[1]
            info = {"port": port, "token": self.token, "pid": os.getpid()}
            claim_engine_info(info)
            if self.handshake and sys.stdout is not None:
                print(json.dumps(info), flush=True)
            log.info("Engine listening on ws://127.0.0.1:%d (pid %d)", port, os.getpid())
            try:
                loop.add_signal_handler(signal.SIGINT, self._stop.set)
            except (NotImplementedError, RuntimeError):
                pass
            await self._stop.wait()
        release_engine_info(os.getpid())
        self.engine.shutdown()
        log.info("Engine stopped")

    def _broadcast(self, msg: dict[str, Any]) -> None:
        for c in list(self._clients):
            c.emit(msg)

    # --------------------------------------------------------------------------------------------
    async def _handler(self, ws) -> None:
        self._open += 1
        try:
            await self._serve_client(ws)
        finally:
            self._open -= 1

    async def _greet(self, ws) -> str | None:
        """The client's name once its hello carries the launch token; None when it does not.
        The connection is then closed and the reason logged - never the token it sent."""
        try:
            first = await asyncio.wait_for(ws.recv(), timeout=HELLO_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._refuse("no hello")
            await ws.close(P.CLOSE_BAD_HELLO, "expected hello")
            return None
        except Exception as e:  # closed, or a hello larger than any real one
            self._refuse(f"hello failed ({e.__class__.__name__})")
            return None
        try:
            hello = P.decode(first) if isinstance(first, str) else {}
        except (ValueError, RecursionError):
            hello = {}
        if hello.get("type") != P.HELLO:
            self._refuse("the first message was not a hello")
            await ws.close(P.CLOSE_BAD_HELLO, "expected hello")
            return None
        token = hello.get("token")
        if not isinstance(token, str) or not secrets.compare_digest(token.encode(), self.token.encode()):
            self._refuse("wrong token")
            await ws.close(P.CLOSE_UNAUTHORIZED, "bad token")
            return None
        name = hello.get("client")
        return name[:P.MAX_CLIENT_NAME_CHARS] if isinstance(name, str) and name else "?"

    async def _serve_client(self, ws) -> None:
        client = ClientConn(ws, asyncio.get_running_loop())
        name = await self._greet(ws)
        if name is None:
            return
        client.name = name
        # Past the token: settings and commands can be large.
        ws.protocol.max_message_size = P.MAX_MESSAGE_BYTES

        self._clients.add(client)
        sender = asyncio.create_task(self._sender(client))
        log.info("client connected: %s", client.name)
        try:
            await ws.send(P.encode({"type": P.HELLO_OK, "version": P.PROTOCOL_VERSION, "status": self.engine.status()}))
            async for message in ws:
                if client.too_slow:
                    break
                # Messages already received are handed over without a pause, so without this
                # one client's backlog held the whole event loop: 5000 status requests at 5 ms
                # each kept every other client, and this one's own disconnection, waiting 25 s.
                await asyncio.sleep(0)
                if isinstance(message, bytes):
                    self._audio(client, message)
                    continue
                try:
                    msg = P.decode(message)
                except (ValueError, RecursionError) as e:
                    log.warning("unreadable message from %s: %s", client.name, e.__class__.__name__)
                    client.emit(P.error(problems.BAD_MESSAGE, "not a JSON object with a string 'type'"))
                    continue
                try:
                    await self._dispatch(client, msg)
                except Exception as e:
                    log.exception("%s from %s failed", msg["type"][:40], client.name)
                    client.emit(P.error(problems.BAD_MESSAGE, str(e)))
        finally:
            self._clients.discard(client)
            if client.session is not None:
                client.session.cancel()
            client.queue.put_nowait(None)
            sender.cancel()
            log.info("client disconnected: %s", client.name)

    @staticmethod
    def _audio(client: ClientConn, frame: bytes) -> None:
        """A frame of the client's current take. One that cannot be audio is dropped and
        reported once per take, without the take's id: the take goes on, and an error naming it
        would end it at the shell."""
        session = client.session
        if session is None:
            return
        problem = None
        if len(frame) > P.MAX_AUDIO_FRAME_BYTES:
            problem = f"an audio frame of {len(frame)} bytes is larger than {P.MAX_AUDIO_FRAME_BYTES}"
        elif len(frame) % 2:
            problem = "an audio frame had an odd number of bytes (expected 16-bit samples)"
        else:
            try:
                session.feed(frame)
            except Exception as e:
                log.exception("audio for %s failed", session.id)
                problem = f"audio could not be used: {e}"
        if problem and client.bad_audio_in != session.id:
            client.bad_audio_in = session.id
            log.warning("%s: %s", session.id, problem)
            client.emit(P.error(problems.BAD_AUDIO, problem))

    async def _sender(self, client: ClientConn) -> None:
        try:
            while True:
                msg = await client.queue.get()
                if msg is None:
                    return
                await client.ws.send(P.encode(msg))
        except Exception as e:
            log.debug("sender for %s ended: %s", client.name, e)

    @staticmethod
    def _names_current(client: ClientConn, msg: dict[str, Any]) -> bool:
        """Whether an end or cancel is about the session running now. A late one for an
        earlier take must not end or cancel the take that has started since. A message with
        no id means the current session, as it always did."""
        if client.session is None:
            return False
        sid = msg.get("id")
        return sid is None or str(sid) == client.session.id

    async def _dispatch(self, client: ClientConn, msg: dict[str, Any]) -> None:
        t = msg["type"]
        try:
            msg = P.checked(msg)
        except ValueError as e:
            sid = msg.get("id")
            sid = sid if isinstance(sid, str) and len(sid) <= P.MAX_ID_CHARS else None
            if t == P.COMMAND_RUN:
                # The shell waits for this answer: give it one that leaves the text alone.
                client.emit({"type": P.COMMAND_RESULT, "id": sid,
                             **CommandResult(text="", rejected=f"bad request: {e}").as_dict()})
            else:
                # Naming a take that could not start ends it at the shell; naming a running
                # one over a malformed end or cancel would end it too early.
                client.emit(P.error(problems.BAD_MESSAGE, f"{t}: {e}", sid if t == P.SESSION_START else None))
            return
        if t == P.SESSION_START:
            # A take still recording is abandoned by a new one. A take that has already ended
            # is finishing its text, and is left to deliver it: pressing the chord again
            # straight after letting go used to cancel it mid-decode, and the sentence just
            # spoken was never typed. Its final carries its own id, so it cannot be mistaken
            # for the new take's.
            old = client.session
            if old is not None and not old.done and not old.ending:
                old.cancel()
            sid = str(msg.get("id") or uuid.uuid4().hex[:8])
            try:
                client.session = self.engine.start_session(sid, msg.get("context") or {}, client.emit,
                                                           msg.get("language"))
            except Exception as e:
                client.emit(P.error(problems.TAKE_REFUSED, str(e), sid))
        elif t == P.SESSION_END:
            if self._names_current(client, msg):
                client.session.end()
        elif t == P.SESSION_CANCEL:
            if self._names_current(client, msg):
                client.session.cancel()
        elif t == P.COMMAND_RUN:
            cid = msg.get("id")
            selection = msg.get("selection") or ""
            instruction = msg.get("instruction") or ""

            def job() -> None:
                result = self.engine.run_command(selection, instruction)
                client.emit({"type": P.COMMAND_RESULT, "id": cid, **result.as_dict()})

            # On the language-model worker, so a command queues behind clean-up rather than
            # competing with it for the GPU.
            self.engine.submit_llm(job)
        elif t in (P.SELFCHECK_RUN, P.SELFCHECK_REPAIR):
            rid = msg.get("id")

            def check() -> None:
                # On its own thread: a full check hashes gigabytes, and neither model worker
                # may wait for that.
                try:
                    if t == P.SELFCHECK_RUN:
                        checks = self.engine.full_check() if msg.get("full") else list(self.engine.checks)
                        client.emit({"type": P.SELFCHECK_RESULT, "id": rid, "checks": checks})
                    else:
                        client.emit({"type": P.SELFCHECK_REPAIRED, "id": rid,
                                     "removed": self.engine.repair_models()})
                except Exception as e:
                    log.exception("self-check failed")
                    client.emit(P.error(problems.UNKNOWN_MESSAGE, f"self-check failed: {e}"))

            threading.Thread(target=check, name="selfcheck", daemon=True).start()
        elif t in (P.MODELS_DOWNLOAD, P.MODELS_CANCEL, P.MODELS_REMOVE):
            def act() -> None:
                # Off the connection's loop: removing a model deletes gigabytes.
                try:
                    if t == P.MODELS_DOWNLOAD:
                        self.engine.download_model(msg["kind"], msg["key"])
                    elif t == P.MODELS_CANCEL:
                        if not self.engine.cancel_download(msg["id"]):
                            raise ValueError("that download cannot be stopped, or has already finished")
                    else:
                        self.engine.remove_model(msg["kind"], msg["key"])
                except Exception as e:
                    client.emit(P.error(problems.SETTING_REFUSED, f"{problems.detail(e)}"))
                client.emit(self.engine.status())

            threading.Thread(target=act, name="models", daemon=True).start()
        elif t == P.STATUS_GET:
            client.emit(self.engine.status())
        elif t == P.SETTINGS_RESET:
            self.engine.reset_preferences()
            client.emit(self.engine.status())
        elif t == P.SETTINGS_SET:
            for problem in self.engine.apply_settings(msg) or []:
                client.emit(P.error(problems.SETTING_REFUSED, problem))  # the Hub shows it
            client.emit(self.engine.status())
        elif t == P.SHUTDOWN:
            log.info("shutdown requested by %s", client.name)
            self._stop.set()
        elif t == P.HELLO:
            pass  # a second hello changes nothing
        else:
            client.emit(P.error(problems.UNKNOWN_MESSAGE, t[:40]))


def serve(cfg: Config, port: int = 0, token: str | None = None, handshake: bool = False) -> int:
    server = EngineServer(cfg, port, token, handshake)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass
    return 0


def serve_in_thread(cfg: Config) -> tuple[threading.Thread, EngineServer, str]:
    """Run the server on a background thread (in-process mode / tests). Returns when the
    port is known."""
    server = EngineServer(cfg, 0, None, False)
    ready = threading.Event()
    result: dict[str, Any] = {}

    def run() -> None:
        async def main() -> None:
            from websockets.asyncio.server import serve as ws_serve

            loop = asyncio.get_running_loop()
            server.engine.add_status_listener(server._broadcast)
            server.engine.load()
            async with ws_serve(server._handler, "127.0.0.1", 0, **server.serve_options()) as s:
                result["port"] = s.sockets[0].getsockname()[1]
                ready.set()
                await server._stop.wait()
            server.engine.shutdown()

        asyncio.run(main())

    th = threading.Thread(target=run, name="engine-server", daemon=True)
    th.start()
    ready.wait(30)
    return th, server, f"ws://127.0.0.1:{result['port']}"
