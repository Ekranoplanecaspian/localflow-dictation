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
import secrets
import signal
import sys
import threading
import uuid
from typing import Any

from localflow.config import CONFIG_DIR, Config
from localflow.service import protocol as P
from localflow.service.engine import Engine, Session

log = logging.getLogger(__name__)

ENGINE_INFO_PATH = CONFIG_DIR / "engine.json"


class ClientConn:
    def __init__(self, ws, loop: asyncio.AbstractEventLoop):
        self.ws = ws
        self.loop = loop
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.session: Session | None = None
        self.name = "?"

    def emit(self, msg: dict[str, Any]) -> None:
        """Thread-safe: called from the engine worker thread."""
        self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)


class EngineServer:
    def __init__(self, cfg: Config, port: int = 0, token: str | None = None, handshake: bool = False):
        self.cfg = cfg
        self.port = port
        self.token = token or secrets.token_hex(16)
        self.handshake = handshake
        self.engine = Engine(cfg)
        self._stop = asyncio.Event()
        self._clients: set[ClientConn] = set()

    # --------------------------------------------------------------------------------------------
    async def run(self) -> None:
        from websockets.asyncio.server import serve

        loop = asyncio.get_running_loop()
        self.engine.add_status_listener(self._broadcast)
        self.engine.load()
        async with serve(self._handler, "127.0.0.1", self.port, max_size=4 * 2**20, max_queue=512) as server:
            port = server.sockets[0].getsockname()[1]
            info = {"port": port, "token": self.token, "pid": os.getpid()}
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            ENGINE_INFO_PATH.write_text(json.dumps(info), encoding="utf-8")
            if self.handshake and sys.stdout is not None:
                print(json.dumps(info), flush=True)
            log.info("Engine listening on ws://127.0.0.1:%d (pid %d)", port, os.getpid())
            try:
                loop.add_signal_handler(signal.SIGINT, self._stop.set)
            except (NotImplementedError, RuntimeError):
                pass
            await self._stop.wait()
        try:
            ENGINE_INFO_PATH.unlink()
        except OSError:
            pass
        self.engine.shutdown()
        log.info("Engine stopped")

    def _broadcast(self, msg: dict[str, Any]) -> None:
        for c in list(self._clients):
            c.emit(msg)

    # --------------------------------------------------------------------------------------------
    async def _handler(self, ws) -> None:
        loop = asyncio.get_running_loop()
        client = ClientConn(ws, loop)
        try:
            first = await asyncio.wait_for(ws.recv(), timeout=5)
            hello = P.decode(first) if isinstance(first, str) else {}
            if hello.get("type") != P.HELLO:
                await ws.close(P.CLOSE_BAD_HELLO, "expected hello")
                return
            if not secrets.compare_digest(str(hello.get("token", "")), self.token):
                await ws.close(P.CLOSE_UNAUTHORIZED, "bad token")
                return
            client.name = str(hello.get("client", "?"))
        except Exception as e:
            log.debug("handshake failed: %s", e)
            return

        self._clients.add(client)
        sender = asyncio.create_task(self._sender(client))
        log.info("client connected: %s", client.name)
        try:
            await ws.send(P.encode({"type": P.HELLO_OK, "version": P.PROTOCOL_VERSION, "status": self.engine.status()}))
            async for message in ws:
                if isinstance(message, bytes):
                    if client.session is not None:
                        client.session.feed(message)
                    continue
                try:
                    msg = P.decode(message)
                    await self._dispatch(client, msg)
                except Exception as e:
                    log.exception("bad message from %s", client.name)
                    client.emit(P.error("bad_message", str(e)))
        finally:
            self._clients.discard(client)
            if client.session is not None:
                client.session.cancel()
            client.queue.put_nowait(None)
            sender.cancel()
            log.info("client disconnected: %s", client.name)

    async def _sender(self, client: ClientConn) -> None:
        try:
            while True:
                msg = await client.queue.get()
                if msg is None:
                    return
                await client.ws.send(P.encode(msg))
        except Exception as e:
            log.debug("sender for %s ended: %s", client.name, e)

    async def _dispatch(self, client: ClientConn, msg: dict[str, Any]) -> None:
        t = msg["type"]
        if t == P.SESSION_START:
            if client.session is not None and not client.session.done:
                client.session.cancel()
            sid = str(msg.get("id") or uuid.uuid4().hex[:8])
            try:
                client.session = self.engine.start_session(sid, msg.get("context") or {}, client.emit,
                                                           msg.get("language"))
            except Exception as e:
                client.emit(P.error("not_ready", str(e), sid))
        elif t == P.SESSION_END:
            if client.session is not None:
                client.session.end()
        elif t == P.SESSION_CANCEL:
            if client.session is not None:
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
        elif t == P.STATUS_GET:
            client.emit(self.engine.status())
        elif t == P.SETTINGS_SET:
            self.engine.apply_settings(msg)
            client.emit(self.engine.status())
        elif t == P.SHUTDOWN:
            log.info("shutdown requested by %s", client.name)
            self._stop.set()
        else:
            client.emit(P.error("unknown_type", t))


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
            async with ws_serve(server._handler, "127.0.0.1", 0, max_size=4 * 2**20) as s:
                result["port"] = s.sockets[0].getsockname()[1]
                ready.set()
                await server._stop.wait()
            server.engine.shutdown()

        asyncio.run(main())

    th = threading.Thread(target=run, name="engine-server", daemon=True)
    th.start()
    ready.wait(30)
    return th, server, f"ws://127.0.0.1:{result['port']}"
