"""Command line entry point: `localflow` or `python -m localflow`."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import wave
from dataclasses import replace
from logging.handlers import RotatingFileHandler

import numpy as np

from localflow import __version__
from localflow.config import CONFIG_DIR, CONFIG_PATH, LOG_PATH, Config

log = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    handlers: list[logging.Handler] = [RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8")]
    if sys.stderr is not None:  # pythonw.exe has no console
        for stream in (sys.stdout, sys.stderr):  # dictated text can be in any script; cp1252 consoles choke
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        handlers.append(logging.StreamHandler())
    for h in handlers:
        h.setFormatter(fmt)
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), handlers=handlers)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------------------------
def cmd_serve(cfg: Config, port: int, token: str | None, handshake: bool) -> int:
    from localflow.service.server import serve

    return serve(cfg, port=port, token=token, handshake=handshake)


def cmd_send_wav(cfg: Config, path: str, realtime: bool) -> int:
    """Stream a WAV through the session protocol like a real dictation and print what comes back."""
    import threading
    import time

    from localflow.service.client import EngineClient, EngineProcess, discover
    from localflow.service.protocol import FRAME_SAMPLES

    audio = _load_wav(path)
    proc = None
    found = discover()
    client = EngineClient("send-wav")
    connected = False
    if found:
        try:
            client.connect(*found)
            connected = True
            print(f"attached to running engine on port {found[0]}")
        except Exception as e:
            print(f"engine.json present but connection failed ({e}); starting a fresh engine")
    if not connected:
        proc = EngineProcess(cfg.log_level)
        client.connect(*proc.start())
    ready = threading.Event()
    done = threading.Event()
    result: dict = {}

    def on_status(st):
        if st.get("stt", {}).get("state") == "ready":
            ready.set()
        elif st.get("stt", {}).get("state") == "error":
            result["error"] = st["stt"].get("error")
            ready.set()

    client.on_status = on_status
    client.on_partial = lambda ev: print(f"  partial: {ev['text']}")
    client.on_final = lambda ev: (result.update(ev), done.set())
    client.on_error = lambda ev: (result.update(error=ev.get("message")), done.set())
    on_status(client.status or {})
    print("waiting for the engine ...", flush=True)
    ready.wait(120)
    if result.get("error"):
        print("engine error:", result["error"])
        return 1
    st = client.status["stt"]
    print(f"engine: {st['model']} on {st['device']} ({st['precision']})")
    client.start_session({"app": "send-wav", "title": path})
    t0 = time.perf_counter()
    for i in range(0, len(audio), FRAME_SAMPLES):
        client.send_audio(audio[i:i + FRAME_SAMPLES])
        if realtime:
            time.sleep(FRAME_SAMPLES / 16000)
    released = time.perf_counter()
    client.end_session()
    done.wait(60)
    dt = (time.perf_counter() - released) * 1000
    if "text" in result:
        t = result["timings"]
        print(f"final: {result['text']}")
        print(f"  {t['audio_s']}s audio | {t['mode']}: {t['live_decodes']} live decodes ({t['stt_live_ms']} ms) over "
              f"{t['phrases']} phrases | final {'reused live' if t['reused_live'] else str(t['stt_final_ms']) + ' ms'} "
              f"({t['chunks']} chunk(s)) | post {t['post_ms']} ms | "
              f"engine release->final {t['release_to_final_ms']} ms | client key-up->final {dt:.0f} ms")
    else:
        print("no final:", result)
    client.close()
    if proc:
        proc.stop()
    return 0 if "text" in result else 1


def cmd_devices(cfg: Config) -> int:
    from localflow.audio import list_input_devices, resolve_input_device

    chosen, _ = resolve_input_device(cfg.audio.device)
    print(f"{'idx':>4}  {'host api':<20} name")
    for idx, name, api in list_input_devices():
        mark = "*" if idx == chosen else " "
        print(f"{mark}{idx:>3}  {api:<20} {name}")
    print("\n* = device LocalFlow will use (config: audio.device = index or name substring)")
    return 0


def _load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    with wave.open(path) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != target_sr:  # linear resample; fine for a debug command
        idx = np.arange(0, len(x), sr / target_sr)
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return x


def cmd_transcribe(cfg: Config, path: str) -> int:
    import time

    from localflow.cleanup.pipeline import CleanupPipeline
    from localflow.stt import build_transcriber

    audio = _load_wav(path)
    stt = build_transcriber(cfg.stt)
    stt.warmup()
    t0 = time.perf_counter()
    raw = stt.transcribe(audio, language=cfg.stt.language)
    dt = time.perf_counter() - t0
    print(f"[{len(audio) / 16000:.1f}s audio, {dt * 1000:.0f} ms]")
    print("raw  :", raw)
    from dataclasses import replace as _replace
    pipe = CleanupPipeline(_replace(cfg.postprocess, llm_cleanup=False))
    print("clean:", pipe.process(raw).text)
    return 0


def cmd_command(cfg: Config, selection: str, instruction: str) -> int:
    """Run one command-mode edit against the real clean-up model and print what came back.

    The unit tests cover the guard with a fake provider, which proves the rules but says
    nothing about whether Qwen actually edits rather than chats. This is how that gets checked.
    """
    from localflow.cleanup.command import CommandRunner
    from localflow.llm.providers import build_provider

    server = None

    def server_factory():
        # The bundled provider needs a llama-server, which the engine normally owns. Starting a
        # second one would not fit in 8 GB beside the running app, so stop LocalFlow first.
        nonlocal server
        from localflow.llm.server import LlamaServer

        if server is None:
            server = LlamaServer(model_key=cfg.postprocess.llm_model, device=cfg.stt.device)
            server.start()
        return server

    try:
        provider = build_provider(cfg.postprocess, server_factory=server_factory)
    except Exception as e:
        print(f"could not reach a clean-up model: {e}")
        return 1
    if provider is None:
        print("auto-edits are off (see `localflow config`)")
        return 1
    result = CommandRunner(provider).run(selection, instruction)
    print(f"instruction : {instruction}")
    print(f"selection   : {selection}")
    print(f"result      : {result.text}")
    verdict = "replaced" if result.changed else f"LEFT ALONE ({result.rejected})"
    print(f"verdict     : {verdict} in {result.ms:.0f} ms" if result.ms else f"verdict     : {verdict}")
    if server is not None:
        server.stop()
    return 0


def cmd_config(cfg: Config) -> int:
    print(CONFIG_PATH)
    print(open(CONFIG_PATH, encoding="utf-8").read())
    print("log:", LOG_PATH)
    return 0


def cmd_bench(cfg: Config, args) -> int:
    from localflow import bench

    sub = args.bench_cmd or "run"
    if sub == "fetch":
        return bench.fetch_public(args.n)
    if sub == "record":
        return bench.record_own(cfg)
    if sub == "stream":
        return 0 if bench.stream(cfg, args.set, args.limit) else 1
    if sub == "cleanup":
        return 0 if bench.cleanup(cfg, args.model, args.rules_only, not args.quiet, args.device) else 1
    stt = cfg.stt
    if args.device:
        stt = replace(stt, device=args.device)
    if args.precision:
        stt = replace(stt, precision=args.precision)
    if args.model_path:
        stt = replace(stt, model_path=args.model_path)
    if args.backend:
        stt = replace(stt, backend=args.backend)
    cfg = replace(cfg, stt=stt)
    return 0 if bench.run(cfg, args.set, args.limit, args.quiet) else 1


# ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="localflow", description="Local push-to-talk dictation for Windows.")
    p.add_argument("--version", action="version", version=f"localflow {__version__}")
    p.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING (default from config)")
    sub = p.add_subparsers(dest="cmd")
    sv = sub.add_parser("serve", help="run the speech engine as a local service")
    sv.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    sv.add_argument("--token", default=None, help="auth token (default: random, printed with --handshake)")
    sv.add_argument("--handshake", action="store_true", help="print {port, token, pid} as JSON on stdout when ready")
    sw = sub.add_parser("send-wav", help="stream a WAV through the engine like a dictation (latency test)")
    sw.add_argument("wav")
    sw.add_argument("--fast", action="store_true", help="send audio as fast as possible instead of real time")
    sub.add_parser("devices", help="list microphones")
    t = sub.add_parser("transcribe", help="transcribe a WAV file (debug the STT pipeline)")
    t.add_argument("wav")
    sub.add_parser("config", help="print config path and contents")
    cm = sub.add_parser("command", help="apply a spoken instruction to a piece of text (command mode)")
    cm.add_argument("instruction", help='e.g. "make this more formal"')
    cm.add_argument("selection", help="the text that would have been selected")
    _add_bench_parser(sub)
    return p


def _add_bench_parser(sub) -> None:
    b = sub.add_parser("bench", help="benchmark accuracy and latency on a golden set")
    bsub = b.add_subparsers(dest="bench_cmd")
    br = bsub.add_parser("run", help="run the benchmark (default)")
    br.add_argument("--set", default="public", choices=["public", "own", "all"])
    br.add_argument("--limit", type=int, default=None, help="only the first N files")
    br.add_argument("--device", default=None, help="override stt.device: auto | cpu | cuda")
    br.add_argument("--precision", default=None, help="override stt.precision: auto | fp32 | fp16 | int8")
    br.add_argument("--model-path", default=None, help="override stt.model_path (directory of model files)")
    br.add_argument("--backend", default=None, help="override stt.backend: parakeet | whisper")
    br.add_argument("--quiet", action="store_true")
    bs = bsub.add_parser("stream", help="stream a set through a real engine at real-time pace (release latency KPI)")
    bs.add_argument("--set", default="own", choices=["public", "own", "all"])
    bs.add_argument("--limit", type=int, default=None)
    bc = bsub.add_parser("cleanup", help="run the clean-up quality set (rules vs rules+LLM)")
    bc.add_argument("--model", default=None, help="bundled model key (qwen3-4b, qwen3-1.7b) or provider model name")
    bc.add_argument("--rules-only", action="store_true")
    bc.add_argument("--device", default="auto", help="auto | cpu (CPU build of llama.cpp, for prompt work while the GPU is busy)")
    bc.add_argument("--quiet", action="store_true")
    bf = bsub.add_parser("fetch", help="download the public LibriSpeech test-clean subset")
    bf.add_argument("--n", type=int, default=60)
    bsub.add_parser("record", help="record the own-voice set from a list of prompts")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = Config.load()
    _setup_logging(args.log_level or cfg.log_level)
    # There is no default command any more. `run` used to be it, because a shortcut to
    # `localflow-bg` with no arguments started the tray app; the shell owns all of that now and
    # always asks for `serve` explicitly, so a bare `localflow` is someone looking for help.
    if not args.cmd:
        parser.print_help()
        sys.exit(0)
    try:
        if args.cmd == "serve":
            code = cmd_serve(cfg, args.port, args.token, args.handshake)
        elif args.cmd == "send-wav":
            code = cmd_send_wav(cfg, args.wav, realtime=not args.fast)
        elif args.cmd == "devices":
            code = cmd_devices(cfg)
        elif args.cmd == "transcribe":
            code = cmd_transcribe(cfg, args.wav)
        elif args.cmd == "config":
            code = cmd_config(cfg)
        elif args.cmd == "command":
            code = cmd_command(cfg, args.selection, args.instruction)
        else:
            code = cmd_bench(cfg, args)
    except Exception:
        # Under pythonw - which is how the shell starts the engine - this is the only place an
        # error is ever visible.
        log.exception("%s failed", args.cmd)
        code = 1
    sys.exit(code)
