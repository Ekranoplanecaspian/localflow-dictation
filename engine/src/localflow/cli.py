"""Command line entry point: `localflow` or `python -m localflow`."""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import sys
import threading
import wave
from dataclasses import replace
from logging.handlers import RotatingFileHandler

import numpy as np

from localflow import __version__
from localflow.config import CONFIG_DIR, CONFIG_PATH, FAULT_PATH, LOG_PATH, SAFE_MODE_ENV, Config, apply_safe_mode

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


_fault_file = None  # held open for the life of the process: faulthandler writes to it at the end


def _install_crash_hooks(fault_path=FAULT_PATH) -> None:
    """Make every way the engine can die leave something in the log.

    Before, an exception outside `main`'s own handler, or in a thread, went to stderr - which
    under the shell is a pipe it keeps only the last few lines of - and a native crash (a fault
    in onnxruntime, say) left nothing anywhere: the shell's "engine exited" was all there was.
    """
    global _fault_file

    def on_exception(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("unhandled exception; the engine is stopping", exc_info=(exc_type, exc, tb))

    def on_thread_exception(args):
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "?"
        log.critical("unhandled exception in thread %s", name,
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = on_exception
    threading.excepthook = on_thread_exception

    # A native crash cannot run Python, so the stack is written by faulthandler, to its own
    # file, and moved into the log by the next engine to start.
    try:
        if fault_path.exists() and fault_path.stat().st_size > 0:
            text = fault_path.read_text(encoding="utf-8", errors="replace").strip()
            log.error("the previous engine ended with a fatal error:\n%s", text)
        _fault_file = open(fault_path, "w", encoding="utf-8")
        faulthandler.enable(file=_fault_file, all_threads=True)
    except OSError as e:
        log.warning("could not set up the crash report file %s: %s", fault_path, e)


# ---------------------------------------------------------------------------------------------
def prestart_speech(cfg: Config) -> None:
    """First thing: the speech worker starts loading the model the engine will most likely ask
    for - the saved one, on the graphics card (the settings Engine._speech_cfg makes) - while
    the engine itself starts up. Only a model already on this PC: one that is not would be
    downloaded by the worker itself, beside the engine's own download of it - on a real first
    run the 2.4 GB of Parakeet v3 came down twice and was kept twice (2026-10-01)."""
    if cfg.stt.device == "cpu":
        return
    from localflow.stt import catalogue, remote

    entry = catalogue.current(cfg.stt)
    here = entry is None or any(catalogue.is_installed(entry, d) for d in ("cuda", "cpu"))
    remote.prestart(replace(cfg.stt, device="cuda" if cfg.stt.device == "cuda" else "auto") if here else None)


def cmd_serve(cfg: Config, port: int, token: str | None, handshake: bool) -> int:
    prestart_speech(cfg)
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


def cmd_hardware(as_json: bool = False) -> int:
    from localflow import gpu, hwinfo
    from localflow.config import MODELS_DIR

    report = hwinfo.detect()
    driver = gpu.monitor().driver_version()
    if as_json:
        free_ram, free_disk = hwinfo.ram_free_gb(), hwinfo.disk_free_gb(MODELS_DIR)
        print(json.dumps({**report.as_dict(), "nvidia_driver": driver,
                          "ram_free_gb": round(free_ram, 1) if free_ram is not None else None,
                          "disk_free_gb": round(free_disk, 1) if free_disk is not None else None}, indent=2))
        return 0
    for line in hwinfo.describe(report, MODELS_DIR):
        print(line)
    if driver:
        print(f"NVIDIA     driver {driver}")
    return 0


def cmd_cuda(install: bool = False) -> int:
    from localflow import cudalibs

    if cudalibs.bundled():
        print("bundled: the nvidia packages beside onnxruntime are used; nothing to download")
    elif cudalibs.installed():
        print(f"downloaded: {cudalibs.lib_dir()}")
    elif install:
        def progress(done: int, total: int) -> None:
            print(f"\r{done / 2**20:,.0f} of {total / 2**20:,.0f} MB", end="", flush=True)

        print(f"ready: {cudalibs.ensure(progress)}")
    else:
        print(f"not downloaded ({cudalibs.DOWNLOAD_BYTES / 2**30:.1f} GB); `localflow cuda --install` fetches them")
    from localflow.stt.parakeet import cuda_available

    ok, reason = cuda_available()
    print(f"speech on the graphics card: {'yes' if ok else 'no'} ({reason})")
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
    if sub == "release":
        from localflow import perfrecord

        return perfrecord.release(cfg)
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
    if args.speech:
        from localflow.stt import catalogue

        stt = replace(stt)
        catalogue.get(args.speech).apply(stt)
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
    sub.add_parser("speech-worker", help=argparse.SUPPRESS)  # the engine's GPU speech process (stt/remote.py)
    fe = sub.add_parser("fetch", help=argparse.SUPPRESS)  # one model download the engine can stop (fetch.py)
    fe.add_argument("kind")
    fe.add_argument("key")
    fe.add_argument("device", nargs="?", default="cpu")
    sw = sub.add_parser("send-wav", help="stream a WAV through the engine like a dictation (latency test)")
    sw.add_argument("wav")
    sw.add_argument("--fast", action="store_true", help="send audio as fast as possible instead of real time")
    sub.add_parser("devices", help="list microphones")
    hw = sub.add_parser("hardware", help="what this computer has: processor, memory, graphics, disk")
    hw.add_argument("--json", action="store_true", help="the report as JSON")
    cu = sub.add_parser("cuda", help="the graphics card libraries speech needs on an NVIDIA card: where they are")
    cu.add_argument("--install", action="store_true", help="download them now (about 1 GB) if they are missing")
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
    br.add_argument("--speech", default=None,
                    help="a speech model from the catalogue: parakeet-v3, parakeet-v2, parakeet-v3-compact, "
                         "whisper-turbo")
    br.add_argument("--quiet", action="store_true")
    bs = bsub.add_parser("stream", help="stream a set through a real engine at real-time pace (release latency KPI)")
    bs.add_argument("--set", default="own", choices=["public", "own", "all"])
    bs.add_argument("--limit", type=int, default=None)
    bc = bsub.add_parser("cleanup", help="run the clean-up quality set (rules vs rules+LLM)")
    bc.add_argument("--model", default=None, help="bundled model key (qwen3-4b, phi-4-mini, gemma-4-e2b, ...; see llm/manifest.py) or provider model name")
    bc.add_argument("--rules-only", action="store_true")
    bc.add_argument("--device", default="auto",
                    help="auto | cpu (CPU build of llama.cpp, for prompt work while the GPU is busy) | "
                         "vulkan (AMD/Intel graphics)")
    bc.add_argument("--quiet", action="store_true")
    bf = bsub.add_parser("fetch", help="download the public LibriSpeech test-clean subset")
    bf.add_argument("--n", type=int, default=60)
    bsub.add_parser("record", help="record the own-voice set from a list of prompts")
    bsub.add_parser("release", help="the performance record for this version: start-up, latency, accuracy, "
                                    "memory and clean-up quality, written to docs/perf and compared with the last")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "speech-worker":
        # Before anything else: it logs through the engine, and its stdout is the channel.
        from localflow.stt.remote import worker_main

        sys.exit(worker_main())
    if args.cmd == "fetch":
        # Its stdout is the channel too. The proxy and mirror come from the engine's environment.
        from localflow.fetch import child_main

        sys.exit(child_main([args.kind, args.key, args.device]))
    cfg = Config.load()
    _setup_logging(args.log_level or cfg.log_level)
    # Before anything imports huggingface_hub (it reads the mirror once), and before any child
    # process starts (they inherit the proxy). The engine looks the proxy up in the background.
    from localflow import net

    net.apply_endpoint(cfg.network.hf_endpoint)
    net.apply_proxy(background=args.cmd == "serve")
    # There is no default command any more. `run` used to be it, because a shortcut to
    # `localflow-bg` with no arguments started the tray app; the shell owns all of that now and
    # always asks for `serve` explicitly, so a bare `localflow` is someone looking for help.
    if not args.cmd:
        parser.print_help()
        sys.exit(0)
    try:
        if args.cmd == "serve":
            _install_crash_hooks()
            if os.environ.get(SAFE_MODE_ENV) == "1":
                apply_safe_mode(cfg)
                log.warning("SAFE MODE: the engine crashed repeatedly, so this one runs on the processor, "
                            "with the default speech model and no AI clean-up. Settings on disk are unchanged.")
            code = cmd_serve(cfg, args.port, args.token, args.handshake)
        elif args.cmd == "send-wav":
            code = cmd_send_wav(cfg, args.wav, realtime=not args.fast)
        elif args.cmd == "devices":
            code = cmd_devices(cfg)
        elif args.cmd == "hardware":
            code = cmd_hardware(as_json=args.json)
        elif args.cmd == "cuda":
            code = cmd_cuda(install=args.install)
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
