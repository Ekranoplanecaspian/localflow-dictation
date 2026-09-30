"""Benchmark harness: word error rate, per-stage latency, and memory on a golden set.

Sets live under engine/bench/data/<set>/ as pairs of NAME.wav (16 kHz mono) + NAME.txt:
  public : LibriSpeech test-clean subset, fetched with `localflow bench fetch`
  own    : your own voice, recorded with `localflow bench record`
Every run writes engine/bench/results/<timestamp>-<set>.json and prints a summary,
so any later change ("faster", "no accuracy regression") is checked against a number.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from localflow.config import Config, PostProcessConfig

log = logging.getLogger(__name__)

BENCH_DIR = Path(__file__).resolve().parents[2] / "bench"
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"
SAMPLE_RATE = 16000

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)
_SPACES = re.compile(r"\s+")
_ORDINAL = re.compile(r"^(\d+)(st|nd|rd|th)$")
_UNIT_SPLIT = re.compile(r"^(\d+)([a-z]+)$")

_ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_ONES.update({"oh": 0})
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
_ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
                  "eighth": 8, "ninth": 9, "tenth": 10, "twelfth": 12, "twentieth": 20, "thirtieth": 30}
_UNITS = {"gigabytes": "gb", "gigabyte": "gb", "megabytes": "mb", "megabyte": "mb", "kilobytes": "kb",
          "percent": "%", "kilograms": "kg", "kilogram": "kg", "grams": "g", "gram": "g",
          "kilometres": "km", "kilometers": "km", "metres": "m", "meters": "m", "okay": "ok"}
_JOIN = {"standup": "stand up", "cleanup": "clean up", "candlelight": "candle light", "email": "e mail",
         "cannot": "can not"}


def _parse_number(tokens: list[str], i: int) -> tuple[str, int]:
    """Parse one spoken number starting at tokens[i]; return (digits, next index).

    "twenty three" -> 23, "two hundred fifty" -> 250, "two thousand twenty six" -> 2026.
    "three thirty" -> 3 (the tens word starts a new number, as in times and "nine fifteen").
    A run of three or more single digits ("four four seven one") is read digit by digit -> 4471,
    the way people say phone, invoice and card numbers."""
    # digit-by-digit run?
    run = []
    j = i
    while j < len(tokens) and tokens[j] in _ONES and _ONES[tokens[j]] < 10:
        run.append(_ONES[tokens[j]])
        j += 1
    if len(run) >= 3 and (j >= len(tokens) or tokens[j] not in _SCALES):
        return "".join(str(d) for d in run), j

    total, current, last, j = 0, 0, None, i
    while j < len(tokens):
        w = tokens[j]
        if w in _ORDINAL_WORDS:
            if last in (None, "tens"):
                current += _ORDINAL_WORDS[w]
                j += 1
            break
        if w in _ONES:
            v = _ONES[w]
            kind = "teen" if v >= 10 else "ones"
            if last is None or (last == "tens" and kind == "ones") or last in ("hundred", "big"):
                current += v
                last = kind
            else:
                break
        elif w in _TENS:
            if last is None or last in ("hundred", "big"):
                current += _TENS[w]
                last = "tens"
            else:
                break
        elif w == "hundred":
            if last in ("ones", "teen", None):
                current = (current or 1) * 100
                last = "hundred"
            else:
                break
        elif w in _SCALES:  # thousand, million, billion
            total += (current or 1) * _SCALES[w]
            current = 0
            last = "big"
        elif w == "and" and last in ("hundred", "big"):
            pass
        else:
            break
        j += 1
    return str(total + current), j


def _words_to_number(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in _ONES or t in _TENS or t in _ORDINAL_WORDS:
            value, i = _parse_number(tokens, i)
            out.append(value)
        else:
            out.append(t)
            i += 1
    return out


# ---------------------------------------------------------------------------------------------
# text normalisation + WER
def normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation, collapse whitespace, and treat digits and number words as
    the same thing ("3:30" == "three thirty", "32GB" == "32 gigabytes", "March 1" == "March 1st").
    Both sides get the same treatment, so a model that punctuates, capitalises, or writes numbers
    as digits is not penalised for it."""
    text = text.lower().replace("-", " ").replace(":", " ").replace("/", " ")
    text = _PUNCT.sub("", text)
    tokens = _SPACES.sub(" ", text).strip().split()
    expanded: list[str] = []
    for t in tokens:
        m = _ORDINAL.match(t)
        if m:
            t = m.group(1)
        m = _UNIT_SPLIT.match(t)
        if m:  # "32gb" -> "32 gb"
            expanded += [m.group(1), m.group(2)]
            continue
        if t in _JOIN:
            expanded += _JOIN[t].split()
            continue
        expanded.append(_UNITS.get(t, t))
    return _words_to_number(expanded)


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------------------------
@dataclass
class FileResult:
    name: str
    seconds: float
    stt_ms: float
    post_ms: float
    errors: int
    ref_words: int
    reference: str
    hypothesis: str


@dataclass
class RunResult:
    timestamp: str
    set: str
    files: int
    backend: str
    model: str
    device_requested: str
    device_used: str
    precision: str
    git_commit: str | None
    load_s: float
    audio_s: float
    wer: float
    stt_ms_p50: float
    stt_ms_p95: float
    ms_per_audio_second: float
    post_ms_p50: float
    rss_mb: float
    vram_mb: float | None
    per_file: list[FileResult]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _vram_used_mb() -> float | None:
    """Dedicated VRAM used by *this* process. Windows (WDDM) hides per-process usage from
    nvidia-smi, but exposes it as a performance counter."""
    import os

    ps = (f"(Get-Counter '\\GPU Process Memory(pid_{os.getpid()}*)\\Dedicated Usage' -ErrorAction SilentlyContinue)"
          ".CounterSamples | Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum")
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=15)
        return round(float(out.stdout.strip() or 0) / 2**20)
    except Exception:
        return None


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return 0.0


def load_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        idx = np.arange(0, len(audio), sr / SAMPLE_RATE)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def list_set(name: str) -> list[tuple[Path, str]]:
    d = DATA_DIR / name
    pairs = []
    for wav in sorted(d.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.exists():
            pairs.append((wav, txt.read_text(encoding="utf-8").strip()))
    return pairs


# ---------------------------------------------------------------------------------------------
def run(cfg: Config, set_name: str = "public", limit: int | None = None, quiet: bool = False) -> RunResult | None:
    from localflow.cleanup.pipeline import CleanupPipeline
    from localflow.stt import build_transcriber

    sets = ["public", "own"] if set_name == "all" else [set_name]
    pairs: list[tuple[Path, str]] = []
    for s in sets:
        pairs += list_set(s)
    if not pairs:
        print(f"No files in {DATA_DIR / set_name}. Run `localflow bench fetch` or `localflow bench record` first.")
        return None
    if limit:
        pairs = pairs[:limit]

    t0 = time.perf_counter()
    stt = build_transcriber(cfg.stt)
    stt.warmup()
    load_s = time.perf_counter() - t0
    post = CleanupPipeline(PostProcessConfig(llm_cleanup=False))
    device_used = getattr(stt, "device", cfg.stt.device)

    results: list[FileResult] = []
    for i, (wav, ref) in enumerate(pairs, 1):
        audio = load_wav(wav)
        t1 = time.perf_counter()
        hyp = stt.transcribe(audio, language=cfg.stt.language)
        t2 = time.perf_counter()
        clean = post.rules(hyp)
        t3 = time.perf_counter()
        r_words, h_words = normalize(ref), normalize(clean)
        errors = edit_distance(r_words, h_words)
        results.append(FileResult(wav.stem, len(audio) / SAMPLE_RATE, (t2 - t1) * 1000, (t3 - t2) * 1000,
                                  errors, len(r_words), ref, clean))
        if not quiet:
            print(f"  [{i:>3}/{len(pairs)}] {wav.stem:<24} {len(audio) / SAMPLE_RATE:5.1f}s  stt {(t2 - t1) * 1000:6.0f} ms  "
                  f"err {errors:>2}/{len(r_words):<3}", flush=True)

    vram_after = _vram_used_mb()
    stt_ms = [r.stt_ms for r in results]
    audio_s = sum(r.seconds for r in results)
    total_err = sum(r.errors for r in results)
    total_ref = sum(r.ref_words for r in results)
    run_result = RunResult(
        timestamp=datetime.now().strftime("%Y%m%d-%H%M%S"),
        set=set_name, files=len(results),
        backend=cfg.stt.backend, model=cfg.stt.model, device_requested=cfg.stt.device, device_used=device_used,
        precision=getattr(stt, "precision", cfg.stt.precision), git_commit=_git_commit(), load_s=round(load_s, 2),
        audio_s=round(audio_s, 1),
        wer=round(100 * total_err / max(total_ref, 1), 2),
        stt_ms_p50=round(statistics.median(stt_ms), 1),
        stt_ms_p95=round(float(np.percentile(stt_ms, 95)), 1),
        ms_per_audio_second=round(sum(stt_ms) / max(audio_s, 1e-6), 1),
        post_ms_p50=round(statistics.median(r.post_ms for r in results), 2),
        rss_mb=round(_rss_mb()),
        vram_mb=(round(vram_after) if vram_after is not None else None),
        per_file=results,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{run_result.timestamp}-{set_name}.json"
    out.write_text(json.dumps(asdict(run_result), indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(run_result, out)
    return run_result


def print_summary(r: RunResult, path: Path | None = None) -> None:
    print()
    print(f"set {r.set}: {r.files} files, {r.audio_s:.0f} s of audio | {r.backend} {r.model} "
          f"({r.device_used}, {r.precision}) | commit {r.git_commit or '-'}")
    print(f"  WER                 {r.wer:6.2f} %")
    print(f"  stt p50 / p95       {r.stt_ms_p50:6.0f} / {r.stt_ms_p95:.0f} ms")
    print(f"  ms per audio second {r.ms_per_audio_second:6.1f}   (lower is faster; 1000 = real time)")
    print(f"  rules p50           {r.post_ms_p50:6.2f} ms")
    print(f"  model load          {r.load_s:6.1f} s")
    print(f"  process RSS         {r.rss_mb:6.0f} MB" + (f"   dedicated VRAM {r.vram_mb} MB" if r.vram_mb else ""))
    worst = sorted(r.per_file, key=lambda f: f.errors / max(f.ref_words, 1), reverse=True)[:3]
    if worst and worst[0].errors:
        print("  worst files:")
        for f in worst:
            if f.errors:
                print(f"    {f.name}: {f.errors}/{f.ref_words}\n      ref: {f.reference[:110]}\n      hyp: {f.hypothesis[:110]}")
    if path:
        print(f"  saved {path}")


# ---------------------------------------------------------------------------------------------
def stream(cfg: Config, set_name: str = "own", limit: int | None = None, probe=None) -> RunResult | None:
    """The phase-1 KPI: stream each file through a real engine process at real-time pace, as the
    tray app does, and measure release-to-final latency and the accuracy of the final text."""
    import threading

    from localflow.service.client import EngineClient, EngineProcess
    from localflow.service.protocol import FRAME_SAMPLES

    pairs = list_set(set_name) if set_name != "all" else list_set("public") + list_set("own")
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        print(f"No files in {DATA_DIR / set_name}.")
        return None
    proc = EngineProcess(cfg.log_level)
    client = EngineClient("bench-stream")
    t0 = time.perf_counter()
    client.connect(*proc.start())
    if probe:  # the performance record (perfrecord.py) watches memory and times the loads
        probe.started(proc.pid)
    ready, done = threading.Event(), threading.Event()
    result: dict = {}

    def on_status(st):
        if probe and st.get("stt", {}).get("state") == "ready":
            probe.ready("speech")
        if probe and st.get("llm", {}).get("state") == "ready":
            probe.ready("cleanup")
        if st.get("stt", {}).get("state") in ("ready", "error"):
            result["status"] = st
            ready.set()

    client.on_status = on_status
    client.on_final = lambda ev: (result.update(final=ev), done.set())
    client.on_error = lambda ev: (result.update(final=None, error=ev), done.set())
    on_status(client.status or {})
    ready.wait(180)
    load_s = time.perf_counter() - t0
    st = result["status"]["stt"]
    if st["state"] != "ready":
        print("engine error:", st.get("error"))
        proc.stop()
        return None
    print(f"engine: {st['model']} on {st['device']} ({st['precision']}), streaming {len(pairs)} files at real time ...")
    results: list[FileResult] = []
    release_ms: list[float] = []
    final_ms: list[float] = []
    reused = 0
    modes: set[str] = set()
    for i, (wav, ref) in enumerate(pairs, 1):
        audio = load_wav(wav)
        done.clear()
        result.pop("final", None)
        client.start_session({"app": "bench", "title": wav.name})
        for j in range(0, len(audio), FRAME_SAMPLES):
            client.send_audio(audio[j:j + FRAME_SAMPLES])
            time.sleep(FRAME_SAMPLES / SAMPLE_RATE)
        released = time.perf_counter()
        client.end_session()
        done.wait(60)
        client_ms = (time.perf_counter() - released) * 1000
        ev = result.get("final")
        if not ev:
            print(f"  [{i:>3}/{len(pairs)}] {wav.stem}: no final ({result.get('error')})")
            continue
        t = ev["timings"]
        hyp = ev["text"]
        r_words, h_words = normalize(ref), normalize(hyp)
        errors = edit_distance(r_words, h_words)
        results.append(FileResult(wav.stem, t["audio_s"], t["stt_final_ms"], t["post_ms"], errors, len(r_words), ref, hyp))
        release_ms.append(client_ms)
        final_ms.append(t["release_to_final_ms"])
        reused += int(t.get("reused_live", False))
        modes.add(t["mode"])
        print(f"  [{i:>3}/{len(pairs)}] {wav.stem:<24} {t['audio_s']:5.1f}s  key-up->final {client_ms:5.0f} ms "
              f"({'reused' if t.get('reused_live') else 'decoded'})  err {errors:>2}/{len(r_words):<3}", flush=True)
        time.sleep(1.5)  # a realistic gap between dictations (lets the GPU idle, as it would)
    client.close()
    proc.stop()
    total_err = sum(r.errors for r in results)
    total_ref = sum(r.ref_words for r in results)
    run_result = RunResult(
        timestamp=datetime.now().strftime("%Y%m%d-%H%M%S"), set=f"stream-{set_name}", files=len(results),
        backend=cfg.stt.backend, model=st["model"], device_requested=cfg.stt.device, device_used=st["device"],
        precision=st["precision"], git_commit=_git_commit(), load_s=round(load_s, 2),
        audio_s=round(sum(r.seconds for r in results), 1),
        wer=round(100 * total_err / max(total_ref, 1), 2),
        stt_ms_p50=round(statistics.median(release_ms), 1), stt_ms_p95=round(float(np.percentile(release_ms, 95)), 1),
        ms_per_audio_second=round(sum(final_ms) / max(sum(r.seconds for r in results), 1e-6), 1),
        post_ms_p50=round(statistics.median(r.post_ms for r in results), 2),
        rss_mb=0, vram_mb=None, per_file=results,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{run_result.timestamp}-stream-{set_name}.json"
    out.write_text(json.dumps(asdict(run_result), indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"stream {set_name}: {len(results)} dictations, {run_result.audio_s:.0f} s of audio | {st['model']} on "
          f"{st['device']} ({st['precision']}), mode {'/'.join(sorted(modes))} | commit {run_result.git_commit or '-'}")
    print(f"  WER of final text        {run_result.wer:6.2f} %")
    print(f"  key-up -> final p50/p95  {run_result.stt_ms_p50:6.0f} / {run_result.stt_ms_p95:.0f} ms   "
          f"(engine-side p50 {statistics.median(final_ms):.0f} ms; live decode reused for {reused}/{len(results)})")
    print(f"  saved {out}")
    return run_result


CLEANUP_CASES = BENCH_DIR / "cleanup_cases.jsonl"
COMMAND_CASES = BENCH_DIR / "command_cases.jsonl"


def check_command(case: dict, result) -> list[str]:
    """What a command-mode result got wrong, by the case's objective checks. Empty = pass.

    The checks are deliberately about substance rather than style: the facts and names that
    have to survive a rewrite, the day a translation must name, the line count of a list. A
    refused edit fails, because the user asked for a change and got none."""
    text = result.text
    problems = []
    if not result.changed:
        problems.append(f"no change ({result.rejected or 'unchanged'})")
        return problems
    low = text.lower()
    for want in case.get("contains", []):
        if want.lower() not in low:
            problems.append(f"missing {want!r}")
    for bad in case.get("must_not", []):
        if bad.lower() in f" {low} ":
            problems.append(f"contains {bad!r}")
    if "min_lines" in case and len([ln for ln in text.splitlines() if ln.strip()]) < case["min_lines"]:
        problems.append(f"fewer than {case['min_lines']} lines")
    if "max_ratio" in case and len(text) > case["max_ratio"] * len(case["selection"]):
        problems.append(f"not shorter ({len(text)} vs {len(case['selection'])} chars)")
    return problems


def _norm_text(t: str) -> str:
    t = t.strip().replace("\r\n", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = "\n".join(line.strip() for line in t.split("\n"))  # trailing spaces are not a difference
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.rstrip(".").strip()


def cleanup(cfg: Config, model_key: str | None = None, rules_only: bool = False, show: bool = True,
            device: str = "auto") -> dict:
    """Run the clean-up quality set: rules-only vs rules+LLM, exact matches, must-not violations, latency.
    device="cpu" runs the bundled model on the CPU build (slow, but lets prompt work happen while the GPU is busy)."""
    from localflow.cleanup.pipeline import CleanupPipeline
    from localflow.config import PostProcessConfig
    from localflow.llm.providers import build_provider

    cases = [json.loads(line) for line in CLEANUP_CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    pp = PostProcessConfig(**{**cfg.postprocess.__dict__, "llm_cleanup": not rules_only})
    if model_key:
        pp.llm_model = model_key
    server = None
    provider = None
    if not rules_only:
        if pp.llm_provider == "bundled":
            from localflow.llm.server import LlamaServer

            server = LlamaServer(model_key=pp.llm_model, device=device)
            t0 = time.perf_counter()
            server.start()
            print(f"llama-server ({pp.llm_model}, {device}) ready in {time.perf_counter() - t0:.1f}s")
        provider = build_provider(pp, server_factory=lambda: server)
    exact = 0
    violations = 0
    used = 0
    rejected = 0
    lat: list[float] = []
    edits: list[float] = []
    command_pass = 0
    command_ms: list[float] = []
    command_cases = [json.loads(line) for line in COMMAND_CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        for case in cases:
            pipe = CleanupPipeline(PostProcessConfig(**{**pp.__dict__, "dictionary_terms": case.get("dictionary", [])}), provider)
            if provider:
                pipe.prefill(case["raw"], case.get("app"), case.get("title"))
            res = pipe.process(case["raw"], case.get("app"), case.get("title"))
            got, want = _norm_text(res.text), _norm_text(case["expect"])
            ok = got == want
            bad = [s for s in case.get("must_not", []) if s in res.text]
            exact += ok
            violations += bool(bad)
            used += res.used_llm
            rejected += bool(res.llm_rejected)
            if res.llm_ms is not None:
                lat.append(res.llm_ms)
            edits.append(edit_distance(normalize(want), normalize(got)) / max(len(normalize(want)), 1))
            if show:
                flag = "ok  " if ok and not bad else ("VIOL" if bad else "diff")
                extra = f" [{'llm' if res.used_llm else 'rules'}{', rejected: ' + res.llm_rejected if res.llm_rejected else ''}]"
                print(f"  {flag} {case['id']:<14}{extra}")
                if not ok or bad:
                    print(f"       want: {want!r}\n       got:  {got!r}" + (f"\n       must not contain: {bad}" if bad else ""))
        if provider:
            from localflow.cleanup.command import CommandRunner

            if show:
                print("  command mode:")
            for case in command_cases:
                res = CommandRunner(provider).run(case["selection"], case["instruction"])
                problems = check_command(case, res)
                command_pass += not problems
                if res.ms is not None:
                    command_ms.append(res.ms)
                if show:
                    print(f"  {'ok  ' if not problems else 'FAIL'} {case['id']:<14}"
                          + (f" {'; '.join(problems)}\n       got:  {res.text!r}" if problems else ""))
    finally:
        if server:
            server.stop()
    n = len(cases)
    summary = {
        "cases": n, "exact": exact, "exact_pct": round(100 * exact / n, 1), "violations": violations,
        "word_error_pct": round(100 * sum(edits) / n, 1), "llm_used": used, "llm_rejected": rejected,
        "llm_ms_p50": round(statistics.median(lat)) if lat else None,
        "llm_ms_p95": round(float(np.percentile(lat, 95))) if lat else None,
        "model": "rules" if rules_only else f"{pp.llm_provider}:{pp.llm_model}",
        "command_cases": len(command_cases) if provider else 0,
        "command_pass": command_pass,
        "command_ms_p50": round(statistics.median(command_ms)) if command_ms else None,
    }
    print(f"\ncleanup set ({summary['model']}): {exact}/{n} exact ({summary['exact_pct']} %), {violations} must-not violations, "
          f"mean word error {summary['word_error_pct']} %" + (f", LLM p50/p95 {summary['llm_ms_p50']}/{summary['llm_ms_p95']} ms, "
                                                              f"used {used}, rejected {rejected}" if lat else "")
          + (f"; command mode {command_pass}/{len(command_cases)} "
             f"(p50 {summary['command_ms_p50']} ms)" if provider else ""))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-cleanup.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def fetch_public(n: int = 60, offset: int = 0) -> int:
    """Download a deterministic slice of LibriSpeech test-clean via the Hugging Face rows API."""
    import urllib.request

    import soundfile as sf

    dest = DATA_DIR / "public"
    dest.mkdir(parents=True, exist_ok=True)
    got = 0
    while got < n:
        length = min(100, n - got)
        url = ("https://datasets-server.huggingface.co/rows?dataset=openslr/librispeech_asr"
               f"&config=clean&split=test&offset={offset + got}&length={length}")
        with urllib.request.urlopen(url, timeout=60) as r:
            rows = json.load(r)["rows"]
        if not rows:
            break
        for row in rows:
            row = row["row"]
            name = f"libri-{row['id'].replace('-', '_')}"
            wav, txt = dest / f"{name}.wav", dest / f"{name}.txt"
            if not wav.exists():
                src = row["audio"][0]["src"] if isinstance(row["audio"], list) else row["audio"]["src"]
                with urllib.request.urlopen(src, timeout=60) as a:
                    data = a.read()
                import io

                audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
                audio = audio.mean(axis=1)
                if sr != SAMPLE_RATE:
                    idx = np.arange(0, len(audio), sr / SAMPLE_RATE)
                    audio = np.interp(idx, np.arange(len(audio)), audio)
                sf.write(str(wav), audio.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
                txt.write_text(row["text"].strip(), encoding="utf-8")
            got += 1
            print(f"  {got:>3}/{n} {name}", flush=True)
    print(f"public set: {len(list(dest.glob('*.wav')))} files in {dest}")
    return 0


RECORD_PROMPTS = [
    "Hey, can you send me the updated budget spreadsheet by Tuesday?",
    "Let's schedule the design review for three thirty on Thursday afternoon.",
    "The API returns a 404 when the user ID is missing from the request.",
    "Please add Priya, Arnab, and Dr. Okonkwo to the calendar invite.",
    "I'll be in Bangalore from the 12th to the 19th of October.",
    "Remind me to renew the domain before it expires on March 1st.",
    "The quarterly revenue came in at 2.4 million, up 18 percent year over year.",
    "Can we move the standup to 9:15 so the London team can join?",
    "Merge the feature branch into main after the tests pass.",
    "Thanks for the quick turnaround, this looks great.",
    "Order two large pizzas, one margherita and one with mushrooms.",
    "The flight lands at 6:45 a.m. local time, so I'll take a cab straight to the office.",
    "Update the README with the new installation steps and the config options.",
    "She said the meeting was productive, but we still need a decision on pricing.",
    "The temperature in the server room hit 31 degrees again last night.",
    "Use snake case for Python variables and camel case in the TypeScript code.",
    "First, back up the database. Second, run the migration. Third, restart the service.",
    "I don't think we should ship on Friday; let's aim for Monday morning instead.",
    "Could you double check the invoice number? I think it's 4471, not 4417.",
    "My email is arnab at example dot com, and the phone number ends in 8823.",
    "The new laptop has 32 gigabytes of RAM and an RTX 4060 graphics card.",
    "We're using Parakeet for speech recognition and Qwen for the clean-up model.",
    "What time does the museum close on Sundays?",
    "Honestly, the second draft reads much better than the first one did.",
    "Push the fix to staging, then ping me on Slack when it's deployed.",
    "The recipe needs 250 grams of flour, two eggs, and a pinch of salt.",
    "Let's revisit the roadmap in two weeks once we have the benchmark numbers.",
    "Set the timeout to 30 seconds and retry three times before giving up.",
    "Happy birthday! Hope you have a wonderful day and a great year ahead.",
    "Turn off the lights, lock the door, and don't forget the keys on the table.",
]


MIN_TAKE_SECONDS = 1.0


def _record_take(dev, extra) -> np.ndarray:
    import sounddevice as sd

    chunks: list[np.ndarray] = []
    kwargs = dict(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=dev,
                  callback=lambda indata, *_: chunks.append(indata[:, 0].copy()))
    stream = sd.InputStream(extra_settings=extra, **kwargs) if extra else sd.InputStream(**kwargs)
    with stream:
        input("  recording... speak, then press Enter > ")
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def record_own(cfg: Config) -> int:
    """Read each prompt aloud in your normal dictation voice. The prompt text is the reference.

    Flow per prompt: Enter starts the take, Enter stops it. The take is transcribed on the spot
    so you can hear what the model heard and redo it if you misspoke. Takes under 1 s are
    rejected (a double Enter is the usual cause)."""
    import soundfile as sf

    from localflow.audio import ensure_com_initialized, resolve_input_device
    from localflow.stt import build_transcriber

    dest = DATA_DIR / "own"
    dest.mkdir(parents=True, exist_ok=True)
    ensure_com_initialized()
    dev, extra = resolve_input_device(cfg.audio.device)
    print("Loading the speech model so each take can be checked as you go...", flush=True)
    stt = build_transcriber(cfg.stt)
    stt.warmup()
    print(f"\nRecording with device {dev} at {SAMPLE_RATE} Hz into {dest}")
    print("Per prompt: Enter to start, read it aloud, Enter to stop. Then Enter to keep it, r to redo.")
    print("At a prompt: s = skip, q = quit. Ctrl+C also quits; finished takes are kept.\n")
    total = len(RECORD_PROMPTS)
    try:
        for i, prompt in enumerate(RECORD_PROMPTS, 1):
            wav, txt = dest / f"own-{i:02d}.wav", dest / f"own-{i:02d}.txt"
            status = " (done, Enter re-records)" if wav.exists() else ""
            cmd = input(f"[{i:02d}/{total}]{status}\n  \"{prompt}\"\n  Enter to record > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "s":
                print()
                continue
            while True:
                audio = _record_take(dev, extra)
                seconds = len(audio) / SAMPLE_RATE
                if seconds < MIN_TAKE_SECONDS:
                    print(f"  only {seconds:.1f}s, that was probably a double Enter. Again: press Enter, speak, Enter.")
                    input("  Enter to record > ")
                    continue
                heard = stt.transcribe(audio)
                ref, hyp = normalize(prompt), normalize(heard)
                errors = edit_distance(ref, hyp)
                verdict = "exact" if errors == 0 else f"{errors} word{'s' if errors > 1 else ''} off"
                print(f"  {seconds:.1f}s | heard: {heard}\n  {verdict}.", end=" ")
                again = input("Enter to keep, r to redo > ").strip().lower()
                if again == "r":
                    input("  Enter to record > ")
                    continue
                sf.write(str(wav), audio, SAMPLE_RATE, subtype="PCM_16")
                txt.write_text(prompt, encoding="utf-8")
                print(f"  saved {wav.name}\n")
                break
    except (KeyboardInterrupt, EOFError):
        print("\nstopped.")
    n = len(list(dest.glob("*.wav")))
    print(f"own set: {n}/{total} takes in {dest}")
    if n:
        print("Run the benchmark on them with:  localflow bench run --set own")
    return 0
