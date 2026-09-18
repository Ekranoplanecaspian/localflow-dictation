# Architecture

## The machine this is designed for

| Component | Spec | Role in LocalFlow |
|---|---|---|
| CPU | AMD Ryzen AI 9 HX 370, 12 cores / 24 threads | Speech-to-text (Parakeet fp32 via onnxruntime) |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB VRAM | LLM auto-edits via Ollama (phase 2); optional STT via CUDA |
| NPU | AMD XDNA 2 (~50 TOPS) | Not used. Tooling (Ryzen AI SW) is immature for ASR; revisit later |
| RAM | 32 GB | Plenty; the fp32 Parakeet model uses ~3 GB |
| Mic | Realtek microphone array | Captured through WASAPI (lowest latency host API on Windows) |

Design decision: **STT on the CPU, LLM on the GPU.** Parakeet 0.6B is small enough that the
12-core Zen 5 transcribes 12 s of speech in ~0.6 s, and a typical 4 s dictation in ~0.2 s.
That leaves the 8 GB of VRAM free for a 4B-parameter cleanup model, which is where a GPU
actually matters (a 4B model on CPU is too slow to feel instant; on the 4060 it is ~80 tok/s).
Moving STT to CUDA later is a config flag plus `pip install -e .[gpu]`.

## Pipeline

```
 hold Ctrl+Win ─┐
                ▼
   ┌─────────────────────┐    ┌──────────────────┐    ┌───────────────────┐
   │ hotkey.py           │    │ audio.py         │    │ context.py        │
   │ pynput low-level    │───▶│ always-on WASAPI │    │ foreground window │
   │ hook: down / up     │    │ stream + 500 ms  │    │ process + title   │
   └─────────────────────┘    │ pre-roll buffer  │    └─────────┬─────────┘
                              └────────┬─────────┘              │
   release ────────────────────────────┤                        │
                                       ▼                        │
                         ┌──────────────────────────┐           │
                         │ stt/parakeet.py          │           │
                         │ Parakeet TDT 0.6B v3     │           │
                         │ onnxruntime, fp32, CPU   │           │
                         │ (VAD split above 25 s)   │           │
                         └────────────┬─────────────┘           │
                                      ▼                         ▼
                         ┌──────────────────────────────────────────┐
                         │ postprocess.py                           │
                         │ L1 rules: fillers, "new line", dictionary,│
                         │           snippets                       │
                         │ L2 (opt): local LLM via Ollama           │
                         │           self-corrections, lists, tone  │
                         └────────────┬─────────────────────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ inject.py                │
                         │ SendInput (unicode) or   │
                         │ clipboard paste + restore│
                         └──────────────────────────┘
```

That pipeline is the shape of a dictation, but the left-hand half of it is no longer Python.
Phase 3 moved the hotkey, the microphone, the foreground-window context and text injection into
the Rust shell, and phase 4.4 deleted the Python originals (`app.py`, `ui.py`, `hotkey.py`,
`inject.py`, `context.py`, `autostart.py`, `sounds.py`) rather than leave a second, worse
dictation app in the tree for someone to start by accident. `pynput`, `pywin32`, `pystray` and
`pillow` went with them.

What Python still owns is everything to the right of the microphone: VAD, the speech models,
the clean-up pipeline and the benchmark harness. The state machine that joins the two halves is
`app/src-tauri/src/session.rs`, and the thread layout it replaced is described under "The shell"
below.

## The engine service (phase 1)

Since phase 1 the models live in their own process, `localflow serve`, and the shell is its
client. The protocol is documented in `engine/src/localflow/service/protocol.py`: a local
WebSocket with a per-launch token, JSON control messages, and 20 ms int16 audio frames
streamed while the hotkey is held.

What the GPU taught us (RTX 4060 Laptop, measured, see `docs/ROADMAP.md` for numbers):

* The GPU idles at 210 MHz and only reaches full clocks under *continuous* load. 5 s of
  audio decodes in 45 ms when it is busy, ~90 ms when merely warm, ~350 ms cold. Tiny
  kernels do not wake it; real inferences do.
* onnxruntime's CUDA path costs ~35 ms extra whenever the encoder's input length differs
  from the previous call. Differences under one 80 ms frame are free.
* Stitching per-phrase transcriptions reads badly because the model punctuates each phrase
  as a sentence. The final text must come from one pass over the whole take.

So while the key is held, the engine re-decodes the whole take back to back ("live"
decodes): that is the live preview text, it keeps the GPU clocked up, and because every
decode is zero-padded to a 1 s bucket, the final decode shares the previous shape. When the
audio that arrived after the last live decode is under 160 ms and silent by VAD, that live
decode simply becomes the final. Takes longer than ~22 s are split at VAD phrase boundaries.
On battery (or CPU) the live decode runs every 1.5 s instead of continuously.

fp16 was tried and shelved: the converted encoder that loads keeps its large Constant
tensors in fp32 and runs ~40 % slower through casts. fp32 with `kSameAsRequested` uses
~3.0 GB of dedicated VRAM.

## The AI layer (phase 2)

The engine runs two single-threaded workers, because neither model is thread-safe and both
share one GPU: `stt` does voice activity detection and every speech decode in order, `llm`
does clean-up and prompt pre-fill so a 250 ms language-model call never delays the next
dictation's live decoding.

Clean-up is four layers, cheapest first (`engine/src/localflow/cleanup/`):

1. **rules** - fillers, "new line"/"new paragraph", snippets. Microseconds.
2. **inverse text normalisation** - only the unambiguous spoken forms: digit runs
   ("four four seven one" -> 4471), spoken emails and URLs, percent, degrees. Anything
   contextual is left to the model.
3. **personal dictionary** - phonetic matching (Metaphone plus Jaro-Winkler and edit
   distance) so "Arnub", "Okonko" and "parasit" become the spellings you taught it, including
   two-word terms and spoken titles ("doctor" -> "Dr"). Common English words are never
   touched, so "sit" cannot become "Smith".
4. **the model** - Qwen3-4B-Instruct on a bundled `llama-server` the engine owns, prompted
   with the style profile of the app you are dictating into and your dictionary terms.
   Skipped entirely for short utterances with no correction or number cues.

Two things keep the model honest. A **guard** rejects output that is not an edit of what you
said (assistant-speak, wrapped text, an answer to a dictated question, a length that moved
too far) and falls back to the rule output. And **the transcript is the whole prompt** - the
model never sees the conversation, so it has nothing to answer.

`llama-server` is downloaded and pinned in `llm/manifest.py`, started on demand, and killed
with the engine. It loads in parallel with the speech model, so dictation works (rules only)
while it downloads. Providers are swappable: bundled, Ollama, any OpenAI-compatible endpoint,
or Anthropic with your own key.

Measured on the 30-case quality set (`localflow bench cleanup`):

| | exact | must-not violations | word error | latency p50 |
|---|---|---|---|---|
| rules only | 8/30 | 7 | 17.4 % | - |
| Qwen3-1.7B | 7/30 | 5 | 18.2 % | 99 ms |
| Qwen3-4B, first prompt | 17/30 | 1 | 8.8 % | 202 ms |
| Qwen3-4B, tuned | 22/30 | 0 | 3.3 % | 246 ms |

The 1.7B model is not merely worse, it is unusable here: it compresses instead of editing, so
the guard rejects most of its output. Model size buys instruction-following, not prose.

Two things the measurements changed:

* **Prompt pre-fill mid-utterance is off by default.** Warming the model's prompt while the
  user is still speaking saves ~40 ms of prompt evaluation, but it steals GPU time from the
  live speech decoding running at the same moment. That made live decodes lag further behind
  the end of the take, so the final decode could be reused far less often (8/30 instead of
  18/30) and cost ~150 ms more. Net loss. The setting stays for machines where speech and
  clean-up are not on the same device.
* **The guard has an anti-deletion check.** With no correction to apply, at least 85 % of the
  speaker's content words must survive the edit. It was added after the model quietly deleted
  "She said" from the front of a dictated sentence; a length ratio alone does not catch that.
  Number and ordinal words are exempt, because turning "the twelfth" into "the 12th" is the job.
* **Generation is capped at the length the guard would reject anyway.** An edit longer than
  1.6x the input is refused however good it looks, so tokens spent past that point can only be
  wasted - and they were: dictating a bare noun phrase made the model answer at 74 words instead
  of editing 7, and the user waited 1.4 s for output that was thrown away. The budget is now
  about two tokens per word of the ceiling, which halves that case (1262 ms -> 702 ms) and
  leaves well-behaved edits untouched (129 -> 132 ms), with the quality set unchanged at 22/30.
  Output that hits the cap is rejected on its own terms, because a truncated runaway's length
  ratio can land inside the guard's window by accident and a half sentence must never be typed.
* **llama-server joins a job object that kills it when the engine dies.** `terminate()` on the
  engine does not touch its children, so every killed engine left a model server behind holding
  ~2.7 GB of VRAM. Two of those filled the 8 GB card and turned a benchmark into 28-second
  dictations. Orphans from earlier runs are also reaped at startup, matched by path shape so a
  different app's llama-server is never touched.

End to end with everything on, streaming the 30-take voice set at real-time pace:
**key-up to final text 362 ms p50 / 522 ms p95** (speech decode ~50 ms, clean-up ~250 ms).

## The native shell (phase 3)

`app/` is a Tauri 2 application: Rust owns the operating system, the webview owns the pixels.
It replaces the Python tray app, and speaks the same session protocol, so both can drive the
same engine (`app/src-tauri/src/`):

| Module | Owns |
|---|---|
| `engine.rs` | Finding or spawning `localflow serve`, the handshake, the socket, health, restart |
| `hotkey.rs` | The low-level keyboard hook, the chord engine, push-to-talk and hands-free |
| `audio.rs` | WASAPI capture, the pre-roll ring, resampling to 16 kHz, the level meter |
| `session.rs` | The dictation state machine that joins the three together |
| `inject.rs` | Typing or pasting the text into whatever was focused |
| `context.rs` | The focused window, and UI Automation for selection, caret and browser URL |
| `tray.rs` | The tray icon, its states, and its menu |

Things that were learned the hard way, and are now enforced by the code:

* **The engine must not outlive the shell.** `terminate()` does not reach grandchildren, so a
  shell killed from Task Manager left an engine behind holding ~4 GB of VRAM (models plus its
  own `llama-server`). One was found running from an earlier session while phase 3 was being
  built. Both shells now put their children in a job object with
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (`localflow/jobobject.py`, `win.rs`), which the kernel
  empties however the parent dies.
* **Two engines do not fit.** Speech (~3.0 GB) plus the clean-up model (~2.7 GB) is most of an
  8 GB card, so the shell attaches to an engine that is already running rather than starting a
  second one, and only shuts down engines it started itself.
* **The microphone is opened at startup, not at the hotkey.** Opening a WASAPI stream takes tens
  of milliseconds, which is exactly where the first word lives. A 500 ms ring buffer is always
  filling, and is flushed to the engine when a take begins.
* **The hook callback must be trivial.** Windows silently removes a low-level hook whose callback
  is slow, with no notification. The callback here only updates a bitset and sends on a channel,
  and a watchdog compares `GetLastInputInfo` against the last event the hook saw so a hook that
  was removed anyway is reinstalled within three seconds.
* **Our own typing must not look like a keypress.** Injected events carry `LLKHF_INJECTED` and
  are ignored, or a dictation containing the chord's letters would retrigger itself.
* **Never call `SendInput` from inside the hook callback.** The Start-menu suppression did, and
  it re-enters the hook and takes long enough that Windows can drop the hook altogether. The
  symptom is the worst kind: the hotkey works once and then silently stops. It now runs on the
  controller thread, which is still before the Win key is released.
* **Typing is not reliable; pasting is.** Synthetic unicode key events arrive intact - a
  low-level hook recording our own injection saw the exact scan codes we sent - but modern XAML
  text controls mangle them beyond a dozen characters. `--type-probe` sent `aaaa bbbb cccc` to
  Windows 11 Notepad and read back `aaaa ccccccccc`: the right length, with each run of
  characters collapsed onto its last one, identically for every batching strategy and every
  pacing. The same text pasted lands perfectly. So the clipboard is the default and typing is
  kept for text short enough to survive (under twelve characters) and as the fallback when the
  clipboard cannot be taken.
* **A chord ending in Win would open the Start menu.** When the chord fires, an unassigned key is
  injected so the eventual Win release is part of a sequence rather than a lone tap.

The shell can be checked without a window, which is how each of those was found:

| Command | What it proves |
|---|---|
| `app.exe --selftest <wav>` | The whole engine link: spawn, handshake, session, partials, final |
| `app.exe --report` | Engine state, the focused window, and the strategy it would inject with |
| `app.exe --inject-test [--type\|--paste]` | Opens Notepad, injects, reads it back through the clipboard and UI Automation, and says whether it survived |
| `app.exe --type-probe <text>...` | Types each string and prints exactly what came back |
| `app.exe --hook-probe <text>` | Records our own injection through a low-level hook: says whether Windows delivered what we sent, or the target app mangled it |

## Living as a desktop app

* The application is the Tauri executable. It starts the engine itself (`-m localflow serve
  --handshake`) and holds it in a job object, so the engine and `llama-server` die with it.
* `win.rs` writes the shell's own path to
  `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` for start-at-sign-in; the Hub and the
  first-run wizard both toggle it.
* Two logs, side by side and in the same format: `%APPDATA%\LocalFlow\shell.log` from the shell
  and `localflow.log` from the engine.
* The flow bar carries `WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TRANSPARENT`: it cannot take
  keyboard focus, does not appear in the taskbar or Alt-Tab, and passes clicks through, so the
  caret stays where you are typing. The styles are read back at startup and logged, because a
  bar that can take focus would swallow the caret and that is not visible until it happens.

## Why these components

| Concern | Choice | Alternatives considered |
|---|---|---|
| STT model | **Parakeet TDT 0.6B v3** (NVIDIA). WER 6.3% vs Whisper large-v3 7.4%, ~10x faster, punctuates itself, 25 European languages | Whisper large-v3-turbo via faster-whisper (kept as optional backend for the other 70+ languages); Nemotron/Canary (bigger, slower) |
| STT runtime | **onnxruntime via `onnx-asr`**: pure pip, no torch, CPU today / CUDA tomorrow | NeMo (huge, torch), sherpa-onnx (fine, but the Python API is clunkier) |
| Hotkey | **pynput** low-level hook: gives key *release*, which `RegisterHotKey` and Electron's `globalShortcut` do not | `keyboard` lib (unmaintained), raw ctypes `SetWindowsHookEx` (do this when we go native) |
| Text injection | **SendInput with KEYEVENTF_UNICODE** for short text, **clipboard paste + restore** for long text | UI Automation `ValuePattern` (per-control, brittle), `WM_CHAR` (does not work across processes reliably) |
| Auto-edits | **Ollama** on the RTX 4060 (`qwen3:4b`), opt-in | Cloud API with your own key (add later as a second backend); rules only (already the L1 layer) |
| Language | **Python 3.12** for the engine because the ASR ecosystem is Python-first | Rust/Tauri for the *shell* (phase 3), not for the model code |

## Roadmap

1. **Skeleton (this repo, done)**: hold-to-talk, hands-free double tap, Parakeet on CPU, rule cleanup,
   injection, per-app context captured, config file, CLI, self-test.
2. **Wispr parity**: Ollama auto-edits on (self-corrections, lists, tone per app), per-app rules
   (e.g. auto-send in Slack, code-mode in VS Code/Cursor), personal dictionary UI, snippets,
   whisper-mode gain boost, usage stats, command mode (select text + speak an instruction).
3. **Proper Windows app**: a Tauri 2 shell (Rust + web UI) that spawns this engine as a sidecar
   over a local JSON-RPC socket. Tauri gives us the tray icon, an always-on-top transparent
   "flow bar" overlay with waveform, a settings window, start-at-login, auto-update, and an MSI
   installer, with the branding living in ordinary HTML/CSS. Alternative if we want one language:
   PySide6 + PyInstaller (simpler, heavier, uglier animations).
4. **Speed**: STT on CUDA (`[gpu]` extra), streaming partials ("type as you speak") using
   Parakeet's transducer decoder, and warm LLM KV-cache.
