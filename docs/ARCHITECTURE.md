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

Only local programs get in (A9): the server listens on 127.0.0.1 only, refuses any request
with an Origin header (every browser sends one) or a Host other than 127.0.0.1/localhost (DNS
rebinding), takes at most 16 connections, and allows one 4 KB hello within 5 s until the token
checks out. After that, every field is checked for type and size (`protocol.checked`). A bad
message gets an `error` - or, for a command, a `command.result` that leaves the text alone -
and changes nothing. A bad audio frame is dropped and reported once per take. A client that
stops reading is dropped after 5000 unread replies. `engine/tests/test_connection.py` covers
all of this, plus a fuzz run.

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
| `app.exe --crash-test [s] [panic\|fault]` | Crashes on purpose after `s` seconds (over 60, or it will not restart): the log and the self-relaunch |

## Checking it

`app.exe --e2e [scenario...]` (`src-tauri/src/e2e.rs`) is the end-to-end harness: the whole app
- engine link, session machine, chord controller, audio pipeline and pre-roll, injector, flow
bar - with only the two ends scripted. The chord goes straight to the controller instead of
coming from the keyboard hook, and speech comes from WAV files (`scripts/make-e2e-audio.ps1`,
Windows' own voices; the WAVs are not committed) through a scripted microphone. Text is typed
into an edit control of the harness's own and read back from it; in a harness run the injector
refuses to type anywhere else. It joins the running LocalFlow's engine (never a second one on the
GPU), keeps its sentences out of the history and puts the clipboard back.

* 12 scenarios: one take, two quick takes in order, hands-free typed once, Escape on and off, a
  copied image surviving a dictation, a command's instruction never typed, a long selection
  edited whole or not at all, a 30-second take, text kept when another window took the front
  (then Win + Alt + V), a password box, the tray's Paste last dictation, and a silent microphone.
* By name only, since they touch the user's screen or engine: `admin-window` (a take into Task
  Manager, administrator without a UAC prompt, is copied, not typed; it leaves Task Manager
  open - LocalFlow cannot close it), `desktop-switch` (input moves to another desktop mid-take,
  as for a UAC prompt: the screen is empty for a moment), `gpu-lost` (the speech worker killed
  mid-take).

**Lock, sleep and the secure desktop** (`power.rs`). The keyboard hook sees nothing on the lock
screen, a UAC prompt's secure desktop or while the machine sleeps, so a hidden window watches
`WM_WTSSESSION_CHANGE`, `WM_POWERBROADCAST` and `EVENT_SYSTEM_DESKTOPSWITCH`. Input leaving ends
the take as a release would (its words are kept: delivery checks input is on this session's own
desktop) and clears the keys the hook thought held; after resume the microphone stream is rebuilt
and the hook reinstalled. **A graphics card failing under speech** (a driver reset, the speech
worker dying) makes `Engine.transcribe` rebuild speech on the processor and decode the same
audio again; the placement moves it back to the card at its next reading.
* `--e2e faults` (quit LocalFlow first: they break the engine they use): microphone unplugged
  mid-take, clean-up server frozen, engine killed mid-take, engine frozen.
* `--e2e soak [minutes]`: bursts of takes and rests long enough for the idle release, nothing
  typed (the machine stays usable); samples the shell, webviews, engine and clean-up server
  every minute into `target/soak.csv` and fits growth per hour against limits.

`localflow bench release` (`engine/src/localflow/perfrecord.py`) is the performance record per
release: start-up, key-up-to-text latency and WER on the own-voice set, peak memory, graphics
memory, and the clean-up and command sets, written to `docs/perf/v<version>.json` and compared
with the last record. It needs LocalFlow quit.

## When things go wrong

* **Shell** (`src-tauri/src/guard.rs`): every panic is logged with thread, place and backtrace;
  locks hand back their data when poisoned (`LockExt::locked`); long-lived workers restart; the
  keyboard hook and the injector lose one event or job rather than their thread; a native fault
  is logged by an unhandled-exception filter. A panic on the main thread or a native fault
  starts a new copy (`--restarted --after <pid>`) unless it came within a minute of start-up.
  Windows' own restart registration was tried and restarted nothing. Release builds unwind
  (`panic = "unwind"`, +4.8 MB).
* **Engine**: `sys.excepthook` and `threading.excepthook` log to `localflow.log`; `faulthandler`
  writes a hard crash's stack to `localflow-fault.log`, which the next engine moves into the log.
* **Crash loop**: three engine crashes in two minutes start the next engine in safe mode
  (processor only, default speech model, no clean-up), in memory only - saves write the user's
  own values back. Said once in a notification; "Leave safe mode" in the tray and the Hub.
* **A take in flight when the engine goes**: the shell keeps every take's audio until its text
  arrives and plays it to the next engine; after 90 s without one it is given up and the user
  told. A frozen engine is noticed after about 35 s (no answer to the heartbeat).
* **engine.json** names the engine clients join. An engine claims it only when no live engine is
  named there and removes it only when it names itself; the shell removes it for an engine it
  killed.
* **Hub**: an error boundary per page ("This page hit a problem" and Reload) and per window;
  faults, including unhandled ones outside rendering, go to `shell.log`.
* **Settings**: the shell refuses unusable or dangerous values with a reason (hotkeys that go
  off while typing or belong to Windows, out-of-range numbers); the engine tidies or refuses
  clean-up settings (`validate.py`) and reports what it refused. An unreadable settings file is
  set aside, never silently replaced by the defaults; saves are atomic; a byte-order mark is fine.
* **Microphone glitches**: an underrun or overrun (Windows' data-discontinuity flag) loses a few
  samples and the stream goes on; only a real device error rebuilds it. Treating a glitch as a
  dead device once put the microphone into a loop of half a second on, five seconds off.

### Status and the problem catalogue

`shared/problems.json` lists every problem LocalFlow detects (46 today): a readable code
(`mic-unavailable`, `cleanup-server-crashed`), its part and level (failed, degraded, info,
handled, internal), how it is detected, a title and message that say what happened and what to
do, and where there is one a fix (a button) and the flow bar's few words. The shell compiles it
in (`problems.rs`); the engine sends codes only (`localflow/problems.py`), and sorts a model that
will not load by cause - download failed, files damaged, out of memory, graphics card
unavailable, clean-up server missing or crashed, cloud key rejected or unreachable - instead of
passing the exception on. Tests on both sides check both ways: every code used has an entry and
every entry is used. The shell's codes are typed constants, so a problem without an entry cannot
be written.

The status model (`health.rs`) assesses every part - engine, speech model, clean-up, graphics
card, microphone, hotkey - as ok, starting, degraded or failed, with the catalogue's words. The
tray shows it, the Hub's Status card lists the parts that need a look (with the code, small), a
problem that lasts 10 s is announced once and recovery once, and the flow bar says why a press
did nothing or when a take lost its engine or microphone.

"Check LocalFlow" (E4) runs the shell's checks (`selfcheck.rs`: microphone privacy, whether the
microphone sends anything, the settings folder, WebView2) and asks the engine for its own
(`selfcheck.run`; `selfcheck.py`: driver, models folder and free space, clean-up server, model
files, download sites). Model files are verified offline: the Hugging Face cache keeps each
snapshot's file listing (`trees/<commit>.json`, with every LFS file's SHA-256), and a clean-up
model downloaded into a folder keeps its ETag (its SHA-256) in `.cache/huggingface/download`.
The engine's quick checks go out with its status; "Download again" (`selfcheck.repair`) deletes
the damaged files a full check found, and only inside the model caches.

## Models and where they run (v0.2)

Three pieces decide what runs and where; all live in the engine, and the Hub only shows and
changes their settings.

**What can be chosen.** `stt/catalogue.py` lists the speech models (Parakeet v3, v2, v3 Compact,
Whisper Large v3 Turbo). Each entry maps onto the existing `stt` settings (backend, model,
precision), names its exact files so "is it downloaded?" is answered from the Hugging Face cache
without the network, and carries speed and accuracy ratings measured with
`localflow bench run --speech <key>`. `llm/manifest.py` does the same for clean-up models; only
entries with `offered=True` (Qwen3 4B, Phi-4 mini, Gemma 4 E2B) appear in the Hub, the rest stay
usable by key for benchmarks. Downloads report byte progress through `hfprogress.py`, which
turns huggingface_hub's tqdm bars into a callback.

**What the machine has.** `hwinfo.py` is the capability report: every graphics adapter from
DXGI's adapter list (vendor, dedicated and shared memory, built in or discrete - judged by
vendor, size and name, since asking D3D12 would wake the card), the processor's name, physical
cores, threads and instruction sets (`IsProcessorFeaturePresent`; an old Windows answering no to
all reads as unknown), RAM, and free RAM and disk. `modelchoice.detect_hardware` carries it; the
compute status sends it to the Hub ("This PC") with free RAM and disk read fresh each time.

**Memory (B5).** Loading anything always leaves 0.75 GB of RAM free (`hwinfo.HEADROOM_GB`), from
measured footprints: speech on the processor 2.3 GB (Parakeet v3) or 0.8 GB (Compact), any
speech model on the card about 1.4 GB of RAM; clean-up the model file + 0.5 GB on the
processor, 1.4 GB on the card. The bundled clean-up server is not started without that room
(`Engine._room_for_cleanup` raises `problems.NotEnoughMemory` -> `cleanup-low-memory`, which
retries by itself); Automatic speech skips a model that would not fit; a PC under 9 GB of RAM
starts with clean-up off (`config.for_this_pc`, first run and reset only).

**Where (placement).** `gpu.py` reads temperature, load and memory from NVML (`nvml.dll`, via
ctypes; returns nothing on a machine without an NVIDIA card). `placement.py` is the pure policy:
a governor turns readings into one of four levels - full, gentle (no keep-warm), light
(clean-up on the processor), off (nothing on the GPU) - from a temperature limit (default 80 C;
light at the limit, off 6 C above it, gentle 8 C below), another app's GPU load while LocalFlow
is idle, battery, and idle time (10 minutes frees the VRAM: speech moves to the processor so
dictation stays instant, and the clean-up model is unloaded rather than moved - the next take
wakes it, launching its server before speech moves back so the two load side by side). It rises
after two readings and
falls one step per 90 s once 5 C under the threshold, so nothing ping-pongs; waking from idle is
immediate. Clean-up leaves the GPU first because it costs least there: Qwen3 4B is 381 ms on the
GPU and 767 ms on the processor at the same quality, where Parakeet is four times slower on the
processor and runs on every dictation. A card under 6 GB never holds both models, under 4 GB
neither. Parakeet Compact always runs on the processor (its int8 kernels fall back to the CPU
under CUDA anyway); Whisper Turbo stays on the GPU short of "off". Graphics memory other apps
have taken counts too (`VramGuard`): LocalFlow's own share is worked out from the models it has
on the card (Parakeet v3 2.9 GB, a clean-up model its file + 150 MB; Windows drivers report no
memory per process), everything else in use is someone else's, and with 0.5 GB spare there must
be room for both models (else light) or for speech (else off). It rises on two readings, or at
once before anything is loaded; it comes back with 768 MB to spare after 90 s, and pauses while
a model loads, whose memory is on the card before the engine reports it there.

**The graphics card's libraries (B2).** The installer carries no CUDA libraries: onnxruntime-gpu's
execution provider is there, but cuDNN, cuBLAS, cuFFT and the CUDA runtime (1.5 GB) are
downloaded by an NVIDIA PC on first run - `cudalibs.py`, pinned wheels from PyPI, their DLLs
kept in `%LOCALAPPDATA%\LocalFlow\bin\cuda\<tag>` and loaded with
`onnxruntime.preload_dlls(directory=...)`. `Engine.fetch_cuda_libs` starts the download after
speech has loaded (and when "Processor only" is turned off); until it is done
`ComputeController.speech_can_use_card` keeps speech on the processor on purpose, and when it is
done the CUDA probe is reset, any spare speech worker dropped, and placement moves speech to the
card. A virtualenv uses its own nvidia packages (`LOCALFLOW_CUDA_FROM_DOWNLOAD=1` ignores them).

**AMD and Intel graphics (B3).** Clean-up has a third place besides the NVIDIA card and the
processor: "vulkan", llama.cpp's Vulkan build on the first adapter that is not NVIDIA's
(`llm/server.py vulkan_device`; the others are hidden from the server with
GGML_VK_VISIBLE_DEVICES, and the list is asked for once per install and cached beside the
build). `ComputeController.cleanup_off_card` decides where clean-up goes when it is not on the
NVIDIA card: there, if the adapter has the model's memory + 512 MB (built-in graphics count what
they may borrow), "Processor only" is not set, it has not failed this session, and its estimate
beats the processor's - priors from the Radeon 890M (0.76 x the processor), then this machine's
measurements. Built-in graphics use RAM, so the memory check treats them like the processor.
Speech stays on CUDA or the processor (B4 is the research on the rest).

**Which (model choice).** `modelchoice.py`, when the user left it to LocalFlow (Automatic, the
default): the most accurate model that is quick enough on the device it will run on - speech
within 0.25 s per second of audio, clean-up within 1.5 s at the median. If nothing is quick
enough, accuracy still wins among models within 20 % of the quickest. "Quick enough" is judged
from this machine: `service/engine.py` times every real decode (a `TimedTranscriber` wrapper)
and every bundled clean-up, per model and device, into `%APPDATA%\LocalFlow\perf.json`; after
five samples the median replaces the estimate. Until then: another model measured on this
machine and device, scaled by how their priors compare; failing that, the development
machine's numbers - for speech on the processor re-measured 2026-09-29 and scaled by the
measured core curve (`cpu_scale`: little slower down to 4 cores, then steeply), for clean-up
scaled by physical cores. Timings under 1 ms are impossible and dropped. It never trades accuracy for a cooler GPU - that is
placement's job - and only switches between models already on disk. Memory is the one hard
limit: a speech model that would not fit in free RAM is skipped (a loaded one fits by
definition), and the reason says so. Choosing a model in the Hub
pins it (`compute.auto_speech` / `auto_cleanup` turn off).

**Carrying it out.** `service/compute.py` polls every 5 s, asks both policies, and compares
the answer with where the models *actually* are (`Engine.devices()`), not where they were meant
to be: a GPU that silently fell back to the processor is recognised once rather than rebuilt
every tick, and a failed move waits two minutes. A move builds the model on its new device and
swaps it in on the worker, so dictation never stops; speech and clean-up move in parallel.
The exception is changing clean-up model while staying on the GPU: two language models and
speech do not fit in 8 GB, so the old server stops first (a few seconds of rules-only clean-up).
New clean-up servers are warmed with the instructions before they take over. Model changes the
user asked for hold the same lock, so the two never interleave. Keep-warm (continuous live
decoding, which holds the GPU at full clocks while the key is down) is now only on at the
"full" level.

Measured on the RTX 4060 laptop with real models (temperature scripted): VRAM 6.8 GB with
everything on the GPU, 4.1 GB with clean-up on the processor, and none of LocalFlow's with
nothing on it.

**Speech on the GPU runs in a process of its own** (`stt/remote.py`, `localflow speech-worker`).
A process that has used CUDA keeps ~1.4 GB of memory (0.8 GB resident) and 0.1-0.35 GB of VRAM
until it exits - the CUDA context, cuDNN and cuBLAS state - however the model is unloaded. So the
engine never touches CUDA itself: a speech model bound for the GPU is built in a worker, and an
idle release, which moves speech to the processor in the engine, ends the worker and all of that
with it. The two talk over the worker's stdin/stdout (length-prefixed pickles, one request at a
time, the worker's log records forwarded ahead of each reply; its own stdout is redirected to
stderr at the descriptor level so no library can write into the channel). The model's own
exceptions come back as themselves, so problem classification is unchanged; a worker that dies
raises `WorkerDied`. The decode round trip costs nothing measurable (31 s of audio: 2.27 s
through the worker, 2.35 s in process; the pipe transfer is 1.8 ms). `serve` starts the worker
first thing and has it load the saved model at once, so its imports, CUDA check and model set-up
overlap the engine's own start-up; the engine takes it if its settings match. The frozen engine
runs itself as the worker. `LOCALFLOW_SPEECH_IN_PROCESS=1` restores the old way (tests use it).

Start-up (2026-09-25): speech ready ~3.3 s after the engine starts (was 4-5 s). Two changes:
the engine says ready before the warm-up decode (a take started meanwhile waits for it inside
the key hold), and clean-up starts loading only once speech has (side by side on the GPU, speech
took 6.7 s instead of 2.5). What is left is onnxruntime setting up Parakeet's 2.3 GB encoder:
~2.2 s whatever the graph optimisation level, prepacking or CUDA module loading.

Memory (2026-09-25): after an idle release the engine holds 3.1 GB private / 2.5 GB resident -
essentially Parakeet on the processor, kept loaded so dictation stays instant - no speech worker,
no clean-up server, and no VRAM (was 4.5 GB and 0.35 GB of VRAM). On the processor the clean-up
server runs without weight repacking (`--no-repack`: 2.2 GB -> 0.5 GB private, and faster).
Whisper is unloaded after ten minutes unused. A long take's live decoding counts as LocalFlow's
own GPU use, so it no longer looks like another app and moves clean-up mid-take. Idle CPU with
the Hub hidden: 0.8 % of one core (the flow bar's waveform only animates during a take, and the
Hub refreshes only while visible).

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
| Hotkey | **`WH_KEYBOARD_LL`** hook in the Rust shell (`hotkey.rs`): gives key *release*, which `RegisterHotKey` and Electron's `globalShortcut` do not | pynput (the Python skeleton's choice, retired in 4.4) |
| Text injection | **SendInput with KEYEVENTF_UNICODE** for short text, **clipboard paste + restore** for long text | UI Automation `ValuePattern` (per-control, brittle), `WM_CHAR` (does not work across processes reliably) |
| Auto-edits | **Bundled llama-server** with a model chosen from `llm/manifest.py` (Qwen3 4B by default), on the GPU or the processor as placement decides | Ollama, any OpenAI-compatible endpoint, Anthropic (all still selectable as providers); rules only (the L1 layer) |
| Language | **Python 3.12** for the engine because the ASR ecosystem is Python-first | Rust/Tauri for the *shell* (phase 3), not for the model code |

## Roadmap

The plan and its status live in [ROADMAP.md](ROADMAP.md): phases 0-6.2 shipped as v0.1.0, and
the "Version 0.2" section there tracks what `main` adds.
