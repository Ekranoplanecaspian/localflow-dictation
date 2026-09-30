# Roadmap

The full design document (architecture, protocol, budgets, decisions, risks) is the
"LocalFlow Build Plan" artifact; this file is the checklist version that lives with the code.

**Where things stand (2026-09-23):** v0.1.0 is released (phases 0-6.2) and is the copy installed
and used every day. `main` is **v0.2**, in development; see [Version 0.2](#version-02-in-development-on-main)
below for what is done, what is verified, and what is left before it can be released.

Target: a background app that starts at sign-in, shows a flow bar the moment you hold the
hotkey, and puts cleaned text at the caret in any app within ~300 ms, with the speech model
and the clean-up model on the GPU and nothing leaving the machine.

Two processes, one boundary: the **shell** (Tauri 2, Rust + WebView2) owns the OS and the
screen (hooks, mic, injection, UI Automation, tray, flow bar, hub); the **engine** (Python)
owns the models (VAD, Parakeet on CUDA, Whisper turbo, clean-up pipeline). A bundled
`llama-server` serves the clean-up model (Qwen3 4B by default) to the engine. They talk over a token-protected local
WebSocket carrying JSON events and 20 ms audio frames.

| Measure | Today | Target |
|---|---|---|
| Release-to-inserted, 5 s dictation, AI on | **362 ms p50 / 522 ms p95** (phase 2) | <= 400 ms p50 |
| Release-to-inserted, AI off | 52 ms p50 to final text (phase 1) + injection | <= 100 ms p50 |
| WER, English | 6.3 % (Parakeet v3 fp32) | no regression from fp16 |
| Idle | CPU ~0, RAM 2.6 GB; v0.2 frees the GPU after 10 idle minutes (VRAM 6.8 -> 1.35 GB) | CPU < 1 %, GPU 0 % |

## Phase 0: Foundations (done 2026-09-06)
- [x] 0.1 Monorepo layout (`engine/`, `app/`, `design/`, `docs/`), Rust + Tauri toolchains, CI on Windows
- [x] 0.2 Benchmark harness: `localflow bench fetch | record | run` (WER, latency p50/p95, RSS, VRAM)
- [x] 0.3 Design tokens (`design/tokens.json` -> `app/src/styles/tokens.css`)
- Done: `tauri dev` opens the LocalFlow window; CI green (engine pytest, vite build, cargo check);
  baseline on 60 LibriSpeech test-clean files, CPU fp32: WER 2.25 %, 45.2 ms per audio second,
  stt p50 249 ms / p95 590 ms, RSS 2.7 GB.
  Own-voice set (30 takes, 168 s, laptop mic): WER 8.75 % raw, of which ~20 of 33 word errors are
  formatting differences (3:30 vs "three thirty", 32GB, 2 eggs, March 1); real misses are proper
  nouns (Priya, Okonkwo, Parakeet, Qwen), a spoken email address, and digit-by-digit numbers.
  Those are exactly the dictionary, ITN and LLM targets of phase 2. Harness to-do: number-aware
  normalisation so digits and number words score as equal.

## Phase 1: Engine service, GPU, streaming (done 2026-09-06)
- [x] 1.1 Engine as a local WebSocket service (`localflow serve`, token, JSON + int16 frames); the tray app spawns it and is its client (`engine/src/localflow/service/`)
- [x] 1.2 GPU STT: onnxruntime-gpu 1.29 (CUDA 13 wheels), probe-based CUDA-then-CPU selection, `kSameAsRequested` arena (3.0 GB). fp16 shelved: the loadable conversion keeps big tensors fp32 and runs slower (see ARCHITECTURE.md)
- [x] 1.3 Live decoding while the key is held: continuous full-take re-decodes (keeps the laptop GPU at boost clocks, gives live text), 1 s length buckets so the final shares the previous shape, live result reused as the final when the tail is silent; VAD phrase boundaries only for chunking takes over ~22 s
- [x] 1.4 Whisper large-v3-turbo through onnx-asr (fp16 on CUDA), routed by `stt.language`, loaded on demand
- [x] 1.5 Conditioning kept conservative: lift only takes peaking under -30 dBFS (a -20 dBFS threshold cost 4 errors in 381 words on laptop-mic audio)
- Done: `localflow bench stream --set own` (30 takes at real-time pace, 1.5 s gaps): key-up to final
  **52 ms p50 / 86 ms p95**, live decode reused 18/30, WER 4.72 % (padding noise on hard words; 4.20 %
  offline padded, 3.15 % offline unpadded GPU, 4.21 % CPU). Offline GPU: 13 ms per audio second.
  Daily app runs on the service. Follow-ups: unload Whisper after idle (both models resident = 4.9 GB),
  a real NeMo fp16 export, and the number-aware scorer's handling of "two point four".

## Phase 2: AI layer (done 2026-09-07)
- [x] 2.1 Bundled `llama-server` (pinned in `llm/manifest.py`, CUDA and CPU builds, downloaded and
      managed by the engine) + Qwen3-4B-Instruct Q4_K_M; providers: bundled / Ollama / any
      OpenAI-compatible endpoint / Anthropic
- [x] 2.2 Clean-up v2 (`cleanup/`): rules -> inverse text normalisation -> phonetic dictionary ->
      model, with per-app style profiles and an output guard that rejects assistant-speak,
      wrapped text, answers to dictated questions and wild length changes
- [x] 2.3 Prompt pre-fill mid-utterance on a separate llm worker; short cue-free utterances skip
      the model entirely
- [x] 2.4 30-case quality set (`localflow bench cleanup`) with expected output and must-not strings
- Quality set: **22/30 exact, 0 must-not violations, 3.3 % word error, model p50 217 ms**
  (rules only: 8/30, 7 violations, 17.4 %. Qwen3-1.7B is unusable here: it compresses instead of
  editing, so the guard rejects most of its output.)
- End to end (`localflow bench stream --set own`, 30 dictations at real-time pace with auto-edits
  on): **key-up to final text 362 ms p50 / 522 ms p95**, live decode reused for 16/30.
- Config v2 migration drops the old Ollama-only settings so the AI layer turns itself on.
- Two measured decisions: prompt pre-fill is off by default (it steals GPU time from live
  decoding and cost more than it saved: 635 ms p50 with it, 362 ms without), and llama-server now
  joins a job object so it dies with the engine (orphans from killed engines had been holding
  2.7 GB of VRAM each, which wrecked a benchmark before it was found).

## Phase 3: Native shell (Tauri, Rust)
- [x] 3.1 Engine sidecar lifetime (`app/src-tauri/src/engine.rs`): attach to a running engine or
      spawn one, read the `{port, token, pid}` handshake from stdout, hello, heartbeat every 10 s,
      reconnect with backoff, and a job object so the engine and `llama-server` die with the shell
- [x] 3.2 `WH_KEYBOARD_LL` hook on its own thread (`hotkey.rs`): chord engine with left/right
      collapsing, push-to-talk, double-tap to latch hands-free, Escape to cancel, injected input
      ignored, Start-menu suppression, and a watchdog that reinstalls a hook Windows dropped
- [x] 3.3 Always-on capture (`audio.rs`): 500 ms pre-roll flushed at the start of a take, a 32-tap
      windowed-sinc resampler to 16 kHz mono, a 50 Hz level meter, and a supervisor that rebuilds
      the stream on a device change, a stream error, or three seconds of silence
- [x] 3.4 Injection (`inject.rs`): unicode `SendInput` or clipboard paste with restore, chosen per
      app; waits for the chord's modifiers to be released; Shift+Enter for newlines in chat apps
- [x] 3.5 UI Automation context (`context.rs`): process, title, selection, text before the caret,
      browser URL, all inside a 120 ms budget so a slow app never delays a dictation
- [x] 3.6 Tray (`tray.rs`): icon drawn per state, tooltip, open / restart engine / start at
      sign-in / quit, and a toast when the engine breaks or comes back
- Headless verification (`app.exe --selftest <wav>` and `--report`): engine spawned, handshake,
  session, live partials, **final text 432 ms after release**, engine and `llama-server` gone the
  moment the shell exits; context reported `brave.exe` with the page URL and picked `type`.
- Still to check by hand: the injection self-test in eight apps, and the hook under a ten-minute
  GPU stress test.

## Phase 4: Flow bar and Hub
- [x] 4.1 Flow bar (`app/src-tauri/src/flowbar.rs`, `app/src/FlowBar.tsx`): a transparent
      always-on-top window that cannot take focus (`WS_EX_NOACTIVATE`), stays out of Alt-Tab and
      the taskbar (`WS_EX_TOOLWINDOW`), passes clicks through, and is placed on the work area of
      the monitor holding the focused window. The waveform is a row of springs written straight
      to the DOM at 60 fps - the newest level enters at the centre and travels outward - with
      states for listening, transcribing and error, and the last words as a caption.
      Verified at startup by reading the window styles back and logging them.
- [x] 4.2 Hub (`app/src/hub/`): Overview (stats, 14-day chart, where you dictate, engine state),
      History (searchable, grouped by day, click to copy, shows what you actually said before
      clean-up), Dictionary (phonetic terms, exact replacements, snippets), Voice (chord captured
      by pressing it, microphone, start at sign-in, history retention) and Models (auto-edits on
      or off, provider, house style). Shell settings live in `shell.json`; clean-up settings are
      sent to the running engine over the protocol, which applies and saves them at once.
- [x] 4.3 Onboarding wizard (`app/src/onboarding/`): five steps on first run - welcome,
      microphone (live level), hotkey (captured by pressing it), a real test dictation, done.
      `#onboarding` forces it for review.
- [x] 4.4 Retire the Python UI (keep the engine CLI): `app.py`, `ui.py`, `hotkey.py`,
      `inject.py`, `context.py`, `autostart.py` and `sounds.py` deleted, along with the `run`,
      `autostart`, `self-test` and `overlay-demo` commands and the `localflow-bg` entry point.
      `pynput`, `pywin32`, `pystray` and `pillow` dropped from the engine's dependencies. The
      CLI is now the engine only: `serve`, `send-wav`, `devices`, `transcribe`, `config`,
      `bench`. A bare `localflow` prints help rather than starting a tray app.

## Phase 5: Wispr parity
- [x] 5.1 Command mode (`cleanup/command.py`, Win+Alt by default): the selection and a spoken
      instruction go to the model, and the answer replaces the selection - but only if it
      survives a guard built for edits rather than clean-up (length and word retention say
      nothing here, since "summarise this" is supposed to drop most of the words). Anything
      that looks like the model leaving its role - refusing, explaining, wrapping in quotes,
      trailing commentary, echoing the instruction, being truncated - leaves the text alone.
      The selection is read by UI Automation, or copied after the chord is released when UIA
      cannot see it. The two chords may not nest, and a nested pair turns command mode off.
- [x] 5.2 Smart spacing and capitalisation (`cleanup/joining.py`): the shell had been sending
      the text before the caret since phase 3 and the engine threw it away, so every take was
      written as a whole sentence and dropped mid-clause with a capital and no space. A leading
      capital is now removed only when the first word is a closed-class function word, because
      lower-casing a name corrupts what was said while an unwanted capital is merely visible.
      The preceding text is also given to the clean-up model for spelling and casing - which
      made it echo that text back into its answer, so `strip_echo` undoes that.
- [x] 5.3 Per-app rules (Hub -> Apps): turn dictation off in an app entirely, force the
      writing style instead of letting the engine guess it from the executable name, force
      typing or pasting, or press Enter after inserting so a dictated message sends itself.
      Rules are keyed on the executable name, every field defaults to "leave it alone", and a
      rule that changes nothing is dropped on save. A disabled app says so on the flow bar,
      because a hotkey that quietly does nothing looks exactly like a broken one.
- [x] 5.4 Hands-free auto-stop (`session::watch_hands_free`): a latched take has no finger to
      end it, so it stops itself after a configurable silence (8s by default). Snippets take
      `{date}`, `{time}`, `{day}`, `{month}`, `{year}` and `{now}`, each with an optional
      strftime format, and an unknown `{...}` is left alone because it is far more likely to be
      part of the user's own template. Dictionary learning (`learn.rs`) reads the history's raw
      and cleaned text, aligns them with a longest-common-subsequence walk, and offers the
      corrections that keep repeating - it only ever suggests, since a wrong entry rewrites a
      word in every future dictation.

## Phase 6: Packaging
- [x] 6.1 PyInstaller engine bundle (`engine/localflow-engine.spec`): a one-directory freeze,
      1.8 GB, carrying onnxruntime's native libraries and the CUDA wheels. Shipped as a Tauri
      resource and found by `engine_command`, so an installed build runs with no virtualenv and
      no Python on the machine. Verified: the shell spawns `localflow-engine.exe`, no
      `python.exe` anywhere, Parakeet on CUDA in 2.4s, engine ready in 6.4s. Resumable
      downloads with checksums were already in place from phase 2 (`llm/downloader.py`).
- [x] 6.2 NSIS per-user installer (`app/src-tauri/nsis/hooks.nsh`), single instance, clean
      uninstall. Verified by installing and uninstalling for real: installs to
      `%LOCALAPPDATA%\LocalFlow` with no administrator prompt (`asInvoker`), registers under
      HKCU and nothing under HKLM, and uninstalling removes every program file, the Run key and
      the Start menu entry while leaving 6.85 GB of models, 1.22 GB of llama-server and the
      settings and history untouched. The single-instance mutex was missing entirely - the
      Python app had one and phase 4.4 deleted it - so two copies meant two keyboard hooks and
      every dictation injected twice; three launches now leave exactly one process. The shell
      also repairs a sign-in entry that points somewhere stale, which this machine had left
      over from the retired Python app.
- [ ] 6.3 Updater + GitHub Actions release pipeline. **Partly done:** the public release
      channel exists - https://github.com/Ekranoplanecaspian/localflow-dictation, kept separate
      from this private development repo and given fresh snapshots rather than its history -
      and v0.1.0 shipped there by hand on 2026-09-18 with the installer attached (SHA-256
      7bf3a755...fb5df, verified against GitHub's own digest after upload). Not open source:
      all rights reserved, installer free to run for personal use. Still to build: the in-app
      updater and a pipeline that produces releases instead of a person doing it.
- [ ] 6.4 Diagnostics export

## Phase 7: Polish and brand
- [ ] 7.1 Name, logo, icons, type, colour, motion, sounds via the tokens file
- [ ] 7.2 Performance pass (idle CPU < 1 %, startup < 3 s, 60 fps bar)
- [ ] 7.3 Accessibility, high contrast, mixed-DPI monitors, remote desktop
- [ ] 7.4 Beta checklist and docs

## Version 0.2 (in development on `main`)

v0.1 is installed from its release (`%LOCALAPPDATA%\LocalFlow`) and is the copy in daily use;
`main` carries version 0.2.0 and `release/0.1` (from tag `v0.1.0`) takes v0.1.x fixes. A
development build shares settings, history and models with the installed copy, and only one
copy can run at a time - quit the installed one first (README, "Working on LocalFlow").

Done, committed and pushed (2026-09-22/23):

- [x] 0.2.1 **Speech model choice** (`engine/src/localflow/stt/catalogue.py`, Hub -> Models):
      Parakeet v3 (default), Parakeet v2 (English only), Parakeet v3 Compact (int8), Whisper
      Large v3 Turbo. Downloads with live progress while the current model keeps working; saved
      only once the new one has loaded. Ratings measured with `localflow bench run --speech`:
      v3 3.7 % / v2 5.5 % word error on the developer's own voice. Whisper Small tried and
      dropped (took English for Hindi and looped).
- [x] 0.2.2 **Clean-up model choice** (`llm/manifest.py`, `offered=True`): Qwen3 4B (default),
      Phi-4 mini, Gemma 4 E2B, from nine measured with `localflow bench cleanup`, which now also
      runs a 12-case command-mode set (`engine/bench/command_cases.jsonl`). The rest of the nine
      stay in the manifest, usable by key but not offered (numbers in the manifest's comment).
- [x] 0.2.3 **Graphics card or processor, by heat and load** (`placement.py`, `gpu.py`,
      `service/compute.py`; Hub -> Models -> Graphics card). NVML readings every 5 s; four
      levels - everything on the GPU; no keep-warm; clean-up on the processor; nothing on the
      GPU - driven by a temperature limit (default 80 C), another app's GPU load, battery, and
      10 idle minutes (which frees the VRAM; the next dictation wakes it at once). Hysteresis:
      up after two readings, down one step per 90 s when properly cool. Moves build the model
      on its new device before swapping, so dictation never stops. Modes: Automatic, Graphics
      card, Processor only.
- [x] 0.2.4 **Automatic model choice** (`modelchoice.py`): the default in both pickers. Quality
      first - the most accurate model quick enough on the device it runs on (speech within
      0.25 s per second of audio, clean-up within 1.5 s); steps down only when the device is too
      slow. Real dictations are timed per model and device (`%APPDATA%\LocalFlow\perf.json`)
      and after five samples the measurement outranks the estimate. Only switches between models
      already on disk (Compact excepted). Picking a model pins it; config v3 keeps existing
      non-default choices pinned. On the development machine it picks Parakeet v3 and Qwen3 4B
      everywhere: both are quick enough even on the processor.
- [x] 0.2.5 Fixes found while testing those in the real app: a second launch now asks the
      running copy to show its window (named event `Local\LocalFlow.ShowWindow`); a stale
      `engine.json` naming a pid Windows had reused left the shell on "Starting..." for good
      (pid is now checked to be an engine, a refused connection drops the file); new clean-up
      servers are warmed before taking over (first CPU clean-up 3 s -> 0.8 s); the frozen
      engine collects every `localflow` module; Hub sections open at the top.

Verified: 207 engine tests and 37 shell tests pass. With real models, a real llama-server and
real VRAM readings (only the temperature, and once the core count, simulated): a full
heat / cool / idle / wake cycle with 35 dictations running and none failing; automatic model
switches on a simulated 6-core processor; the pickers and all three modes clicked through in
the real app.

Verified since (A11, 2026-09-24): real heat and the frozen engine - see A11 below. Still not
verified: a real game as "another app using the GPU" (skipped for now), and the installer itself
(0.2.6).

Left before v0.2 can be released:

- [x] 0.2.6 Build the frozen engine and the installer from `main`, install it over v0.1, and
      check dictation, the pickers and the Graphics card section in the installed copy
      (README, "Packaging"). Then publish to the public repo as for v0.1 (fresh snapshot,
      noreply author, all rights reserved). Done 2026-09-30 as D11 below.
- [x] 0.2.7 Backport the stale-`engine.json` fix to `release/0.1`? Decided 2026-09-24: no, v0.2
      replaces v0.1.
- [x] 0.2.8 Delete the clean-up models downloaded for evaluation and not offered (~11.7 GB in
      `%LOCALAPPDATA%\LocalFlow\models\llm`)? Decided 2026-09-24: keep them, for comparisons.
- [x] 0.2.9 Code review of the whole codebase (v0.1 included): 10 findings and 6 smaller items,
      all fixed and committed one per fix (2026-09-24, `bfc805e`..`657d8ff`). The six hand checks
      (hands-free stop types once; two quick takes both land, in order; a command followed at
      once by a dictation is not typed; a long selection is edited whole or left alone; a copied
      screenshot survives a dictation; Escape-cancels off takes effect) are now scenarios of the
      A1 harness, and all pass (2026-09-23).

Since 2026-09-24 the development machine runs the v0.2 development build every day instead of
the installed v0.1 (sign-in, Start menu and desktop point at `app\src-tauri\target\release\app.exe`;
the engine runs from the repository). Build it with `npm run tauri build -- --no-bundle` - a plain
`cargo build --release` leaves the UI pointed at the Vite dev server.

### Road to release (planned 2026-09-24; under way)

Order: A with E, then B, C, D. Each item: scope agreed, built with tests, verified for real, shown, then
committed on its own. GPU benchmarks one at a time, below 60 C.

**Progress (2026-09-24):** E3 and A1-A10 done, committed locally, not pushed.
**Next:** group B (hardware), under way since 2026-09-28 in the order B1, B5, B3, B2, B4, B6; then
C (design, waits for the brief) and D (ship). Groups A and E are done: E1, E2, E4 and E5
2026-09-24; A4, E5's live checks and E6 2026-09-25; E7 2026-09-28. B1, B5 and B3 done 2026-09-28, B2, B4 and B6 2026-09-29: group B is done.
**v0.2.0 was released on 2026-09-30** (D1 and D11, with a start on D7); group C, the design,
moves to 0.3 together with the rest of D. Nothing is known broken.

How the checks are run (details in ARCHITECTURE.md, "Checking it"): `app.exe --e2e` (12 scenarios),
`app.exe --e2e faults` and `localflow bench release` with LocalFlow quit, `app.exe --e2e soak`.
Run them from your own terminal. From inside the Claude desktop app, %APPDATA% is sandboxed: the
harness cannot see the running engine and starts a second one, and settings changes land in a
private copy. There, run them through a one-off scheduled task (`/IT`) and **delete the task
right after `/Run`** - a task left behind fires by itself at its start time.

**A. Works flawlessly, uses little**
- [x] A1 End-to-end harness: real takes through the shell's session and injection code (quick
      re-presses, commands, long takes) - the hands-on checks, minus the physical keyboard.
      Done 2026-09-23: `app.exe --e2e [scenario...]` runs the real app with a scripted chord and
      recorded speech (`scripts/make-e2e-audio.ps1`) into a text box of its own; 8 scenarios, all
      passing, sharing the running engine and keeping out of the history. Run it from your own
      terminal - from inside the Claude app, %APPDATA% is sandboxed and it cannot find the engine
- [x] A2 Fault injection: engine crash mid-take, microphone unplugged, sleep/resume, GPU driver
      reset, full disk, hung clean-up server
      - [x] Harness faults (2026-09-23, `app.exe --e2e faults`, LocalFlow quit first): microphone
            dies mid-take (words before it kept), clean-up server hangs (typed without clean-up
            after its 8 s timeout), engine crashes mid-take (back in 2.6 s), engine freezes
            (replaced after 37 s). Found and fixed: a take in flight when the engine went away
            left the app stuck in "finishing"
      - [x] The take in flight when the engine dies is kept: its audio is replayed to the next
            engine and typed (both engine faults now require it; given up after 90 s)
      - Moved to E6 (decided 2026-09-23): sleep/resume, graphics driver reset, full disk -
        they cannot be triggered safely by a script, and E6 builds their handling
- [x] A3 Soak test: hours of synthetic dictation, no memory or handle growth. Done 2026-09-23:
      `app.exe --e2e soak [minutes]` - bursts of takes and rests long enough for the idle
      release, nothing typed so the machine stays usable, samples to `target/soak.csv`. Two
      hours, 90 takes, none failed; per hour: shell +2 MB and +4 handles, webviews +10 MB,
      engine flat - all well inside the limits
- [x] A4 Budgets: idle CPU < 1 %, idle RAM (2.6 GB today) released with the GPU, the 0.35 GB
      VRAM left after an idle release, Whisper unloaded when unused, start-up < 3 s. Done
      2026-09-25. Decided: idle RAM stays 3.1 GB (the full speech model, not the int8 one, so
      the first take after a rest is as good as any), and start-up 3.3 s is accepted (no
      dictating while it loads)
      - [x] Idle CPU (2026-09-23): 0.8 % of one core with the Hub hidden, was 11.4 % - the flow
            bar's waveform animated all day, and the Hub refreshed every 4 s while hidden
      - [x] Clean-up unloaded while idle (decided 2026-09-23), woken by the next take: the first
            take after a rest is typed 1.1 s after release, with clean-up. The processor build
            runs without weight repacking: 2.2 GB -> 0.5 GB, and faster
      - [x] Whisper unloaded after ten minutes unused
      - [x] A long take no longer moves clean-up to the processor mid-take (16 s -> 2.7 s)
      - [x] Idle engine RAM 4.5 -> 3.1 GB private (2.5 GB resident), and the ~0.35 GB of
            graphics memory left after an idle release -> none (2026-09-25). Both were CUDA
            state a process keeps until it exits, so speech on the GPU now runs in a speech
            worker process that the idle release ends. What is left is the speech model on the
            processor, kept loaded so dictation stays instant
      - [x] Start-up: speech ready ~3.3 s after the engine starts, was 4-5 s (2026-09-25): ready
            before the warm-up decode, clean-up loaded after speech rather than beside it, and the
            speech worker loading from the engine's first moment. Of the rest, ~2.2 s is
            onnxruntime setting up the 2.3 GB encoder, which no session option shortens
- [x] A5 Disk: delete download archives after extracting (~0.6 GB), old llama.cpp builds after
      an update. Done 2026-09-23: archives are deleted once unpacked, and at each start the
      engine removes leftover archives and other llama.cpp builds (one still in use is skipped
      until a later start)
- [x] A6 Performance record per release (latency, memory, accuracy) so regressions show.
      Done 2026-09-24: `localflow bench release` writes `docs/perf/v<version>.json` and flags
      regressions against the last record. v0.2.0 baseline: speech ready 3.3 s, key-up to text
      345 / 441 ms (p50 / p95), WER 5.5 % on the own set, engine 5.4 GB and clean-up 3.4 GB peak
      private memory, 5.9 GB graphics memory, clean-up 22/30 exact, commands 12/12
- [x] A7 Every Hub setting works end to end (found: "Show the flow bar" is saved, never read).
      Done 2026-09-24: every Hub setting traced to where it is used. Fixed: the flow bar switch
      (checked for real: no bar with it off); a cloud provider's new address or API key now
      applies at once; an unreadable shell settings file (a byte-order mark, half a file) is set
      aside instead of silently replaced by the defaults, saves are atomic, and both the shell
      and the engine read files saved with a byte-order mark
- [x] A8 Input validation in shell and engine: dictionary/snippets, reserved hotkeys (Win+L),
      provider URLs, retention, app rules. Done 2026-09-24: the shell refuses hotkeys that would
      go off while typing (a lone letter or modifier, Shift+letter), contain Escape, or belong to
      Windows or every app (Win+L, Alt+Tab, Ctrl+C...), with the reason shown in the Hub; ranges
      for retention, double tap and hands-free; app rules normalised. The engine tidies and
      checks the clean-up settings from the Hub and from its file (empty snippet triggers,
      instructions too long for the model, addresses, providers, pasted keys, numbers) and
      reports what it refused
- [x] A9 Local connection hardening: localhost only, token, size limits, refuse browser
      origins, malformed-message tests. Done 2026-09-24: the engine refuses web pages (Origin
      header) and other host names (DNS rebinding) before the WebSocket opens, and allows 16
      connections at most. Until the token checks out, a connection gets one 4 KB hello within
      5 s. Every field is checked for type and size. Found and fixed: a context of the wrong
      type left a take that never ended; a command with a non-text selection was never
      answered; a bad audio frame dropped the whole connection; a client that stopped reading
      grew the engine's memory without limit. The shell takes at most 16 MB per message from
      the engine, sends audio without Nagle delay, and refuses a command on a selection over
      100,000 characters. Checked for real: a web page in a browser cannot connect, and the
      live engine refuses a wrong token, a wrong host and an oversized hello. The 8 end-to-end
      scenarios pass
- [x] A10 Leftovers: template `react.svg`, placeholder brand values, dead code; demo mode kept
      out of release builds; a Content Security Policy for the Hub (added 2026-09-24).
      Done 2026-09-24: the template images and favicon are gone; the review routes
      (`#hub?demo=1`, `#flowbar?demo=...`, `#onboarding`, test faults) and their sample data exist
      in `npm run dev` only. Removed: four phase-3 developer commands the Hub never called (one
      typed any text into any window, one read any WAV path), the unused opener plugin that let
      the page open URLs, and the Python tray app's recorder and its hotkey/injection/overlay
      settings. The Hub runs under a strict policy: its own scripts and styles only, no network
      but the shell's IPC, no frames, forms or plugins. Brand values stay placeholders until the
      design brief (group C). Found on the way: one engine client's backlog of messages held
      every other client up (5000 status requests blocked the engine for 25 s); fixed. Checked
      for real: the Hub renders and updates live under the policy, the 8 end-to-end scenarios
      pass, and the review routes still work in development
- [ ] A11 The v0.2 "not verified yet" items: real sustained heat, a real game, frozen engine
      - [x] Frozen engine (2026-09-24): built from `main` with PyInstaller (1.8 GB); all 36 engine
            modules are in the bundle, the app ran on it (it started llama-server itself) and the
            8 end-to-end scenarios pass
      - [x] Real heat (2026-09-24, on the frozen engine): the GPU heated by LocalFlow's own
            back-to-back dictations (`app.exe --stress`), real sensor readings, the limit lowered
            for the test so the laptop never went past 65 C. Run 1 (limit 65 C): keep-warm off at
            63 C after 15 s, clean-up to the processor at 66 C after 35 s; the GPU then held
            ~65 C, so speech never had to move; 760 dictations, none failed. Run 2 (limit 60 C):
            clean-up to the processor at 60 C; load stopped a minute later; back on the graphics
            card 90 s after the GPU fell under 55 C, as designed; 140 dictations, none failed.
            Not reached: speech moving too (limit + 6 C), and keep-warm returning (needs the GPU
            under limit - 13 C, below this laptop's idle temperature at a lowered limit).
            Seen: moving clean-up to the processor warms the GPU a little (63-65 C at "light"):
            they share the laptop's cooling, as the known limit below says
      - [ ] A real game as "another app using the GPU" - skipped for now (decided 2026-09-24)

**B. Runs on the hardware people have** (Windows 10 and 11, x64; ARM64 after v0.2)
- [x] B1 Capability report: GPU vendor/VRAM, cores, instruction sets, RAM, free disk (2026-09-28).
      `hwinfo.py`: every graphics adapter through DXGI (vendor, own and shared memory, built in
      or not; no device is created, so a sleeping NVIDIA card stays asleep), the processor's
      name, cores, threads and instruction sets, RAM, free RAM and disk read fresh. Shown as
      "This PC" under the Hub's graphics card, printed by `localflow hardware [--json]`, in the
      diagnostics file and its summary line. Decided 2026-09-28 for the rest of B: CUDA libraries
      download on first run (NVIDIA only); clean machines are VirtualBox VMs; weak PCs target a
      4-core processor with no NVIDIA card, English-only allowed there
- [x] B2 Slim installer: CUDA libraries downloaded on first run, NVIDIA machines only
      (engine bundle 1.8 GB -> 295 MB, 2026-09-29; decided 2026-09-28: download on first run,
      NVIDIA only; offline NVIDIA PCs run speech on the processor until they can)
      - [x] `cudalibs.py`: the six nvidia-* wheels the engine is tested with, pinned by version,
            size and SHA-256 (PyPI, about 1.0 GB; cuRAND, never loaded, left out), downloaded
            through the proxy with resume, DLLs kept flat in bin\cuda\<tag>, loaded with
            `onnxruntime.preload_dlls(directory=...)`; `localflow cuda [--install]`. Checked: a
            Parakeet decode on the RTX 4060 loaded every CUDA DLL from that folder
      - [x] First run on an NVIDIA PC (recent enough driver, not "Processor only"): the download
            starts once speech has loaded; placement keeps speech on the processor on purpose
            meanwhile, then moves it to the card. The Status card's graphics card row says
            "Getting ready" with the progress; a failure is `gpu-libs-download-failed`, retried
            by itself for a connection, space or clock
      - [x] The spec leaves the nvidia packages out (and filters any a hook brings in). Checked
            on the frozen build: without the libraries it says so and keeps speech off the card;
            with them in place it runs speech on the card
- [x] B3 Clean-up on AMD/Intel graphics: llama.cpp Vulkan build (test on the Radeon 890M)
      (2026-09-28; decided that day: only when measured quicker than the processor, and on
      NVIDIA + built-in graphics PCs as the second place when the NVIDIA card is hot or full).
      Measured on the Radeon 890M, Qwen3 4B: clean-up 788 ms at the median against 1042 ms on
      the processor in the same session (prompt 353 vs 171 tokens/s, answer 21 vs 20; same
      quality, 22/30 and 12/12)
      - [x] The Vulkan build (b10819, 33 MB, SHA-256 pinned); `LlamaServer(device="vulkan")`
            uses the first adapter that is not NVIDIA's and hides the rest from the server
            (GGML_VK_VISIBLE_DEVICES), so the NVIDIA card is never touched; the adapter list is
            asked once per install; `localflow bench cleanup --device vulkan`
      - [x] "vulkan" as a third place for clean-up: off the NVIDIA card (or with none) it goes
            there when the adapter has room and the estimate - the 890M's, 0.76 x the processor,
            then this machine's measurements - beats the processor. "Processor only" means no
            graphics at all; a failed start leaves it on the processor for the session. Without
            an NVIDIA card placement says so and puts speech on the processor directly. The Hub
            says "Built-in graphics" or the card's name
- [x] B4 Speech on weak/non-NVIDIA machines (research): a 4-core CPU has no model within budget
      today; try Windows ML/DirectML, a smaller English model, int8 (2026-09-29). The premise
      was wrong: it came from estimates, not measurements. Streamed at real time through a real
      engine pinned to fewer cores (`scripts/weak_pc.py`: N cores, N speech threads, no CUDA,
      its own settings), the final text came this long after key-up (median / p95):

                             12 cores        4 cores         2 cores
        Parakeet v3          327 / 532 ms    462 / 830 ms    794 / 1463 ms   (WER 4.7 %)
        Parakeet v3 Compact  309 / 512 ms    340 / 552 ms    623 / 1072 ms   (WER 5.8 %)

      Per decode, 67 / 82 / 190 and 57 / 60 / 144 ms per second of audio: all inside the 250
      budget. The bug was Automatic's: priors 2.4x too slow, scaled straight by cores, so a
      4-core PC looked six times slower than measured and got Compact for good (a model never
      chosen is never measured). Fixed with the new priors, speech's measured core curve, and
      a model not yet measured judged by one that has been on the same device. Also found: the
      test suite had been writing 0 ms timings into perf.json (fixed; such timings are now
      ignored). Decided 2026-09-29: close here, no DirectML or smaller English model (neither
      is needed on a 4-core PC); an older CPU's slower cores are left to its own timings - if
      v3 is too slow there, Automatic steps down to Compact after the first five dictations.
      Caveat: fewer cores of a fast CPU, not slower ones; read 2 cores here as an older 4-core
      laptop. B6's virtual machines check a real first run
- [x] B5 Low memory: check RAM/VRAM before loading; clean-up off with a message on 8 GB PCs
      (2026-09-28, in three parts; decided that day: clean-up off by default on 8 GB but can be
      turned on, too little free RAM skips it and it retries by itself, Automatic speech takes
      Compact when the full model won't fit, a card other apps filled means the processor).
      Measured RAM: Parakeet v3 2.3 GB on the processor, Compact 0.8 GB, any speech model about
      1.4 GB on the card; clean-up the model file + 0.5 GB on the processor, 1.4 GB on the card.
      Measured VRAM: Parakeet v3 2.9 GB, Qwen3 4B 2.7 GB. Always 0.75 GB of RAM left free
      - [x] Clean-up waits for free memory: `cleanup-low-memory` (degraded, retries by itself)
            says how much it needs and how much is free; first run and reset start with
            clean-up off below 9 GB of RAM, and Models says why
      - [x] Automatic speech skips a model that won't fit in free RAM (Compact instead, with the
            reason under the picker); a loaded model fits by definition, a hand-picked one is
            never replaced, and it goes back to the full model once there is room
      - [x] A graphics card other apps have filled: LocalFlow's own share worked out from the
            models it has there (Windows drivers report no per-process memory), too little room
            for both -> clean-up to the processor, for speech -> both, "the graphics card is
            full: other apps are using X of its Y GB"; back with 768 MB to spare after 90 s
- [x] B6 Clean-machine installs in Sandbox/VM: Windows 10 and 11, no GPU, offline first run,
      proxy, low disk (2026-09-29). Decided that day: VirtualBox on this PC, Windows 11 only
      (Windows 10's support ended October 2025), and the online and offline first runs; proxy
      and low disk not tried on a VM (E6's unit tests cover them). The VM: Windows 11
      Enterprise evaluation 25H2, 4 cores, 8 GB, no GPU, no microphone (AC97 audio, which
      Windows 11 has no driver for), made and driven from the host by scripts in
      app\src-tauri\target (vm-create.ps1, vbox.sh, vm\lf-test.ps1; not in git)
      - [x] Online: the 163 MB installer installs in 34 s to 305 MB; the first run downloads
            Parakeet v3 (2.6 GB) and is ready in 5.5 min; "no NVIDIA graphics card", so no CUDA
            download; clean-up starts off (8 GB, B5); the 12 standard e2e scenarios pass there
      - [x] Offline: speech-download-failed, retried by itself after 15 / 30 / 60 / 120 s;
            plugged back in, it downloaded and was ready 2.5 min later with nobody touching it
      - [x] Found and fixed: a PC with no microphone showed "Opening" for good, and the app
            "Starting up" (the capture thread's one announcement came before the status model
            listened; it is now replayed); the offline message quoted "[Errno 11001]
            getaddrinfo failed" (now "This PC could not reach the internet"); both checked on the VM
            with the rebuilt installer, offline: "Microphone: Unavailable" with its fix, and
            the plain message
      - Found, for later: the welcome says the models run "on your own GPU" (D7, honest
        claims); on the slow VM a later start took 35 s against 3.7 s here; huggingface_hub
        fills the log with HF_TOKEN and hf_xet warnings

**C. Consistent design and brand** (waits for the design brief; name stays LocalFlow for now,
and a later rename changes the display name only, never the folders)
- [ ] C1 `docs/DESIGN.md`: colour, type, spacing, motion, voice, sound, accessibility rules
- [ ] C2 Tokens for everything: Rust constants for the tray icon; last hard-coded colours
- [ ] C3 Every surface: flow bar states, Hub pages, onboarding, tray, notifications, error and
      empty states with a next step, success/error feedback after every action
- [ ] C4 Identity: icon set, installer look, window titles, exe/installer metadata, README,
      release page with screenshots, demo GIF, one clear Download button
- [ ] C5 Ship the fonts (Bricolage Grotesque is referenced, not bundled; OFL)
- [ ] C6 Motion and sound, respecting reduced motion
- [ ] C7 Accessibility: contrast, keyboard navigation with visible focus, screen-reader labels,
      high contrast, 100-200 % scaling, mixed DPI, small windows without overflow
- [ ] C8 A language setting in the Hub (Voice): today a language outside Parakeet's 25 can only be
      chosen by editing `stt.language` in config.json (found in E7, added 2026-09-28). Automatic
      by default; a named language goes to Whisper, with its download size said first; perhaps
      per app too, beside the other app rules

**D. Ready to ship** (unsigned for now; no analytics of any kind)
- [x] D1 No dictated text in logs (today `shell.log` holds every final); clear what is there.
      Done 2026-09-30 for v0.2.0: shell.log writes finals, commands and window titles as their
      length (the diagnostics redaction, now applied as each line is written;
      `LOCALFLOW_LOG_WORDS=1` keeps the words for development). The engine's log had none.
      Lines written before 0.2.0 are left in place until the log rolls at 2 MB
- [ ] D2 API keys in Windows Credential Manager, never sent to the Hub, out of logs and exports
- [ ] D3 Pinned checksums for every llama.cpp download, pinned model revisions, signed updates,
      warning on a non-local `http://` provider
- [ ] D4 Onboarding consent: explains the half-second microphone buffer, asks about history
      (nothing kept until yes) and start at sign-in
- [ ] D5 Delete everything (history, logs, settings, dictionary, timings); uninstaller offers to
      remove the models
- [ ] D6 Privacy statement and licence (drafted from what the app does), third-party notices
      (Parakeet CC-BY attribution, Gemma terms, fonts, CUDA, PyInstaller), About page with
      version, publisher, support link, current copyright year
- [ ] D7 Honest claims: numbers say what they were measured on; "nothing leaves your machine"
      lists its exceptions; Wispr named only in factual comparisons. Started 2026-09-30: the
      welcome, the installer's description and the README say "your own PC", not "your own GPU"
- [ ] D8 Costs up front: cloud providers bill your account; download sizes shown first
- [ ] D9 Dependency audit: licences, `cargo audit`, `pip-audit`, `npm audit`,
      `HF_HUB_DISABLE_TELEMETRY`
- [ ] D10 Updater (6.3), release pipeline with a link check, diagnostics export (6.4). With the
      update rules from E6 (PRODUCTION.md, "Install, update, uninstall"): download quietly,
      install on the next quit and never while a take is recording, finishing or owed (nor
      during a model switch); a failed download, signature check or installer keeps the
      running version, is logged, tried again later, and said on the Status card - never quit
      without a new version to start; settings from a newer version are already safe (E6)
- [x] D11 0.2.6: build, install over v0.1, publish. Done 2026-09-30, ahead of group C (the
      design waits for 0.3), after D1 and the welcome's GPU claim. Frozen engine 294 MB,
      installer 171 MB (SHA-256 755d19d6...17e0b1). Installed silently over v0.1.0 in 15 s: 299 MB
      of program, "LocalFlow 0.2.0" in Add or remove programs, the installed copy ready on its own
      bundled engine 1.4 s after starting, the 12 standard e2e scenarios passing there, and its
      shell.log holding lengths only. Published to the public repo as v0.2.0 (Latest), a fresh
      snapshot as for v0.1.0. Found and fixed on the way: the installer's hook stopped every
      `llama-server.exe` and `app.exe` by name - LM Studio's server included; now only processes
      running from the install folder or %LOCALAPPDATA%\LocalFlow. Found, for later: Tauri's own
      "is it running" check still stops any `app.exe` of the current user by name - give the
      binary its own name (`mainBinaryName`) in 0.3, with the Run key and shortcuts moved over.
      Also for 0.3: installing over v0.1.0 leaves v0.1's `localflow-engine\_internal\nvidia`
      (1.5 GB of CUDA libraries) in place - NSIS overwrites files but removes none - and
      `cudalibs.bundled()` then finds and uses them instead of the pinned download. The
      installer should empty `localflow-engine\` before copying the new one
- [x] D12 CI green (GitHub Actions, both repos; red since mid-September). Fixed 2026-09-30: the
      engine tests set what they assumed of the development PC (CUDA libraries, 12 cores,
      onnxruntime-gpu, no first-run download for the module-wide engine); CI gives cargo check a
      stand-in engine folder, keeps Hugging Face downloads as plain files and fetches the speech
      model once before the tests; the elevation test holds on an administrator's runner. Fixed
      on the way: a finished first download showed at 100 % beside "ready"; closing a busy
      speech worker returned before it had ended. Found, for later:
      onnxruntime 1.30 refuses a model whose external data is a symlink out of its folder, which
      is how huggingface_hub stores downloads for an administrator or with Developer Mode on.
      v0.2.0 ships 1.29 and is fine; before moving to 1.30 the engine should download with
      `HF_HUB_DISABLE_SYMLINKS` (and repair an existing symlinked cache)

**E. Fails well** (details and the edge-case list: [PRODUCTION.md](PRODUCTION.md))
- [x] E1 Status model: each part ok/degraded/failed + reason + action, shown in flow bar, tray,
      Hub Status card and one notification per change. Done 2026-09-24 (`app/src-tauri/src/
      health.rs`): engine, speech model, clean-up, graphics card, microphone and hotkey, each ok /
      starting / degraded / failed with a plain reason and a fix. Hub: a Status card on Overview
      (one line when all is well, the parts that need a look otherwise) and the sidebar pill;
      tray: colour and tooltip, amber for "needs attention"; notifications once per problem that
      lasts 10 s, and once on recovery; work moved off the graphics card on purpose reads as fine,
      with a note (decisions 2026-09-24). Flow bar: a refused press says why (engine restarting
      or not running, speech loading or failed, no microphone, dictation off in this app) - the
      engine-down case used to show nothing at all; mid-take, a tag when the engine drops ("your
      words are kept") or the microphone goes. Checked in the real app: an engine killed is back
      in 1.1 s with no notification; a missing chosen microphone turns the card amber, is announced
      after 17 s and "working normally again" 7 s after it is fixed; the fault scenarios and the
      8 end-to-end scenarios pass. Disk, network and updates join with E4, E6 and D10
- [x] E2 Problem catalogue: a code, detection, plain message and recovery for every failure.
      Done 2026-09-24: `shared/problems.json`, 46 problems, each with a readable code, level,
      detection, title, message and (where there is one) a fix and the flow bar's words; the
      shell compiles it in, the engine sends codes only, and tests on both sides check both ways
      (every code used has an entry, every entry is used). The engine now sorts a model that
      will not load by cause (download, damaged files, memory, graphics card, clean-up server,
      cloud key or address) instead of passing the exception on; codes show small on the Status
      card. Found on the way: every microphone glitch (a buffer underrun or overrun) was treated
      as a dead device, which on this machine turned into half a second of microphone, five
      seconds of none, over and over - glitches are now ignored and logged, and the stream
      stayed up. Decided 2026-09-24: one shared JSON file; codes are words, shown in the Hub's
      details and logs only; common engine causes classified
- [x] E3 Never crash: panic hook to the log, poison-tolerant locks, restarting workers, Restart
      Manager; engine exception hooks; crash-loop safe mode; Hub error boundaries
      - [x] Shell (2026-09-23): panic and native-fault logging, poison-tolerant locks, restarting
            workers, self-relaunch after a crash (Windows' restart registration restarted
            nothing in testing), `--crash-test`
      - [x] Engine (2026-09-23): exception hooks and faulthandler into the log; 3 crashes in 2
            minutes start safe mode (processor, default speech model, no clean-up, never saved),
            left from the tray or the Hub
      - [x] Hub (2026-09-23): an error boundary per page and one per window, "This page hit a
            problem - Reload", every fault (render or unhandled) in shell.log, at most 20 a load
- [x] E4 Self-check: WebView2, microphone privacy, driver version, disk, write access, model
      checksums, download hosts - each with a fix. Done 2026-09-24: "Check LocalFlow" on the
      Status card runs every check and lists them with a fix beside each problem. Shell
      (`selfcheck.rs`): Windows' microphone privacy switches (a blocked microphone opens fine
      and hears only silence, so it now shows as blocked, and a press says "Microphone
      blocked"), whether the microphone sends anything, the settings folder, WebView2. Engine
      (`selfcheck.py`): NVIDIA driver (580+ for CUDA 13), the models folder and its free space,
      the clean-up server, every model file hashed offline against the hashes its download
      recorded (the Hugging Face cache's file listing; a clean-up model's ETag), and the
      download sites (one request each, only on the button, through the Windows proxy). Quick
      checks run at start and feed the Status card (a new Storage row, and the driver on the
      Graphics card row); "Download again" deletes exactly the damaged files - never anything
      outside the model caches - and restarts the engine, which fetches them afresh. On this
      machine all pass: 2.4 GB of speech model and 2.3 GB of clean-up model verified in about
      2 s each. Decided 2026-09-24: quick at start, full on demand; offline hashes; a button on
      the Status card; network only on the button
- [x] E5 Where text goes: admin windows, password fields, focus moved, clipboard history
      exclusion, "Paste last dictation". Done 2026-09-24: text goes only into the window it was
      spoken into - if another window (or the desktop, or the taskbar) is in front when it is
      ready, it is kept and the flow bar says so; Paste last dictation (Win + Alt + V, and the
      tray) puts it where the user clicks. An app running as administrator silently drops
      LocalFlow's keystrokes, so there the text goes on the clipboard and the bar says "press
      Ctrl + V". A password field (UI Automation's IsPassword) is never read, and a take into one
      is typed exactly as heard - no clean-up, never pasted, not in history, not in the log, not
      offered by Paste last dictation, dots on the flow bar and in the Hub. Every dictation that
      passes through the clipboard is marked to stay out of clipboard history and the cloud
      clipboard. Found on the way: reading "the text before the caret" through a value pattern
      read a password field's whole value. Checked for real: two new end-to-end scenarios
      (window-changed: kept, then pasted with Win + Alt + V; password-field: a real Windows
      password box, typed exactly as heard, logged as "<password, 44 chars>"), all 10 pass.
      Decided 2026-09-24: Win + Alt + V; admin apps get the clipboard; a different window keeps
      the text; dots for passwords
      - [x] The rest checked live (2026-09-25), and a bug found: the tray's Paste last dictation
            could never paste - clicking the tray puts the taskbar in front, so it found no
            window and kept the text again. It now goes back to the topmost app window first.
            New scenarios: tray-paste, and admin-window (by name only: Task Manager runs as
            administrator without a UAC prompt; the take is copied, the copy carries the
            clipboard-history exclusion marks, the bar says press Ctrl + V, nothing is typed).
            All 11 pass. Clipboard history itself cannot be read by a script reliably (Windows
            only answers the app in front); seeing a dictation stay out of Win + V is a manual
            glance
- [x] E6 Microphone, hotkey, sleep/lock, network/proxy/mirror, disk and install/update edge cases
      from PRODUCTION.md; with A2's untested faults: sleep/resume, graphics driver reset, full disk.
      Done 2026-09-25, in four parts below; the fifth, the update rules, went into D10 (decided
      2026-09-25: there is no updater yet for them to govern)
      - [x] Microphone (2026-09-25). Another app taking the microphone in exclusive mode (tried
            for real: an exclusive WASAPI stream knocks LocalFlow's off) used to read "OS Error
            -2004287478"; it is now "Another app has the microphone to itself", with the setting
            that prevents it, and the microphone comes back by itself a second after that app
            lets go. A dictation that comes back empty from a microphone sending silence says
            "Heard nothing — is the microphone muted?" (new scenario silent-take; 12 pass). A
            Bluetooth headset's microphone is marked in the pickers and on the Status card, with
            why it recognises worse (call quality); told apart by its device's parent
            (BTHHFENUM, BTHLE), not its name - "Headset (HD 450BT)" says nothing of Bluetooth.
            Already there from E2/E4/A2: blocked by Windows privacy, none plugged in, the chosen
            one missing (falls back to the default, said), unplugged mid-take
      - [x] Lock, sleep and graphics driver reset (2026-09-25). The hook sees nothing on the lock
            screen, a UAC prompt's secure desktop or in sleep, so a chord held when one came up
            was released unseen and the take recorded on. Now the session locking or
            disconnecting, suspend and the input desktop changing end the take (its words kept
            for Win + Alt + V) and clear the held keys; resume also rebuilds the microphone and
            reinstalls the hook. Speech on the graphics card failing mid-take (driver reset, the
            speech worker dying) is decoded again on the processor instead of lost, and goes
            back to the card by itself. Checked for real: desktop-switch (input really moved to
            another desktop with the chord held: ended, kept, pasted; the late release did
            nothing) and gpu-lost (worker killed mid-take: typed whole, back on the card six
            seconds later); all 12 standard scenarios pass. Not tried live: a real lock (Win + L)
            and a real sleep - the lock screen is the same desktop switch; a real driver reset
            needs admin tools
      - [x] Downloads (2026-09-25). A company proxy handed out by an automatic configuration
            script (PAC) or automatic detection (WPAD) is now followed: the engine asks WinHTTP
            and exports the answer as HTTPS_PROXY for itself and its children (Python's clients
            only read a typed-in proxy, Hugging Face's Rust downloader only the environment).
            Checked for real: a local PAC naming a local proxy, a real Hugging Face download
            tunnelled through it. A mirror for blocked networks: Models > Downloads > "Download
            from" (checked with a local mirror: a first run's requests all went there). A first
            run without a connection is said, shows its download progress, and is retried by
            itself (15 s doubling to 5 min) until it goes through - no "Try again" needed; the
            clean-up model likewise. Every download checks its room first ("needs about 2.6 GB
            free on C:, and there is 1.0 GB"), and a wrong clock is named as such, each with the
            Settings page that fixes it. Found on the way: a retry timer could outlive the
            engine that set it
      - [x] Machine and install (2026-09-25). Windows shutting down, restarting for an update or
            signing out mid-take: the take ends and the engine is stopped in order (e2e
            session-end). Windows refusing to run the engine or the clean-up server - a company
            policy (AppLocker, Smart App Control, WDAC) or antivirus - is named as that, not as
            "it stopped while starting". Settings a newer LocalFlow wrote are read and never
            saved over (shell.json has a version now), and the Status card says so. The
            installer brings WebView2 where Windows 10 lacks it. Checked, nothing to change: a
            non-ASCII Windows user name (the packaged engine, onnxruntime and llama-server all
            work through such paths), monitor and scaling changes (the flow bar is placed afresh
            at every take)
- [x] E7 Help: troubleshooting per status code, FAQ, known issues, Report a problem; reset and
      export/import of settings. In three parts; decided 2026-09-28: Help lives in the Hub, the
      diagnostics zip replaces dictated words with their length, export/import covers the
      user's own words (dictionary, snippets, app rules, house style), reset covers preferences
      only (not those, history or models)
      - [x] Help page (2026-09-28): every problem from the catalogue with its meaning and fix
            button, searchable, those happening now first, linked from the Status card's codes;
            ten questions and eight known issues. Found: no language setting in the Hub (the
            answer points to the settings file; now C8)
      - [x] Report a problem (2026-09-28): Help > Report a problem makes a zip in Downloads,
            built by the shell so it works when the engine does not (logs, status, settings,
            model timings, system details, a README saying what is and is not inside); dictated
            words and window titles become their length, the dictionary, snippets and house
            style counts, an API key "<set>", no history. "Open a report on GitHub" opens the
            public repository's new-issue page with only version, Windows, graphics card and
            problem codes filled in. Checked against this machine's real logs: none of 227
            dictations or any window title in the zip
      - [x] Reset and your words (2026-09-28): Help > Your settings. Reset preferences puts
            everything back but the user's own words, history, models, API key, language and
            mirror (asks first, says what stays). Export writes the dictionary, terms, snippets,
            app rules and house style to one file; import merges one back, adding what is new
            and keeping the user's own where both have an entry. Checked on the real app with
            its settings backed up and restored byte for byte afterwards

Known limits: moving work to the processor frees the graphics card and its memory but does not
remove heat on a laptop whose CPU and GPU share cooling - the biggest heat saving is the
"no keep-warm" step. (The 0.35 GB of VRAM that used to stay after an idle release is gone since
A4: speech on the graphics card runs in a worker process that the idle release ends.)

## Version 0.2.1: models you can see (planned 2026-09-30)

Found installing v0.2.0 from GitHub on a wiped PC: setup's "Try it" says "a few seconds" while
the 2.6 GB speech model is still downloading (the hotkey is ignored silently meanwhile); the
clean-up model's first download reports progress to the log only, so the Hub says "loading" for
minutes; nothing says which other models there are, which suit this PC, or what they are for.
Decided 2026-09-30: ship as v0.2.1 in today's look (0.3's design restyles it); downloads shown
in a strip on every Hub page, in the tray, and by a notification when done; LocalFlow suggests
models once after setup and on the Models page, nowhere else.

- [x] M1 Engine downloads (2026-10-01): one queue for every download (speech, clean-up, its
      runtime, the CUDA libraries), one at a time, most urgent first (first run, switch,
      LocalFlow's own parts, library), with bytes, speed, time left and state in
      status.downloads; the clean-up model's first download, which showed in the log only, now
      shows. Download without switching, cancel, remove (never the model in use), and each
      model rated for this PC (modelchoice.fit). Found: Xet downloads - 7-15x quicker here than
      huggingface_hub's plain HTTP (11-22 against 1.5-3 MB/s) - cannot be stopped from their
      progress callback and never resume, so each model downloads in a process of its own
      (`localflow fetch`) that a cancel ends at once (0.03 s, nothing left on disk); pause was
      dropped for that reason. Progress follows bytes received as well as written: 130 updates
      in 31 s where there were 7
- [x] M2 Model library (Hub, Models): every speech and clean-up model as a card - in use / on
      this PC with its disk / not downloaded with its size, how it suits this PC and why, its
      download's progress - with Download only, Cancel download and Remove (asks once more)
- [x] M3 Downloads you cannot miss: a strip on every Hub page (model, progress, speed, time
      left, Cancel, what is next, what just arrived or failed), the percentage beside Models,
      the tray ring filling with the download and its tooltip, a notification per model that
      arrives, and "Auto-edits are ready" once clean-up has downloaded and loaded. Speech and
      Auto-edits say where they stand at the top of their cards; the Status card says
      "Downloading 46 %" for clean-up
- [x] M4 Suggestions: "Recommended for this PC" on Models - a more accurate Parakeet, Whisper
      only for a language Parakeet does not know, auto-edits when off and suited (never below
      9 GB, B5), a quicker clean-up model when the one in use is slow; never a model Automatic
      already has and passed over - and "Your models" once on Overview after setup
- [x] M5 First run: the welcome lists what downloads and how big; "Try it" shows the speech
      model's download and turns ready by itself; finishing or skipping setup early brings one
      "LocalFlow is ready" notification; a press during the download says "Still downloading
      the speech model - 45 %, about 2 min left" (speech-downloading)
- [x] M6 Build, install on a wiped PC, publish v0.2.1 (2026-10-01). The installer (171 MB)
      installed silently in 13 s on the development PC wiped of LocalFlow; the real first run
      downloaded Parakeet v3, the clean-up runtime, Qwen3 4B and the CUDA libraries one at a
      time, "LocalFlow is ready" and "Auto-edits are ready" arrived as notifications, and the
      strip, the Models percentage and "Your models" showed in the real window. Found on it and
      fixed: the speech worker, started early, downloaded the speech model a second time beside
      the queue's download - 4.9 GB kept instead of 2.5 and 100 s more before speech was ready;
      v0.2.0 had it too (now it preloads only a model already here; checked on a scratch first
      run: 2.55 GB, ready in 94 s); and "100 %" was shown for the 20 s Xet spends writing the
      file. Seen but not a product bug: an app started from a program that was running before
      LOCALFLOW_ENGINE_EXE was removed inherits it and uses the repository's engine

## Open decisions (yours)
- Product name - LocalFlow for now (2026-09-24); a rename changes the display name only
- ~~Default clean-up model~~ - decided: Automatic, which picks Qwen3 4B on capable hardware
- ~~Public release channel for auto-update~~ - decided 2026-09-18: a separate public repo, all rights reserved
- ~~Code-signing certificate~~ - decided 2026-09-24: unsigned for now
- ~~Platforms, analytics, history, legal text~~ - decided 2026-09-24: Windows 10 and 11 x64
  (ARM64 after v0.2); no analytics; history asked during setup; privacy and licence text drafted
  from the app, no outside review
