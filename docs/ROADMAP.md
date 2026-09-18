# Roadmap

The full design document (architecture, protocol, budgets, decisions, risks) is the
"LocalFlow Build Plan" artifact; this file is the checklist version that lives with the code.

Target: a background app that starts at sign-in, shows a flow bar the moment you hold the
hotkey, and puts cleaned text at the caret in any app within ~300 ms, with the speech model
and the clean-up model on the GPU and nothing leaving the machine.

Two processes, one boundary: the **shell** (Tauri 2, Rust + WebView2) owns the OS and the
screen (hooks, mic, injection, UI Automation, tray, flow bar, hub); the **engine** (Python)
owns the models (VAD, Parakeet fp16 on CUDA, Whisper turbo, clean-up pipeline). A bundled
`llama-server` serves Qwen3-4B to the engine. They talk over a token-protected local
WebSocket carrying JSON events and 20 ms audio frames.

| Measure | Today | Target |
|---|---|---|
| Release-to-inserted, 5 s dictation, AI on | **362 ms p50 / 522 ms p95** (phase 2) | <= 400 ms p50 |
| Release-to-inserted, AI off | 52 ms p50 to final text (phase 1) + injection | <= 100 ms p50 |
| WER, English | 6.3 % (Parakeet v3 fp32) | no regression from fp16 |
| Idle | CPU ~0, RAM 2.6 GB | CPU < 1 %, GPU 0 %, VRAM ~4.5 GB |

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
- [ ] 4.3 Onboarding wizard
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
- [ ] 6.3 Updater + GitHub Actions release pipeline (needs a public release channel)
- [ ] 6.4 Diagnostics export

## Phase 7: Polish and brand
- [ ] 7.1 Name, logo, icons, type, colour, motion, sounds via the tokens file
- [ ] 7.2 Performance pass (idle CPU < 1 %, startup < 3 s, 60 fps bar)
- [ ] 7.3 Accessibility, high contrast, mixed-DPI monitors, remote desktop
- [ ] 7.4 Beta checklist and docs

## Open decisions (yours)
- Product name (LocalFlow is a placeholder; free to change before 6.2)
- Default clean-up model (Qwen3-4B recommended for the 8 GB budget)
- Public release channel for auto-update (needed by 6.3)
- Code-signing certificate (optional, decided at 6.2)
