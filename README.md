# LocalFlow

Local, private push-to-talk dictation for Windows. Hold **Ctrl+Win**, talk, let go, and the
text appears wherever your caret is. No subscription, no cloud: the speech model
(NVIDIA Parakeet TDT 0.6B v3) and the clean-up model (Qwen3 4B) both run on your PC - on its
graphics card when it has a suitable one, otherwise on the processor.

> **Not open source.** Copyright (c) 2026 Arnab Arya, all rights reserved. The source is here
> to read; it is not licensed for reuse. The installer on the Releases page may be downloaded
> and run for personal use. See [LICENSE](LICENSE).

**Download:** the installer is on the [Releases](../../releases) page. It installs for your user
only, with no administrator prompt, and downloads its speech and clean-up models (several
gigabytes) on first run.

* [docs/ROADMAP.md](docs/ROADMAP.md): the phased plan to a polished product
* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): current design and rationale
* [docs/FEATURE_PARITY.md](docs/FEATURE_PARITY.md): Wispr Flow feature checklist

## Layout

```
engine/   Python: the models only - speech, clean-up, benchmarks  -> pip install -e "./engine[dev]"
app/      Tauri 2 shell: Rust core + React UI. The application.  -> cd app && npm install
design/   tokens.json = visual system; build_tokens.py -> app/src/styles/tokens.css
docs/     roadmap, architecture, feature parity
```

## Engine setup (once)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -e ".\engine[dev]"
```

The first run downloads the speech model (~2.4 GB) to `%USERPROFILE%\.cache\huggingface`.

## Run

LocalFlow is the Tauri app in `app/`. Build it with `npm run tauri build` and run the executable
it produces; it starts the speech engine itself as a child process and stops it on exit. The
hotkey, the flow bar, the tray icon, start-at-sign-in and text injection all belong to that
shell - the Python side no longer has a user interface of its own.

The `localflow` CLI is the engine, and is for development and diagnosis:

```powershell
.\.venv\Scripts\localflow serve --port 7788 --token dev  # run the engine by hand; the shell will attach to it
.\.venv\Scripts\localflow send-wav some.wav              # stream a file like a dictation: live + final text and timings
.\.venv\Scripts\localflow transcribe some.wav            # one pass over a file, no session
.\.venv\Scripts\localflow devices                        # list microphones
.\.venv\Scripts\localflow config                         # where the settings live
.\.venv\Scripts\localflow bench run --set own            # accuracy and latency
```

The shell has its own headless checks, which cover what the Python `self-test` and
`overlay-demo` commands used to: `app.exe --selftest <wav>` streams a WAV through the whole link
and prints the final text, `app.exe --report` prints engine state and the focused window, and
`#flowbar?demo=recording` renders the flow bar in a browser. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

GPU: `pip uninstall onnxruntime; pip install -e ".\engine[gpu]"` (CUDA 13 libraries, ~1.6 GB). The
engine picks CUDA automatically when it works and falls back to the CPU otherwise.

Languages: Parakeet handles 25 European languages and detects which one you speak. Set
`stt.language` to any other code (`"hi"`, `"ja"`, `"ar"`, ...) and dictation goes to Whisper
large-v3-turbo, downloaded on first use (~1.6 GB).

## Auto-edits

On by default. After the speech model produces a transcript, LocalFlow removes fillers,
applies what you took back ("send it Monday, no wait, Tuesday"), formats dictated lists,
writes numbers, times, emails and URLs the way you would type them, fixes the spelling of
names you have taught it, and matches the tone of the app you are dictating into.

Everything runs on your machine: the engine downloads `llama-server` and Qwen3-4B (~2.5 GB)
on first use and manages that process itself. Nothing is sent anywhere.

| Setting | What it does |
|---|---|
| `postprocess.llm_cleanup` | `false` turns auto-edits off; the rule-based clean-up still runs |
| `postprocess.llm_provider` | `bundled` (default), `ollama`, `openai` (any compatible endpoint), `anthropic` |
| `postprocess.llm_model` | bundled: `qwen3-4b` (default) or `qwen3-1.7b` (faster, weaker); otherwise the provider's model name |
| `postprocess.dictionary_terms` | `["Arnab", "Okonkwo", "Parakeet"]`: correct spellings, matched by sound |
| `postprocess.custom_instructions` | your own style notes, added to the model's instructions |
| `postprocess.llm_min_words` | shorter utterances skip the model unless they contain a correction or a number |
| `postprocess.llm_prefill` | warm the model's prompt while you are still speaking |

A style profile is chosen from the app you are dictating into: chat, email, document, code
editor, terminal, or general. Anything the model returns that does not look like an edit of
what you said (an answer to your question, an apology, a wildly different length) is thrown
away and the rule-based text is used instead.

Cloud models are supported for people who want them: set `llm_provider` to `anthropic` or
`openai`, put your key in `llm_api_key`, and install the extra with
`pip install -e ".\engine[cloud]"`.

Settings live in `%APPDATA%\LocalFlow\config.json`; logs in `%APPDATA%\LocalFlow\localflow.log`.

| Key | What it does |
|---|---|
| `hotkey.keys` | e.g. `["ctrl", "win"]`, `["f8"]`, `["ctrl", "alt", "space"]` |
| `audio.device` | mic index or name substring (`"NVIDIA Broadcast"`) |
| `stt.backend` | `auto` (default: Parakeet + Whisper for other languages), `parakeet`, `whisper`, `whisper-ct2` |
| `stt.device` | `auto` / `cpu` / `cuda` (CUDA needs `pip install -e ".\engine[gpu]"`) |
| `stt.language` | `null` = detect (Parakeet languages); `"hi"`, `"ja"`, ... = Whisper |
| `stt.gpu_keep_warm` | `auto` (on mains only) / `always` / `never`: keep the GPU clocked up while you hold the key |
| `postprocess.llm_cleanup` | auto-edits, on by default (see below) |
| `postprocess.dictionary` | exact replacements: `{"arnub": "Arnab"}` |
| `postprocess.snippets` | `{"my email": "you@example.com"}` |
| `inject.method` | `auto` / `type` / `paste` |
| `compute.mode` | `adaptive` (default), `gpu` (always the graphics card), `cpu` (never) |
| `compute.temp_limit_c` | adaptive: clean-up moves to the processor at this temperature, speech 6 °C above it (default 80) |
| `compute.auto_speech`, `compute.auto_cleanup` | `true` (default): LocalFlow picks the model; `false`: the one set in `stt` / `postprocess` |
| `compute.idle_release_min` | adaptive: free the graphics card after this many minutes unused; `0` = never (default 10) |

## Models and where they run

The Hub's **Models** page picks the speech model (Parakeet v3, Parakeet v2, Parakeet v3 Compact,
Whisper Large v3 Turbo) and the clean-up model (Qwen3 4B, Phi-4 mini, Gemma 4 E2B). A model
downloads the first time it is chosen, and the current one keeps working until the new one has
loaded.

LocalFlow also decides where each model runs. In the default **Automatic** mode it watches the
graphics card's temperature and load: as it warms up it first stops keeping it busy between
words, then moves clean-up to the processor (about 0.4 s slower per dictation), then speech;
another app using the GPU moves clean-up off it too; and after ten idle minutes both leave the
GPU, freeing its memory, until the next dictation takes it back. Models move without
interrupting dictation. Parakeet Compact always runs on the processor, where it is faster. The
policy is `engine/src/localflow/placement.py`.

Which model runs is **Automatic** by default too, and it puts quality first: the most accurate
model that is quick enough on the device it is running on (speech within 0.25 s per second of
audio, clean-up within 1.5 s). On a fast machine that means Parakeet v3 and Qwen3 4B everywhere;
a slower processor gets Parakeet Compact or Phi-4 mini when work lands on it. Every dictation's
timings are recorded (`perf.json` beside the settings), so the choice follows what this machine
actually does rather than an estimate. Automatic only switches between models already
downloaded (apart from Compact, 0.7 GB). Choosing a model in the Hub pins it; choosing
Automatic again hands it back. The policy is `engine/src/localflow/modelchoice.py`.

## Benchmark

Every speed or accuracy change is measured against a golden set.

```powershell
.\.venv\Scripts\localflow bench fetch          # 60 LibriSpeech test-clean utterances (once)
.\.venv\Scripts\localflow bench record         # your own voice, 30 prompts (optional, once)
.\.venv\Scripts\localflow bench run --set public
.\.venv\Scripts\localflow bench run --set all --device cuda
.\.venv\Scripts\localflow bench stream --set own     # the real thing: stream each take through the engine at real time
.\.venv\Scripts\localflow bench cleanup              # the auto-edit quality set (30 dictations with expected output)
.\.venv\Scripts\localflow bench cleanup --rules-only # ... the same set without the model, for comparison
```

Results go to `engine/bench/results/` as JSON. `bench run` prints WER, per-file latency
p50/p95, milliseconds per second of audio, RSS and VRAM; `bench stream` prints the number
that matters for dictation, key-up to final text, p50/p95.

## Tests

```powershell
cd engine; ..\.venv\Scripts\python -m pytest -q
```

## App (Tauri shell)

Needs Rust (rustup) and the Visual Studio C++ build tools.

```powershell
cd app
npm install
npm run tauri dev
```

## Packaging

Two steps, in this order. The shell bundles the frozen engine as a resource, so the engine has
to exist before the installer is built.

```powershell
.\.venv\Scripts\pip install -e ".\engine[package]"
cd engine; ..\.venv\Scripts\python -m PyInstaller localflow-engine.spec --noconfirm
cd ..\app; npm run tauri build
```

The first step produces `engine/dist/localflow-engine/` - about 300 MB, most of it
onnxruntime. The second produces an installer under `app/src-tauri/target/release/bundle/`.

NVIDIA's CUDA libraries (cuDNN, cuBLAS, cuFFT: 1.5 GB unpacked) are left out on purpose. A PC
with an NVIDIA card downloads them on first run (about 1 GB, from PyPI, checked against pinned
checksums) while speech runs on the processor, and moves speech to the graphics card once they
are there; `localflow cuda --install` fetches them by hand. A PC without one never needs them.

The installer is per-user: it needs no administrator prompt, installs under `%LOCALAPPDATA%`,
and the start-at-sign-in entry the app writes lives in `HKCU` beside it. Uninstalling removes
that entry and offers to delete your settings, history and downloaded models - keeping them is
the default, because the models are a multi-gigabyte download and reinstalling is the usual
reason to be there.

The speech and clean-up models are **not** in the installer. They download on first run to
`%LOCALAPPDATA%\LocalFlow`, resumably and with checksums, which keeps the installer to the
program itself rather than roughly five gigabytes of weights.

## Working on LocalFlow

The copy you use every day is the **installed** one, in `%LOCALAPPDATA%\LocalFlow`, started from
the Start menu. Builds from this repository are separate and never replace it.

| | |
|---|---|
| `main` | v0.2, under development - status in [docs/ROADMAP.md](docs/ROADMAP.md#version-02-in-development-on-main) |
| `release/0.1` | fixes for the shipped v0.1, branched from the `v0.1.0` tag |

Only one copy of LocalFlow can run at a time - the second one hands over to the first and
exits - so quit the installed one from the tray before starting a development build. Both share
your settings, history and downloaded models. A development build will not take start-at-sign-in
away from the installed copy: it only repairs a sign-in entry that is actually broken.
