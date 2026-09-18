# Wispr Flow feature parity

What Wispr Flow does (from wisprflow.ai/features, the help center and reviews), how it does it,
and where LocalFlow stands. Wispr's pipeline: audio is uploaded, transcribed in the cloud
(OpenAI subprocessor), cleaned by a fine-tuned Llama, then pasted into the active app.
Ours is the same shape, entirely on this laptop.

| Wispr Flow feature | How Wispr does it | LocalFlow | Status |
|---|---|---|---|
| Push-to-talk (hold Ctrl+Win on Windows) | low-level key hook | `hotkey.py` + `app.py` | done |
| Hands-free mode (double-tap the hotkey) | same | double-tap detection in `app.py` | done |
| Works in every text field | clipboard paste into the focused app | `inject.py` (SendInput / paste) | done |
| Filler word removal ("um", "uh") | LLM cleanup | rules layer, always on | done |
| Auto punctuation and capitalisation | ASR + LLM | Parakeet punctuates natively | done |
| "New line" / "new paragraph" | LLM | L1 rules | done |
| Backtracking / self-correction ("Monday, no, Tuesday") | LLM | Qwen3-4B on a bundled llama-server the engine owns; guard falls back to rules | done |
| Numbered / bulleted list formatting | LLM | same model, prompted with the target app's style profile | done |
| Tone per app (formal in Gmail, casual in Slack) | detects active app, prompts LLM | `cleanup/profiles.py`: chat, email, docs, code, terminal, general, chosen from the process and window title | done |
| Personal dictionary | word list fed to the models | `dictionary_terms` matched phonetically (Metaphone + Jaro-Winkler + edit distance), and given to the model as spellings to use | done |
| Snippets ("my email" -> address) | phrase triggers | `postprocess.snippets` | done |
| Contextual name spelling | LLM sees on-screen context | dictionary terms are in the prompt; reading the screen is still to do | partial |
| Command mode (select text, say "make this concise") | LLM rewrite in place | not started | todo (read selection via Ctrl+C, rewrite, paste) |
| Whisper mode (quiet speech) | gain + model | takes peaking under -30 dBFS are lifted before the model | done |
| 100+ languages | Whisper-class models | Parakeet auto-detects 25 European languages; any other `stt.language` routes to Whisper turbo on the same runtime, loaded on demand | done |
| Live text while speaking | not offered (waveform only) | engine re-decodes the take continuously while the key is held; `partial` events carry the text (overlay shows it in phase 4) | done (engine) |
| Developer mode (camelCase, file tags in Cursor) | prompt + app detection | code profile preserves identifiers, paths, flags; file tagging not started | partial |
| Flow bar overlay (waveform pill) | native overlay window | `ui.Overlay`: Tk, colour-keyed, WS_EX_NOACTIVATE, live mic level | done (basic); prettier version in the Tauri shell |
| Tray icon, start at login, quiet start | app shell | `tray.rs` (icon drawn per state) + `win.rs` (HKCU Run key); the wizard offers it on first run | done |
| Privacy mode / no cloud | opt-in setting | everything is local by default | done |
| Virtual microphone support (Krisp, NVIDIA Broadcast) | device picker | `audio.device` by name/index | done |
| History of dictations | local DB | not started | todo (SQLite log, opt-in) |
| Usage analytics, team dictionary, Notetaker | SaaS | out of scope | n/a |

Sources: [Wispr features](https://wisprflow.ai/features), [Wispr changelog](https://wisprflow.ai/whats-new),
[Zapier review](https://zapier.com/blog/wispr-flow/), [eesel review](https://www.eesel.ai/blog/wispr-flow-review),
[Wispr Flow 101](https://sidsaladi.substack.com/p/wispr-flow-101-the-complete-guide).
