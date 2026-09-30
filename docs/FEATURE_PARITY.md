# Wispr Flow feature parity

What Wispr Flow does (from wisprflow.ai/features, the help center and reviews), how it does it,
and where LocalFlow stands. Wispr's pipeline: audio is uploaded, transcribed in the cloud
(OpenAI subprocessor), cleaned by a fine-tuned Llama, then pasted into the active app.
Ours is the same shape, entirely on this laptop. Updated 2026-09-23 (v0.1.0 released, v0.2 on
`main`).

| Wispr Flow feature | How Wispr does it | LocalFlow | Status |
|---|---|---|---|
| Push-to-talk (hold Ctrl+Win on Windows) | low-level key hook | `WH_KEYBOARD_LL` hook in the shell (`hotkey.rs`), chord set by pressing it in the Hub | done |
| Hands-free mode (double-tap the hotkey) | same | double-tap latch in `hotkey.rs`; stops after a few seconds of silence (`session::watch_hands_free`) | done |
| Works in every text field | clipboard paste into the focused app | `inject.rs`: unicode `SendInput` or clipboard paste with restore, per app | done |
| Filler word removal ("um", "uh") | LLM cleanup | rules layer, always on | done |
| Auto punctuation and capitalisation | ASR + LLM | Parakeet punctuates natively; `cleanup/joining.py` fits the text to the sentence it lands in | done |
| "New line" / "new paragraph" | LLM | L1 rules | done |
| Backtracking / self-correction ("Monday, no, Tuesday") | LLM | bundled clean-up model (Qwen3 4B by default); a guard falls back to rules | done |
| Numbered / bulleted list formatting | LLM | same model, prompted with the target app's style profile | done |
| Tone per app (formal in Gmail, casual in Slack) | detects active app, prompts LLM | `cleanup/profiles.py`, overridable per app in Hub -> Apps | done |
| Personal dictionary | word list fed to the models | phonetic matching plus the model's prompt; suggestions learned from your own corrections | done |
| Snippets ("my email" -> address) | phrase triggers | `postprocess.snippets`, with `{date}`, `{time}` placeholders | done |
| Contextual name spelling | LLM sees on-screen context | dictionary terms and the text before the caret (UI Automation) are in the prompt | done |
| Command mode (select text, say "make this concise") | LLM rewrite in place | Win+Alt: `cleanup/command.py`; leaves the text alone if the edit looks wrong | done |
| Whisper mode (quiet speech) | gain + model | takes peaking under -30 dBFS are lifted before the model | done |
| 100+ languages | Whisper-class models | Parakeet auto-detects 25 European languages; other languages go to Whisper turbo | done |
| Live text while speaking | not offered (waveform only) | the flow bar shows live text while the key is held | done |
| Developer mode (camelCase, file tags in Cursor) | prompt + app detection | code profile preserves identifiers, paths, flags; file tagging not started | partial |
| Flow bar overlay (waveform pill) | native overlay window | `flowbar.rs` + `FlowBar.tsx`: no-activate, click-through, live level and text | done |
| Tray icon, start at login, quiet start | app shell | `tray.rs` + `win.rs` (HKCU Run key); the wizard offers it on first run | done |
| Per-app rules (off in some apps, auto-send in chat) | settings | Hub -> Apps: disable, force paste/type, press Enter after, force a style | done |
| Privacy mode / no cloud | opt-in setting | everything is local by default | done |
| Virtual microphone support (Krisp, NVIDIA Broadcast) | device picker | Hub -> Voice | done |
| History of dictations | local DB | Hub -> History: searchable, shows what you said before clean-up, retention setting | done |
| Choice of models | none (cloud) | v0.2: four speech and three clean-up models, or Automatic (best that is quick enough here) | done on `main` |
| GPU heat and memory management | n/a (cloud) | v0.2: moves models between the graphics card and the processor by temperature, load and idle time | done on `main` |
| Usage analytics, team dictionary, Notetaker | SaaS | out of scope | n/a |

Sources: [Wispr features](https://wisprflow.ai/features), [Wispr changelog](https://wisprflow.ai/whats-new),
[Zapier review](https://zapier.com/blog/wispr-flow/), [eesel review](https://www.eesel.ai/blog/wispr-flow-review),
[Wispr Flow 101](https://sidsaladi.substack.com/p/wispr-flow-101-the-complete-guide).
