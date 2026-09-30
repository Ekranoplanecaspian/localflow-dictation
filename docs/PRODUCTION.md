# Production readiness

What v0.2 needs beyond features to be something other people install and rely on: how it
fails, how it tells the user, and what a full application ships with. Written 2026-09-24 from a
code audit and online research (sources at the end). The checklist lives in ROADMAP.md, "Road to
release", group E. Built so far: E3 "never crash" (2026-09-23) - see "Never crash" below and
ARCHITECTURE.md, "When things go wrong". E1 (status model) and E2 (problem catalogue) followed on 2026-09-24. The rest of group E is
still a plan.

## Decided (2026-09-24)

* Text that cannot be typed is kept and offered ("Paste last dictation", tray and a shortcut),
  never typed into whatever window is in front now.
* Password fields: dictated, but never saved in history or sent to the clean-up model.
* Dictations are kept out of Windows clipboard history and the cloud clipboard.
* Engine crash loop (3 in 2 minutes): safe mode, said, with one click back to normal.
* A crashed app is restarted, after the crash is logged. Built by LocalFlow relaunching itself:
  Windows' restart registration (Restart Manager) recorded test crashes but restarted none.
* Updates download quietly and install on the next quit, never during a dictation.
* Problems are reported as a prefilled GitHub issue on the public repo plus a diagnostics zip.
* Models download on first run; no separate offline installer.
* Unsigned for v0.2; every release submitted to Microsoft's malware analysis before publishing.
* Downloads use the Windows proxy; an advanced setting for a Hugging Face mirror.
* Group E is built together with group A, first.

## The rule

**LocalFlow never just stops working.** Every failure is one of three things:

1. **Handled silently** - it recovers by itself and nobody needs to know (a dropped keyboard
   hook reinstalled, a microphone stream rebuilt).
2. **Degraded and said once** - it keeps working in a lesser way and says so, with a way back
   (clean-up model failed, so rules only; graphics card failed, so the processor).
3. **Stopped and explained** - it cannot do the job, says exactly why in plain words, and offers
   the fix (the microphone is blocked in Windows privacy settings: button opens that page).

A dictation is never lost: if the text cannot be typed, it is kept and offered.

Every message follows Microsoft's guidance: say what happened, why if known, and what to do - no
"error", no blame, one message per cause rather than one generic message.

## Found missing today (code audit)

| Gap | Effect now |
|---|---|
| ~~Release builds use `panic = "abort"` with no panic hook~~ | fixed 2026-09-23 (E3): panics and native faults logged, workers restart, the app relaunches itself |
| ~~No `sys.excepthook` / `threading.excepthook` / `faulthandler` in the engine~~ | fixed 2026-09-23 (E3); three crashes in two minutes start safe mode |
| ~~No React error boundary~~ | fixed 2026-09-23 (E3): per page and per window |
| ~~No elevated-window check~~ | fixed 2026-09-24 (E5): the text goes on the clipboard and the bar says "press Ctrl + V" |
| ~~No password-field check~~ | fixed 2026-09-24 (E5): typed as heard, never cleaned up, saved, logged or shown |
| ~~Dictations pass through the clipboard unmarked~~ | fixed 2026-09-24 (E5): marked to stay out of clipboard history and the cloud clipboard |
| ~~No microphone-privacy check~~ | fixed 2026-09-24 (E4): "Windows is blocking the microphone", with the settings page |
| ~~No free-disk check, no proxy support, no sleep/lock handling~~ | fixed 2026-09-25 (E6): space checked before every download, the Windows proxy (PAC and WPAD too) used for all of them, a take ended and kept when the screen locks |
| CUDA 13 needs NVIDIA driver 580+ | older drivers quietly fall back to the processor with no hint to update |
| ~~PyInstaller engine under a non-ASCII Windows user name~~ | checked 2026-09-25 (E6): not so with today's PyInstaller - the engine runs from, and keeps settings in, folders like `Ünïcødé 用户`; onnxruntime and llama-server load models through them |
| ~~Tauri installer's WebView2 mode not set~~ | fixed 2026-09-25 (E6): `embedBootstrapper` |

## How errors will work

### One status model
Each part reports **ok / degraded / failed + reason + action**: engine link, speech model,
clean-up model, microphone, keyboard hook, graphics card, disk, network, updates. One source,
shown in four places, each at its own volume:

* **Flow bar** - a word or two, only when it affects the take in hand ("Microphone blocked").
* **Tray** - icon state and tooltip, always current.
* **Hub, Status card** - every part, the reason, and a Fix button where one exists.
* **Notification** - once per change of state, never repeated for the same problem.

### A problem catalogue
Every known failure gets a code, how it is detected, the message, and the recovery. Engine and
shell share the codes. The catalogue is a table in the code, so a new failure cannot ship
without a message. Built 2026-09-24 (E2): `shared/problems.json`, 46 entries; see ARCHITECTURE.md,
"Status and the problem catalogue".

### Never crash, and when it happens, recover
* **Shell:** a panic hook that writes the message, location and backtrace to `shell.log` before
  anything else; poisoned-mutex-tolerant locks instead of `lock().unwrap()`; worker threads that
  catch and restart instead of taking the app down; a crashed app comes back (it starts a new
  copy of itself; a crash in its first minute does not, so start-up faults cannot loop).
* **Engine:** `sys.excepthook`, `threading.excepthook` and `faulthandler` into `localflow.log`.
  The shell already restarts it with backoff; add **crash-loop detection** - three crashes in two
  minutes starts **safe mode** (processor only, AI clean-up off, default speech model) and says so.
* **Hub:** an error boundary per page: "This page hit a problem - Reload", with the fault logged.

### Self-check ("Check LocalFlow")
Runs at start (quick parts) and from the Hub (all parts), each with a fix: WebView2 present;
microphone allowed in Windows privacy settings and actually hearing something; a default
microphone exists; the NVIDIA driver is new enough; enough free disk; settings folder writable;
model files present and matching their checksums (antivirus quarantine shows up here);
`llama-server.exe` present; download hosts reachable; engine start time.

## Edge cases, by area

**Microphone**
* Blocked in Windows privacy settings (`CapabilityAccessManager\ConsentStore\microphone`) -
  failed, with a button to the settings page.
* No microphone, or the chosen one unplugged - fall back to the default and say so.
* Unplugged mid-take - finish with the audio so far instead of losing it.
* Muted or silent - "LocalFlow heard nothing - is the microphone muted?" when a take's level
  never rises.
* Bluetooth headset - using its microphone switches it to the hands-free profile (8 or 16 kHz
  audio, music drops to mono); note it in the microphone picker and suggest the built-in one.
* Held in exclusive mode by another app - failed, naming the reason. Done 2026-09-25 (E6).

**Hotkey**
* Lock screen, UAC prompt, secure desktop - the hook sees nothing; a take running when the
  session locks is ended (kept, not lost). Done 2026-09-25 (E6), with the desktop switch a UAC
  prompt makes.
* Sleep and resume - rebuild audio, re-check the graphics card, reset the hook state. Done
  2026-09-25 (E6); the graphics card is re-checked by the next decode (below).
* Conflicts - the chord also used by another app or by Windows (Win+H is Windows dictation).
* AltGr on international layouts is Ctrl+Alt (already why command mode avoids Ctrl+Alt).
* Remote desktop, games with anti-cheat - the hook may not see keys; documented.

**Where the text goes**
* Admin window - detect the foreground window's elevation; paste instead of typing, and if even
  that fails keep the text and say "Windows doesn't let LocalFlow type into apps running as
  administrator".
* Password field (UI Automation `IsPassword`) - no history, no clean-up model, typed as heard.
* Focus moved while the text was being prepared - do not type into whatever is in front now;
  keep the text and offer it ("Paste last dictation", tray and a shortcut).
* Read-only field, nothing focused, fullscreen game - same: keep and offer.
* Clipboard held by another app - already falls back to typing.
* Keep dictations out of clipboard history and cloud clipboard (the registered formats
  `ExcludeClipboardContentFromMonitorProcessing`, `CanIncludeInClipboardHistory`,
  `CanUploadToCloudClipboard`).

**Engine and models**
* Engine will not start - say which: files missing (antivirus), non-ASCII path, port refused.
* Crash loop - safe mode (above).
* Clean-up server crash, port conflict or out of graphics memory - rules only, said once, retried.
* Model file damaged - checksum mismatch, re-download.
* NVIDIA driver older than 580 - processor, with "update your NVIDIA driver for faster dictation".
* Graphics driver reset mid-decode - redo that decode on the processor. Done 2026-09-25 (E6).
* Cloud provider: wrong key (401), rate limited (429), unreachable - named, rules-only meanwhile.

**Downloads and network**
* Offline first run - the app cannot dictate until the speech model is here; say so plainly,
  resume when back online. Done 2026-09-25 (E6): retried by itself, progress on the Status card.
* Proxy - use the Windows proxy settings for every download (Hugging Face honours only
  environment variables). Done 2026-09-25 (E6): WinHTTP's answer, PAC and WPAD included,
  exported to the environment. (Python's clients do read a typed-in Windows proxy; not a PAC.)
* Hugging Face blocked - a mirror setting (`HF_ENDPOINT`); file contents come from separate CDN
  hosts, which firewalls often block even when huggingface.co is allowed. Done 2026-09-25 (E6):
  Models > Downloads > "Download from".
* Not enough disk - checked before a download starts, with the size needed. Done 2026-09-25 (E6).
* Interrupted or corrupt - resume, checksum, retry; never a half file in place. Already so:
  Hugging Face writes `.incomplete` files and renames them, llama.cpp archives download to
  `.part` and are checked; E4's "Download again" repairs a damaged file.
* Clock wrong - TLS fails; say to fix the clock. Done 2026-09-25 (E6).

**Windows and the machine**
* Windows 10 without WebView2 - the installer provides it (`embedBootstrapper`, +1.8 MB). Done
  2026-09-25 (E6); to be seen working on a Windows 10 machine in group D.
* Several users on one PC - per-session single instance already; per-user data already.
* Enterprise: AppLocker blocking programs in AppData, roaming AppData on a network share,
  Controlled Folder Access - detect write/launch failures and say what is blocking. Done
  2026-09-25 (E6) for launches (engine-blocked, cleanup-blocked: policy 1260, Smart App Control
  or WDAC 4551, antivirus 225/226); writes were E4's (folder-not-writable). Controlled Folder
  Access guards Documents, Desktop and the like, not AppData, where LocalFlow writes. Roaming
  AppData holds only settings, history and logs; models are in local AppData.
* Low memory, battery saver, display scaling or monitor changes, theme change live. Already so:
  a model that does not fit is out-of-memory (E2), battery moves work off the graphics card
  (placement), the flow bar is placed on the current monitor's work area and scale at every
  take, and the Hub follows the Windows theme.
* Windows restarting for an update, or shutting down mid-take - end cleanly, save state. Done
  2026-09-25 (E6): the take ends, the engine is told to stop; settings and history are saved as
  they change, so there is nothing else to save.

**Install, update, uninstall**
* Update while dictating - wait until idle.
* Update fails (download, signature, installer) - keep the running version, log it, retry later,
  say so; never quit without a new version to start.
  (These two are built with the updater itself, ROADMAP D10: decided 2026-09-25.)
* Settings from a newer version (after a downgrade) - read what is known, warn, never overwrite.
  Done 2026-09-25 (E6): shell.json now carries a version too; raise it with every new setting.
* Antivirus false positive on the unsigned PyInstaller engine - submit every release to
  Microsoft's malware analysis before publishing; known-issues page explains.

## What a full application also ships with

* **Updates** (ROADMAP D10): signed update files, staged, release notes shown in the app.
* **Help**: a troubleshooting page per status code, FAQ, known issues, "Report a problem"
  (diagnostics zip + prefilled issue). Done 2026-09-28 (E7): all in the Hub's Help page.
* **Recovery**: "Paste last dictation"; reset settings to defaults; export and import the
  dictionary, snippets and app rules. Done (E5, E7).
* **Versioning**: semantic versions, a changelog, release notes per version.
* **Resource honesty**: publish the measured idle cost (Wispr Flow is reported at ~800 MB RAM and
  8 % CPU idle on Windows - a place to beat it, and say so with numbers).
* **Distribution trust**: unsigned for now (decided). Azure Artifact Signing ($9.99/month) is open
  to individuals only in the USA and Canada; elsewhere the options are a traditional certificate
  or the Microsoft Store as MSIX, where Microsoft signs for free.

## Sources
* Tauri panic handling: https://aptabase.com/blog/catching-panics-on-tauri-apps ,
  https://github.com/tauri-apps/tauri/discussions/9649
* Tauri updater: https://v2.tauri.app/plugin/updater/ ; installer and WebView2 modes:
  https://v2.tauri.app/distribute/windows-installer/
* UIPI and SendInput: https://codenote.net/en/posts/windows-admin-elevation-blocks-chatgpt-computer-use-uipi/ ,
  https://dev.to/howmindswork/how-i-inject-text-into-any-windows-app-including-elevated-processes-4jl6
* Microphone privacy: https://support.microsoft.com/en-us/windows/privacy/turn-on-app-permissions-for-your-microphone-in-windows ,
  https://sysmansquad.com/2023/01/21/microphone_app_permissions/
* Bluetooth hands-free audio: https://learn.microsoft.com/en-us/windows-hardware/drivers/bluetooth/bluetooth-classic-audio
* CUDA 13 driver: https://forums.developer.nvidia.com/t/compatible-nvidia-gpu-drivers-on-windows-for-cuda-toolkit-13-0/371618
* PyInstaller and non-ASCII paths: https://github.com/pyinstaller/pyinstaller/issues/1295 ;
  antivirus: https://www.pythonguis.com/faq/problems-with-antivirus-software-and-pyinstaller/
* Hugging Face behind firewalls: https://huggingface.co/docs/hub/models-downloading
* Error message guidance: https://learn.microsoft.com/en-us/windows/win32/uxguide/mess-error
* Signing: https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options ,
  https://azure.microsoft.com/en-us/products/artifact-signing ,
  https://learn.microsoft.com/en-us/windows/apps/publish/publish-your-app/msi/app-package-requirements
* Wispr Flow on Windows: https://www.getvoibe.com/resources/is-wispr-flow-reliable/
