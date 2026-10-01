//! LocalFlow's native shell.
//!
//! The shell owns the operating system: the keyboard hook, the microphone, text injection,
//! window context and the tray. The Python engine owns the models. They meet over a local
//! WebSocket (`engine.rs`), and the dictation state machine that joins them lives in
//! `session.rs`.

mod audio;
mod context;
mod diagnostics;
mod downloads;
mod words;
mod e2e;
mod engine;
mod flowbar;
mod guard;
mod health;
mod problems;
mod hotkey;
mod inject;
mod learn;
#[macro_use]
mod log;
mod history;
mod paths;
mod power;
mod selfcheck;
mod selftest;
mod session;
mod settings;
mod soak;
mod tray;
mod win;

use std::sync::Arc;

use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Listener, Manager, WindowEvent};

use crate::audio::Capture;
use crate::guard::LockExt;
use crate::engine::{Engine, Sink};
use crate::hotkey::{Action, Hotkeys};
use crate::session::SessionManager;

pub use crate::engine::Engine as EngineHandle;

/// Engine events, delivered to the webview and to the parts of the shell that act on them.
struct TauriSink(AppHandle);

impl Sink for TauriSink {
    fn emit(&self, event: &str, payload: Value) {
        // The session machine has to see finals and errors before the UI does, because it
        // decides whether the take is over and hands the text to the injector.
        match event {
            "final" => session::on_final(&self.0, &payload),
            "command-result" => session::on_command_result(&self.0, &payload),
            "engine-error" => session::on_error(&self.0, &payload),
            "selfcheck-result" | "selfcheck-repaired" => selfcheck::deliver(&payload),
            "engine-link" if payload.get("link").and_then(Value::as_str) != Some("ready") => {
                on_link_down(&self.0)
            }
            _ => {}
        }
        let mut payload = payload;
        session::for_windows(event, &mut payload);
        let _ = self.0.emit(event, payload);
    }
}

/// The engine went away. The takes it owed wait for the next one (session.rs), and a take
/// still being spoken carries on recording for it.
fn on_link_down(app: &AppHandle) {
    let Some(sessions) = app.try_state::<Arc<SessionManager>>() else { return };
    sessions.on_link_down();
}

/// Everything the commands need, kept in Tauri's state.
pub struct Shell {
    pub capture: Capture,
    pub hotkeys: std::sync::Mutex<Option<Hotkeys>>,
}

#[tauri::command]
fn shell_status(
    engine: tauri::State<'_, Engine>,
    sessions: tauri::State<'_, Arc<SessionManager>>,
    shell: tauri::State<'_, Shell>,
) -> Value {
    json!({
        "link": engine.link(),
        "engine": engine.status(),
        "phase": sessions.phase(),
        "device": shell.capture.device_name(),
        "autostart": win::autostart_enabled(),
        "hotkey": hotkey::Config::load().describe(),
        "audio_live": shell.capture.is_live(),
        "level": shell.capture.level(),
        "version": env!("CARGO_PKG_VERSION"),
        "log": log::location(),
    })
}

/// A fault in the Hub or the flow bar, so it is in the same log as everything else.
#[tauri::command]
fn log_ui_error(page: String, message: String, stack: Option<String>) {
    // From a web page, so bounded: a runaway error loop must not fill the disk.
    let clip = |s: &str, n: usize| s.chars().take(n).collect::<String>();
    shell_log!(
        "UI fault on the {} page: {}\n{}",
        clip(&page, 40),
        clip(&message, 500),
        clip(stack.as_deref().unwrap_or(""), 4000)
    );
}

#[tauri::command]
fn restart_engine(engine: tauri::State<'_, Engine>) {
    engine.restart();
}

#[tauri::command]
fn set_autostart(on: bool) -> bool {
    win::set_autostart(on);
    win::autostart_enabled()
}

/// Everything the Hub renders in one call: stats, history, both sets of settings, and the
/// microphones on offer. One round trip keeps the window from painting in pieces.
#[tauri::command]
fn hub_data(app: AppHandle, engine: tauri::State<'_, Engine>) -> Value {
    let settings = settings::load();
    let entries = if settings.history { history::load(settings.retention_days) } else { Vec::new() };
    let stats = history::stats(&entries);
    // The list can grow to thousands; the Hub pages through the rest on demand.
    let recent: Vec<&history::Entry> = entries.iter().take(200).collect();
    json!({
        // From Cargo.toml, so the Hub can never drift from the build it is part of: it used to
        // carry its own "0.1.0", which would have gone on saying so in every later version.
        "version": env!("CARGO_PKG_VERSION"),
        "settings": settings,
        "engine_config": settings::engine_config(),
        "engine": engine.status(),
        "link": engine.link(),
        "health": health::current(&app),
        "stats": stats,
        "history": recent,
        "history_total": entries.len(),
        "microphones": audio::input_devices(),
        "bluetooth_microphones": win::bluetooth_microphones(),
        // Corrections the user keeps making by hand, offered as dictionary entries.
        "suggestions": learn::suggestions(&entries, &known_terms(&settings)),
        "paths": {
            "settings": settings::path().map(|p| p.display().to_string()),
            "history": history::location(),
            "log": log::location(),
        },
    })
}

/// Terms already in the engine's dictionary, so they are not suggested a second time.
fn known_terms(_settings: &settings::Settings) -> Vec<String> {
    let cfg = settings::engine_config();
    let pp = cfg.get("postprocess");
    let mut terms: Vec<String> = pp
        .and_then(|p| p.get("dictionary_terms"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(str::to_owned).collect())
        .unwrap_or_default();
    if let Some(map) = pp.and_then(|p| p.get("dictionary")).and_then(Value::as_object) {
        terms.extend(map.values().filter_map(Value::as_str).map(str::to_owned));
        terms.extend(map.keys().cloned());
    }
    terms
}

#[tauri::command]
fn clear_history() -> bool {
    history::clear()
}

/// Save the shell's own settings and apply the ones that can change while running.
#[tauri::command]
fn save_settings(app: AppHandle, settings: settings::Settings) -> Result<Value, String> {
    apply_shell_settings(&app, settings)
}

/// Check, save and apply the shell's settings: the Hub's changes, a reset, an import.
fn apply_shell_settings(app: &AppHandle, mut settings: settings::Settings) -> Result<Value, String> {
    let before = settings::load();
    settings.prune_rules();
    // Refused with a reason the Hub shows, rather than saved and found out the hard way.
    let settings = settings.validated()?;
    settings::save(&settings)?;

    // Everything the hook reads - both chords, double tap, Escape - applies at once, without
    // reinstalling it: they are masks and flags it reads on every key event.
    let hotkeys = hotkey::Config::from_settings(&settings);
    hotkey::configure(&hotkeys);
    let chord = settings.chord();
    if !chord.is_empty() && chord != before.chord() {
        shell_log!("hotkey changed to {}", hotkeys.describe());
    }
    let command = settings.command_chord().unwrap_or_default();
    if command != before.command_chord().unwrap_or_default() {
        shell_log!(
            "command hotkey {}",
            if command.is_empty() { "off".to_owned() } else { format!("set ({} keys)", command.len()) }
        );
    }
    if settings.microphone != before.microphone {
        if let Some(shell) = app.try_state::<Shell>() {
            shell.capture.rebuild();
        }
    }
    if settings.retention_days != before.retention_days {
        history::prune(settings.retention_days);
    }
    // The chosen microphone and the chord are part of the status.
    health::refresh(app);
    Ok(json!({"ok": true, "hotkey": hotkey::Config::from_settings(&settings).describe()}))
}

/// Help > Reset preferences: every preference to its default, in the shell and the engine -
/// but not the user's own words (dictionary, snippets, app rules, house style), their history
/// or whether it is kept, or the models (decided 2026-09-28). Onboarding is not shown again.
#[tauri::command]
fn reset_preferences(app: AppHandle, engine: tauri::State<'_, Engine>) -> Result<Value, String> {
    if !engine.is_connected() {
        return Err("the engine isn't running; reset once it is back".into());
    }
    let current = settings::load();
    let fresh = settings::Settings {
        app_rules: current.app_rules,
        onboarded: current.onboarded,
        history: current.history,
        retention_days: current.retention_days,
        ..settings::Settings::default()
    };
    let result = apply_shell_settings(&app, fresh)?;
    if let Some(shell) = app.try_state::<Shell>() {
        shell.capture.rebuild(); // back on the Windows default microphone
    }
    engine.send(json!({"type": "settings.reset"}));
    shell_log!("preferences reset to their defaults; the user's words, history and models kept");
    Ok(result)
}

/// Help > Your words > Export: dictionary, terms, snippets, app rules and house style, one file
/// in Downloads. Returns its path.
#[tauri::command]
fn export_words() -> Result<String, String> {
    let words = words::Words::gather(&settings::engine_config(), &settings::load().app_rules);
    let dir = diagnostics::destination().ok_or("there is no folder to save it in")?;
    let path = dir.join(format!("LocalFlow words {}.json", diagnostics::today()));
    let text = serde_json::to_string_pretty(&words).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    diagnostics::remember(&path);
    shell_log!(
        "words exported: {} dictionary entries, {} terms, {} snippets, {} app rules",
        words.dictionary.len(), words.dictionary_terms.len(), words.snippets.len(), words.app_rules.len()
    );
    Ok(path.display().to_string())
}

/// Help > Your words > Import: the file's contents (read by the Hub), merged in - new entries
/// added, the user's own kept where both have one.
#[tauri::command]
fn import_words(app: AppHandle, engine: tauri::State<'_, Engine>, text: String) -> Result<words::Merged, String> {
    if text.len() > 4 << 20 {
        return Err("that file is far too big to be a words file".into());
    }
    let theirs = words::Words::parse(&text)?;
    if !engine.is_connected() {
        return Err("the engine isn't running; import once it is back".into());
    }
    let current = settings::load();
    let mine = words::Words::gather(&settings::engine_config(), &current.app_rules);
    let (merged, summary) = words::merge(&mine, &theirs);
    if merged.app_rules != current.app_rules {
        apply_shell_settings(&app, settings::Settings { app_rules: merged.app_rules.clone(), ..current })?;
    }
    // The engine checks what it is sent as it checks the Hub's own edits, and saves it.
    engine.send(json!({"type": "settings.set", "postprocess": merged.postprocess_patch()}));
    shell_log!("words imported: {} added, {} of the user's own kept", summary.added, summary.kept);
    Ok(summary)
}

/// "Check LocalFlow": every check, the slow ones (model files, download sites) only when `full`.
#[tauri::command]
async fn run_selfcheck(app: AppHandle, full: bool) -> Value {
    let items = selfcheck::run(&app, full).await;
    health::recheck_settings_folder(&app);
    json!(items)
}

/// "Download again": delete the damaged model files the last full check found and restart the
/// engine, which downloads them afresh.
#[tauri::command]
async fn repair_models(app: AppHandle) -> Result<usize, String> {
    selfcheck::repair(&app).await
}

/// Help > Report a problem: the diagnostics zip (words replaced by their length). Returns its path.
#[tauri::command]
async fn export_diagnostics(app: AppHandle) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || diagnostics::export(&app).map(|p| p.display().to_string()))
        .await
        .map_err(|e| e.to_string())?
}

/// The diagnostics file just made, selected in File Explorer.
#[tauri::command]
fn reveal_diagnostics() -> Result<(), String> {
    diagnostics::reveal()
}

/// GitHub's new-issue page on the public repository, filled in with what is not personal.
#[tauri::command]
fn report_issue(app: AppHandle) -> Result<(), String> {
    diagnostics::open_issue(&app)
}

/// A page of Windows Settings, for a status Fix button. Only pages named here: the Hub never
/// gets to open an arbitrary address.
#[tauri::command]
fn open_windows_settings(page: String) -> Result<(), String> {
    let uri = match page.as_str() {
        "privacy_microphone" => "ms-settings:privacy-microphone",
        "sound" => "ms-settings:sound",
        "storage" => "ms-settings:storagesense",
        "date_time" => "ms-settings:dateandtime",
        _ => return Err(format!("no settings page called {page:?}")),
    };
    std::process::Command::new("explorer.exe").arg(uri).spawn().map(|_| ()).map_err(|e| e.to_string())
}

/// Change the engine's clean-up settings, its speech model (`stt: {model: <key>}`) or its
/// bundled clean-up model (`llm: {model: <key>}`), or where the models run (`compute: {mode,
/// temp_limit_c, idle_release_min}`). Sent
/// over the protocol rather than written to its file, so the running engine applies them at
/// once; the engine then saves them itself. A model change reports its download and load
/// progress through the ordinary status messages.
#[tauri::command]
fn save_engine_settings(
    engine: tauri::State<'_, Engine>,
    postprocess: Option<Value>,
    stt: Option<Value>,
    llm: Option<Value>,
    compute: Option<Value>,
    network: Option<Value>,
) -> Result<(), String> {
    if !engine.is_connected() {
        return Err("the engine is not connected".into());
    }
    let mut msg = json!({"type": "settings.set"});
    if let Some(pp) = postprocess {
        msg["postprocess"] = pp;
    }
    if let Some(stt) = stt {
        msg["stt"] = stt;
    }
    if let Some(llm) = llm {
        msg["llm"] = llm;
    }
    if let Some(compute) = compute {
        msg["compute"] = compute;
    }
    if let Some(network) = network {
        msg["network"] = network;
    }
    engine.send(msg);
    Ok(())
}

/// Whether the window opens when LocalFlow starts. Started by the user: yes. Started by Windows
/// at sign-in: only if they asked for it. Started again after a crash: no - the user is busy
/// in another app, and the window would take the focus from what they are typing; the
/// "restarted" notification says what happened. Setup that has not been done needs the window
/// whatever started it.
fn show_window_at_start(at_sign_in: bool, restarted: bool, onboarded: bool, open_window_at_sign_in: bool) -> bool {
    if !onboarded {
        return true;
    }
    !restarted && (!at_sign_in || open_window_at_sign_in)
}

/// Setup was finished before the speech model was ready: say so, once, when it is (M5).
#[tauri::command]
fn notify_when_ready(chord: String) {
    downloads::tell_when_ready(chord.chars().take(40).collect());
}

/// The model library (Hub > Models): download a model without switching to it, stop a
/// download, or remove a model. The engine answers with a status, or with an error the Hub shows.
#[tauri::command]
fn model_action(
    engine: tauri::State<'_, Engine>,
    action: String,
    kind: Option<String>,
    key: Option<String>,
    id: Option<String>,
) -> Result<(), String> {
    if !engine.is_connected() {
        return Err("the engine is not connected".into());
    }
    engine.send(model_message(&action, kind.as_deref(), key.as_deref(), id.as_deref())?);
    Ok(())
}

fn model_message(action: &str, kind: Option<&str>, key: Option<&str>, id: Option<&str>) -> Result<Value, String> {
    let short = |s: Option<&str>, what: &str| match s {
        Some(v) if !v.is_empty() && v.len() <= 64 => Ok(v.to_owned()),
        _ => Err(format!("a model action needs its {what}")),
    };
    match action {
        "download" | "remove" => {
            let kind = short(kind, "kind")?;
            if kind != "speech" && kind != "cleanup" {
                return Err(format!("no such kind of model: {kind}"));
            }
            Ok(json!({"type": format!("models.{action}"), "kind": kind, "key": short(key, "key")?}))
        }
        "cancel" => Ok(json!({"type": "models.cancel", "id": short(id, "download")?})),
        other => Err(format!("no such model action: {other}")),
    }
}

/// Minimal RIFF/WAVE reader: 16-bit PCM only, downmixed to mono. The benchmark corpus is
/// already 16 kHz, and anything else is rejected rather than silently resampled.
pub(crate) fn read_wav_16k_mono(path: &str) -> anyhow::Result<Vec<i16>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("not a WAV file");
    }
    let mut pos = 12;
    let (mut channels, mut rate, mut bits) = (0u16, 0u32, 0u16);
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into()?) as usize;
        let body = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        match id {
            b"fmt " if body.len() >= 16 => {
                channels = u16::from_le_bytes(body[2..4].try_into()?);
                rate = u32::from_le_bytes(body[4..8].try_into()?);
                bits = u16::from_le_bytes(body[14..16].try_into()?);
            }
            b"data" => data = Some(body),
            _ => {}
        }
        pos += 8 + size + (size & 1);
    }
    let data = data.ok_or_else(|| anyhow::anyhow!("no data chunk"))?;
    if bits != 16 {
        anyhow::bail!("only 16-bit PCM is supported, this file is {bits}-bit");
    }
    if rate != audio::SAMPLE_RATE {
        anyhow::bail!("expected {} Hz audio, this file is {rate} Hz", audio::SAMPLE_RATE);
    }
    let samples: Vec<i16> = data
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();
    if channels <= 1 {
        return Ok(samples);
    }
    Ok(samples
        .chunks(channels as usize)
        .map(|f| (f.iter().map(|s| *s as i32).sum::<i32>() / f.len() as i32) as i16)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_in_starts_quietly_in_the_tray_unless_asked_otherwise() {
        assert!(show_window_at_start(false, false, true, false), "opened by the user: the window");
        assert!(!show_window_at_start(true, false, true, false), "at sign-in: the tray only");
        assert!(show_window_at_start(true, false, true, true), "at sign-in, window asked for");
        assert!(show_window_at_start(true, false, false, false), "setup not done yet: it needs the window");
    }

    #[test]
    fn a_restart_after_a_crash_stays_in_the_tray() {
        assert!(!show_window_at_start(false, true, true, false), "opened by the user, then crashed");
        assert!(!show_window_at_start(true, true, true, true), "even with the window asked for at sign-in");
        assert!(show_window_at_start(false, true, false, false), "setup not done yet: it needs the window");
    }

    #[test]
    fn model_actions_become_engine_messages_and_nothing_else_does() {
        assert_eq!(
            model_message("download", Some("cleanup"), Some("phi-4-mini"), None).unwrap(),
            json!({"type": "models.download", "kind": "cleanup", "key": "phi-4-mini"})
        );
        assert_eq!(
            model_message("remove", Some("speech"), Some("parakeet-v2"), None).unwrap(),
            json!({"type": "models.remove", "kind": "speech", "key": "parakeet-v2"})
        );
        assert_eq!(model_message("cancel", None, None, Some("d3")).unwrap(), json!({"type": "models.cancel", "id": "d3"}));
        assert!(model_message("shutdown", Some("speech"), Some("x"), None).is_err());
        assert!(model_message("download", Some("tools"), Some("x"), None).is_err());
        assert!(model_message("remove", Some("speech"), None, None).is_err());
        assert!(model_message("cancel", None, None, Some(&"x".repeat(65))).is_err());
    }

    fn wav(channels: u16, rate: u32, bits: u16, samples: &[i16]) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
        out.extend_from_slice(&(channels * 2).to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(&data);
        out
    }

    fn write(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn reads_mono_and_downmixes_stereo() {
        let mono = write("localflow-mono.wav", &wav(1, 16_000, 16, &[1, -2, 3]));
        assert_eq!(read_wav_16k_mono(mono.to_str().unwrap()).unwrap(), vec![1, -2, 3]);

        let stereo = write("localflow-stereo.wav", &wav(2, 16_000, 16, &[10, 20, -30, -10]));
        assert_eq!(read_wav_16k_mono(stereo.to_str().unwrap()).unwrap(), vec![15, -20]);
    }

    /// Silently resampling would make a latency measurement a lie, so the wrong rate is an
    /// error rather than a conversion.
    #[test]
    fn refuses_audio_it_would_have_to_convert() {
        let wrong = write("localflow-44k.wav", &wav(1, 44_100, 16, &[1, 2, 3]));
        let err = read_wav_16k_mono(wrong.to_str().unwrap()).unwrap_err().to_string();
        assert!(err.contains("44100"), "{err}");

        let not_wav = write("localflow-not.wav", b"this is not a wav file at all, not even close");
        assert!(read_wav_16k_mono(not_wav.to_str().unwrap()).is_err());
    }
}

/// `--selftest <wav>`: stream a file through the engine and print the result.
pub fn selftest_dictate(wav: &str, realtime: bool) -> i32 {
    selftest::dictate(wav, realtime)
}

/// `--inject-test [--type|--paste]`: put text into a Notepad we open, then read it back.
pub fn selftest_injection(forced: Option<&str>) -> i32 {
    selftest::injection(match forced {
        Some("type") => Some(inject::Method::Type),
        Some("paste") => Some(inject::Method::Paste),
        _ => None,
    })
}

/// `--batching-probe`: try each way of sending key events and report which survive.
pub fn selftest_batching() -> i32 {
    selftest::batching_probe()
}

/// `--type-probe <text>...`: type each string into Notepad and print what came back.
pub fn selftest_type_probe(samples: &[String]) -> i32 {
    selftest::type_probe(samples)
}

/// `--hook-probe <text>`: compare what we sent with what Windows delivered.
pub fn selftest_hook_probe(text: &str) -> i32 {
    selftest::hook_probe(text)
}

/// `--stress <minutes> <wav>`: phase 3's acceptance test for the hook under GPU load.
pub fn selftest_stress(minutes: u64, wav: &str) -> i32 {
    selftest::stress(minutes, wav)
}

/// `--report`: print what the shell can see of the engine and the focused window.
pub fn selftest_report() -> i32 {
    selftest::report()
}

/// Quit for real: stop the engine we own, then exit.
pub fn shutdown<R: tauri::Runtime>(app: &AppHandle<R>) {
    if let Some(engine) = app.try_state::<Engine>() {
        engine.shutdown();
    }
    if let Some(shell) = app.try_state::<Shell>() {
        shell.capture.stop();
        if let Some(hotkeys) = shell.hotkeys.locked().take() {
            hotkeys.stop();
        }
    }
    // Give the engine a moment to die with its job object rather than being orphaned.
    std::thread::sleep(std::time::Duration::from_millis(150));
    app.exit(0);
}

// ---------------------------------------------------------------------------------------------

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // First, so that even a fault during start-up is written down.
    guard::install_panic_hook();
    guard::install_fault_handler();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let restarted = args.iter().any(|a| a == guard::RESTARTED_ARG);
    let at_sign_in = args.iter().any(|a| a == win::AUTOSTART_FLAG);
    let crash_test = args.iter().position(|a| a == "--crash-test").map(|i| {
        let seconds = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(65);
        let kind = args.get(i + 2).cloned().unwrap_or_default();
        (seconds, kind)
    });
    guard::wait_for_predecessor(&args);
    // `--e2e [scenario...]`: the end-to-end harness (e2e.rs). It runs beside the user's own copy,
    // so it claims nothing that copy owns: no single-instance mutex, keyboard hook, tray icon or
    // show-window event, and it attaches to that copy's engine rather than start a second one.
    let harness = args.iter().position(|a| a == "--e2e").map(|i| args[i + 1..].to_vec());
    if harness.is_some() {
        e2e::activate();
    }

    // Before anything else claims a keyboard hook or a microphone. A second copy would install
    // a second low-level hook on the same chord, so one press would start two dictations and
    // inject the result twice - and the usual way that happens is not deliberate, it is
    // double-clicking a shortcut for an app that is already sitting in the tray.
    if harness.is_none() && !win::claim_single_instance() {
        shell_log!("another copy of LocalFlow is already running; this one is exiting");
        if !win::request_show() {
            win::show_existing_window();
        }
        return;
    }
    if restarted {
        shell_log!("started again after LocalFlow closed unexpectedly");
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .setup(move |app| {
            let handle = app.handle().clone();
            // Read once, for the history clean-up below and whether to show the window.
            let start_settings = harness.is_none().then(settings::load);

            if let Some(s) = &start_settings {
                // A sign-in entry left pointing at a program that has moved or been replaced
                // fails silently once per sign-in, so fix it before anything else needs attention.
                win::repair_autostart();

                // History past its retention window is deleted from disk, not merely hidden.
                history::prune_if_due(s.retention_days);

                // A second launch (a Start-menu click while this copy sits in the tray) asks for
                // the window instead of starting another copy.
                let shower = handle.clone();
                win::on_show_request(move || {
                    let app = shower.clone();
                    let _ = shower.run_on_main_thread(move || tray::show_window(&app));
                });
            }

            let mut keys_for_harness = None;
            let engine = Engine::start(Arc::new(TauriSink(handle.clone())));
            let sessions = SessionManager::new(handle.clone(), engine.clone());
            let (capture, tape) = if harness.is_some() {
                let (capture, tape) = Capture::start_scripted(handle.clone(), sessions.clone());
                (capture, Some(tape))
            } else {
                (Capture::start(handle.clone(), sessions.clone()), None)
            };

            app.manage(engine.clone());
            app.manage(sessions.clone());

            // The hotkey drives everything: context, then the session, then the microphone.
            let hotkeys = {
                let sessions = sessions.clone();
                let capture = capture.clone();
                let handle = handle.clone();
                let cfg = hotkey::Config::load();
                let chord = cfg.describe();
                let _ = handle.emit("hotkey", json!({"chord": chord}));
                let act = move |action| on_hotkey(&handle, &sessions, &capture, action);
                if harness.is_some() {
                    let (hotkeys, keys) = Hotkeys::scripted(cfg, act);
                    keys_for_harness = Some(keys);
                    hotkeys
                } else {
                    Hotkeys::install(cfg, act)
                }
            };

            app.manage(Shell { capture, hotkeys: std::sync::Mutex::new(Some(hotkeys)) });

            if harness.is_none() {
                tray::create(&handle)?;
            }
            flowbar::create(&handle)?;
            wire_events(&handle);
            health::start(&handle, harness.is_some());
            // The microphone may have spoken before anyone was listening (no microphone at
            // all fails at once): tell the status model what it last said.
            if let Some(msg) = audio::last_device_event() {
                apply_mic(&handle, &msg);
            }
            power::watch(handle.clone());

            if let (Some(names), Some(keys), Some(tape)) = (harness.clone(), keys_for_harness.take(), tape) {
                e2e::start(handle.clone(), keys, tape, names);
            }

            // The window starts hidden (tauri.conf.json) and is shown here - unless Windows
            // started LocalFlow at sign-in, when it waits quietly in the tray, ready to dictate.
            // The harness never shows it: it would only take the focus from the window the
            // harness types into.
            if let Some(s) = &start_settings {
                if show_window_at_start(at_sign_in, restarted, s.onboarded, s.open_window_at_sign_in) {
                    tray::show_window(&handle);
                } else if restarted {
                    shell_log!("started again after a crash: waiting in the tray");
                } else {
                    shell_log!("started at sign-in: waiting in the tray");
                }
            }

            if restarted {
                let said = problems::show(problems::APP_RESTARTED, &[]);
                notify(&handle, &said.title, &said.message);
            }
            if let Some((seconds, kind)) = &crash_test {
                guard::schedule_crash_test(&handle, std::time::Duration::from_secs(*seconds), kind);
            }

            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the window leaves the app running in the tray, like every other
            // background dictation tool.
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .invoke_handler(tauri::generate_handler![
            shell_status,
            log_ui_error,
            restart_engine,
            set_autostart,
            hub_data,
            clear_history,
            save_settings,
            save_engine_settings,
            model_action,
            notify_when_ready,
            open_windows_settings,
            export_diagnostics,
            reset_preferences,
            export_words,
            import_words,
            reveal_diagnostics,
            report_issue,
            run_selfcheck,
            repair_models,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn on_hotkey(
    app: &AppHandle,
    sessions: &Arc<SessionManager>,
    capture: &Capture,
    action: Action,
) {
    match action {
        Action::Start | Action::StartHandsFree => {
            if matches!(action, Action::StartHandsFree) {
                // The first tap of the double tap started a take; it was not speech.
                sessions.cancel();
            }
            // The context is captured before the session starts, because the moment text is
            // injected the user may have moved on, and because the engine's style profile
            // depends on which app is in front right now.
            let mut ctx = context::foreground();
            context::enrich(&mut ctx);
            context::remember(&ctx);
            let settings = settings::load();
            let rule = settings.rule_for(&ctx.app);
            if rule.disabled {
                // Silently doing nothing looks identical to a broken hotkey, so say why.
                shell_log!("hotkey ignored: dictation is turned off in {}", ctx.app);
                let name = ctx.app.trim_end_matches(".exe");
                let _ = app.emit("notice", json!({"code": problems::DICTATION_OFF_IN_APP.as_str(),
                    "text": problems::bar(problems::DICTATION_OFF_IN_APP, &[("app", name)])}));
                return;
            }
            let mut context = ctx.to_json();
            if !rule.profile.is_empty() && rule.profile != "auto" {
                // The engine guesses a style from the app name; a rule overrules the guess.
                context["profile"] = json!(rule.profile);
            }
            match sessions.begin(context, None) {
                Some(id) => {
                    shell_log!(
                        "{} {} in {} ({})",
                        id,
                        if matches!(action, Action::StartHandsFree) { "hands-free" } else { "recording" },
                        if ctx.app.is_empty() { "?" } else { &ctx.app },
                        ctx.title
                    );
                    capture.begin();
                    hotkey::set_recording(true);
                    // Push-to-talk ends when the chord is released; a latched take has no such
                    // signal, so it needs something to notice the user has stopped talking.
                    if matches!(action, Action::StartHandsFree) && settings.hands_free_timeout_s > 0 {
                        session::watch_hands_free(
                            app,
                            &id,
                            std::time::Duration::from_secs(settings.hands_free_timeout_s),
                        );
                    }
                }
                None => shell_log!(
                    "hotkey ignored: {}",
                    if sessions.engine_connected() {
                        "the speech model is still loading"
                    } else {
                        "the engine is not connected"
                    }
                ),
            }
        }
        Action::PasteLast => inject::paste_last(app),
        Action::StartCommand => {
            let mut ctx = context::foreground();
            context::enrich(&mut ctx);
            context::remember(&ctx);
            // Read the selection now, while the window is definitely still focused. UI
            // Automation cannot see into a lot of Electron and the web, and the fallback is to
            // copy it - but that cannot happen until the chord is released, so it waits until
            // the take is over.
            let selection = ctx.selection.clone();
            if settings::load().rule_for(&ctx.app).disabled {
                shell_log!("command hotkey ignored: dictation is turned off in {}", ctx.app);
                let name = ctx.app.trim_end_matches(".exe");
                let _ = app.emit("notice", json!({"code": problems::DICTATION_OFF_IN_APP.as_str(),
                    "text": problems::bar(problems::DICTATION_OFF_IN_APP, &[("app", name)])}));
                return;
            }
            match sessions.begin_command(ctx.to_json(), selection.clone()) {
                Some(id) => {
                    shell_log!(
                        "{} command in {} ({}) | selection: {}",
                        id,
                        if ctx.app.is_empty() { "?" } else { &ctx.app },
                        ctx.title,
                        match &selection {
                            Some(s) => format!("{} chars from UI Automation", s.chars().count()),
                            None => "not visible, will copy after the take".to_owned(),
                        }
                    );
                    capture.begin();
                    hotkey::set_recording(true);
                }
                None => shell_log!("command hotkey ignored: the engine is not ready"),
            }
        }
        Action::Stop => {
            capture.end();
            hotkey::set_recording(false);
            sessions.finish();
        }
        Action::Cancel => {
            capture.end();
            hotkey::set_recording(false);
            sessions.cancel();
            shell_log!("cancelled");
            let _ = app.emit("cancelled", json!({}));
        }
    }
}

/// Keep the tray icon and the flow bar showing what is actually happening.
fn wire_events(app: &AppHandle) {
    let handle = app.clone();
    app.listen("phase", move |event| {
        let Ok(msg) = serde_json::from_str::<Value>(event.payload()) else { return };
        let phase = msg.get("phase").and_then(Value::as_str).unwrap_or("");
        match phase {
            "recording" => tray::set_state(&handle, tray::State::Recording, None),
            "finishing" => tray::set_state(&handle, tray::State::Processing, None),
            _ => health::show_in_tray(&handle),
        }
        flowbar::on_phase(&handle, phase);
    });

    // A cancelled take never reaches the idle phase through `final`, so the bar is told
    // directly or it would hang on screen until the next dictation.
    let handle = app.clone();
    app.listen("cancelled", move |_| flowbar::on_phase(&handle, "idle"));

    // The tray's Paste last dictation: let its menu close, then go back to the window the text
    // is for - clicking the tray took the front from it.
    let handle = app.clone();
    app.listen("paste-last-request", move |_| {
        let app = handle.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(350));
            inject::paste_last_from_tray(&app);
        });
    });

    // The hotkey was pressed and no take could start (the engine restarting, the speech model
    // loading, no microphone, dictation off in this app), or text was kept rather than typed.
    // The bar says why rather than letting it look like nothing happened.
    let handle = app.clone();
    app.listen("notice", move |event| {
        let hold = serde_json::from_str::<Value>(event.payload())
            .ok()
            .and_then(|m| m.get("hold_ms").and_then(Value::as_u64));
        flowbar::show_notice(&handle, hold.map(std::time::Duration::from_millis));
    });

    // The status model (health.rs) decides what the tray shows and what is announced: an
    // engine that cannot start, safe mode, a model that failed, a microphone that went away.
    let handle = app.clone();
    let safe = std::sync::Mutex::new(false);
    app.listen("engine-link", move |_| {
        let link = handle.state::<Engine>().link();
        let mut was = safe.locked();
        if link.safe_mode != *was {
            *was = link.safe_mode;
            tray::set_safe_mode(&handle, link.safe_mode);
        }
        drop(was);
        health::refresh(&handle);
    });

    let handle = app.clone();
    app.listen("engine-status", move |event| {
        health::refresh(&handle);
        // the tray's download ring and the "downloaded" notifications (M3)
        if let Ok(status) = serde_json::from_str::<Value>(event.payload()) {
            downloads::on_status(&handle, &status);
        }
    });

    let handle = app.clone();
    app.listen("audio-device", move |event| {
        if let Ok(msg) = serde_json::from_str::<Value>(event.payload()) {
            apply_mic(&handle, &msg);
        }
    });
}

/// A microphone announcement ("audio-device") into the status model.
fn apply_mic(app: &AppHandle, msg: &Value) {
    let text = |k: &str| msg.get(k).and_then(Value::as_str).map(str::to_owned);
    let ok = msg.get("ok").and_then(Value::as_bool).unwrap_or(false);
    let flag = |k: &str| msg.get(k).and_then(Value::as_bool).unwrap_or(false);
    // When it failed, "device" is the one that could not be opened, for the words.
    health::set_mic(app, text("device"), if ok { None } else { text("error") }, flag("busy"), flag("bluetooth"));
}

/// A Windows toast. Best effort: an unsigned development build has no registered app id, so
/// this can be silently dropped, which is why the tray icon carries the same information.
pub(crate) fn notify(app: &AppHandle, title: &str, body: &str) {
    use tauri_plugin_notification::NotificationExt;
    let _ = app.notification().builder().title(title).body(body).show();
}
