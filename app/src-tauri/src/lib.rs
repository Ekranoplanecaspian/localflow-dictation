//! LocalFlow's native shell.
//!
//! The shell owns the operating system: the keyboard hook, the microphone, text injection,
//! window context and the tray. The Python engine owns the models. They meet over a local
//! WebSocket (`engine.rs`), and the dictation state machine that joins them lives in
//! `session.rs`.

mod audio;
mod context;
mod engine;
mod flowbar;
mod hotkey;
mod inject;
mod learn;
#[macro_use]
mod log;
mod history;
mod paths;
mod selftest;
mod session;
mod settings;
mod tray;
mod win;

use std::sync::Arc;

use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Listener, Manager, WindowEvent};

use crate::audio::Capture;
use crate::engine::{Engine, Link, Sink};
use crate::hotkey::{Action, Hotkeys};
use crate::session::{Phase, SessionManager};

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
            _ => {}
        }
        let _ = self.0.emit(event, payload);
    }
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

#[tauri::command]
fn restart_engine(engine: tauri::State<'_, Engine>) {
    engine.restart();
}

#[tauri::command]
fn set_autostart(on: bool) -> bool {
    win::set_autostart(on);
    win::autostart_enabled()
}

/// Type a sample into whatever is focused. The self-test for phase 3.4: run it with each of the
/// target apps in front and read the result.
///
/// It waits before looking at the foreground window, because clicking the button in this window
/// makes *this* window the foreground one - the first version typed its sample into LocalFlow's
/// own page every time, which looks exactly like injection being broken.
#[tauri::command]
async fn inject_self_test(text: Option<String>, delay_ms: Option<u64>) -> Result<Value, String> {
    let text = text.unwrap_or_else(|| "LocalFlow injection self-test 1 2 3.".to_owned());
    tokio::time::sleep(std::time::Duration::from_millis(delay_ms.unwrap_or(4000))).await;

    tauri::async_runtime::spawn_blocking(move || {
        let ctx = context::foreground();
        if ctx.app == "app.exe" || ctx.app == "localflow.exe" {
            return Err(
                "LocalFlow is still the focused window - click into the app you want to test \
                 while the countdown runs"
                    .to_owned(),
            );
        }
        let method = inject::method_for(&ctx.app, &text);
        let started = std::time::Instant::now();
        let used = inject::inject(&text, method, &ctx.app).map_err(|e| e.to_string())?;
        Ok(json!({
            "app": ctx.app,
            "title": ctx.title,
            "method": if used == inject::Method::Paste { "paste" } else { "type" },
            "chars": text.chars().count(),
            "ms": started.elapsed().as_millis() as u64,
        }))
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Everything the Hub renders in one call: stats, history, both sets of settings, and the
/// microphones on offer. One round trip keeps the window from painting in pieces.
#[tauri::command]
fn hub_data(engine: tauri::State<'_, Engine>) -> Value {
    let settings = settings::load();
    let entries = if settings.history { history::load(settings.retention_days) } else { Vec::new() };
    let stats = history::stats(&entries);
    // The list can grow to thousands; the Hub pages through the rest on demand.
    let recent: Vec<&history::Entry> = entries.iter().take(200).collect();
    json!({
        "settings": settings,
        "engine_config": settings::engine_config(),
        "engine": engine.status(),
        "link": engine.link(),
        "stats": stats,
        "history": recent,
        "history_total": entries.len(),
        "microphones": audio::input_devices(),
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

/// More of the history list, oldest-ward from `offset`.
#[tauri::command]
fn history_page(offset: usize, limit: usize) -> Value {
    let settings = settings::load();
    if !settings.history {
        return json!([]);
    }
    let entries = history::load(settings.retention_days);
    let page: Vec<&history::Entry> = entries.iter().skip(offset).take(limit.min(500)).collect();
    json!(page)
}

#[tauri::command]
fn clear_history() -> bool {
    history::clear()
}

/// Save the shell's own settings and apply the ones that can change while running.
#[tauri::command]
fn save_settings(app: AppHandle, mut settings: settings::Settings) -> Result<Value, String> {
    let before = settings::load();
    settings.prune_rules();
    let settings = settings;
    settings::save(&settings)?;

    // The chord can change without reinstalling the hook: it is just a mask.
    let chord = settings.chord();
    if !chord.is_empty() && chord != before.chord() {
        hotkey::set_chord(&chord);
        shell_log!("hotkey changed to {}", hotkey::Config::from_settings(&settings).describe());
    }
    let command = settings.command_chord().unwrap_or_default();
    if command != before.command_chord().unwrap_or_default() {
        hotkey::set_command_chord(&command);
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
    Ok(json!({"ok": true, "hotkey": hotkey::Config::from_settings(&settings).describe()}))
}

/// Change the engine's clean-up settings. Sent over the protocol rather than written to its
/// file, so the running engine applies them at once; the engine then saves them itself.
#[tauri::command]
fn save_engine_settings(engine: tauri::State<'_, Engine>, postprocess: Value) -> Result<(), String> {
    if !engine.is_connected() {
        return Err("the engine is not connected".into());
    }
    engine.send(json!({"type": "settings.set", "postprocess": postprocess}));
    Ok(())
}

/// What the shell can see about the focused window, including UI Automation.
#[tauri::command]
fn peek_context() -> Value {
    let mut ctx = context::foreground();
    context::enrich(&mut ctx);
    ctx.to_json()
}

/// Push a WAV file through the engine exactly as a dictation would go, and return the final
/// message. This is how the engine link is verified without touching the microphone.
#[tauri::command]
async fn dictate_wav(
    app: AppHandle,
    path: String,
    realtime: Option<bool>,
) -> Result<Value, String> {
    let pcm = read_wav_16k_mono(&path).map_err(|e| e.to_string())?;
    let sessions = app.state::<Arc<SessionManager>>().inner().clone();
    let id = sessions
        .begin(json!({"app": "bench.exe", "title": "dictate_wav"}), None)
        .ok_or_else(|| "the engine is not connected".to_owned())?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    sessions.set_waiter(&id, tx);

    let frame = audio::SAMPLE_RATE as usize / 50; // 20 ms
    let pace = realtime.unwrap_or(true);
    for chunk in pcm.chunks(frame) {
        sessions.feed(chunk);
        if pace {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    }
    sessions.finish();
    match tokio::time::timeout(std::time::Duration::from_secs(120), rx).await {
        Ok(Ok(msg)) => Ok(msg),
        Ok(Err(_)) => Err("the session ended without a result".into()),
        Err(_) => Err("timed out waiting for the engine".into()),
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
        if let Some(hotkeys) = shell.hotkeys.lock().unwrap().take() {
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
    // Before anything else claims a keyboard hook or a microphone. A second copy would install
    // a second low-level hook on the same chord, so one press would start two dictations and
    // inject the result twice - and the usual way that happens is not deliberate, it is
    // double-clicking a shortcut for an app that is already sitting in the tray.
    if !win::claim_single_instance() {
        shell_log!("another copy of LocalFlow is already running; this one is exiting");
        win::show_existing_window();
        return;
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_notification::init())
        .setup(|app| {
            let handle = app.handle().clone();

            // A sign-in entry left pointing at a program that has moved or been replaced fails
            // silently once per sign-in, so fix it before anything else needs attention.
            win::repair_autostart();

            let engine = Engine::start(Arc::new(TauriSink(handle.clone())));
            let sessions = SessionManager::new(handle.clone(), engine.clone());
            let capture = Capture::start(handle.clone(), sessions.clone());

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
                Hotkeys::install(cfg, move |action| {
                    on_hotkey(&handle, &sessions, &capture, action)
                })
            };

            app.manage(Shell { capture, hotkeys: std::sync::Mutex::new(Some(hotkeys)) });

            tray::create(&handle)?;
            flowbar::create(&handle)?;
            wire_events(&handle);

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
            restart_engine,
            set_autostart,
            inject_self_test,
            peek_context,
            dictate_wav,
            hub_data,
            history_page,
            clear_history,
            save_settings,
            save_engine_settings,
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
                let _ = app.emit("app-disabled", json!({"app": ctx.app}));
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
                let _ = app.emit("app-disabled", json!({"app": ctx.app}));
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
        let state = match phase {
            "recording" => tray::State::Recording,
            "finishing" => tray::State::Processing,
            _ => engine_tray_state(&handle),
        };
        tray::set_state(&handle, state, None);
        flowbar::on_phase(&handle, phase);
    });

    // A cancelled take never reaches the idle phase through `final`, so the bar is told
    // directly or it would hang on screen until the next dictation.
    let handle = app.clone();
    app.listen("cancelled", move |_| flowbar::on_phase(&handle, "idle"));

    // The hotkey was pressed before the speech model finished loading. Nothing was recorded,
    // so say so on the bar rather than letting the press look like it did nothing.
    let handle = app.clone();
    app.listen("engine-warming", move |_| flowbar::on_phase(&handle, "warming"));

    // The hotkey was pressed somewhere the user turned dictation off. Nothing happened, and a
    // hotkey that does nothing is indistinguishable from a broken one.
    let handle = app.clone();
    app.listen("app-disabled", move |_| flowbar::on_phase(&handle, "warming"));

    let handle = app.clone();
    let last_told = std::sync::Mutex::new(false);
    app.listen("engine-link", move |_| {
        let link = handle.state::<Engine>().link();
        if matches!(handle.state::<Arc<SessionManager>>().phase(), Phase::Idle) {
            let state = engine_tray_state(&handle);
            tray::set_state(&handle, state, link.detail.as_deref().or(Some("")));
        }
        // Tell the user once when dictation stops being possible, and once when it comes
        // back. A background app that silently does nothing is the worst kind.
        let broken = matches!(link.link, Link::Failed);
        let mut told = last_told.lock().unwrap();
        if broken != *told {
            *told = broken;
            notify(
                &handle,
                if broken { "LocalFlow can't start its engine" } else { "LocalFlow is ready again" },
                &link.detail.clone().unwrap_or_else(|| "Dictation is available.".into()),
            );
        }
    });

    let handle = app.clone();
    app.listen("engine-status", move |_| {
        if matches!(handle.state::<Arc<SessionManager>>().phase(), Phase::Idle) {
            tray::set_state(&handle, engine_tray_state(&handle), None);
        }
    });
}

/// A Windows toast. Best effort: an unsigned development build has no registered app id, so
/// this can be silently dropped, which is why the tray icon carries the same information.
fn notify(app: &AppHandle, title: &str, body: &str) {
    use tauri_plugin_notification::NotificationExt;
    let _ = app.notification().builder().title(title).body(body).show();
}

fn engine_tray_state(app: &AppHandle) -> tray::State {
    let engine = app.state::<Engine>();
    match engine.link().link {
        Link::Ready if engine.stt_ready() => tray::State::Idle,
        Link::Ready => tray::State::Loading,
        Link::Starting | Link::Connecting => tray::State::Loading,
        Link::Reconnecting => tray::State::Offline,
        Link::Failed => tray::State::Error,
        Link::Stopped => tray::State::Offline,
    }
}
