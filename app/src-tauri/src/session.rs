//! The dictation state machine: what happens between pressing the hotkey and text appearing.
//!
//! `Idle -> Recording -> Finishing -> Idle`. The hook thread and the audio callback both call
//! in here, so every method is short and never blocks: the engine handle queues, it does not
//! wait.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::Serialize;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::oneshot;

use crate::engine::Engine;

/// Presses shorter than this are accidental, not dictation.
const MIN_TAKE: std::time::Duration = std::time::Duration::from_millis(300);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    /// Nothing is being dictated.
    Idle,
    /// The hotkey is held (or hands-free is on): audio is streaming to the engine.
    Recording,
    /// The hotkey is released and we are waiting for the final text.
    Finishing,
}

/// What the words being spoken are *for*.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    /// Type them at the caret.
    Dictate,
    /// Treat them as an instruction for the text the user had selected when they started.
    /// The selection is captured at the start when UI Automation can see it, and copied at the
    /// end when it cannot - by then the chord is released, so Ctrl+C means Ctrl+C.
    Command { selection: Option<String> },
}

struct Active {
    id: String,
    mode: Mode,
    started: Instant,
    /// Frames sent since `session.start`, for the "did we actually hear anything" check.
    frames: u64,
    /// A caller waiting for this session's final text (the WAV test command).
    waiter: Option<oneshot::Sender<Value>>,
}

pub struct SessionManager {
    engine: Engine,
    app: AppHandle,
    active: Mutex<Option<Active>>,
    phase: Mutex<Phase>,
    counter: AtomicU64,
    /// Finals that arrive for a session we already forgot (a cancel racing a final).
    waiters: Mutex<HashMap<String, oneshot::Sender<Value>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PhaseEvent {
    pub phase: Phase,
    pub id: Option<String>,
}

impl SessionManager {
    pub fn new(app: AppHandle, engine: Engine) -> Arc<Self> {
        Arc::new(Self {
            engine,
            app,
            active: Mutex::new(None),
            phase: Mutex::new(Phase::Idle),
            counter: AtomicU64::new(0),
            waiters: Mutex::new(HashMap::new()),
        })
    }

    pub fn phase(&self) -> Phase {
        *self.phase.lock().unwrap()
    }

    /// Whether the link is up, which is not the same as being able to dictate: the models
    /// load for some time after it comes up.
    pub fn engine_connected(&self) -> bool {
        self.engine.is_connected()
    }

    fn set_phase(&self, phase: Phase, id: Option<String>) {
        *self.phase.lock().unwrap() = phase;
        let _ = self.app.emit("phase", PhaseEvent { phase, id });
    }

    /// Start a dictation. Returns the session id, or None if the engine cannot take one.
    pub fn begin(&self, context: Value, language: Option<String>) -> Option<String> {
        self.begin_in(context, language, Mode::Dictate)
    }

    /// Start a command-mode take. The spoken words become an instruction rather than text.
    pub fn begin_command(&self, context: Value, selection: Option<String>) -> Option<String> {
        self.begin_in(context, None, Mode::Command { selection })
    }

    fn begin_in(&self, context: Value, language: Option<String>, mode: Mode) -> Option<String> {
        if !self.engine.is_connected() {
            let _ = self.app.emit(
                "engine-error",
                json!({"code": "not_connected", "message": "the engine is not connected yet"}),
            );
            return None;
        }
        // The link comes up in a second but the speech model takes ten or twenty more, and a
        // take started in that window used to be sent, refused by the engine, and dropped -
        // the first dictation after every cold start failed with nothing on screen to say
        // why. Refuse it here instead, where we can say so.
        if !self.engine.stt_ready() {
            let _ = self.app.emit("engine-warming", json!({}));
            return None;
        }
        // A second press while one is still finishing is normal impatience: let the previous
        // one land on its own and start a new session anyway.
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let id = format!("s{n}");
        self.engine.session_start(&id, context, language);
        *self.active.lock().unwrap() =
            Some(Active { id: id.clone(), mode, started: Instant::now(), frames: 0, waiter: None });
        self.set_phase(Phase::Recording, Some(id.clone()));
        Some(id)
    }

    /// Audio from the capture thread. Silently dropped when nothing is recording.
    pub fn feed(&self, pcm: &[i16]) {
        let mut active = self.active.lock().unwrap();
        let Some(a) = active.as_mut() else { return };
        a.frames += pcm.len() as u64;
        drop(active);
        self.engine.send_audio(pcm);
    }

    /// The hotkey was released. A press too short to be speech is thrown away rather than
    /// sent: it is almost always a mistyped shortcut.
    pub fn finish(&self) {
        let (id, too_short) = {
            let active = self.active.lock().unwrap();
            match active.as_ref() {
                Some(a) => (Some(a.id.clone()), a.started.elapsed() < MIN_TAKE && a.waiter.is_none()),
                None => (None, false),
            }
        };
        let Some(id) = id else { return };
        if too_short {
            self.cancel();
            return;
        }
        self.engine.session_end(&id);
        self.set_phase(Phase::Finishing, Some(id));
    }

    pub fn cancel(&self) {
        let taken = self.active.lock().unwrap().take();
        if let Some(a) = taken {
            self.engine.session_cancel(&a.id);
        }
        self.set_phase(Phase::Idle, None);
    }

    /// How long the current take has been running, for the flow bar.
    pub fn elapsed_ms(&self) -> Option<u128> {
        self.active.lock().unwrap().as_ref().map(|a| a.started.elapsed().as_millis())
    }

    fn take_waiter(&self, id: &str) -> Option<oneshot::Sender<Value>> {
        let mut active = self.active.lock().unwrap();
        if let Some(a) = active.as_mut() {
            if a.id == id {
                return a.waiter.take();
            }
        }
        drop(active);
        self.waiters.lock().unwrap().remove(id)
    }

    pub fn set_waiter(&self, id: &str, tx: oneshot::Sender<Value>) {
        let mut active = self.active.lock().unwrap();
        match active.as_mut() {
            Some(a) if a.id == id => a.waiter = Some(tx),
            _ => {
                drop(active);
                self.waiters.lock().unwrap().insert(id.to_owned(), tx);
            }
        }
    }
}

/// The engine produced final text for a session.
pub fn on_final(app: &AppHandle, msg: &Value) {
    let Some(sessions) = app.try_state::<Arc<SessionManager>>() else { return };
    let id = msg.get("id").and_then(Value::as_str).unwrap_or_default().to_owned();
    if let Some(tx) = sessions.take_waiter(&id) {
        let _ = tx.send(msg.clone());
    }
    let mode = {
        let mut active = sessions.active.lock().unwrap();
        match active.as_ref() {
            Some(a) if a.id == id => {
                let mode = a.mode.clone();
                *active = None;
                Some(mode)
            }
            _ => None,
        }
    };
    let text = msg.get("text").and_then(Value::as_str).unwrap_or("");

    // Command mode: what was just transcribed is an instruction, not something to type. The
    // take is only half the work, so the phase stays at Finishing until the edit comes back.
    if let Some(Mode::Command { selection }) = mode {
        start_command(app, &id, selection, text);
        return;
    }
    if mode.is_some() {
        sessions.set_phase(Phase::Idle, None);
    }
    let t = msg.get("timings");
    let num = |k: &str| t.and_then(|t| t.get(k)).and_then(Value::as_f64).unwrap_or(0.0);
    crate::shell_log!(
        "{id} final: {:.1}s audio | stt {:.0} ms | {} {:.0} ms | release->final {:.0} ms | {:?}",
        num("audio_s"),
        num("stt_final_ms"),
        if t.and_then(|t| t.get("used_llm")).and_then(Value::as_bool).unwrap_or(false) {
            "auto-edits"
        } else {
            "rules"
        },
        num("post_ms"),
        num("release_to_final_ms"),
        text
    );
    if !text.is_empty() {
        let settings = crate::settings::load();
        crate::history::record(
            msg,
            &crate::context::last().map(|c| c.app).unwrap_or_default(),
            settings.history,
        );
        crate::inject::deliver(app, text);
    }
}

/// Level below which the microphone is considered quiet. Room tone and a fan sit well under
/// this; even quiet speech goes over it.
const SILENCE_LEVEL: f32 = 0.012;
/// How often the watcher looks. Frequent enough to stop promptly, rare enough to be free.
const WATCH_EVERY: std::time::Duration = std::time::Duration::from_millis(200);

/// End a latched hands-free take once the room has been quiet for long enough.
///
/// Push-to-talk needs nothing like this: the take ends when the fingers leave the chord.
/// Hands-free has no such signal, so a take started and then forgotten sits there recording -
/// and every second of it is audio the engine will eventually be asked to transcribe.
///
/// The watcher belongs to one take. It stops as soon as that take is no longer the active one,
/// so a second dictation started in the meantime is never cut short by the first one's timer.
pub fn watch_hands_free(app: &AppHandle, id: &str, timeout: std::time::Duration) {
    let app = app.clone();
    let id = id.to_owned();
    std::thread::Builder::new()
        .name("hands-free-watch".into())
        .spawn(move || {
            let Some(sessions) = app.try_state::<Arc<SessionManager>>() else { return };
            let Some(shell) = app.try_state::<crate::Shell>() else { return };
            let mut quiet_since: Option<Instant> = None;
            loop {
                std::thread::sleep(WATCH_EVERY);
                // Someone else's take now, or none at all: this watcher is done.
                let still_ours = matches!(
                    sessions.active.lock().unwrap().as_ref(),
                    Some(a) if a.id == id
                );
                if !still_ours {
                    return;
                }
                if shell.capture.level() > SILENCE_LEVEL {
                    quiet_since = None;
                    continue;
                }
                let since = *quiet_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= timeout {
                    crate::shell_log!(
                        "{id} hands-free stopped after {:.0}s of silence",
                        timeout.as_secs_f32()
                    );
                    shell.capture.end();
                    crate::hotkey::set_recording(false);
                    // Tell the hook too, or it still believes it is latched and the next tap
                    // of the chord would be read as "stop" instead of starting a new take.
                    crate::hotkey::clear_hands_free();
                    sessions.finish();
                    return;
                }
            }
        })
        .ok();
}

/// Send a command-mode take to the engine: the selection, and the instruction just spoken.
///
/// Runs on its own thread because the fallback way of reading a selection is to copy it, which
/// means waiting for the user's fingers to leave the chord and then for the target app to
/// answer Ctrl+C. Neither belongs on the websocket task.
fn start_command(app: &AppHandle, id: &str, selection: Option<String>, instruction: &str) {
    let app = app.clone();
    let id = id.to_owned();
    let instruction = instruction.trim().to_owned();
    let selection = selection.filter(|s| !s.trim().is_empty());
    std::thread::Builder::new()
        .name("command".into())
        .spawn(move || {
            let Some(sessions) = app.try_state::<Arc<SessionManager>>() else { return };
            let done = |reason: &str| {
                crate::shell_log!("{id} command: {reason}");
                let _ = app.emit("command-result", json!({"id": id, "changed": false, "rejected": reason}));
                sessions.set_phase(Phase::Idle, None);
            };
            if instruction.is_empty() {
                return done("nothing said");
            }
            // UI Automation saw it at the start; otherwise copy it now that the chord is free.
            let selection = match selection {
                Some(s) => s,
                None => match crate::inject::copy_selection() {
                    Some(s) => s,
                    None => return done("no text selected"),
                },
            };
            crate::shell_log!(
                "{id} command: {:?} on {} chars",
                instruction,
                selection.chars().count()
            );
            app.state::<Engine>().run_command(&id, &selection, &instruction);
        })
        .ok();
}

/// The engine answered a command. Only a `changed` result touches the user's document.
pub fn on_command_result(app: &AppHandle, msg: &Value) {
    let Some(sessions) = app.try_state::<Arc<SessionManager>>() else { return };
    let id = msg.get("id").and_then(Value::as_str).unwrap_or_default().to_owned();
    let changed = msg.get("changed").and_then(Value::as_bool).unwrap_or(false);
    let text = msg.get("text").and_then(Value::as_str).unwrap_or("");
    let ms = msg.get("ms").and_then(Value::as_f64).unwrap_or(0.0);
    if changed && !text.is_empty() {
        crate::shell_log!("{id} command applied in {ms:.0} ms | {text:?}");
        crate::inject::deliver_replacement(app, text);
    } else {
        // The selection is left exactly as it was. This is the common outcome when the model
        // wanders off, and it must stay cheap and quiet rather than destroying the text.
        crate::shell_log!(
            "{id} command left the text alone ({}) after {ms:.0} ms",
            msg.get("rejected").and_then(Value::as_str).unwrap_or("no reason given")
        );
    }
    sessions.set_phase(Phase::Idle, None);
}

/// The engine reported a problem. A session-scoped error ends the take.
pub fn on_error(app: &AppHandle, msg: &Value) {
    let Some(sessions) = app.try_state::<Arc<SessionManager>>() else { return };
    let id = msg.get("id").and_then(Value::as_str).unwrap_or_default().to_owned();
    crate::shell_log!(
        "engine error{}: {}",
        if id.is_empty() { String::new() } else { format!(" in {id}") },
        msg.get("message").and_then(Value::as_str).unwrap_or("?")
    );
    if id.is_empty() {
        return;
    }
    if let Some(tx) = sessions.take_waiter(&id) {
        let _ = tx.send(msg.clone());
    }
    let mut active = sessions.active.lock().unwrap();
    if matches!(active.as_ref(), Some(a) if a.id == id) {
        *active = None;
        drop(active);
        sessions.set_phase(Phase::Idle, None);
    }
}
