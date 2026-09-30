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

use crate::engine::Engine;
use crate::guard::LockExt;

/// Presses shorter than this are accidental, not dictation.
const MIN_TAKE: std::time::Duration = std::time::Duration::from_millis(300);
/// How long takes wait for a new engine after theirs went away, before they are given up.
const RECOVER_WITHIN: std::time::Duration = std::time::Duration::from_secs(90);
/// Replayed audio goes to the engine a second at a time, well inside its message size limit.
const REPLAY_CHUNK: usize = crate::audio::SAMPLE_RATE as usize;

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

/// What a take needs to be heard again by a different engine: how it was started, and every
/// sample sent so far. Kept until its text arrives, so an engine that dies mid-take costs a
/// restart and a moment, not the words.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Spoken {
    pub context: Value,
    pub language: Option<String>,
    pub audio: Vec<i16>,
    /// The window it was spoken into, which its text may go to and no other.
    pub window: isize,
    /// Spoken into a password field.
    pub private: bool,
}

/// The take the hotkey is driving right now.
struct Active {
    id: String,
    mode: Mode,
    started: Instant,
    /// The app it was spoken into, whose per-app rules apply when its text is typed.
    target: String,
    /// `session.end` has been sent: the take is finishing, and ending it again would only ask
    /// the engine for a second copy of the same final.
    ended: bool,
    spoken: Spoken,
    /// The engine it was being sent to went away while it was still being spoken. It keeps
    /// recording, and goes to the next engine once it ends.
    orphaned: bool,
}

/// A take whose text is still owed.
#[derive(Debug, Clone, PartialEq)]
pub struct Owed {
    pub mode: Mode,
    pub target: String,
    /// Whether it was the current take. Only the current take's text moves the phase: an
    /// earlier one landing must not end the take being spoken now.
    pub was_active: bool,
    pub spoken: Spoken,
}

/// Which takes are still owed their text, and what each one's text is for.
///
/// Kept apart from the window and the engine so it can be tested on its own. It used to be one
/// slot holding the current take, and a final for anything else was typed as a dictation: a
/// cancelled take's text, or a command's spoken instruction when a dictation had started
/// before the command's text came back.
#[derive(Default)]
struct Takes {
    active: Option<Active>,
    /// Takes that had ended and were then overtaken by a newer one. Their text still arrives
    /// (the engine lets an ended take finish) and still belongs where it was spoken.
    earlier: HashMap<String, Owed>,
    /// Takes whose engine went away before their text came, waiting for the next one.
    recovering: Vec<(String, Owed)>,
}

impl Takes {
    /// Make `take` the current one. Returns the id of a take that was still recording, which
    /// the caller must cancel: it was abandoned rather than finished.
    fn begin(&mut self, take: Active) -> Option<String> {
        let old = self.active.replace(take)?;
        if old.ended {
            self.earlier.insert(
                old.id,
                Owed { mode: old.mode, target: old.target, was_active: false, spoken: old.spoken },
            );
            None
        } else {
            Some(old.id)
        }
    }

    /// A final (or an error) arrived for `id`. Returns what that take's text is for, or None
    /// when nothing is owed to it - it was cancelled, or already settled.
    fn settle(&mut self, id: &str) -> Option<Owed> {
        if self.active.as_ref().is_some_and(|a| a.id == id) {
            let a = self.active.take()?;
            return Some(Owed { mode: a.mode, target: a.target, was_active: true, spoken: a.spoken });
        }
        self.earlier.remove(id)
    }

    /// Throw the current take away. Its text, if the engine still sends it, is owed to nobody.
    fn cancel(&mut self) -> Option<String> {
        self.active.take().map(|a| a.id)
    }

    /// The engine that owed these takes has gone. Ended takes wait for the next engine; the
    /// take still being spoken carries on recording and joins them when it ends. Returns
    /// whether that take is still recording.
    fn orphan_all(&mut self) -> bool {
        let mut earlier: Vec<_> = self.earlier.drain().collect();
        // Oldest first, so they are typed in the order they were spoken.
        earlier.sort_by_key(|(id, _)| id[1..].parse::<u64>().unwrap_or(0));
        self.recovering.extend(earlier);
        match self.active.as_mut() {
            Some(a) if !a.ended => {
                a.orphaned = true;
                true
            }
            Some(_) => {
                let a = self.active.take().expect("checked");
                self.recovering.push((
                    a.id,
                    Owed { mode: a.mode, target: a.target, was_active: true, spoken: a.spoken },
                ));
                false
            }
            None => false,
        }
    }

    /// Hand the takes waiting for an engine to it: from here they are owed as usual.
    fn take_recovering(&mut self) -> Vec<(String, Owed)> {
        let batch = std::mem::take(&mut self.recovering);
        for (id, owed) in &batch {
            self.earlier.insert(id.clone(), owed.clone());
        }
        batch
    }

    /// Nothing waits for an engine any more, and nothing still recording will.
    fn nothing_to_recover(&self) -> bool {
        self.recovering.is_empty() && !self.active.as_ref().is_some_and(|a| a.orphaned)
    }
}

pub struct SessionManager {
    engine: Engine,
    app: AppHandle,
    takes: Mutex<Takes>,
    phase: Mutex<Phase>,
    counter: AtomicU64,
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
            takes: Mutex::new(Takes::default()),
            phase: Mutex::new(Phase::Idle),
            counter: AtomicU64::new(0),
        })
    }

    pub fn phase(&self) -> Phase {
        *self.phase.locked()
    }

    /// Whether the link is up, which is not the same as being able to dictate: the models
    /// load for some time after it comes up.
    pub fn engine_connected(&self) -> bool {
        self.engine.is_connected()
    }

    fn set_phase(&self, phase: Phase, id: Option<String>) {
        *self.phase.locked() = phase;
        let _ = self.app.emit("phase", PhaseEvent { phase, id });
    }

    /// Back to idle, unless a newer take has started meanwhile - for work that finishes after
    /// its take stopped being the current one (a command's edit, say).
    fn idle_unless_busy(&self) {
        // Checked and set under one lock, as `begin_in` does, so a take starting in between
        // cannot be marked idle while it records.
        let takes = self.takes.locked();
        if takes.active.is_none() {
            self.set_phase(Phase::Idle, None);
        }
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
        // A take that cannot work is refused here, where the flow bar can say why: the engine
        // restarting or down, the speech model loading (the first dictation after every cold
        // start used to be sent, refused by the engine and dropped with nothing on screen), or
        // no microphone. A press that did nothing and said nothing looked like a broken hotkey.
        if let Some((code, why)) = crate::health::refusal_now(&self.app) {
            crate::shell_log!("hotkey refused [{code}]: {why}");
            let _ = self.app.emit("notice", json!({ "code": code, "text": why }));
            return None;
        }
        // A second press while one is still finishing is normal impatience: the previous take
        // is kept as owed and lands on its own, and the new one starts at once.
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let id = format!("s{n}");
        let target = context.get("app").and_then(Value::as_str).unwrap_or_default().to_owned();
        let window = crate::context::last().map(|c| c.hwnd).unwrap_or(0);
        let private = context.get("password").and_then(Value::as_bool).unwrap_or(false);
        let spoken = Spoken { context: context.clone(), language: language.clone(), audio: Vec::new(), window, private };
        let take =
            Active { id: id.clone(), mode, started: Instant::now(), target, ended: false, spoken, orphaned: false };
        let mut takes = self.takes.locked();
        if let Some(abandoned) = takes.begin(take) {
            self.engine.session_cancel(&abandoned);
        }
        self.engine.session_start(&id, context, language);
        self.set_phase(Phase::Recording, Some(id.clone()));
        if private {
            // The flow bar and the Hub show dots instead of the words.
            let _ = self.app.emit("take-private", json!({ "id": id }));
            crate::shell_log!("[{}] {id}", crate::problems::PASSWORD_FIELD.as_str());
        }
        Some(id)
    }

    /// Audio from the capture thread. Silently dropped when nothing is recording. Also kept
    /// with the take, in case its engine goes away before the text comes back.
    pub fn feed(&self, pcm: &[i16]) {
        let send = {
            let mut takes = self.takes.locked();
            match takes.active.as_mut() {
                Some(a) if !a.ended => {
                    a.spoken.audio.extend_from_slice(pcm);
                    !a.orphaned
                }
                _ => false,
            }
        };
        if send {
            self.engine.send_audio(pcm);
        }
    }

    /// The hotkey was released. A press too short to be speech is thrown away rather than
    /// sent: it is almost always a mistyped shortcut.
    pub fn finish(&self) {
        let (id, too_short) = {
            let mut takes = self.takes.locked();
            match takes.active.as_mut() {
                // Ended once already: its text is on the way.
                Some(a) if !a.ended => {
                    let too_short = a.started.elapsed() < MIN_TAKE;
                    a.ended = !too_short;
                    if a.orphaned && !too_short {
                        // Its engine is gone: it goes to the next one with the others.
                        takes.orphan_all();
                        drop(takes);
                        self.set_phase(Phase::Finishing, None);
                        return;
                    }
                    (a.id.clone(), too_short)
                }
                _ => return,
            }
        };
        if too_short {
            self.cancel();
            return;
        }
        self.engine.session_end(&id);
        self.set_phase(Phase::Finishing, Some(id));
    }

    pub fn cancel(&self) {
        let taken = self.takes.locked().cancel();
        if let Some(id) = taken {
            self.engine.session_cancel(&id);
        }
        self.set_phase(Phase::Idle, None);
    }

    /// The engine link went down, and whatever it owed is not coming: a new engine knows
    /// nothing of the old one's takes. They used to wait for ever, the app stuck in
    /// "finishing"; then they were dropped, and the words with them. Now each keeps its audio,
    /// and it is played to the next engine once that one can take it (`recover`). Returns
    /// whether a take is still being spoken - it carries on recording.
    pub fn on_link_down(self: &Arc<Self>) -> bool {
        let (recording, newly, waiting) = {
            let mut takes = self.takes.locked();
            // A restart is several link changes (lost, starting, connecting); the take is told
            // once, when it first loses its engine.
            let was = takes.active.as_ref().is_some_and(|a| a.orphaned);
            let recording = takes.orphan_all();
            (recording, recording && !was, takes.recovering.len())
        };
        if waiting == 0 && !recording {
            if self.phase() != Phase::Idle {
                self.set_phase(Phase::Idle, None);
            }
            return false;
        }
        crate::shell_log!(
            "the engine went away with {waiting} take(s) unfinished{}; they wait for the next one",
            if recording { " and one still being spoken" } else { "" }
        );
        if recording {
            // The take goes on recording and is played to the next engine when it ends.
            if newly {
                crate::health::take_notice(&self.app, crate::problems::ENGINE_LOST_MID_TAKE);
            }
        } else {
            self.set_phase(Phase::Finishing, None);
        }
        self.recover();
        recording
    }

    /// Wait for an engine that can take dictation, then play it the takes the last one owed.
    /// One recovery at a time; a second engine loss during a replay starts another afterwards.
    fn recover(self: &Arc<Self>) {
        static RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if RUNNING.swap(true, Ordering::SeqCst) {
            return;
        }
        let this = self.clone();
        let spawned = std::thread::Builder::new().name("take-recovery".into()).spawn(move || {
            let deadline = Instant::now() + RECOVER_WITHIN;
            loop {
                std::thread::sleep(std::time::Duration::from_millis(250));
                let ready = this.engine.is_connected() && this.engine.stt_ready();
                let mut takes = this.takes.locked();
                if takes.nothing_to_recover() {
                    break;
                }
                // Not while anything is being spoken: an orphaned take still recording goes with
                // the rest once it ends, and a new take must not be cut off by the replay (the
                // engine abandons a take that is still recording when another one starts).
                let speaking = takes.active.as_ref().is_some_and(|a| !a.ended);
                if ready && !speaking {
                    let batch = takes.take_recovering();
                    drop(takes);
                    this.replay(&batch);
                    continue;
                }
                if Instant::now() > deadline {
                    let lost: Vec<String> = takes.recovering.drain(..).map(|(id, _)| id).collect();
                    let recording = takes.active.as_ref().is_some_and(|a| a.orphaned);
                    if recording {
                        takes.active = None;
                    }
                    drop(takes);
                    crate::shell_log!("no engine came back in {:?}; lost: {}", RECOVER_WITHIN, lost.join(", "));
                    this.set_phase(Phase::Idle, None);
                    let lost = crate::problems::show(crate::problems::TAKE_LOST, &[]);
                    let _ = this.app.emit("engine-error", json!({"code": lost.code, "message": lost.message}));
                    break;
                }
            }
            RUNNING.store(false, Ordering::SeqCst);
        });
        if spawned.is_err() {
            RUNNING.store(false, Ordering::SeqCst);
        }
    }

    /// Play takes to the engine as if they were being spoken again, all at once.
    fn replay(&self, batch: &[(String, Owed)]) {
        for (id, owed) in batch {
            let spoken = &owed.spoken;
            crate::shell_log!(
                "{id} replayed to the new engine ({:.1}s of audio)",
                spoken.audio.len() as f32 / crate::audio::SAMPLE_RATE as f32
            );
            self.engine.session_start(id, spoken.context.clone(), spoken.language.clone());
            for chunk in spoken.audio.chunks(REPLAY_CHUNK) {
                self.engine.send_audio(chunk);
            }
            self.engine.session_end(id);
        }
    }

    /// Whether `id` is the take being spoken right now (not yet ended).
    fn is_recording(&self, id: &str) -> bool {
        self.takes.locked().active.as_ref().is_some_and(|a| a.id == id && !a.ended)
    }
}

/// The engine produced final text for a session.
pub fn on_final(app: &AppHandle, msg: &Value) {
    let Some(sessions) = app.try_state::<Arc<SessionManager>>() else { return };
    let id = msg.get("id").and_then(Value::as_str).unwrap_or_default().to_owned();
    let text = msg.get("text").and_then(Value::as_str).unwrap_or("");
    let owed = {
        let mut takes = sessions.takes.locked();
        let owed = takes.settle(&id);
        // The current take's text is in: back to idle - except for a command, whose take is
        // only half the work, so it stays at Finishing until the edit comes back.
        if matches!(&owed, Some(o) if o.was_active && o.mode == Mode::Dictate) {
            sessions.set_phase(Phase::Idle, None);
        }
        owed
    };
    let Some(owed) = owed else {
        // Cancelled, or settled already. Typing it would put words the user threw away - or an
        // instruction meant for command mode - into their document.
        crate::shell_log!("{id} final for a take nobody is waiting for; not typed");
        return;
    };

    // Command mode: what was just transcribed is an instruction, not something to type.
    if let Mode::Command { selection } = owed.mode {
        start_command(app, &id, selection, text);
        return;
    }
    let t = msg.get("timings");
    let num = |k: &str| t.and_then(|t| t.get(k)).and_then(Value::as_f64).unwrap_or(0.0);
    let private = owed.spoken.private;
    let logged = if private { format!("<password, {} chars>", text.chars().count()) } else { format!("{text:?}") };
    crate::shell_log!(
        "{id} final: {:.1}s audio | stt {:.0} ms | {} {:.0} ms | release->final {:.0} ms | {}",
        num("audio_s"),
        num("stt_final_ms"),
        if t.and_then(|t| t.get("used_llm")).and_then(Value::as_bool).unwrap_or(false) {
            "auto-edits"
        } else {
            "rules"
        },
        num("post_ms"),
        num("release_to_final_ms"),
        logged
    );
    if text.trim().is_empty() && heard_nothing(&owed.spoken.audio) {
        // Not "nothing recognisable was said" - the microphone itself sent silence, which a
        // blank flow bar would leave looking like a broken app.
        let code = crate::problems::TAKE_SILENT;
        crate::shell_log!(
            "[{}] {id}: {:.1}s of audio that never rose above -60 dBFS",
            code.as_str(),
            owed.spoken.audio.len() as f32 / crate::audio::SAMPLE_RATE as f32
        );
        let _ = app.emit("notice", json!({"code": code.as_str(), "text": crate::problems::bar(code, &[]), "hold_ms": 5000}));
    }
    if !text.is_empty() {
        let settings = crate::settings::load();
        // A test run's sentences are not the user's dictations, and a password is nobody's
        // history.
        let keep = settings.history && !crate::e2e::active() && !private;
        crate::history::record(msg, &owed.target, keep, settings.retention_days);
        crate::inject::deliver(app, text, &owed.target, owed.spoken.window, private);
    }
}

/// A take long enough to have been spoken, whose loudest sample stayed under -60 dBFS: the
/// microphone sent silence (muted, switched off, or its volume at zero). Room tone on a laptop
/// microphone sits around -50 dBFS, so a quiet room does not count.
pub fn heard_nothing(audio: &[i16]) -> bool {
    const QUIET_PEAK: i32 = 33; // -60 dBFS
    audio.len() >= crate::audio::SAMPLE_RATE as usize
        && audio.iter().all(|s| (*s as i32).abs() < QUIET_PEAK)
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
                // Ended by the user, someone else's take now, or none at all: this watcher is done.
                if !sessions.is_recording(&id) {
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
            let done = |code: crate::problems::Code| {
                let p = crate::problems::get(code);
                crate::shell_log!("{id} command [{}]: {}", p.code, p.title);
                let _ = app.emit(
                    "command-result",
                    json!({"id": id, "changed": false, "rejected": p.title, "code": p.code}),
                );
                sessions.idle_unless_busy();
            };
            if instruction.is_empty() {
                return done(crate::problems::COMMAND_NOTHING_SAID);
            }
            // UI Automation saw it at the start; otherwise copy it now that the chord is free.
            let selection = match selection {
                Some(s) => s,
                None => match crate::inject::copy_selection() {
                    Some(s) => s,
                    None => return done(crate::problems::COMMAND_NO_SELECTION),
                },
            };
            // The engine refuses a longer one; a copied document could otherwise be many
            // megabytes, and no model rewrites that in one go.
            if selection.chars().count() > crate::engine::COMMAND_SELECTION_MAX {
                return done(crate::problems::COMMAND_SELECTION_TOO_LONG);
            }
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
    sessions.idle_unless_busy();
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
    // Whatever the take was, nothing more is coming for it.
    let mut takes = sessions.takes.locked();
    if matches!(takes.settle(&id), Some(o) if o.was_active) {
        sessions.set_phase(Phase::Idle, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "Heard nothing" is for a microphone sending silence, never for a quiet room or a short
    /// press.
    #[test]
    fn only_a_silent_microphone_counts_as_hearing_nothing() {
        let second = crate::audio::SAMPLE_RATE as usize;
        assert!(heard_nothing(&vec![0; 2 * second]), "digital silence");
        assert!(heard_nothing(&vec![20; 2 * second]), "under -60 dBFS");
        // Room tone on a laptop microphone: about -50 dBFS.
        let room: Vec<i16> = (0..2 * second).map(|i| if i % 7 == 0 { 100 } else { -60 }).collect();
        assert!(!heard_nothing(&room));
        assert!(!heard_nothing(&vec![0; second / 2]), "too short to have been spoken");
        let mut one_word = vec![0i16; 2 * second];
        one_word[second] = 4000;
        assert!(!heard_nothing(&one_word));
    }

    fn take(id: &str, mode: Mode, app: &str) -> Active {
        Active {
            id: id.into(),
            mode,
            started: Instant::now(),
            target: app.into(),
            ended: false,
            spoken: Spoken::default(),
            orphaned: false,
        }
    }

    fn end(t: &mut Takes) {
        t.active.as_mut().unwrap().ended = true;
    }

    /// The bug: pressing again straight after letting go lost the first take's text, or typed
    /// it as whatever the new take was.
    #[test]
    fn a_take_overtaken_while_finishing_still_gets_its_text_as_what_it_was() {
        let mut t = Takes::default();
        assert_eq!(t.begin(take("s0", Mode::Dictate, "slack.exe")), None);
        end(&mut t);
        assert_eq!(t.begin(take("s1", Mode::Dictate, "code.exe")), None, "an ended take is not cancelled");

        let s0 = t.settle("s0").expect("s0 is still owed its text");
        assert_eq!((s0.mode, s0.target.as_str(), s0.was_active), (Mode::Dictate, "slack.exe", false));
        let s1 = t.settle("s1").expect("the current take");
        assert!(s1.was_active);
        assert_eq!(t.settle("s0"), None, "settled once only");
    }

    /// A command's spoken instruction came back as a dictation once a newer take had started,
    /// and "make this more formal" was typed into the document.
    #[test]
    fn an_earlier_command_is_still_a_command() {
        let mut t = Takes::default();
        t.begin(take("s0", Mode::Command { selection: Some("hi".into()) }, "word.exe"));
        end(&mut t);
        t.begin(take("s1", Mode::Dictate, "word.exe"));
        let s0 = t.settle("s0").unwrap();
        assert_eq!(s0.mode, Mode::Command { selection: Some("hi".into()) });
    }

    /// An engine that died mid-take: first the app stuck waiting for ever, then the words were
    /// dropped. The takes wait for the next engine instead, in the order they were spoken.
    #[test]
    fn takes_an_engine_owed_wait_for_the_next_one_in_order() {
        let mut t = Takes::default();
        t.begin(take("s9", Mode::Dictate, "a.exe"));
        end(&mut t);
        t.begin(take("s10", Mode::Dictate, "a.exe"));
        end(&mut t);
        assert!(!t.orphan_all(), "nothing is still being spoken");
        assert_eq!(t.settle("s9"), None, "not owed by the new engine until handed to it");
        let batch = t.take_recovering();
        let ids: Vec<&str> = batch.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["s9", "s10"], "oldest first, numerically");
        assert!(t.settle("s9").is_some() && t.settle("s10").is_some(), "owed as usual once replayed");
        assert!(t.nothing_to_recover());
    }

    /// The take still being spoken when the engine goes keeps recording, and joins the others
    /// only once it ends.
    #[test]
    fn a_take_still_being_spoken_keeps_recording_through_an_engine_loss() {
        let mut t = Takes::default();
        t.begin(take("s0", Mode::Dictate, "a.exe"));
        assert!(t.orphan_all(), "still recording");
        assert!(t.active.as_ref().unwrap().orphaned);
        assert!(!t.nothing_to_recover());
        assert!(t.take_recovering().is_empty(), "not until it ends");
        end(&mut t);
        assert!(!t.orphan_all());
        assert_eq!(t.take_recovering().len(), 1);
    }

    #[test]
    fn a_cancelled_take_is_owed_nothing() {
        let mut t = Takes::default();
        t.begin(take("s0", Mode::Dictate, "a.exe"));
        end(&mut t);
        assert_eq!(t.cancel().as_deref(), Some("s0"));
        assert_eq!(t.settle("s0"), None, "its final, if one still comes, is not typed");
    }

    /// A take still recording when another starts is abandoned: the caller cancels it.
    #[test]
    fn a_take_still_recording_is_abandoned_by_a_new_one() {
        let mut t = Takes::default();
        t.begin(take("s0", Mode::Dictate, "a.exe"));
        assert_eq!(t.begin(take("s1", Mode::Command { selection: None }, "a.exe")).as_deref(), Some("s0"));
        assert_eq!(t.settle("s0"), None);
    }
}
