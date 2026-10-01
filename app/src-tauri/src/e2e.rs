//! The end-to-end harness: real dictations through the real shell, with a script in place of
//! the keyboard and a recording in place of the microphone.
//!
//! `app.exe --e2e [scenario...]` starts the whole app - engine link, session machine, chord
//! controller, audio pipeline, injector, flow bar - and then plays takes through it into a text
//! box of its own, reading back what landed. Only two things are not the real ones: the chord
//! arrives from this script rather than the keyboard hook, and the speech comes from WAV files
//! (`scripts/make-e2e-audio.ps1`) rather than the microphone. Everything between is exercised
//! as a user would exercise it, which is what the hand checks after the v0.2 code review were
//! doing with a real keyboard.
//!
//! It runs beside the user's own LocalFlow and shares its engine; nothing it dictates goes into
//! the history. The text box must keep the focus while it runs: every take checks that it has
//! it and refuses to type anywhere else.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use tauri::{AppHandle, Listener, Manager};

use crate::audio::Tape;
use crate::engine::Engine;
use crate::guard::LockExt;
use crate::hotkey::Raw;

static ACTIVE: AtomicBool = AtomicBool::new(false);
/// The test text box's window. In a harness run, text is typed only while it is in front.
static TARGET: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);

/// This process is the harness: its dictations are tests, not the user's.
pub fn activate() {
    ACTIVE.store(true, Ordering::SeqCst);
}

pub fn active() -> bool {
    ACTIVE.load(Ordering::Relaxed)
}

/// Off during a soak: takes go all the way through, and the injector reports what it would
/// have typed without typing it, so the run never needs the focus.
static TYPING: AtomicBool = AtomicBool::new(true);

pub fn set_typing(on: bool) {
    TYPING.store(on, Ordering::SeqCst);
}

pub fn typing() -> bool {
    !active() || TYPING.load(Ordering::Relaxed)
}

/// In a harness run, whether typing now would land in the test text box. Checked by the
/// injector at the moment it types: something taking the focus during a 30-second take once
/// sent a test paragraph into whatever window was in front instead.
pub fn may_type_now() -> Result<(), String> {
    if !active() {
        return Ok(());
    }
    let front = unsafe { windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow() };
    if front.0 as isize == TARGET.load(Ordering::SeqCst) {
        Ok(())
    } else {
        let ctx = crate::context::foreground();
        Err(format!(
            "the end-to-end test's text box is not in front ({} {:?} is); not typing anywhere else",
            if ctx.app.is_empty() { "?" } else { &ctx.app },
            ctx.title
        ))
    }
}

/// The test text box: the one window of LocalFlow's own that stands in for a user's app.
pub fn is_target(hwnd: isize) -> bool {
    active() && hwnd != 0 && hwnd == TARGET.load(Ordering::SeqCst)
}

type Scenario = fn(&Run) -> Result<String, String>;

/// Faults, run only when asked for by name or with `faults`. They kill and freeze the engine,
/// so they need one of their own: with the user's LocalFlow running, the harness shares its
/// engine and these refuse to run.
const FAULTS: &[(&str, &str, Scenario)] = &[
    ("mic-unplugged", "a microphone that dies mid-take still gives the words said before it died", mic_unplugged),
    ("cleanup-hang", "a hung clean-up server delays a take by its timeout, not forever", cleanup_hang),
    ("engine-crash", "a take in flight when the engine dies is typed once a new one is up", engine_crash),
    ("engine-hang", "a frozen engine is replaced, and the take spoken to it still typed", engine_hang),
];

/// Name, what it proves, and the test. Run in this order; the cheap ones first.
const SCENARIOS: &[(&str, &str, Scenario)] = &[
    ("one-take", "a take is typed where it was spoken", one_take),
    ("quick-retake", "two quick takes both land, in order", quick_retake),
    ("hands-free", "a hands-free take stops on a press and is typed once", hands_free),
    ("escape", "Escape cancels a take, and does nothing when that is turned off", escape),
    ("clipboard-image", "a copied image is still on the clipboard after a dictation", clipboard_image),
    ("command-then-dictation", "a command's spoken instruction is never typed", command_then_dictation),
    ("long-selection", "a command edits a long selection whole or leaves it alone", long_selection),
    ("long-take", "a 30-second take arrives whole and in order", long_take),
    ("window-changed", "text is kept, not typed, when another window took the front; Win+Alt+V pastes it", window_changed),
    ("password-field", "a take into a password field is typed exactly as heard, and shown as dots", password_field),
    ("tray-paste", "the tray's Paste last dictation goes back to the window you were in and pastes there", tray_paste),
    ("silent-take", "a take from a microphone sending silence says 'heard nothing', and types nothing", silent_take),
];

/// Run only when asked for by name: they leave something on the user's screen.
const BY_NAME: &[(&str, &str, Scenario)] = &[
    // Task Manager runs as administrator without asking, and LocalFlow cannot close it again.
    ("admin-window", "a take into an administrator app is copied, with 'press Ctrl+V', not typed", admin_window),
    // The screen shows an empty desktop for a few seconds, as it would for a UAC prompt.
    ("desktop-switch", "input moving to another desktop mid-take (a UAC prompt) ends the take and keeps its words", desktop_switch),
    // Breaks the running engine's speech on the graphics card for a few seconds.
    ("gpu-lost", "speech on the graphics card dying mid-take (a driver reset) still types the take", gpu_lost),
    // Tells this harness Windows is shutting down: it stops supervising its engine. Run it last.
    ("session-end", "Windows shutting down mid-take ends the take and stops the engine in order", session_end),
];

/// Start the harness on its own thread; it ends the app when it is done.
pub fn start(app: AppHandle, keys: Sender<Raw>, tape: Tape, names: Vec<String>) {
    let _ = std::thread::Builder::new().name("e2e".into()).spawn(move || {
        let code = run_all(&app, keys, tape, &names);
        if let Some(engine) = app.try_state::<Engine>() {
            engine.shutdown(); // ours only; an engine we attached to is left running
        }
        std::thread::sleep(Duration::from_millis(300));
        // Not `app.exit`: Tauri's exit reports 0 whatever it is given, and the exit code is the
        // result. The engine, if it was ours, dies with the job object.
        std::process::exit(code);
    });
}

fn report(line: &str) {
    println!("{line}");
    crate::shell_log!("[e2e] {line}");
}

fn run_all(app: &AppHandle, keys: Sender<Raw>, tape: Tape, names: &[String]) -> i32 {
    let all_faults = names.iter().any(|n| n == "faults");
    let soaking = names.first().map(String::as_str) == Some("soak");
    let chosen: Vec<_> = SCENARIOS
        .iter()
        .filter(|(name, _, _)| names.is_empty() || names.iter().any(|n| n == name))
        .chain(BY_NAME.iter().filter(|(name, _, _)| names.iter().any(|n| n == name)))
        .chain(FAULTS.iter().filter(|(name, _, _)| all_faults || names.iter().any(|n| n == name)))
        .collect();
    if chosen.is_empty() && !soaking {
        report(&format!(
            "no such scenario; choose from: {}, by name only: {}, or faults: {}, or soak [minutes]",
            SCENARIOS.iter().map(|s| s.0).collect::<Vec<_>>().join(", "),
            BY_NAME.iter().map(|s| s.0).collect::<Vec<_>>().join(", "),
            FAULTS.iter().map(|s| s.0).collect::<Vec<_>>().join(", ")
        ));
        return 2;
    }

    // Ready means the speech model is loaded, not merely that the link is up.
    let deadline = Instant::now() + Duration::from_secs(180);
    let engine = app.state::<Engine>();
    while !engine.stt_ready() {
        if Instant::now() > deadline {
            report("the engine did not become ready in 3 minutes");
            return 2;
        }
        std::thread::sleep(Duration::from_millis(250));
    }

    if names.first().map(String::as_str) == Some("soak") {
        let minutes = names.get(1).and_then(|m| m.parse().ok()).unwrap_or(120);
        let Some(audio) = audio_dir() else {
            report(r"no test speech found: run scripts\make-e2e-audio.ps1");
            return 2;
        };
        return crate::soak::run(app, keys, tape, audio, minutes);
    }

    let faulting = chosen.iter().any(|c| FAULTS.iter().any(|f| f.0 == c.0));
    if faulting && engine.link().attached {
        report("the fault scenarios kill and freeze the engine, and this one is your LocalFlow's: quit LocalFlow first");
        return 2;
    }

    let Some(audio) = audio_dir() else {
        report("no test speech found: run scripts\\make-e2e-audio.ps1");
        return 2;
    };
    let target = match Target::open() {
        Ok(t) => t,
        Err(e) => {
            report(&format!("could not open the text box to dictate into: {e}"));
            return 2;
        }
    };
    // The harness uses the clipboard for one scenario and the injector for all of them, so the
    // user's clipboard is set aside first and put back at the end, whatever happens.
    let users_clipboard = crate::inject::clipboard_snapshot();

    let events = Arc::new(Mutex::new(Vec::new()));
    for kind in ["final", "injected", "command-result", "phase", "engine-link", "notice", "take-private"] {
        let events = events.clone();
        app.listen(kind, move |e| {
            let payload = serde_json::from_str(e.payload()).unwrap_or(Value::Null);
            events.locked().push(Event { kind, payload });
        });
    }
    let run = Run { app: app.clone(), keys, tape, events, target, audio, base: crate::hotkey::Config::load() };

    let mut failed = 0;
    report(&format!("running {} scenario(s) through the real shell", chosen.len()));
    for (name, what, test) in chosen {
        run.reset_hotkeys();
        let started = Instant::now();
        let result = crate::guard::catch("an e2e scenario", || test(&run))
            .unwrap_or_else(|| Err("the scenario panicked (see the log)".into()));
        let secs = started.elapsed().as_secs_f32();
        match result {
            Ok(detail) => report(&format!("PASS {name} ({secs:.1}s): {what} - {detail}")),
            Err(why) => {
                failed += 1;
                report(&format!("FAIL {name} ({secs:.1}s): {what} - {why}"));
            }
        }
        // Let anything late (a stray second paste) land in this scenario's box, not the next.
        std::thread::sleep(Duration::from_millis(600));
    }

    run.target.close();
    if let Some(snapshot) = users_clipboard {
        let _ = snapshot.restore();
    }
    report(&format!("{} failed", failed));
    if failed == 0 { 0 } else { 1 }
}

/// `engine/tests/fixtures/e2e`, beside the source in a development build; or wherever
/// `LOCALFLOW_E2E_AUDIO` says.
fn audio_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LOCALFLOW_E2E_AUDIO").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../engine/tests/fixtures/e2e")
    });
    dir.join("fox.wav").is_file().then_some(dir)
}

// ---------------------------------------------------------------------------------------------
// the run

struct Event {
    kind: &'static str,
    payload: Value,
}

struct Run {
    app: AppHandle,
    keys: Sender<Raw>,
    tape: Tape,
    events: Arc<Mutex<Vec<Event>>>,
    target: Target,
    audio: PathBuf,
    /// The user's hotkey settings, which scenarios change a copy of and then put back.
    base: crate::hotkey::Config,
}

impl Run {
    fn key(&self, raw: Raw) {
        let _ = self.keys.send(raw);
        std::thread::sleep(Duration::from_millis(40));
    }

    /// Press the chord, making sure the text box is still where the text will go first.
    fn press(&self, raw: Raw) -> Result<(), String> {
        if !self.target.focus() {
            return Err("the test text box lost the focus (something else was clicked?)".into());
        }
        self.key(raw);
        Ok(())
    }

    /// Queue a recording; `say` waits for it to finish, `start_saying` does not.
    fn start_saying(&self, name: &str) -> Result<(), String> {
        let path = self.audio.join(format!("{name}.wav"));
        let pcm = crate::read_wav_16k_mono(&path.to_string_lossy()).map_err(|e| format!("{name}.wav: {e}"))?;
        self.tape.play(&pcm);
        Ok(())
    }

    fn say(&self, name: &str) -> Result<(), String> {
        self.start_saying(name)?;
        self.finish_saying();
        Ok(())
    }

    /// Wait for the queued speech to end.
    fn finish_saying(&self) {
        while self.tape.playing() {
            std::thread::sleep(Duration::from_millis(20));
        }
        // People let go a moment after the last word, not on it.
        std::thread::sleep(Duration::from_millis(250));
    }

    /// One push-to-talk take: press, speak, let go.
    fn take(&self, name: &str) -> Result<(), String> {
        self.press(Raw::ChordDown)?;
        self.say(name)?;
        // A user letting go is still in their document. Over a long take another window can
        // take the front (the Claude app does, when the harness is run from it).
        self.target.focus();
        self.key(Raw::ChordUp);
        Ok(())
    }

    fn mark(&self) -> usize {
        self.events.locked().len()
    }

    fn since(&self, mark: usize, kind: &str) -> Vec<Value> {
        self.events.locked()[mark..].iter().filter(|e| e.kind == kind).map(|e| e.payload.clone()).collect()
    }

    /// Wait until `n` events of `kind` have arrived since `mark`.
    fn wait(&self, mark: usize, kind: &str, n: usize, timeout: Duration) -> Result<Vec<Value>, String> {
        let deadline = Instant::now() + timeout;
        loop {
            let got = self.since(mark, kind);
            if got.len() >= n {
                return Ok(got);
            }
            if Instant::now() > deadline {
                return Err(format!("waited {:?} for {n} {kind} event(s), got {}", timeout, got.len()));
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Finals since `mark`, as text, in the order they arrived.
    fn finals(&self, mark: usize) -> Vec<String> {
        self.since(mark, "final").iter().map(|f| text_of(f).to_owned()).collect()
    }

    /// Wait for `n` injections and check none of them failed. The text box is kept in front
    /// meanwhile, as a user waiting for their words would keep it; the injector refuses to type
    /// anywhere else.
    fn injected(&self, mark: usize, n: usize) -> Result<Vec<Value>, String> {
        let deadline = Instant::now() + Duration::from_secs(45);
        while self.since(mark, "injected").len() < n && Instant::now() < deadline {
            self.target.focus();
            std::thread::sleep(Duration::from_millis(50));
        }
        let done = self.wait(mark, "injected", n, Duration::from_millis(1))?;
        if let Some(err) = done.iter().find_map(|d| d.get("error").and_then(Value::as_str)) {
            return Err(format!("injection failed: {err}"));
        }
        // The text box processes the input queue after the injector has sent it.
        std::thread::sleep(Duration::from_millis(400));
        Ok(done)
    }

    fn reset_hotkeys(&self) {
        let mut cfg = self.base.clone();
        cfg.double_tap = true;
        cfg.double_tap_ms = 400;
        cfg.escape_cancels = true;
        crate::hotkey::configure(&cfg);
    }
}

fn text_of(v: &Value) -> &str {
    v.get("text").and_then(Value::as_str).unwrap_or("")
}

/// Words only: spacing, and the space every dictation ends with, are not what is being tested.
fn norm(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn expect_text(got: &str, want: &str) -> Result<(), String> {
    if norm(got) == norm(want) {
        Ok(())
    } else {
        Err(format!("the box holds {:?}, expected {:?}", norm(got), norm(want)))
    }
}

// ---------------------------------------------------------------------------------------------
// scenarios

fn one_take(r: &Run) -> Result<String, String> {
    r.target.set("");
    let m = r.mark();
    r.take("fox")?;
    let done = r.injected(m, 1)?;
    let finals = r.finals(m);
    let [heard] = finals.as_slice() else { return Err(format!("{} finals for one take", finals.len())) };
    expect_text(&r.target.text(), heard)?;
    Ok(format!("{heard:?} by {}", done[0].get("method").and_then(Value::as_str).unwrap_or("?")))
}

fn window_changed(r: &Run) -> Result<String, String> {
    r.target.set("");
    let other = Target::window("LocalFlow end-to-end test - another window", false)?;
    // An edit control's title is its text; start it empty, and the box in front again.
    other.set("");
    let m = r.mark();
    r.press(Raw::ChordDown)?;
    let result = (|| {
        r.say("fox")?;
        // The user clicks into another window before letting go, so it is in front when the
        // words are ready.
        if !other.focus() {
            return Err("Windows would not bring the second window to the front".to_owned());
        }
        r.key(Raw::ChordUp);
        let done = r.wait(m, "injected", 1, Duration::from_secs(45))?;
        let method = done[0].get("method").and_then(Value::as_str).unwrap_or("?").to_owned();
        if method != "kept" {
            return Err(format!("the text was delivered by {method} instead of kept"));
        }
        std::thread::sleep(Duration::from_millis(400));
        if !norm(&r.target.text()).is_empty() || !norm(&other.text()).is_empty() {
            return Err("text was typed somewhere".into());
        }
        let notices = r.since(m, "notice");
        let said = notices.iter().any(|n| n.get("code").and_then(Value::as_str) == Some("text-kept-window-changed"));
        if !said {
            return Err(format!("the flow bar was not told it was kept: {notices:?}"));
        }
        // Back in the box, Win+Alt+V pastes it.
        let heard = r.finals(m).first().cloned().unwrap_or_default();
        let m2 = r.mark();
        r.press(Raw::PasteLast(false))?;
        r.injected(m2, 1)?;
        expect_text(&r.target.text(), &heard)?;
        Ok(format!("kept, then pasted: {heard:?}"))
    })();
    other.close();
    result
}

fn password_field(r: &Run) -> Result<String, String> {
    let pw = Target::window("", true)?;
    // For this scenario the password box is where text may go.
    let before = TARGET.swap(pw.hwnd, Ordering::SeqCst);
    let result = (|| {
        if !pw.focus() {
            return Err("Windows would not bring the password box to the front".to_owned());
        }
        let m = r.mark();
        r.key(Raw::ChordDown);
        r.say("fox")?;
        pw.focus();
        r.key(Raw::ChordUp);
        let done = r.wait(m, "injected", 1, Duration::from_secs(45))?;
        if let Some(err) = done[0].get("error").and_then(Value::as_str) {
            return Err(format!("injection failed: {err}"));
        }
        if r.since(m, "take-private").is_empty() {
            return Err("UI Automation did not report a password field (no take-private)".into());
        }
        let finals = r.since(m, "final");
        let final_ = finals.first().ok_or("no final")?;
        let timings = final_.get("timings");
        if timings.and_then(|t| t.get("private")).and_then(Value::as_bool) != Some(true) {
            return Err("the engine did not treat it as private".into());
        }
        // The windows (and so this harness) see dots only, never the words.
        let shown = text_of(final_);
        let raw = final_.get("raw").and_then(Value::as_str).unwrap_or("");
        if shown.is_empty() || shown.chars().any(|c| c != '•') || raw.chars().any(|c| c != '•') {
            return Err(format!("the words reached the windows: {shown:?} / {raw:?}"));
        }
        if shown.chars().count() != raw.chars().count() {
            return Err(format!("the text was cleaned up: {} chars from {}", shown.chars().count(), raw.chars().count()));
        }
        std::thread::sleep(Duration::from_millis(400));
        if pw.text().chars().count() != shown.chars().count() {
            return Err(format!("the box holds {} chars, expected exactly {}", pw.text().chars().count(), shown.chars().count()));
        }
        Ok(format!("typed exactly as heard ({} chars), dots everywhere else", shown.chars().count()))
    })();
    TARGET.store(before, Ordering::SeqCst);
    pw.close();
    r.target.focus();
    result
}

fn tray_paste(r: &Run) -> Result<String, String> {
    use windows::Win32::UI::WindowsAndMessaging::{FindWindowW, SetForegroundWindow};
    r.target.set("");
    let m = r.mark();
    r.take("fox")?;
    r.injected(m, 1)?;
    let heard = r.finals(m).first().cloned().unwrap_or_default();
    r.target.set("");
    // Clicking the tray icon puts the taskbar in front; the menu item then fires.
    let class: Vec<u16> = "Shell_TrayWnd".encode_utf16().chain(Some(0)).collect();
    let taskbar = unsafe { FindWindowW(windows::core::PCWSTR(class.as_ptr()), None) }
        .map_err(|e| format!("no taskbar: {e}"))?;
    crate::inject::tap_unassigned();
    let _ = unsafe { SetForegroundWindow(taskbar) };
    std::thread::sleep(Duration::from_millis(150));
    if crate::win::foreground_window() != taskbar.0 as isize {
        return Err("Windows would not bring the taskbar to the front".into());
    }
    let m2 = r.mark();
    let _ = tauri::Emitter::emit(&r.app, "paste-last-request", ());
    // Not `injected`: that keeps the box in front, which is the very thing being tested.
    let done = r.wait(m2, "injected", 1, Duration::from_secs(10))?;
    let method = done[0].get("method").and_then(Value::as_str).unwrap_or("?").to_owned();
    if let Some(err) = done[0].get("error").and_then(Value::as_str) {
        return Err(format!("injection failed: {err}"));
    }
    if method == "kept" {
        return Err("the text was kept: the taskbar was still in front".into());
    }
    std::thread::sleep(Duration::from_millis(400));
    expect_text(&r.target.text(), &heard)
        .map_err(|e| format!("{e}; in front: {}", crate::win::describe(crate::win::foreground_window())))?;
    Ok(format!("back in the box, pasted {heard:?} by {method}"))
}

fn silent_take(r: &Run) -> Result<String, String> {
    r.target.set("");
    let m = r.mark();
    // Nothing on the tape: the scripted microphone sends digital silence, as a muted one does.
    r.press(Raw::ChordDown)?;
    std::thread::sleep(Duration::from_millis(2000));
    r.target.focus();
    r.key(Raw::ChordUp);
    let finals = r.wait(m, "final", 1, Duration::from_secs(30))?;
    if !text_of(&finals[0]).trim().is_empty() {
        return Err(format!("silence was heard as {:?}", text_of(&finals[0])));
    }
    let deadline = Instant::now() + Duration::from_secs(3);
    let said = loop {
        let notices = r.since(m, "notice");
        if let Some(n) = notices.iter().find(|n| n.get("code").and_then(Value::as_str) == Some("take-silent")) {
            break text_of(n).to_owned();
        }
        if Instant::now() > deadline {
            return Err(format!("the flow bar was not told it heard nothing: {notices:?}"));
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    std::thread::sleep(Duration::from_millis(300));
    if !r.target.text().trim().is_empty() {
        return Err(format!("something was typed: {:?}", r.target.text()));
    }
    Ok(format!("nothing typed; the bar said {said:?}"))
}

fn desktop_switch(r: &Run) -> Result<String, String> {
    r.target.set("");
    let m = r.mark();
    r.press(Raw::ChordDown)?;
    r.say("fox")?;
    // The chord is still held when a UAC prompt takes input to its secure desktop - done here
    // for real, with a desktop of the harness's own. The release happens over there, unseen.
    let away = OtherDesktop::switch()?;
    let result = (|| {
        let done = r.wait(m, "injected", 1, Duration::from_secs(45)).map_err(|e| format!("the take did not end by itself: {e}"))?;
        let method = done[0].get("method").and_then(Value::as_str).unwrap_or("?").to_owned();
        if method != "kept" {
            return Err(format!("delivered by {method} while input was on another desktop"));
        }
        Ok(())
    })();
    drop(away);
    std::thread::sleep(Duration::from_millis(300));
    result?;
    // The release, arriving late, must not start or stop anything.
    let m_late = r.mark();
    r.key(Raw::ChordUp);
    std::thread::sleep(Duration::from_millis(800));
    if !r.since(m_late, "final").is_empty() || r.phase() != crate::session::Phase::Idle {
        return Err("the late release did something".into());
    }
    let heard = r.finals(m).first().cloned().unwrap_or_default();
    let m2 = r.mark();
    r.press(Raw::PasteLast(false))?;
    r.injected(m2, 1)?;
    expect_text(&r.target.text(), &heard)?;
    Ok(format!("ended on the switch, kept, then pasted with Win+Alt+V: {heard:?}"))
}

fn session_end(r: &Run) -> Result<String, String> {
    use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
    use windows::Win32::UI::WindowsAndMessaging::{SendMessageW, WM_ENDSESSION, WM_QUERYENDSESSION};
    let window = crate::power::window();
    if window == 0 {
        return Err("the session watcher has no window".into());
    }
    let h = HWND(window as *mut _);
    r.target.set("");
    let m = r.mark();
    r.press(Raw::ChordDown)?;
    r.say("fox")?;
    // What Windows sends when it shuts down, restarts for an update, or signs out.
    let allowed = unsafe { SendMessageW(h, WM_QUERYENDSESSION, Some(WPARAM(0)), Some(LPARAM(0))) };
    if allowed.0 == 0 {
        return Err("LocalFlow held the shutdown up".into());
    }
    let before = r.engine().link();
    let (engine_pid, attached) = (before.pid, before.attached);
    unsafe { SendMessageW(h, WM_ENDSESSION, Some(WPARAM(1)), Some(LPARAM(0))) };
    // The take stops recording at once (nothing can be typed while Windows shuts down), and the
    // engine is told to stop - unless it is not ours: here, the user's LocalFlow's, left running.
    std::thread::sleep(Duration::from_millis(500));
    if r.phase() == crate::session::Phase::Recording {
        return Err("the take was still recording after Windows said it was shutting down".into());
    }
    let pid = engine_pid.ok_or("no engine pid")?;
    if attached && !crate::win::pid_alive(pid) {
        return Err("an engine this app had only attached to was stopped".into());
    }
    let _ = m;
    Ok(format!(
        "shutdown not held up; the take stopped recording at once; {}",
        if attached { "the user's engine left running" } else { "our engine told to stop" }
    ))
}

/// Input on another desktop, as a UAC prompt or the lock screen puts it, and back again when
/// dropped - or after ten seconds whatever happens, so the user's screen always comes back.
struct OtherDesktop {
    back: isize,
    other: isize,
    done: Arc<AtomicBool>,
}

impl OtherDesktop {
    fn switch() -> Result<OtherDesktop, String> {
        use windows::Win32::System::StationsAndDesktops::*;
        unsafe {
            let back = OpenInputDesktop(DESKTOP_CONTROL_FLAGS(0), false, DESKTOP_SWITCHDESKTOP)
                .map_err(|e| format!("cannot open this desktop: {e}"))?;
            let name: Vec<u16> = "LocalFlowE2EDesktop".encode_utf16().chain(Some(0)).collect();
            let other = CreateDesktopW(
                windows::core::PCWSTR(name.as_ptr()),
                None,
                None,
                DESKTOP_CONTROL_FLAGS(0),
                0x1000_0000, // GENERIC_ALL
                None,
            )
            .map_err(|e| format!("cannot create a desktop: {e}"))?;
            SwitchDesktop(other).map_err(|e| format!("cannot switch desktops: {e}"))?;
            let done = Arc::new(AtomicBool::new(false));
            let (flag, raw_back) = (done.clone(), back.0 as isize);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(10));
                if !flag.load(Ordering::SeqCst) {
                    let _ = SwitchDesktop(HDESK(raw_back as *mut _));
                }
            });
            Ok(OtherDesktop { back: back.0 as isize, other: other.0 as isize, done })
        }
    }
}

impl Drop for OtherDesktop {
    fn drop(&mut self) {
        use windows::Win32::System::StationsAndDesktops::*;
        self.done.store(true, Ordering::SeqCst);
        unsafe {
            let _ = SwitchDesktop(HDESK(self.back as *mut _));
            let _ = CloseDesktop(HDESK(self.other as *mut _));
            let _ = CloseDesktop(HDESK(self.back as *mut _));
        }
    }
}

fn admin_window(r: &Run) -> Result<String, String> {
    use windows::Win32::UI::WindowsAndMessaging::FindWindowW;
    const CF_UNICODETEXT: u32 = 13;
    let class: Vec<u16> = "TaskManagerWindow".encode_utf16().chain(Some(0)).collect();
    let find = || unsafe { FindWindowW(windows::core::PCWSTR(class.as_ptr()), None) }.ok();
    // Windows does not let an app that is not administrator bring an administrator's window to
    // the front, so Task Manager is asked to come forward itself: started through the shell
    // (which elevates it; CreateProcess would refuse), it opens in front, or brings its open
    // window forward. It may take the front because this process lets it.
    unsafe {
        let _ = windows::Win32::UI::WindowsAndMessaging::AllowSetForegroundWindow(
            windows::Win32::UI::WindowsAndMessaging::ASFW_ANY,
        );
    }
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000; // a console flashing up would take the front
        let _ = std::process::Command::new("cmd")
            .args(["/c", "start", "", "taskmgr.exe"])
            .creation_flags(CREATE_NO_WINDOW)
            .status();
    }
    let in_front = || find().is_some_and(|tm| crate::win::foreground_window() == tm.0 as isize);
    let deadline = Instant::now() + Duration::from_secs(15);
    while !in_front() {
        if Instant::now() > deadline {
            return Err(match find() {
                None => "Task Manager did not open".into(),
                Some(_) => "Task Manager opened, but not in front".into(),
            });
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let tm = find().ok_or("Task Manager went away")?;
    if !crate::win::keystrokes_blocked(tm.0 as isize) {
        return Err("Task Manager is not running as administrator here (UAC off, or not an admin account?)".into());
    }
    let result = (|| {
        let m = r.mark();
        r.key(Raw::ChordDown);
        let said = r.say("fox");
        let still = in_front();
        r.key(Raw::ChordUp); // always let go: a take left open would outlive the harness
        said?;
        if !still {
            return Err(format!(
                "something else took the front during the take: {}",
                crate::win::describe(crate::win::foreground_window())
            ));
        }
        let done = r.wait(m, "injected", 1, Duration::from_secs(45))?;
        let method = done[0].get("method").and_then(Value::as_str).unwrap_or("?").to_owned();
        if method != "kept" {
            return Err(format!("the text was delivered by {method}: its keystrokes would have been dropped"));
        }
        let notices = r.since(m, "notice");
        if !notices.iter().any(|n| n.get("code").and_then(Value::as_str) == Some("text-copied-admin")) {
            return Err(format!("the flow bar was not told to press Ctrl+V: {notices:?}"));
        }
        let heard = r.finals(m).first().cloned().unwrap_or_default();
        let clip = crate::inject::clipboard_snapshot().ok_or("could not read the clipboard")?;
        let text = clip
            .format(CF_UNICODETEXT)
            .map(|b| {
                let wide: Vec<u16> = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                String::from_utf16_lossy(&wide).trim_end_matches('\0').to_owned()
            })
            .unwrap_or_default();
        expect_text(&text, &heard).map_err(|e| format!("on the clipboard: {e}"))?;
        // Marked to stay out of clipboard history and cloud sync.
        let marked = ["ExcludeClipboardContentFromMonitorProcessing", "CanIncludeInClipboardHistory"]
            .iter()
            .all(|name| {
                let wide: Vec<u16> = name.encode_utf16().chain(Some(0)).collect();
                let id = unsafe {
                    windows::Win32::System::DataExchange::RegisterClipboardFormatW(windows::core::PCWSTR(wide.as_ptr()))
                };
                clip.format(id).is_some()
            });
        if !marked {
            return Err("the copy is not marked to stay out of clipboard history".into());
        }
        Ok(format!("copied {heard:?}, kept out of clipboard history; the bar said press Ctrl+V"))
    })();
    // Task Manager is the user's to close: an app that is not administrator cannot.
    r.target.focus();
    result
}

fn quick_retake(r: &Run) -> Result<String, String> {
    r.target.set("");
    let m = r.mark();
    r.take("report")?;
    // Straight back in, while the first take is still being finished.
    std::thread::sleep(Duration::from_millis(60));
    r.take("meeting")?;
    r.injected(m, 2)?;
    let finals = r.finals(m);
    if finals.len() != 2 {
        return Err(format!("{} finals for two takes", finals.len()));
    }
    let both = format!("{} {}", finals[0], finals[1]);
    expect_text(&r.target.text(), &both)?;
    Ok("both typed, first one first".into())
}

fn hands_free(r: &Run) -> Result<String, String> {
    r.target.set("");
    let m = r.mark();
    // Double tap: latch on.
    r.press(Raw::ChordDown)?;
    std::thread::sleep(Duration::from_millis(80));
    r.key(Raw::ChordUp);
    std::thread::sleep(Duration::from_millis(80));
    r.press(Raw::ChordDown)?;
    r.key(Raw::ChordUp);
    // Hands off the keys while speaking - that is the point of it.
    r.say("fox")?;
    // One press stops it; its release must not count as a second stop.
    r.press(Raw::ChordDown)?;
    r.key(Raw::ChordUp);
    r.injected(m, 1)?;
    // Anything typed twice would arrive within this.
    std::thread::sleep(Duration::from_secs(3));
    let typed = r.since(m, "injected").len();
    if typed != 1 {
        return Err(format!("typed {typed} times"));
    }
    let finals = r.finals(m);
    let [heard] = finals.as_slice() else { return Err(format!("{} finals", finals.len())) };
    expect_text(&r.target.text(), heard)?;
    Ok("typed once".into())
}

fn escape(r: &Run) -> Result<String, String> {
    // Turned off: Escape is just a key, and the take carries on.
    let mut cfg = r.base.clone();
    cfg.escape_cancels = false;
    crate::hotkey::configure(&cfg);
    r.target.set("");
    let m = r.mark();
    r.press(Raw::ChordDown)?;
    r.start_saying("meeting")?;
    std::thread::sleep(Duration::from_millis(1200));
    r.key(Raw::Escape);
    r.finish_saying();
    r.key(Raw::ChordUp);
    r.injected(m, 1).map_err(|e| format!("with Escape-cancels off: {e}"))?;
    let finals = r.finals(m);
    let [heard] = finals.as_slice() else { return Err(format!("{} finals with it off", finals.len())) };
    expect_text(&r.target.text(), heard).map_err(|e| format!("with Escape-cancels off: {e}"))?;

    // Turned on: the take is thrown away and nothing is typed.
    r.reset_hotkeys();
    r.target.set("");
    let m = r.mark();
    r.press(Raw::ChordDown)?;
    r.start_saying("fox")?;
    std::thread::sleep(Duration::from_millis(1200));
    r.key(Raw::Escape);
    r.finish_saying();
    r.key(Raw::ChordUp);
    std::thread::sleep(Duration::from_secs(4));
    let typed = r.since(m, "injected").len();
    if typed != 0 || !norm(&r.target.text()).is_empty() {
        return Err(format!("with Escape-cancels on, {typed} injection(s) and {:?} typed", r.target.text()));
    }
    Ok("off: kept the take; on: nothing typed".into())
}

fn clipboard_image(r: &Run) -> Result<String, String> {
    // A 1x1 32-bit DIB, the way a screenshot arrives.
    let mut dib = Vec::new();
    for v in [40u32, 1, 1] {
        dib.extend_from_slice(&v.to_le_bytes());
    }
    dib.extend_from_slice(&1u16.to_le_bytes());
    dib.extend_from_slice(&32u16.to_le_bytes());
    dib.extend_from_slice(&[0u8; 24]);
    dib.extend_from_slice(&[0x11, 0x22, 0x33, 0xFF]);
    const CF_DIB: u32 = 8;
    crate::inject::clipboard_set(CF_DIB, &dib).map_err(|e| format!("could not put an image on the clipboard: {e}"))?;

    r.target.set("");
    let m = r.mark();
    r.take("fox")?;
    let done = r.injected(m, 1)?;
    let method = done[0].get("method").and_then(Value::as_str).unwrap_or("?").to_owned();
    // The clipboard is put back just after the paste lands.
    std::thread::sleep(Duration::from_millis(1500));
    let back = crate::inject::clipboard_snapshot().ok_or("could not read the clipboard back")?;
    match back.format(CF_DIB) {
        Some(bytes) if bytes.starts_with(&dib) => {}
        Some(_) => return Err("an image came back, but not the same one".into()),
        None => return Err(format!("the image is gone (the dictation went in by {method})")),
    }
    let finals = r.finals(m);
    expect_text(&r.target.text(), finals.first().map(String::as_str).unwrap_or(""))?;
    Ok(format!("image intact after a {method}"))
}

fn command_then_dictation(r: &Run) -> Result<String, String> {
    r.target.set("hello world, this is a small test.");
    r.target.select_all();
    let m = r.mark();
    r.press(Raw::CommandDown)?;
    r.say("shout")?;
    r.key(Raw::CommandUp);
    // A dictation straight after, while the command is still being worked on.
    std::thread::sleep(Duration::from_millis(100));
    r.take("report")?;
    let result = r.wait(m, "command-result", 1, Duration::from_secs(60))?;
    let changed = result[0].get("changed").and_then(Value::as_bool).unwrap_or(false);
    r.injected(m, if changed { 2 } else { 1 })?;
    std::thread::sleep(Duration::from_secs(1));

    let finals = r.finals(m);
    let instruction = finals.first().cloned().unwrap_or_default();
    let dictation = finals.get(1).cloned().unwrap_or_default();
    let now = r.target.text().to_lowercase();
    if now.contains("capital letters") {
        return Err(format!("the instruction was typed: {:?}", r.target.text()));
    }
    if dictation.is_empty() || !norm(&now).contains(&norm(&dictation.to_lowercase())) {
        return Err(format!("the dictation {dictation:?} is missing from {:?}", r.target.text()));
    }
    Ok(format!("instruction {instruction:?} kept out; the command {}", if changed { "edited" } else { "left the text" }))
}

fn long_selection(r: &Run) -> Result<String, String> {
    let sentence = "The committee reviewed the budget and agreed to meet again next month. ";
    let original: String = sentence.repeat(20).trim_end().to_owned();
    r.target.set(&original);
    r.target.select_all();
    let m = r.mark();
    r.press(Raw::CommandDown)?;
    r.say("shout")?;
    r.key(Raw::CommandUp);
    let result = r.wait(m, "command-result", 1, Duration::from_secs(120))?;
    let changed = result[0].get("changed").and_then(Value::as_bool).unwrap_or(false);
    if changed {
        r.injected(m, 1)?;
    }
    std::thread::sleep(Duration::from_secs(1));
    let now = r.target.text();
    if norm(&now) == norm(&original) {
        return Ok(format!("left alone ({})", result[0].get("rejected").and_then(Value::as_str).unwrap_or("no change")));
    }
    // Edited: then all of it, once - not the edit with the original still around it.
    let lower = now.chars().filter(|c| c.is_ascii_lowercase()).count();
    let ratio = now.len() as f32 / original.len() as f32;
    if lower == 0 && (0.8..1.2).contains(&ratio) {
        return Ok(format!("edited whole ({} chars)", now.len()));
    }
    Err(format!(
        "half-edited: {lower} lowercase letters left, {:.0} % of the original length",
        ratio * 100.0
    ))
}

fn long_take(r: &Run) -> Result<String, String> {
    r.target.set("");
    let m = r.mark();
    r.take("long")?;
    r.injected(m, 1)?;
    let finals = r.finals(m);
    let [heard] = finals.as_slice() else { return Err(format!("{} finals for one take", finals.len())) };
    let low = heard.to_lowercase();
    for part in ["dictation should keep working", "the weather was cold", "end of the long take"] {
        if !low.contains(part) {
            return Err(format!("{part:?} is missing from {heard:?}"));
        }
    }
    expect_text(&r.target.text(), heard)?;
    Ok(format!("{} words, start to end", heard.split_whitespace().count()))
}

// ---------------------------------------------------------------------------------------------
// faults

impl Run {
    fn engine(&self) -> tauri::State<'_, Engine> {
        self.app.state::<Engine>()
    }

    fn phase(&self) -> crate::session::Phase {
        self.app.state::<Arc<crate::session::SessionManager>>().phase()
    }

    /// Wait until an engine other than `old` is up and able to dictate - a frozen engine still
    /// looks ready until the shell notices. Returns how long that took.
    fn engine_back(&self, old: u32, timeout: Duration) -> Result<Duration, String> {
        let started = Instant::now();
        loop {
            let link = self.engine().link();
            if link.link == crate::engine::Link::Ready && link.pid != Some(old) && self.engine().stt_ready() {
                return Ok(started.elapsed());
            }
            if started.elapsed() > timeout {
                return Err(format!("the engine was not back after {timeout:?} (link {:?}: {:?})", link.link, link.detail));
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    /// Wait until the take machine is idle again, as it must be after any fault.
    fn idle_within(&self, timeout: Duration) -> Result<(), String> {
        let deadline = Instant::now() + timeout;
        while self.phase() != crate::session::Phase::Idle {
            if Instant::now() > deadline {
                return Err(format!("left in the {:?} phase {timeout:?} after the fault", self.phase()));
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        Ok(())
    }

    /// A normal take after a fault: the proof that the fault is over.
    fn recovers(&self) -> Result<(), String> {
        self.target.set("");
        let m = self.mark();
        self.take("meeting")?;
        self.injected(m, 1).map_err(|e| format!("the take after the fault: {e}"))?;
        let finals = self.finals(m);
        expect_text(&self.target.text(), finals.last().map(String::as_str).unwrap_or(""))
            .map_err(|e| format!("the take after the fault: {e}"))
    }
}

fn mic_unplugged(r: &Run) -> Result<String, String> {
    r.target.set("");
    let m = r.mark();
    r.press(Raw::ChordDown)?;
    r.start_saying("long")?;
    // About a dozen words in, the device goes.
    std::thread::sleep(Duration::from_millis(5000));
    r.tape.unplug();
    std::thread::sleep(Duration::from_millis(1500));
    r.key(Raw::ChordUp);
    let typed = r.injected(m, 1);
    r.tape.plug_in();
    typed.map_err(|e| format!("the words before the unplug were lost: {e}"))?;
    let heard = r.finals(m).pop().unwrap_or_default();
    if !heard.to_lowercase().contains("dictation should keep working") {
        return Err(format!("typed {heard:?}, not the start of what was said"));
    }
    r.idle_within(Duration::from_secs(2))?;
    Ok(format!("kept {:?}", heard))
}

fn cleanup_hang(r: &Run) -> Result<String, String> {
    let engine_pid = r.engine().link().pid.ok_or("no engine pid")?;
    let server = proc::children(engine_pid, "llama-server.exe").into_iter().next().ok_or(
        "the engine has no clean-up server to freeze (is AI clean-up on, with the bundled model?)",
    )?;
    proc::suspend(server)?;
    r.target.set("");
    let m = r.mark();
    let released = Instant::now();
    let outcome = (|| {
        r.take("fox")?;
        r.injected(m, 1)
    })();
    let waited = released.elapsed();
    proc::resume(server)?;
    outcome.map_err(|e| format!("with the clean-up server frozen: {e}"))?;
    let heard = r.finals(m).pop().unwrap_or_default();
    expect_text(&r.target.text(), &heard)?;
    // The clean-up timeout is 8 s; the take itself is about 4.
    if waited > Duration::from_secs(16) {
        return Err(format!("typed, but {:.1}s after the take began", waited.as_secs_f32()));
    }
    r.recovers()?;
    Ok(format!("typed without clean-up {:.1}s after the take began; clean-up back afterwards", waited.as_secs_f32()))
}

fn engine_crash(r: &Run) -> Result<String, String> {
    let pid = r.engine().link().pid.ok_or("no engine pid")?;
    r.target.set("");
    let m = r.mark();
    r.press(Raw::ChordDown)?;
    r.start_saying("fox")?;
    std::thread::sleep(Duration::from_millis(1500));
    proc::kill(pid)?;
    r.finish_saying();
    r.key(Raw::ChordUp);
    let back = r.engine_back(pid, Duration::from_secs(60))?;
    // The take in flight is played to the new engine, and typed.
    r.injected(m, 1).map_err(|e| format!("the take in flight was lost: {e}"))?;
    let heard = r.finals(m).pop().unwrap_or_default();
    if !heard.to_lowercase().contains("lazy dog") {
        return Err(format!("the take in flight came back as {heard:?}"));
    }
    expect_text(&r.target.text(), &heard)?;
    r.idle_within(Duration::from_secs(5))?;
    r.recovers()?;
    Ok(format!("engine back in {:.1}s; the take in flight typed whole: {heard:?}", back.as_secs_f32()))
}

/// The speech worker process: speech on the graphics card lives there (engine/…/stt/remote.py).
fn speech_worker(engine_pid: u32) -> Option<u32> {
    ["python.exe", "localflow-engine.exe"].iter().find_map(|n| proc::children(engine_pid, n).first().copied())
}

fn gpu_lost(r: &Run) -> Result<String, String> {
    let pid = r.engine().link().pid.ok_or("no engine pid")?;
    let worker = speech_worker(pid).ok_or("speech is not on the graphics card now: nothing to break")?;
    r.target.set("");
    let m = r.mark();
    r.press(Raw::ChordDown)?;
    r.start_saying("fox")?;
    std::thread::sleep(Duration::from_millis(1500));
    // What a graphics driver reset does to the process using the card.
    proc::kill(worker)?;
    r.finish_saying();
    r.key(Raw::ChordUp);
    r.injected(m, 1).map_err(|e| format!("the take was lost: {e}"))?;
    let heard = r.finals(m).pop().unwrap_or_default();
    if !heard.to_lowercase().contains("lazy dog") {
        return Err(format!("the take came back as {heard:?}"));
    }
    expect_text(&r.target.text(), &heard)?;
    // Speech goes back to the card by itself, in a new worker.
    let deadline = Instant::now() + Duration::from_secs(60);
    let back = loop {
        if let Some(w) = speech_worker(pid).filter(|w| *w != worker) {
            break w;
        }
        if Instant::now() > deadline {
            return Err("typed, but speech did not go back to the graphics card within a minute".into());
        }
        std::thread::sleep(Duration::from_millis(500));
    };
    Ok(format!("typed whole on the processor: {heard:?}; back on the graphics card in a new worker (pid {back})"))
}

fn engine_hang(r: &Run) -> Result<String, String> {
    let pid = r.engine().link().pid.ok_or("no engine pid")?;
    proc::suspend(pid)?;
    let frozen = Instant::now();
    // The shell only knows the engine is frozen when it stops answering; a take started
    // meanwhile is the case that matters.
    r.target.set("");
    let m = r.mark();
    r.take("fox")?;
    // The frozen engine is replaced, not thawed: this waits for the new one.
    let back = r.engine_back(pid, Duration::from_secs(90));
    if back.is_err() {
        let _ = proc::resume(pid); // do not leave a frozen process behind
    }
    back?;
    let replaced_after = frozen.elapsed();
    // The take spoken into the frozen engine is played to the new one.
    r.injected(m, 1).map_err(|e| format!("the take spoken to the frozen engine was lost: {e}"))?;
    let heard = r.finals(m).pop().unwrap_or_default();
    expect_text(&r.target.text(), &heard)?;
    r.idle_within(Duration::from_secs(5))?;
    r.recovers()?;
    Ok(format!("replaced {:.0}s after it froze; the take spoken to it typed", replaced_after.as_secs_f32()))
}

/// Killing, freezing and finding processes, for the fault scenarios.
mod proc {
    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
    };
    use windows::Win32::System::Threading::{OpenProcess, TerminateProcess, PROCESS_ACCESS_RIGHTS, PROCESS_TERMINATE};

    windows_core::link!("ntdll.dll" "system" fn NtSuspendProcess(process : HANDLE) -> i32);
    windows_core::link!("ntdll.dll" "system" fn NtResumeProcess(process : HANDLE) -> i32);
    const PROCESS_SUSPEND_RESUME: PROCESS_ACCESS_RIGHTS = PROCESS_ACCESS_RIGHTS(0x0800);

    fn with<T>(pid: u32, access: PROCESS_ACCESS_RIGHTS, f: impl FnOnce(HANDLE) -> Result<T, String>) -> Result<T, String> {
        unsafe {
            let h = OpenProcess(access, false, pid).map_err(|e| format!("could not open process {pid}: {e}"))?;
            let out = f(h);
            let _ = CloseHandle(h);
            out
        }
    }

    pub fn kill(pid: u32) -> Result<(), String> {
        with(pid, PROCESS_TERMINATE, |h| unsafe { TerminateProcess(h, 1).map_err(|e| e.to_string()) })
    }

    pub fn suspend(pid: u32) -> Result<(), String> {
        with(pid, PROCESS_SUSPEND_RESUME, |h| match unsafe { NtSuspendProcess(h) } {
            0 => Ok(()),
            s => Err(format!("could not freeze process {pid}: 0x{s:08X}")),
        })
    }

    pub fn resume(pid: u32) -> Result<(), String> {
        with(pid, PROCESS_SUSPEND_RESUME, |h| match unsafe { NtResumeProcess(h) } {
            0 => Ok(()),
            s => Err(format!("could not thaw process {pid}: 0x{s:08X}")),
        })
    }

    /// Processes named `name` whose parent is `parent`.
    pub fn children(parent: u32, name: &str) -> Vec<u32> {
        let mut found = Vec::new();
        unsafe {
            let Ok(snap) = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) else { return found };
            let mut entry = PROCESSENTRY32W { dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32, ..Default::default() };
            let mut ok = Process32FirstW(snap, &mut entry).is_ok();
            while ok {
                let len = entry.szExeFile.iter().position(|c| *c == 0).unwrap_or(entry.szExeFile.len());
                let exe = String::from_utf16_lossy(&entry.szExeFile[..len]);
                if entry.th32ParentProcessID == parent && exe.eq_ignore_ascii_case(name) {
                    found.push(entry.th32ProcessID);
                }
                ok = Process32NextW(snap, &mut entry).is_ok();
            }
            let _ = CloseHandle(snap);
        }
        found
    }
}

// ---------------------------------------------------------------------------------------------
// the text box

/// A plain Windows edit control in a window of its own, which the harness types into and reads
/// back directly. Not Notepad: closing Notepad afterwards could close the user's own documents
/// with it, and reading it back meant the clipboard.
struct Target {
    hwnd: isize,
    thread: u32,
}

const EM_SETSEL: u32 = 0x00B1;
const EM_SETLIMITTEXT: u32 = 0x00C5;

impl Target {
    fn open() -> Result<Target, String> {
        let target = Target::window("LocalFlow end-to-end test - leave this window in front", false)?;
        TARGET.store(target.hwnd, Ordering::SeqCst);
        if !target.focus() {
            target.close();
            return Err("Windows would not bring the test text box to the front".into());
        }
        Ok(target)
    }

    /// A text box window of the harness's own; `open` makes one the target.
    fn window(title: &str, password: bool) -> Result<Target, String> {
        use windows::Win32::UI::WindowsAndMessaging::*;
        let title: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("e2e-text-box".into())
            .spawn(move || unsafe {
                let class: Vec<u16> = "EDIT\0".encode_utf16().collect();
                // A password box is a single-line edit; Windows ignores ES_PASSWORD on a multi-line one.
                let style = if password {
                    WS_OVERLAPPEDWINDOW | WS_VISIBLE | WINDOW_STYLE((ES_PASSWORD | ES_AUTOHSCROLL) as u32)
                } else {
                    WS_OVERLAPPEDWINDOW
                        | WS_VISIBLE
                        | WS_VSCROLL
                        | WINDOW_STYLE((ES_MULTILINE | ES_AUTOVSCROLL) as u32)
                };
                let hwnd = CreateWindowExW(
                    WINDOW_EX_STYLE(0),
                    windows::core::PCWSTR(class.as_ptr()),
                    windows::core::PCWSTR(title.as_ptr()),
                    style,
                    120,
                    120,
                    720,
                    420,
                    None,
                    None,
                    None,
                    None,
                );
                let tid = windows::Win32::System::Threading::GetCurrentThreadId();
                match hwnd {
                    Ok(h) => {
                        SendMessageW(h, EM_SETLIMITTEXT, Some(windows::Win32::Foundation::WPARAM(0)), None);
                        let _ = tx.send(Ok((h.0 as isize, tid)));
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e.to_string()));
                        return;
                    }
                }
                let mut msg = MSG::default();
                while GetMessageW(&mut msg, None, 0, 0).as_bool() {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            })
            .map_err(|e| e.to_string())?;
        let (hwnd, thread) = rx.recv_timeout(Duration::from_secs(5)).map_err(|e| e.to_string())??;
        Ok(Target { hwnd, thread })
    }

    fn h(&self) -> windows::Win32::Foundation::HWND {
        windows::Win32::Foundation::HWND(self.hwnd as *mut core::ffi::c_void)
    }

    /// Bring the box to the front and check it is there.
    fn focus(&self) -> bool {
        use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, SetForegroundWindow};
        unsafe {
            if GetForegroundWindow() == self.h() {
                return true;
            }
            // Windows only lets the process with the most recent input take the foreground; an
            // injected key nobody listens for makes that this one.
            crate::inject::tap_unassigned();
            let _ = SetForegroundWindow(self.h());
            std::thread::sleep(Duration::from_millis(150));
            GetForegroundWindow() == self.h()
        }
    }

    fn text(&self) -> String {
        use windows::Win32::Foundation::{LPARAM, WPARAM};
        use windows::Win32::UI::WindowsAndMessaging::{SendMessageW, WM_GETTEXT, WM_GETTEXTLENGTH};
        unsafe {
            let len = SendMessageW(self.h(), WM_GETTEXTLENGTH, None, None).0.max(0) as usize;
            let mut buf = vec![0u16; len + 1];
            let got = SendMessageW(
                self.h(),
                WM_GETTEXT,
                Some(WPARAM(buf.len())),
                Some(LPARAM(buf.as_mut_ptr() as isize)),
            )
            .0
            .max(0) as usize;
            String::from_utf16_lossy(&buf[..got.min(len)])
        }
    }

    /// Replace the contents, leaving the caret at the end.
    fn set(&self, text: &str) {
        use windows::Win32::Foundation::{LPARAM, WPARAM};
        use windows::Win32::UI::WindowsAndMessaging::{SendMessageW, WM_SETTEXT};
        let wide: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
        unsafe {
            SendMessageW(self.h(), WM_SETTEXT, None, Some(LPARAM(wide.as_ptr() as isize)));
            let end = text.encode_utf16().count();
            SendMessageW(self.h(), EM_SETSEL, Some(WPARAM(end)), Some(LPARAM(end as isize)));
        }
        self.focus();
    }

    fn select_all(&self) {
        use windows::Win32::Foundation::{LPARAM, WPARAM};
        use windows::Win32::UI::WindowsAndMessaging::SendMessageW;
        unsafe {
            SendMessageW(self.h(), EM_SETSEL, Some(WPARAM(0)), Some(LPARAM(-1)));
        }
    }

    fn close(&self) {
        use windows::Win32::Foundation::{LPARAM, WPARAM};
        use windows::Win32::UI::WindowsAndMessaging::{PostMessageW, PostThreadMessageW, WM_CLOSE, WM_QUIT};
        unsafe {
            // Closed by its own thread, which then stops.
            let _ = PostMessageW(Some(self.h()), WM_CLOSE, WPARAM(0), LPARAM(0));
            let _ = PostThreadMessageW(self.thread, WM_QUIT, WPARAM(0), LPARAM(0));
        }
    }
}
