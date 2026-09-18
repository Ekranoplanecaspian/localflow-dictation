//! Put text where the caret is, in whatever app is focused.
//!
//! Two strategies:
//!   * **paste** - clipboard plus Ctrl+V, then the previous clipboard is put back. The default,
//!     because it is the only one that is reliable in every app (see `method_for`).
//!   * **type**  - `SendInput` with `KEYEVENTF_UNICODE`, one key event pair per UTF-16 unit.
//!     Used for short text, where it is reliable and leaves the clipboard alone, and as the
//!     fallback when the clipboard cannot be taken.
//!
//! Everything runs on one injector thread: injections stay in order, and the websocket task
//! never blocks waiting for a target app to accept keystrokes.

use std::sync::mpsc::{self, Sender};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use serde_json::json;
use tauri::{AppHandle, Emitter};
use windows::Win32::Foundation::{HANDLE, HGLOBAL, HWND};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, GetClipboardData, IsClipboardFormatAvailable, OpenClipboard,
    SetClipboardData,
};
use windows::Win32::System::Memory::{GlobalAlloc, GlobalLock, GlobalUnlock, GMEM_MOVEABLE};
use windows::Win32::System::Ole::CF_UNICODETEXT;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS,
    KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, VIRTUAL_KEY, VK_CONTROL, VK_LWIN, VK_MENU, VK_RETURN,
    VK_RWIN, VK_SHIFT, VK_TAB, VK_V,
};

/// Typed text survives up to about a dozen characters even in the controls that mangle it, and
/// staying off the clipboard for a short correction is worth it. Anything longer is pasted.
const SHORT_ENOUGH_TO_TYPE: usize = 12;
/// The hotkey is a chord, so Ctrl and Win are usually still down when the text is ready.
const MODIFIER_WAIT: Duration = Duration::from_millis(2000);
/// How long the target app is given to read the clipboard before we put the old contents back.
const CLIPBOARD_SETTLE: Duration = Duration::from_millis(300);
const MODIFIERS: [VIRTUAL_KEY; 5] = [VK_CONTROL, VK_MENU, VK_SHIFT, VK_LWIN, VK_RWIN];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Type,
    Paste,
}

impl Method {
    fn name(self) -> &'static str {
        match self {
            Method::Type => "type",
            Method::Paste => "paste",
        }
    }
}

/// In these apps a bare Enter sends the message, so a dictated line break must be Shift+Enter.
fn is_chat_app(app: &str) -> bool {
    matches!(
        app,
        "slack.exe"
            | "discord.exe"
            | "teams.exe"
            | "ms-teams.exe"
            | "whatsapp.exe"
            | "telegram.exe"
            | "signal.exe"
            | "element.exe"
    )
}

fn is_terminal(app: &str) -> bool {
    matches!(
        app,
        "windowsterminal.exe"
            | "wt.exe"
            | "conhost.exe"
            | "cmd.exe"
            | "powershell.exe"
            | "pwsh.exe"
            | "alacritty.exe"
            | "wezterm-gui.exe"
    )
}

/// Which strategy suits this text in this app.
///
/// Pasting is the default, because typing is not reliable. Synthetic unicode key events reach
/// the target intact - a low-level hook confirms Windows delivers the exact scan codes - but
/// modern XAML text controls mangle them past a dozen characters or so: `--type-probe` sent
/// "aaaa bbbb cccc" to Windows 11 Notepad and got back "aaaa ccccccccc", the same length with
/// each run collapsed onto its last character. Pasting the same text lands perfectly. Typing
/// survives as a fallback for when the clipboard cannot be used.
pub fn method_for(app: &str, text: &str) -> Method {
    let multiline = text.contains('\n');
    if multiline && (is_chat_app(app) || is_terminal(app)) {
        // A typed Enter sends the message or runs the line; a paste inserts it.
        return Method::Paste;
    }
    if text.chars().count() < SHORT_ENOUGH_TO_TYPE {
        // Short enough that typing is reliable, and not worth disturbing the clipboard for.
        return Method::Type;
    }
    Method::Paste
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Typing is only trusted for text short enough to survive the controls that mangle it.
    #[test]
    fn only_very_short_text_is_typed() {
        assert_eq!(method_for("notepad.exe", "hello"), Method::Type);
        assert_eq!(method_for("notepad.exe", "a dictated sentence of ordinary length"), Method::Paste);
        // The measured failure: this exact string came back as "aaaa ccccccccc" when typed.
        assert_eq!(method_for("notepad.exe", "aaaa bbbb cccc"), Method::Paste);
    }

    #[test]
    fn a_dictation_ends_with_a_space_so_the_next_one_does_not_run_into_it() {
        assert_eq!(with_spacing("Hello there.", true), "Hello there. ");
        assert_eq!(with_spacing("the", true), "the ");
        // Already separated: adding another space would creep.
        assert_eq!(with_spacing("Hello ", true), "Hello ");
        // A line break separates on its own; a space after it would land on the next line.
        assert_eq!(with_spacing("Hello
", true), "Hello
");
        assert_eq!(with_spacing("Hello there.", false), "Hello there.");
    }

    /// In a chat app a typed Enter sends the message, and in a shell it runs the line. Both
    /// are irreversible, so multi-line text goes in as one paste instead.
    #[test]
    fn multiline_text_avoids_typing_enter_where_that_would_send_or_run() {
        assert_eq!(method_for("slack.exe", "one\ntwo"), Method::Paste);
        assert_eq!(method_for("windowsterminal.exe", "one\ntwo"), Method::Paste);
        assert_eq!(method_for("notepad.exe", "one\ntwo"), Method::Type);
    }
}

// ---------------------------------------------------------------------------------------------
// the injector thread

struct Job {
    app: AppHandle,
    text: String,
    target: String,
    /// The per-app method override, already resolved. Resolved by the caller rather than here
    /// so the injector thread never touches the settings file.
    method: Option<Method>,
    /// Press Enter once the text has landed, so a dictated chat message sends itself.
    auto_send: bool,
}

fn injector() -> &'static Sender<Job> {
    static TX: OnceLock<Sender<Job>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("injector".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    let started = Instant::now();
                    let wanted = job.method.unwrap_or_else(|| method_for(&job.target, &job.text));
                    let result = inject(&job.text, wanted, &job.target);
                    let method = *result.as_ref().unwrap_or(&wanted);
                    let payload = json!({
                        "method": method.name(),
                        "chars": job.text.chars().count(),
                        "app": job.target,
                        "ms": started.elapsed().as_millis() as u64,
                        "error": result.as_ref().err().map(|e| e.to_string()),
                    });
                    if result.is_ok() && job.auto_send {
                        // After the text, not before: pressing Enter first would send an empty
                        // message, and pressing it too soon can beat the paste into the field.
                        std::thread::sleep(Duration::from_millis(60));
                        if let Err(e) = tap_enter() {
                            crate::shell_log!("auto-send failed in {}: {e}", job.target);
                        } else {
                            crate::shell_log!("auto-sent in {}", job.target);
                        }
                    }
                    match result {
                        Ok(_) => crate::shell_log!(
                            "injected {} chars by {} into {} in {} ms",
                            job.text.chars().count(),
                            method.name(),
                            if job.target.is_empty() { "the caret" } else { &job.target },
                            started.elapsed().as_millis()
                        ),
                        Err(ref e) => crate::shell_log!(
                            "injection FAILED into {}: {e}",
                            if job.target.is_empty() { "the caret" } else { &job.target }
                        ),
                    }
                    let _ = job.app.emit("injected", payload);
                }
            })
            .expect("injector thread");
        tx
    })
}

/// A dictation ends with a space so the next one does not run into it.
///
/// Without this, speaking two sentences in a row types "terrible.Basically" and "the flow
/// bar.You show me" - every sentence boundary in a dictated paragraph loses its space, because
/// each take is injected at the caret with nothing between them.
pub fn with_spacing(text: &str, trailing_space: bool) -> String {
    if !trailing_space {
        return text.to_owned();
    }
    // Nothing to separate if it already ends in whitespace, and a line break is its own
    // separator - a trailing space after one would sit at the start of the next line.
    match text.chars().last() {
        Some(c) if c.is_whitespace() => text.to_owned(),
        Some(_) => format!("{text} "),
        None => text.to_owned(),
    }
}

/// Queue replacement text for a command-mode edit.
///
/// No trailing space, unlike a dictation: this is going *over* a selection rather than after
/// the caret, and a space here would push whatever follows the selection along by one.
pub fn deliver_replacement(app: &AppHandle, text: &str) {
    if text.is_empty() {
        return;
    }
    let target = crate::context::last().map(|c| c.app).unwrap_or_default();
    let rule = crate::settings::load().rule_for(&target);
    // No auto-send on a command-mode edit: the user was rewriting text in place, not composing
    // a message, and sending it would be irreversible.
    let _ = injector().send(Job {
        app: app.clone(),
        text: text.to_owned(),
        target,
        method: method_override(&rule.method),
        auto_send: false,
    });
}

/// Queue text for delivery to the app that was focused when the dictation started.
pub fn deliver(app: &AppHandle, text: &str) {
    if text.is_empty() {
        return;
    }
    let settings = crate::settings::load();
    let target = crate::context::last().map(|c| c.app).unwrap_or_default();
    let rule = settings.rule_for(&target);
    // A message that sends itself needs no trailing space: there is nothing coming after it.
    let text = with_spacing(text, settings.trailing_space && !rule.auto_send);
    let _ = injector().send(Job {
        app: app.clone(),
        text,
        target,
        method: method_override(&rule.method),
        auto_send: rule.auto_send,
    });
}

/// A per-app method setting, or None to let `method_for` decide.
pub fn method_override(setting: &str) -> Option<Method> {
    match setting {
        "type" => Some(Method::Type),
        "paste" => Some(Method::Paste),
        _ => None,
    }
}

/// Press Enter on its own, for auto-send.
fn tap_enter() -> windows::core::Result<()> {
    let mut inputs = Vec::with_capacity(2);
    tap(VK_RETURN, &mut inputs);
    send(&inputs)
}

/// Synchronous injection. Returns the method that actually delivered the text, which is not
/// always the one asked for: a clipboard held open by another application must not cost the
/// user their dictation, so a failed paste is typed instead.
pub fn inject(text: &str, method: Method, app: &str) -> windows::core::Result<Method> {
    wait_for_modifiers_released();
    match method {
        Method::Type => type_text(text, is_chat_app(app)).map(|()| Method::Type),
        Method::Paste => match paste_text(text) {
            Ok(()) => Ok(Method::Paste),
            Err(e) => {
                crate::shell_log!("paste failed ({e}); typing instead");
                type_text(text, is_chat_app(app)).map(|()| Method::Type)
            }
        },
    }
}

// ---------------------------------------------------------------------------------------------
// keyboard

fn key(vk: VIRTUAL_KEY, scan: u16, flags: KEYBD_EVENT_FLAGS) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT { wVk: vk, wScan: scan, dwFlags: flags, time: 0, dwExtraInfo: 0 },
        },
    }
}

fn tap(vk: VIRTUAL_KEY, out: &mut Vec<INPUT>) {
    out.push(key(vk, 0, KEYBD_EVENT_FLAGS(0)));
    out.push(key(vk, 0, KEYEVENTF_KEYUP));
}

fn send(inputs: &[INPUT]) -> windows::core::Result<()> {
    if inputs.is_empty() {
        return Ok(());
    }
    let sent = unsafe { SendInput(inputs, std::mem::size_of::<INPUT>() as i32) };
    if sent as usize != inputs.len() {
        return Err(windows::core::Error::from_thread());
    }
    Ok(())
}

/// Tap a key nothing is bound to. Used as a heartbeat by the stress test and as the hook's own
/// liveness probe; every application ignores it.
pub fn tap_unassigned() {
    let vk = VIRTUAL_KEY(0x07);
    let mut inputs = Vec::with_capacity(2);
    tap(vk, &mut inputs);
    let _ = send(&inputs);
}

pub fn modifiers_down() -> bool {
    MODIFIERS
        .iter()
        .any(|vk| unsafe { GetAsyncKeyState(vk.0 as i32) as u16 & 0x8000 != 0 })
}

/// Typing while the chord is still physically held would turn every letter into a shortcut.
/// Returns false if the user never let go, in which case we type anyway rather than lose text.
pub fn wait_for_modifiers_released() -> bool {
    let deadline = Instant::now() + MODIFIER_WAIT;
    while modifiers_down() {
        if Instant::now() > deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(8));
    }
    true
}

/// How the unicode key events are handed to Windows. Batching them all into one `SendInput`
/// is the fastest, but not every target keeps up with it; see `Batching::probe`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Batching {
    /// Everything in one call.
    All,
    /// One call per character (a key-down and key-up pair).
    PerChar,
    /// One call per character, with a pause between them.
    PerCharSlow,
    /// One call per key event.
    PerEvent,
}

static BATCHING: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn set_batching(b: Batching) {
    let v = match b {
        Batching::All => 0,
        Batching::PerChar => 1,
        Batching::PerCharSlow => 2,
        Batching::PerEvent => 3,
    };
    BATCHING.store(v, std::sync::atomic::Ordering::Relaxed);
}

fn batching() -> Batching {
    match BATCHING.load(std::sync::atomic::Ordering::Relaxed) {
        1 => Batching::PerChar,
        2 => Batching::PerCharSlow,
        3 => Batching::PerEvent,
        _ => Batching::All,
    }
}

fn type_text(text: &str, shift_enter: bool) -> windows::core::Result<()> {
    let mut inputs: Vec<INPUT> = Vec::with_capacity(text.len() * 2);
    for ch in text.chars() {
        match ch {
            '\r' => continue,
            '\n' => {
                if shift_enter {
                    inputs.push(key(VK_SHIFT, 0, KEYBD_EVENT_FLAGS(0)));
                    tap(VK_RETURN, &mut inputs);
                    inputs.push(key(VK_SHIFT, 0, KEYEVENTF_KEYUP));
                } else {
                    tap(VK_RETURN, &mut inputs);
                }
            }
            '\t' => tap(VK_TAB, &mut inputs),
            _ => {
                let mut buf = [0u16; 2];
                for unit in ch.encode_utf16(&mut buf) {
                    inputs.push(key(VIRTUAL_KEY(0), *unit, KEYEVENTF_UNICODE));
                    inputs.push(key(VIRTUAL_KEY(0), *unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
                }
            }
        }
    }
    match batching() {
        Batching::All => {
            for chunk in inputs.chunks(128) {
                send(chunk)?;
            }
        }
        Batching::PerChar => {
            for pair in inputs.chunks(2) {
                send(pair)?;
            }
        }
        Batching::PerCharSlow => {
            for pair in inputs.chunks(2) {
                send(pair)?;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        Batching::PerEvent => {
            for one in inputs.chunks(1) {
                send(one)?;
            }
        }
    }
    Ok(())
}

fn paste_text(text: &str) -> windows::core::Result<()> {
    let previous = clipboard_set(text)?;
    let mut inputs = Vec::with_capacity(4);
    inputs.push(key(VK_CONTROL, 0, KEYBD_EVENT_FLAGS(0)));
    tap(VK_V, &mut inputs);
    inputs.push(key(VK_CONTROL, 0, KEYEVENTF_KEYUP));
    send(&inputs)?;
    if let Some(previous) = previous {
        // The target reads the clipboard when it gets round to handling Ctrl+V, so restoring
        // it synchronously would race that read. Put it back once the paste has landed.
        std::thread::Builder::new()
            .name("clipboard-restore".into())
            .spawn(move || {
                std::thread::sleep(CLIPBOARD_SETTLE);
                let _ = clipboard_set(&previous);
            })
            .ok();
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// clipboard

struct ClipboardGuard;

impl ClipboardGuard {
    /// The clipboard is a single global lock; another app may be holding it for a moment.
    fn open() -> windows::core::Result<Self> {
        let mut last = windows::core::Error::from_thread();
        for _ in 0..12 {
            if unsafe { OpenClipboard(Some(HWND(std::ptr::null_mut()))) }.is_ok() {
                return Ok(ClipboardGuard);
            }
            last = windows::core::Error::from_thread();
            std::thread::sleep(Duration::from_millis(20));
        }
        Err(last)
    }
}

impl Drop for ClipboardGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseClipboard();
        }
    }
}

/// Whatever text is on the clipboard right now.
pub fn clipboard_text() -> Option<String> {
    let _guard = ClipboardGuard::open().ok()?;
    unsafe { clipboard_get_locked() }
}

/// Send a chord like Ctrl+A: hold the modifier, tap the key, let go.
pub fn chord(modifier: VIRTUAL_KEY, key_vk: VIRTUAL_KEY) -> windows::core::Result<()> {
    let mut inputs = Vec::with_capacity(4);
    inputs.push(key(modifier, 0, KEYBD_EVENT_FLAGS(0)));
    tap(key_vk, &mut inputs);
    inputs.push(key(modifier, 0, KEYEVENTF_KEYUP));
    send(&inputs)
}

pub const CTRL: VIRTUAL_KEY = VK_CONTROL;

/// A marker the selection can never equal, so "Ctrl+C did nothing" is distinguishable from
/// "Ctrl+C copied text identical to what was already on the clipboard".
const COPY_PROBE: &str = "\u{200b}LocalFlow\u{200b}";
/// How long the focused app is given to answer Ctrl+C. Generous: this runs once per command,
/// after the user has stopped speaking, and a selection read as empty silently cancels the
/// whole thing.
const COPY_WAIT: Duration = Duration::from_millis(220);

/// Read the focused app's selection by copying it, and put the clipboard back.
///
/// The fallback for when UI Automation cannot see the selection, which is most of Electron and
/// a good deal of the web. It must not run while the command chord is still held - Ctrl+C on
/// top of Win+Alt is not Ctrl+C - so the caller waits for the modifiers first.
pub fn copy_selection() -> Option<String> {
    if !wait_for_modifiers_released() {
        return None;
    }
    // Writing the probe also gets us the real clipboard contents to restore afterwards.
    let previous = clipboard_set(COPY_PROBE).ok()?;
    let _ = chord(CTRL, VIRTUAL_KEY(0x43)); // Ctrl+C
    let deadline = Instant::now() + COPY_WAIT;
    let mut copied = None;
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
        match clipboard_text() {
            Some(text) if text != COPY_PROBE => {
                copied = Some(text);
                break;
            }
            _ => {}
        }
    }
    match previous {
        Some(prev) => {
            let _ = clipboard_set(&prev);
        }
        // There was nothing on the clipboard before; leave our probe off it either way.
        None => {
            let _ = clipboard_set("");
        }
    }
    copied.filter(|t| !t.trim().is_empty())
}

/// Replace the clipboard text, returning what was there so it can be restored.
fn clipboard_set(text: &str) -> windows::core::Result<Option<String>> {
    let guard = ClipboardGuard::open()?;
    let previous = unsafe { clipboard_get_locked() };

    let mut wide: Vec<u16> = text.encode_utf16().collect();
    wide.push(0);
    unsafe {
        let bytes = wide.len() * 2;
        let handle: HGLOBAL = GlobalAlloc(GMEM_MOVEABLE, bytes)?;
        let ptr = GlobalLock(handle) as *mut u16;
        if ptr.is_null() {
            return Err(windows::core::Error::from_thread());
        }
        std::ptr::copy_nonoverlapping(wide.as_ptr(), ptr, wide.len());
        let _ = GlobalUnlock(handle);
        EmptyClipboard()?;
        // Ownership of the block passes to the clipboard on success.
        SetClipboardData(CF_UNICODETEXT.0 as u32, Some(HANDLE(handle.0)))?;
    }
    drop(guard);
    Ok(previous)
}

/// Caller must already hold the clipboard open.
unsafe fn clipboard_get_locked() -> Option<String> {
    if !IsClipboardFormatAvailable(CF_UNICODETEXT.0 as u32).is_ok() {
        return None;
    }
    let handle = GetClipboardData(CF_UNICODETEXT.0 as u32).ok()?;
    let hglobal = HGLOBAL(handle.0);
    let ptr = GlobalLock(hglobal) as *const u16;
    if ptr.is_null() {
        return None;
    }
    let mut len = 0usize;
    while *ptr.add(len) != 0 && len < 4 * 1024 * 1024 {
        len += 1;
    }
    let text = String::from_utf16_lossy(std::slice::from_raw_parts(ptr, len));
    let _ = GlobalUnlock(hglobal);
    Some(text)
}
