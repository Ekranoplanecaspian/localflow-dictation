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
use windows::Win32::Foundation::{GlobalFree, HANDLE, HGLOBAL, HWND};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, EnumClipboardFormats, GetClipboardData,
    GetClipboardSequenceNumber, IsClipboardFormatAvailable, OpenClipboard, SetClipboardData,
};
use windows::Win32::System::Memory::{GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock, GMEM_MOVEABLE};
use windows::Win32::System::Ole::CF_UNICODETEXT;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS,
    KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, VIRTUAL_KEY, VK_CONTROL, VK_LWIN, VK_MENU, VK_RETURN,
    VK_RWIN, VK_SHIFT, VK_TAB, VK_V,
};

use crate::guard::LockExt;

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

    /// Text goes only where it was spoken, never into whatever is in front now.
    #[test]
    fn text_is_kept_when_the_window_changed_or_nothing_can_take_it() {
        let never = |_: isize| false;
        // Numbers that are not real windows: each is its own top-level window.
        assert_eq!(obstacle(1001, 1001, never), None);
        assert_eq!(obstacle(1001, 2002, never), Some(Obstacle::WindowChanged));
        assert_eq!(obstacle(0, 2002, never), None, "Paste last dictation: whatever is in front");
        assert_eq!(obstacle(1001, 0, never), Some(Obstacle::NoWindow));
        assert_eq!(obstacle(1001, 1001, |_| true), Some(Obstacle::Elevated));
        // The taskbar is in front: nowhere to type.
        use windows::Win32::UI::WindowsAndMessaging::FindWindowW;
        let class: Vec<u16> = "Shell_TrayWnd".encode_utf16().chain(Some(0)).collect();
        if let Ok(taskbar) = unsafe { FindWindowW(windows::core::PCWSTR(class.as_ptr()), None) } {
            let taskbar = taskbar.0 as isize;
            assert_eq!(obstacle(0, taskbar, never), Some(Obstacle::NoWindow));
        }
    }

    /// Keys for a window that is not in front are not sent at all - checked right before they
    /// go out, not only when the job started (found by an outside review of 0.2.3). The taskbar
    /// stands in for "a window the user has since left": it is never the one being typed into.
    #[test]
    fn keys_for_a_window_no_longer_in_front_are_not_sent() {
        use windows::Win32::UI::WindowsAndMessaging::FindWindowW;
        let class: Vec<u16> = "Shell_TrayWnd".encode_utf16().chain(Some(0)).collect();
        let Ok(taskbar) = (unsafe { FindWindowW(windows::core::PCWSTR(class.as_ptr()), None) }) else {
            return; // no taskbar (a service session): nothing to test against
        };
        let left = taskbar.0 as isize;
        assert!(!still_in_front(left));
        assert!(still_in_front(0), "no window in particular: whatever is in front");
        let typed = type_text("must not be typed", false, left);
        assert_eq!(typed.map_err(|e| e.code()), Err(WINDOW_CHANGED));
    }

    /// Typing that stops part-way keeps only what had not gone out (found reviewing 0.2.5).
    #[test]
    fn only_complete_characters_count_as_typed() {
        // "ab" then a Shift+Enter: two events each for a and b, four for the line break.
        let ends = [2, 4, 8];
        assert_eq!(chars_sent(&ends, 0), 0);
        assert_eq!(chars_sent(&ends, 4), 2);
        assert_eq!(chars_sent(&ends, 6), 2, "half a line break is not one");
        assert_eq!(chars_sent(&ends, 128), 3);
    }

    #[test]
    fn this_process_does_not_block_its_own_keystrokes() {
        assert!(!crate::win::keystrokes_blocked(0));
        // Asked by process id, this process's elevation is what its own token says: a normal
        // run is not taken for an elevated one. Not simply "not elevated": a CI runner is an
        // administrator.
        assert_eq!(crate::win::process_elevated_for_tests(std::process::id()), Some(own_token_elevated()));
    }

    fn own_token_elevated() -> bool {
        use windows::Win32::Foundation::{CloseHandle, HANDLE};
        use windows::Win32::Security::{GetTokenInformation, TokenElevation, TOKEN_ELEVATION, TOKEN_QUERY};
        use windows::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};
        unsafe {
            let mut token = HANDLE::default();
            OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token).expect("own token");
            let mut elevation = TOKEN_ELEVATION::default();
            let mut len = 0u32;
            GetTokenInformation(
                token,
                TokenElevation,
                Some(&mut elevation as *mut _ as *mut _),
                std::mem::size_of::<TOKEN_ELEVATION>() as u32,
                &mut len,
            )
            .expect("token elevation");
            let _ = CloseHandle(token);
            elevation.TokenIsElevated != 0
        }
    }

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

    /// Uses the real Windows clipboard, so it only runs when asked for
    /// (`cargo test -- --ignored`). Whatever was on the clipboard before is put back at the end.
    ///
    /// The bug: an image on the clipboard was not text, so nothing was saved, and a dictation
    /// replaced it for good.
    #[test]
    #[ignore]
    fn a_paste_puts_back_an_image_that_was_on_the_clipboard() {
        let users = {
            let _g = ClipboardGuard::open().unwrap();
            unsafe { Snapshot::take_locked() }.unwrap()
        };
        let result = std::panic::catch_unwind(|| {
            // A 1x1 32-bit DIB, the way screenshots arrive: header, then one pixel.
            let mut dib = Vec::new();
            for v in [40u32, 1, 1] {
                dib.extend_from_slice(&v.to_le_bytes());
            }
            dib.extend_from_slice(&1u16.to_le_bytes());
            dib.extend_from_slice(&32u16.to_le_bytes());
            dib.extend_from_slice(&[0u8; 24]);
            dib.extend_from_slice(&[0x11, 0x22, 0x33, 0xFF]);
            {
                let _g = ClipboardGuard::open().unwrap();
                unsafe {
                    EmptyClipboard().unwrap();
                    set_locked(8, &dib).unwrap(); // CF_DIB
                }
            }

            let saved = clipboard_swap("dictated text").unwrap();
            assert_eq!(clipboard_text().as_deref(), Some("dictated text"));
            saved.restore().unwrap();

            let back = {
                let _g = ClipboardGuard::open().unwrap();
                unsafe { Snapshot::take_locked() }.unwrap()
            };
            let image = back.0.iter().find(|(f, _)| *f == 8).expect("the image is back");
            assert_eq!(&image.1[..dib.len()], &dib[..]);
            assert_eq!(clipboard_text(), None, "and the dictated text is gone again");
        });
        users.restore().unwrap();
        result.unwrap();
    }

    /// Uses the real Windows clipboard (`cargo test -- --ignored`). Something copied while a paste
    /// was landing stays: the restore used to put the older clipboard back over it (found by an
    /// outside review of 0.2.3).
    #[test]
    #[ignore]
    fn a_restore_leaves_what_was_copied_after_the_paste() {
        let users = clipboard_snapshot().unwrap();
        let result = std::panic::catch_unwind(|| {
            clipboard_set(CF_UNICODETEXT.0 as u32, &wide_bytes("the user's old clipboard")).unwrap();
            let saved = clipboard_swap("dictated text").unwrap();
            let ours = unsafe { GetClipboardSequenceNumber() };
            // Another program copies during the settle time.
            clipboard_set(CF_UNICODETEXT.0 as u32, &wide_bytes("copied just now")).unwrap();
            saved.restore_unless_replaced(ours).unwrap();
            assert_eq!(clipboard_text().as_deref(), Some("copied just now"));

            // Nothing copied meanwhile: the old clipboard comes back as before.
            let saved = clipboard_swap("dictated text").unwrap();
            let ours = unsafe { GetClipboardSequenceNumber() };
            saved.restore_unless_replaced(ours).unwrap();
            assert_eq!(clipboard_text().as_deref(), Some("copied just now"));
        });
        users.restore().unwrap();
        result.unwrap();
    }

    fn wide_bytes(text: &str) -> Vec<u8> {
        text.encode_utf16().chain(std::iter::once(0)).flat_map(u16::to_le_bytes).collect()
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
    /// The window the take was spoken into; 0 for "whatever is in front" (Paste last dictation).
    window: isize,
    /// A password: typed, never pasted, and not kept for Paste last dictation.
    private: bool,
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
                    // One bad job must not end the thread every later dictation is typed by.
                    let app = job.app.clone();
                    if crate::guard::catch("injection", move || run_job(job)).is_none() {
                        let _ = app.emit("injected", json!({"error": "an internal fault while typing"}));
                    }
                }
            })
            .expect("injector thread");
        tx
    })
}

/// Why a job's text cannot go where it was meant to, right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Obstacle {
    /// A different window is in front than the one the take was spoken into.
    WindowChanged,
    /// The desktop or the taskbar is in front.
    NoWindow,
    /// The window's app runs as administrator; Windows drops LocalFlow's keystrokes to it.
    Elevated,
}

/// What stands between `text` and the window in front, if anything. `spoken_into` is the take's
/// window (0: none in particular).
pub fn obstacle(spoken_into: isize, in_front: isize, blocked: impl Fn(isize) -> bool) -> Option<Obstacle> {
    if in_front == 0 || crate::win::is_shell_window(in_front) {
        return Some(Obstacle::NoWindow);
    }
    if spoken_into != 0 && crate::win::root_window(spoken_into) != crate::win::root_window(in_front) {
        return Some(Obstacle::WindowChanged);
    }
    if blocked(in_front) {
        return Some(Obstacle::Elevated);
    }
    None
}

/// Whether `window` (0: none in particular) is still the window in front, on this session's own
/// desktop. Checked again right before keys go out, not only when a job starts: the waits in
/// between (for the chord's keys to come up, for the last paste's clipboard) are long enough for
/// the user to click somewhere else.
pub fn still_in_front(window: isize) -> bool {
    if window == 0 {
        return true;
    }
    crate::power::on_own_desktop() && obstacle(window, crate::win::foreground_window(), |_| false).is_none()
}

/// The most recent dictation, for Paste last dictation. Only in memory, and never a password.
static LAST: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

fn remember_last(text: &str) {
    *LAST.locked() = Some(text.to_owned());
}

/// Tell the flow bar, and the log, why text was kept rather than typed.
fn say_kept(app: &AppHandle, code: crate::problems::Code, chars: usize) {
    let text = crate::problems::bar(code, &[]);
    crate::shell_log!("[{}] kept {chars} chars: {text}", code.as_str());
    let _ = app.emit("notice", json!({"code": code.as_str(), "text": text, "hold_ms": 5000}));
}

/// A job whose text is not typed: say why, keep it for Win + Alt + V (never a password), and
/// report it.
fn keep(job: &Job, code: crate::problems::Code) {
    keep_after(job, code, 0);
}

/// `keep`, for a job whose first `typed` characters already reached its window: only the rest is
/// kept. Keeping the whole text made Win + Alt + V type the first part a second time.
fn keep_after(job: &Job, code: crate::problems::Code, typed: usize) {
    let rest: String = job.text.chars().skip(typed).collect();
    let chars = rest.chars().count();
    if !job.private && !rest.trim().is_empty() {
        remember_last(rest.trim_end());
    }
    if typed > 0 {
        crate::shell_log!("{typed} chars were typed before the window changed; the other {chars} are kept");
    }
    say_kept(&job.app, code, chars);
    let _ = job.app.emit(
        "injected",
        json!({"method": "kept", "chars": chars, "app": job.target, "ms": 0, "error": null}),
    );
}

thread_local! {
    /// How many characters `type_text` had sent when it stopped for a window change.
    static TYPED_BEFORE_STOP: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How many characters are complete once `sent` input events have gone out, given where each
/// character's events end.
fn chars_sent(char_ends: &[usize], sent: usize) -> usize {
    char_ends.iter().take_while(|end| **end <= sent).count()
}

/// The error for keys that stopped because their window was no longer in front.
const WINDOW_CHANGED: windows::core::HRESULT = windows::core::HRESULT(0x8004_1F01u32 as i32);

fn window_changed() -> windows::core::Error {
    windows::core::Error::new(WINDOW_CHANGED, "the window changed")
}

/// Type or paste one job's text, then report how it went.
fn run_job(job: Job) {
    let started = Instant::now();
    // A password is typed: pasting would put it on the clipboard.
    let wanted = if job.private {
        Method::Type
    } else {
        job.method.unwrap_or_else(|| method_for(&job.target, &job.text))
    };
    if !crate::e2e::typing() {
        // A soak run: everything up to here was the real thing; the keystrokes are left out.
        let _ = job.app.emit(
            "injected",
            json!({"method": "none", "chars": job.text.chars().count(), "app": job.target, "ms": 0, "error": null}),
        );
        return;
    }
    // Typed while Ctrl, Alt or Win is still held, every letter is a shortcut: Win + L locks the
    // PC, Ctrl + W closes the tab. Kept for Win + Alt + V instead (a password, never kept, is
    // said again). Waited for before the window is checked, not after: the user can click
    // another window during the wait, and the check has to see that.
    if !wait_for_modifiers_released() {
        let code = if job.private {
            crate::problems::PASSWORD_NOT_TYPED_KEYS_HELD
        } else {
            crate::problems::TEXT_KEPT_KEYS_HELD
        };
        return keep(&job, code);
    }
    // On the lock screen or a UAC prompt's secure desktop, nothing of this session can take
    // keystrokes, whatever window was in front before.
    let in_front = if crate::power::on_own_desktop() { crate::win::foreground_window() } else { 0 };
    if let Some(why) = obstacle(job.window, in_front, crate::win::keystrokes_blocked) {
        return match why {
            // A password never goes on the clipboard, where every program can read it and the
            // next Ctrl+V anywhere pastes it: the user types it themselves.
            Obstacle::Elevated if job.private => keep(&job, crate::problems::PASSWORD_NOT_TYPED_ADMIN),
            // Windows would drop the keystrokes, but not the user's own Ctrl+V.
            Obstacle::Elevated => match copy_private(&job.text) {
                Ok(()) => keep(&job, crate::problems::TEXT_COPIED_ADMIN),
                Err(e) => {
                    crate::shell_log!("could not copy for an administrator app: {e}");
                    keep(&job, crate::problems::TEXT_KEPT_NO_WINDOW)
                }
            },
            Obstacle::WindowChanged => keep(&job, crate::problems::TEXT_KEPT_WINDOW_CHANGED),
            Obstacle::NoWindow => keep(&job, crate::problems::TEXT_KEPT_NO_WINDOW),
        };
    }
    let result = match crate::e2e::may_type_now() {
        Ok(()) => inject(&job.text, wanted, &job.target, job.window),
        Err(why) => {
            crate::shell_log!("{why}");
            Err(windows::core::Error::new(windows::core::HRESULT(0x80004004u32 as i32), why))
        }
    };
    // The window went out of front while the keys were going out (or before the paste):
    // whatever had not reached it is not sent anywhere else, and that rest is kept.
    if matches!(&result, Err(e) if e.code() == WINDOW_CHANGED) {
        let typed = TYPED_BEFORE_STOP.with(|t| t.get());
        return keep_after(&job, crate::problems::TEXT_KEPT_WINDOW_CHANGED, typed);
    }
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
        if !still_in_front(job.window) {
            // Enter is irreversible: in another window it sends or runs something else.
            crate::shell_log!("auto-send skipped in {}: the window changed", job.target);
        } else if let Err(e) = tap_enter() {
            crate::shell_log!("auto-send failed in {}: {e}", job.target);
        } else {
            crate::shell_log!("auto-sent in {}", job.target);
        }
    }
    if result.is_ok() && !job.private {
        remember_last(job.text.trim_end());
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
///
/// `target` and `window` are where the command was spoken: the edit goes there or is kept, never
/// to whichever window the newest take used.
pub fn deliver_replacement(app: &AppHandle, text: &str, target: &str, window: isize) {
    if text.is_empty() {
        return;
    }
    let target = target.to_owned();
    let rule = crate::settings::load().rule_for(&target);
    // No auto-send on a command-mode edit: the user was rewriting text in place, not composing
    // a message, and sending it would be irreversible.
    let _ = injector().send(Job {
        app: app.clone(),
        text: text.to_owned(),
        target,
        window,
        private: false,
        method: method_override(&rule.method),
        auto_send: false,
    });
}

/// Queue text for delivery. `target` is the app that was focused when this take started, whose
/// rules apply - not the most recent one, which is a newer take's when takes overlap - and
/// `window` the window it was spoken into: text is never typed into a different one.
pub fn deliver(app: &AppHandle, text: &str, target: &str, window: isize, private: bool) {
    if text.is_empty() {
        return;
    }
    let settings = crate::settings::load();
    let target = target.to_owned();
    let rule = settings.rule_for(&target);
    // A message that sends itself needs no trailing space: there is nothing coming after it.
    // Nor does a password: a space would become part of it.
    let text = with_spacing(text, settings.trailing_space && !rule.auto_send && !private);
    let _ = injector().send(Job {
        app: app.clone(),
        text,
        target,
        window,
        private,
        method: method_override(&rule.method),
        // A password field's Enter is the user's to press.
        auto_send: rule.auto_send && !private,
    });
}

/// Paste last dictation (Win+Alt+V, and the tray): the most recent dictation again, into
/// whatever is in front now - the way to rescue text that was kept rather than typed.
pub fn paste_last(app: &AppHandle) {
    let Some(text) = LAST.locked().clone() else {
        let code = crate::problems::PASTE_LAST_EMPTY;
        let _ = app.emit("notice", json!({"code": code.as_str(), "text": crate::problems::bar(code, &[])}));
        return;
    };
    let ctx = crate::context::foreground();
    let rule = crate::settings::load().rule_for(&ctx.app);
    crate::shell_log!("paste last dictation ({} chars) into {}", text.chars().count(), ctx.app);
    let _ = injector().send(Job {
        app: app.clone(),
        text: with_spacing(&text, crate::settings::load().trailing_space),
        target: ctx.app,
        window: 0,
        private: false,
        method: method_override(&rule.method),
        auto_send: false,
    });
}

/// The tray's Paste last dictation. By the time its item fires, the taskbar or LocalFlow's own
/// menu window is in front, not the app the text is for: go back to that app first.
pub fn paste_last_from_tray(app: &AppHandle) {
    let front = crate::win::foreground_window();
    let mut front_pid = 0u32;
    unsafe {
        windows::Win32::UI::WindowsAndMessaging::GetWindowThreadProcessId(
            windows::Win32::Foundation::HWND(front as *mut _),
            Some(&mut front_pid),
        )
    };
    if crate::win::is_shell_window(front) || front_pid == std::process::id() {
        let back = crate::win::last_app_window();
        let ok = back != 0 && crate::win::bring_to_front(back);
        crate::shell_log!(
            "paste last dictation from the tray: back to {} ({})",
            crate::win::describe(back),
            if ok { "in front" } else { "Windows would not bring it to the front" }
        );
    }
    paste_last(app);
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
///
/// `window` (0: none in particular) is checked again before every batch of keys and before the
/// paste; once it is not in front, nothing more is sent and the error is `WINDOW_CHANGED`.
pub fn inject(text: &str, method: Method, app: &str, window: isize) -> windows::core::Result<Method> {
    wait_for_modifiers_released();
    match method {
        Method::Type => type_text(text, is_chat_app(app), window).map(|()| Method::Type),
        Method::Paste => match paste_text(text, window) {
            Ok(()) => Ok(Method::Paste),
            Err(e) if e.code() == WINDOW_CHANGED => Err(e),
            Err(e) => {
                crate::shell_log!("paste failed ({e}); typing instead");
                type_text(text, is_chat_app(app), window).map(|()| Method::Type)
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
/// Returns false if the user never let go; a dictation is then kept rather than typed (`run_job`).
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

fn type_text(text: &str, shift_enter: bool, window: isize) -> windows::core::Result<()> {
    let mut inputs: Vec<INPUT> = Vec::with_capacity(text.len() * 2);
    let mut char_ends: Vec<usize> = Vec::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\r' => {}
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
        char_ends.push(inputs.len());
    }
    // Each batch goes only to the window it is for: one that left the front part-way through
    // gets no more of it, and neither does whatever came in front instead.
    let mut sent = 0usize;
    let mut send_to = |batch: &[INPUT]| {
        if !still_in_front(window) {
            TYPED_BEFORE_STOP.with(|t| t.set(chars_sent(&char_ends, sent)));
            return Err(window_changed());
        }
        send(batch)?;
        sent += batch.len();
        Ok(())
    };
    match batching() {
        Batching::All => {
            for chunk in inputs.chunks(128) {
                send_to(chunk)?;
            }
        }
        Batching::PerChar => {
            for pair in inputs.chunks(2) {
                send_to(pair)?;
            }
        }
        Batching::PerCharSlow => {
            for pair in inputs.chunks(2) {
                send_to(pair)?;
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        Batching::PerEvent => {
            for one in inputs.chunks(1) {
                send_to(one)?;
            }
        }
    }
    Ok(())
}

fn paste_text(text: &str, window: isize) -> windows::core::Result<()> {
    // Waits for the last paste's clipboard to be put back first: the window is checked after.
    let previous = clipboard_swap(text)?;
    // Which clipboard is ours: anything copied after this, the restore leaves alone.
    let ours = unsafe { GetClipboardSequenceNumber() };
    if !still_in_front(window) {
        let _ = previous.restore_unless_replaced(ours);
        TYPED_BEFORE_STOP.with(|t| t.set(0));
        return Err(window_changed());
    }
    let mut inputs = Vec::with_capacity(4);
    inputs.push(key(VK_CONTROL, 0, KEYBD_EVENT_FLAGS(0)));
    tap(VK_V, &mut inputs);
    inputs.push(key(VK_CONTROL, 0, KEYEVENTF_KEYUP));
    if let Err(e) = send(&inputs) {
        let _ = previous.restore();
        return Err(e);
    }
    // The target reads the clipboard when it gets round to handling Ctrl+V, so restoring it
    // synchronously would race that read. Put it back once the paste has landed - and have
    // the next clipboard user wait for that, or a second paste within the settle time would
    // take this dictation for the user's clipboard and restore it instead of theirs.
    let restore = std::thread::Builder::new().name("clipboard-restore".into()).spawn(move || {
        std::thread::sleep(CLIPBOARD_SETTLE);
        let _ = previous.restore_unless_replaced(ours);
    });
    if let Ok(handle) = restore {
        *PENDING_RESTORE.locked() = Some(handle);
    }
    Ok(())
}

/// The restore still to come from the last paste.
static PENDING_RESTORE: std::sync::Mutex<Option<std::thread::JoinHandle<()>>> =
    std::sync::Mutex::new(None);

/// Let the last paste put the user's clipboard back before anything else takes it.
fn wait_for_restore() {
    let pending = PENDING_RESTORE.locked().take();
    if let Some(handle) = pending {
        let _ = handle.join();
    }
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

/// Everything on the clipboard now, to be put back with `restore`.
pub fn clipboard_snapshot() -> Option<Snapshot> {
    let _guard = ClipboardGuard::open().ok()?;
    unsafe { Snapshot::take_locked() }
}

/// Replace the clipboard with one block of data in one format.
pub fn clipboard_set(format: u32, bytes: &[u8]) -> windows::core::Result<()> {
    let _guard = ClipboardGuard::open()?;
    unsafe {
        EmptyClipboard()?;
        set_locked(format, bytes)
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
    let previous = clipboard_swap(COPY_PROBE).ok()?;
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
    // Whatever was there - text, an image, files, or nothing - goes back, and the probe with it.
    let _ = previous.restore();
    copied.filter(|t| !t.trim().is_empty())
}

/// Clipboard formats whose data is a GDI object rather than a block of memory, so it cannot be
/// copied as bytes: bitmap, metafile picture, palette, enhanced metafile, and the owner-display
/// family. Nothing is lost by skipping them: Windows synthesises a bitmap from the DIB that
/// every image copy also carries.
const GDI_FORMATS: [u32; 8] = [2, 3, 9, 14, 0x80, 0x82, 0x83, 0x8E];
/// More than this on the clipboard is not copied aside; the paste is typed instead.
const SNAPSHOT_MAX: usize = 256 << 20;

/// Everything on the clipboard, as (format, bytes), so it can be put back exactly.
///
/// The clipboard used to be saved as text only. A screenshot or a copied file was not text, so
/// nothing was saved, and dictating replaced it for good with the dictated sentence.
pub struct Snapshot(Vec<(u32, Vec<u8>)>);

impl Snapshot {
    /// The bytes held in one format, if it is there.
    pub fn format(&self, format: u32) -> Option<&[u8]> {
        self.0.iter().find(|(f, _)| *f == format).map(|(_, b)| b.as_slice())
    }

    /// Caller must hold the clipboard open. None when it holds more than is sensible to copy.
    unsafe fn take_locked() -> Option<Snapshot> {
        let mut formats = Vec::new();
        let mut total = 0usize;
        let mut format = 0u32;
        loop {
            format = EnumClipboardFormats(format);
            if format == 0 {
                break;
            }
            if GDI_FORMATS.contains(&format) {
                continue;
            }
            let Ok(handle) = GetClipboardData(format) else { continue };
            let block = HGLOBAL(handle.0);
            let size = GlobalSize(block);
            if size == 0 {
                continue;
            }
            total += size;
            if total > SNAPSHOT_MAX {
                return None;
            }
            let ptr = GlobalLock(block) as *const u8;
            if ptr.is_null() {
                continue;
            }
            formats.push((format, std::slice::from_raw_parts(ptr, size).to_vec()));
            let _ = GlobalUnlock(block);
        }
        Some(Snapshot(formats))
    }

    /// Put it all back. An empty snapshot leaves the clipboard empty, as it was.
    pub fn restore(&self) -> windows::core::Result<()> {
        let _guard = ClipboardGuard::open()?;
        unsafe { self.restore_locked() }
    }

    /// Put it back unless something else has been copied since the clipboard was LocalFlow's
    /// (`ours`, its sequence number then). A paste puts the user's clipboard back a moment
    /// after it lands, and whatever the user or another program copied in that moment was
    /// overwritten by the older contents. Checked with the clipboard held open, so nothing can
    /// copy between the check and the restore.
    pub fn restore_unless_replaced(&self, ours: u32) -> windows::core::Result<()> {
        let _guard = ClipboardGuard::open()?;
        if unsafe { GetClipboardSequenceNumber() } != ours {
            crate::shell_log!("something was copied while a paste was landing; leaving it on the clipboard");
            return Ok(());
        }
        unsafe { self.restore_locked() }
    }

    /// Caller must hold the clipboard open.
    unsafe fn restore_locked(&self) -> windows::core::Result<()> {
        EmptyClipboard()?;
        for (format, bytes) in &self.0 {
            let _ = set_locked(*format, bytes);
        }
        Ok(())
    }
}

/// Caller must hold the clipboard open. The block belongs to the clipboard once it is set.
unsafe fn set_locked(format: u32, bytes: &[u8]) -> windows::core::Result<()> {
    let block: HGLOBAL = GlobalAlloc(GMEM_MOVEABLE, bytes.len().max(1))?;
    let ptr = GlobalLock(block) as *mut u8;
    if ptr.is_null() {
        let _ = GlobalFree(Some(block));
        return Err(windows::core::Error::from_thread());
    }
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
    let _ = GlobalUnlock(block);
    if let Err(e) = SetClipboardData(format, Some(HANDLE(block.0))) {
        let _ = GlobalFree(Some(block));
        return Err(e);
    }
    Ok(())
}

/// Put `text` on the clipboard, returning everything that was there so it can be restored.
/// Fails rather than lose the user's clipboard when it cannot be saved first.
fn clipboard_swap(text: &str) -> windows::core::Result<Snapshot> {
    wait_for_restore();
    let _guard = ClipboardGuard::open()?;
    unsafe {
        let Some(previous) = Snapshot::take_locked() else {
            crate::shell_log!("the clipboard holds too much to set aside; typing instead of pasting");
            return Err(windows::core::Error::from_hresult(windows::core::HRESULT(0x8007000Eu32 as i32)));
        };
        let wide: Vec<u8> =
            text.encode_utf16().chain(std::iter::once(0)).flat_map(u16::to_le_bytes).collect();
        EmptyClipboard()?;
        set_locked(CF_UNICODETEXT.0 as u32, &wide)?;
        mark_private_locked();
        Ok(previous)
    }
}

/// Clipboard formats Windows reads to keep an item out of clipboard history (Win+V) and the
/// cloud clipboard. A dictation passes through the clipboard for a moment on its way into a
/// paste; it is the user's words, not something they copied, and must not be kept there.
unsafe fn mark_private_locked() {
    use windows::Win32::System::DataExchange::RegisterClipboardFormatW;
    let format = |name: &str| {
        let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        RegisterClipboardFormatW(windows::core::PCWSTR(wide.as_ptr()))
    };
    let zero = 0u32.to_le_bytes();
    for (name, bytes) in [
        ("ExcludeClipboardContentFromMonitorProcessing", &zero[..]),
        ("CanIncludeInClipboardHistory", &zero[..]),
        ("CanUploadToCloudClipboard", &zero[..]),
    ] {
        let id = format(name);
        if id != 0 {
            let _ = set_locked(id, bytes);
        }
    }
}

/// Put `text` on the clipboard for the user to paste themselves, kept out of clipboard history.
pub fn copy_private(text: &str) -> windows::core::Result<()> {
    wait_for_restore();
    let _guard = ClipboardGuard::open()?;
    unsafe {
        let wide: Vec<u8> = text.encode_utf16().chain(std::iter::once(0)).flat_map(u16::to_le_bytes).collect();
        EmptyClipboard()?;
        set_locked(CF_UNICODETEXT.0 as u32, &wide)?;
        mark_private_locked();
    }
    Ok(())
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
    // Bounded by the block's own size rather than a fixed limit: text cut off at a limit would
    // come back as a selection shorter than the real one, and command mode would paste its
    // edit of that over everything the user had selected.
    let units = GlobalSize(hglobal) / 2;
    let mut len = 0usize;
    while len < units && *ptr.add(len) != 0 {
        len += 1;
    }
    let text = String::from_utf16_lossy(std::slice::from_raw_parts(ptr, len));
    let _ = GlobalUnlock(hglobal);
    Some(text)
}
