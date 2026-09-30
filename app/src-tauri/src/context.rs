//! What is on the other end of the caret: the focused window, its process and title, and
//! (through UI Automation) the text around the caret and the browser URL.
//!
//! The cheap half runs on every hotkey press. The UI Automation half is best-effort and is
//! given a hard time budget, because some apps answer slowly and a dictation must not wait.

use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};
use windows::Win32::Foundation::{CloseHandle, HWND, MAX_PATH};
use windows::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_FORMAT, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetForegroundWindow, GetWindowTextLengthW, GetWindowTextW, GetWindowThreadProcessId,
};

use crate::guard::LockExt;

/// How long UI Automation gets before we dictate without it.
const UIA_BUDGET: Duration = Duration::from_millis(120);
/// Enough text before the caret to decide spacing and capitalisation.
const BEFORE_CARET_CHARS: usize = 160;
/// The most selected text read through UI Automation. It is read on every hotkey press, so it
/// is capped; a selection longer than this is copied instead when command mode needs it.
const SELECTION_MAX: usize = 4096;

/// A selection read through UI Automation, or None when it may have been cut short.
///
/// Command mode pastes its answer over the *whole* selection. Given only the first 4096
/// characters of a longer one, it used to replace all of it with an edit of that first part,
/// deleting the rest of the user's text. So `read` is asked for one character more than the
/// cap: getting that many back means there may be more, and a selection that may be partial is
/// no selection at all. Counted in UTF-16 units, which is how UI Automation counts; a count
/// that disagrees can only err towards copying the selection, which is always safe.
fn complete_selection(read: impl FnOnce(i32) -> Option<String>) -> Option<String> {
    let text = read(SELECTION_MAX as i32 + 1)?;
    if text.encode_utf16().count() > SELECTION_MAX {
        crate::shell_log!("the selection is longer than UI Automation reads; it will be copied instead");
        return None;
    }
    Some(text).filter(|s| !s.is_empty())
}

/// How much of the document the fallback reads before giving up on it.
const DOCUMENT_READ_MAX: usize = 4096;

/// The text just before the caret, clipped to `BEFORE_CARET_CHARS`.
///
/// `near` reads the characters right before the caret. `from_start(max)` reads the document
/// from its start up to the caret, at most `max` characters, for providers that cannot do the
/// first. That second read used to be the only one, and in a document longer than its cap it
/// returned the document's *opening* instead - so spacing, capitalisation and the clean-up
/// model's context all came from text nowhere near the caret. Now it is only believed when it
/// provably reached the caret (it came back shorter than the cap), and otherwise there is no
/// context at all, which is always better than the wrong one.
fn before_caret_text(
    near: impl FnOnce() -> Option<String>,
    from_start: impl FnOnce(i32) -> Option<String>,
) -> Option<String> {
    let text = match near() {
        Some(t) => t,
        None => {
            let t = from_start(DOCUMENT_READ_MAX as i32 + 1)?;
            if t.encode_utf16().count() > DOCUMENT_READ_MAX {
                return None;
            }
            t
        }
    };
    let chars: Vec<char> = text.chars().collect();
    let start = chars.len().saturating_sub(BEFORE_CARET_CHARS);
    Some(chars[start..].iter().collect::<String>()).filter(|s| !s.is_empty())
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Context {
    /// Executable name, lowercase, without the path: "code.exe", "slack.exe".
    pub app: String,
    pub title: String,
    pub pid: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_caret: Option<String>,
    /// The focused control is a password field: the take is typed exactly as heard, and kept
    /// out of the history, the log, the clean-up model and Paste last dictation.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub password: bool,
    #[serde(skip)]
    pub hwnd: isize,
}

impl Context {
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|_| json!({}))
    }
}

fn window_title(hwnd: HWND) -> String {
    unsafe {
        let len = GetWindowTextLengthW(hwnd);
        if len <= 0 {
            return String::new();
        }
        let mut buf = vec![0u16; len as usize + 1];
        let n = GetWindowTextW(hwnd, &mut buf);
        String::from_utf16_lossy(&buf[..n as usize])
    }
}

fn process_name(pid: u32) -> String {
    unsafe {
        let Ok(handle) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else {
            return String::new();
        };
        let mut buf = vec![0u16; MAX_PATH as usize];
        let mut len = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(
            handle,
            PROCESS_NAME_FORMAT(0),
            windows::core::PWSTR(buf.as_mut_ptr()),
            &mut len,
        )
        .is_ok();
        let _ = CloseHandle(handle);
        if !ok {
            return String::new();
        }
        let path = String::from_utf16_lossy(&buf[..len as usize]);
        path.rsplit(['\\', '/']).next().unwrap_or(&path).to_lowercase()
    }
}

/// The foreground window, cheaply. Microseconds; safe to call from the hook thread.
pub fn foreground() -> Context {
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd.0.is_null() {
            return Context::default();
        }
        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        Context {
            app: process_name(pid),
            title: window_title(hwnd),
            pid,
            hwnd: hwnd.0 as isize,
            ..Default::default()
        }
    }
}

/// Fill in what UI Automation can tell us about the text: the selection and the text just
/// before the caret. Best-effort and time-boxed. Runs on every hotkey press.
pub fn enrich(ctx: &mut Context) {
    let deadline = std::time::Instant::now() + UIA_BUDGET;
    if let Some(found) = uia::focused_text(deadline) {
        ctx.selection = found.selection.filter(|s| !s.is_empty());
        ctx.before_caret = found.before_caret.filter(|s| !s.is_empty());
        ctx.password = found.password;
    }
}

/// The address of the page, in a browser. For the diagnostics only.
///
/// This used to run on every hotkey press, and nothing used the answer - the engine never reads
/// it. It is also the one lookup the time budget could not bound: it searches the browser's
/// whole UI tree for the address bar, and in a heavy page that search is one long call. Every
/// millisecond of it came before the microphone was switched on.
pub fn add_url(ctx: &mut Context) {
    if is_browser(&ctx.app) {
        ctx.url = uia::browser_url(ctx.hwnd, std::time::Instant::now() + UIA_BUDGET);
    }
}

pub fn is_browser(app: &str) -> bool {
    matches!(
        app,
        "chrome.exe" | "msedge.exe" | "brave.exe" | "firefox.exe" | "opera.exe" | "vivaldi.exe"
            | "arc.exe" | "zen.exe" | "librewolf.exe" | "chromium.exe"
    )
}

/// The last foreground context captured at the start of a dictation. Injection needs it after
/// the fact, and by then the user may have clicked elsewhere.
static LAST: Mutex<Option<Context>> = Mutex::new(None);

pub fn remember(ctx: &Context) {
    *LAST.locked() = Some(ctx.clone());
}

pub fn last() -> Option<Context> {
    LAST.locked().clone()
}

// ---------------------------------------------------------------------------------------------

mod uia {
    use super::*;
    use std::cell::RefCell;
    use windows::core::BSTR;
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
    };
    use windows::Win32::UI::Accessibility::{
        CUIAutomation, IUIAutomation, IUIAutomationElement, IUIAutomationTextPattern,
        IUIAutomationTextRange, IUIAutomationValuePattern, TextPatternRangeEndpoint_End,
        TextPatternRangeEndpoint_Start, TextUnit_Character, TreeScope_Descendants,
        UIA_ControlTypePropertyId, UIA_EditControlTypeId, UIA_TextPatternId, UIA_ValuePatternId,
        UIA_ValueValuePropertyId,
    };

    pub struct Found {
        pub selection: Option<String>,
        pub before_caret: Option<String>,
        pub password: bool,
    }

    thread_local! {
        /// One automation object per thread; creating it costs milliseconds.
        static AUTOMATION: RefCell<Option<IUIAutomation>> = const { RefCell::new(None) };
    }

    fn automation() -> Option<IUIAutomation> {
        AUTOMATION.with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                unsafe {
                    // UI Automation wants an initialised apartment; multi-threaded suits a
                    // worker that only reads.
                    let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
                    *slot = CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER).ok();
                }
            }
            slot.clone()
        })
    }

    fn expired(deadline: std::time::Instant) -> bool {
        std::time::Instant::now() >= deadline
    }

    /// Selection and the text just before the caret, from whatever control has focus.
    pub fn focused_text(deadline: std::time::Instant) -> Option<Found> {
        let auto = automation()?;
        let element: IUIAutomationElement = unsafe { auto.GetFocusedElement().ok()? };
        if expired(deadline) {
            return None;
        }
        // A password field is never read: its value is the password. (Reading "the text before
        // the caret" through the value pattern used to fetch exactly that.)
        if unsafe { element.CurrentIsPassword() }.is_ok_and(|b| b.as_bool()) {
            return Some(Found { selection: None, before_caret: None, password: true });
        }
        unsafe {
            let Ok(pattern) = element.GetCurrentPatternAs::<IUIAutomationTextPattern>(UIA_TextPatternId)
            else {
                // No text pattern: a value pattern still gives us the whole field.
                let value = element
                    .GetCurrentPatternAs::<IUIAutomationValuePattern>(UIA_ValuePatternId)
                    .ok()
                    .and_then(|v| v.CurrentValue().ok())
                    .map(|s: BSTR| s.to_string())
                    .filter(|s| !s.is_empty());
                return Some(Found {
                    selection: None,
                    before_caret: value.map(|v| tail(&v)),
                    password: false,
                });
            };

            let mut selection = None;
            if let Ok(ranges) = pattern.GetSelection() {
                if ranges.Length().unwrap_or(0) > 0 {
                    if let Ok(range) = ranges.GetElement(0) {
                        selection = complete_selection(|max| range.GetText(max).ok().map(|t| t.to_string()));
                    }
                }
            }
            if expired(deadline) {
                return Some(Found { selection, before_caret: None, password: false });
            }

            // Text before the caret.
            let mut before = None;
            if let Ok(ranges) = pattern.GetSelection() {
                if ranges.Length().unwrap_or(0) > 0 {
                    if let Ok(caret) = ranges.GetElement(0) {
                        before = before_caret_text(
                            || near_caret(&caret),
                            |max| from_document_start(&pattern, &caret, max),
                        );
                    }
                }
            }
            Some(Found { selection, before_caret: before, password: false })
        }
    }

    /// The characters just before the caret: the caret's range, collapsed to its start, with
    /// that start moved back. Right wherever the caret is, and cheap in a document of any size.
    pub(super) unsafe fn near_caret(caret: &IUIAutomationTextRange) -> Option<String> {
        let head = caret.Clone().ok()?;
        head.MoveEndpointByRange(TextPatternRangeEndpoint_End, caret, TextPatternRangeEndpoint_Start)
            .ok()?;
        head.MoveEndpointByUnit(TextPatternRangeEndpoint_Start, TextUnit_Character, -(BEFORE_CARET_CHARS as i32))
            .ok()?;
        // Twice the count, in case the provider counts a surrogate pair as two.
        Some(head.GetText(2 * BEFORE_CARET_CHARS as i32).ok()?.to_string())
    }

    /// For providers that cannot move a range by characters: the document from its start to the
    /// caret, at most `max` characters of it.
    unsafe fn from_document_start(
        pattern: &IUIAutomationTextPattern,
        caret: &IUIAutomationTextRange,
        max: i32,
    ) -> Option<String> {
        let head = pattern.DocumentRange().ok()?;
        head.MoveEndpointByRange(TextPatternRangeEndpoint_End, caret, TextPatternRangeEndpoint_Start)
            .ok()?;
        Some(head.GetText(max).ok()?.to_string())
    }

    fn tail(s: &str) -> String {
        let chars: Vec<char> = s.chars().collect();
        let start = chars.len().saturating_sub(BEFORE_CARET_CHARS);
        chars[start..].iter().collect()
    }

    /// The URL in a Chromium or Firefox window, read from the address bar's value.
    pub fn browser_url(hwnd: isize, deadline: std::time::Instant) -> Option<String> {
        if hwnd == 0 || expired(deadline) {
            return None;
        }
        let auto = automation()?;
        unsafe {
            let root = auto.ElementFromHandle(HWND(hwnd as *mut std::ffi::c_void)).ok()?;
            let cond = auto
                .CreatePropertyCondition(
                    UIA_ControlTypePropertyId,
                    &windows::Win32::System::Variant::VARIANT::from(UIA_EditControlTypeId.0),
                )
                .ok()?;
            let edit = root.FindFirst(TreeScope_Descendants, &cond).ok()?;
            let value = edit.GetCurrentPropertyValue(UIA_ValueValuePropertyId).ok()?;
            let text = BSTR::try_from(&value).ok()?.to_string();
            if text.is_empty() || text.contains(' ') {
                return None; // a search box, or the user is mid-typing
            }
            Some(text)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_password_field_is_marked_and_an_ordinary_one_is_not() {
        let ctx = Context { app: "chrome.exe".into(), password: true, ..Context::default() };
        assert_eq!(ctx.to_json()["password"], serde_json::json!(true));
        let ctx = Context { app: "chrome.exe".into(), ..Context::default() };
        assert!(ctx.to_json().get("password").is_none(), "only sent when it is one");
    }


    #[test]
    fn a_selection_within_the_cap_is_used_whole() {
        assert_eq!(complete_selection(|_| Some("make this formal".into())).as_deref(), Some("make this formal"));
        let exactly = "a".repeat(SELECTION_MAX);
        assert_eq!(complete_selection(|_| Some(exactly.clone())), Some(exactly));
    }

    /// The data-loss case: a read that comes back full may have been cut short, and command
    /// mode would paste its edit of that part over the whole selection.
    #[test]
    fn a_selection_that_may_have_been_cut_short_is_not_used() {
        let asked = std::cell::Cell::new(0);
        let got = complete_selection(|max| {
            asked.set(max);
            Some("a".repeat(max as usize))
        });
        assert_eq!(got, None);
        assert_eq!(asked.get() as usize, SELECTION_MAX + 1, "asks for one more than it keeps");
    }

    #[test]
    fn the_text_near_the_caret_is_used_and_clipped() {
        let long = format!("{}the end", "x".repeat(500));
        let got = before_caret_text(|| Some(long.clone()), |_| panic!("not needed")).unwrap();
        assert_eq!(got.chars().count(), BEFORE_CARET_CHARS);
        assert!(got.ends_with("the end"));
    }

    /// The bug: reading from the start of a long document returned its opening, not the text
    /// at the caret. The fallback is only believed when it provably reached the caret.
    #[test]
    fn a_document_read_that_may_not_reach_the_caret_is_not_used() {
        let got = before_caret_text(|| None, |max| Some("opening of the document ".repeat(max as usize)));
        assert_eq!(got, None, "no context beats the wrong context");
    }

    #[test]
    fn a_short_document_is_read_from_its_start_when_nothing_else_works() {
        let asked = std::cell::Cell::new(0);
        let got = before_caret_text(
            || None,
            |max| {
                asked.set(max);
                Some("Dear Priya,\nthanks for".into())
            },
        );
        assert_eq!(got.as_deref(), Some("Dear Priya,\nthanks for"));
        assert_eq!(asked.get() as usize, DOCUMENT_READ_MAX + 1);
    }

    /// Real UI Automation against a real (hidden) Windows edit box holding a long document, with
    /// the caret at the end. Only runs when asked for (`cargo test -- --ignored`).
    #[test]
    #[ignore]
    fn the_text_before_the_caret_of_a_long_document_is_read_at_the_caret() {
        use windows::core::w;
        use windows::Win32::Foundation::{LPARAM, WPARAM};
        use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED};
        use windows::Win32::UI::Accessibility::{CUIAutomation, IUIAutomation, IUIAutomationTextPattern, UIA_TextPatternId};
        use windows::Win32::UI::WindowsAndMessaging::{
            CreateWindowExW, DestroyWindow, DispatchMessageW, PeekMessageW, SendMessageW, SetWindowTextW,
            TranslateMessage, ES_MULTILINE, MSG, PM_REMOVE, WINDOW_EX_STYLE, WINDOW_STYLE, WS_OVERLAPPEDWINDOW,
        };

        let doc = format!("{}and the caret is right here", "The opening of a long document. ".repeat(320));
        assert!(doc.len() > 10_000);
        unsafe {
            // RichEdit, which speaks UI Automation's text pattern itself (a plain EDIT does not).
            windows::Win32::System::LibraryLoader::LoadLibraryW(w!("Msftedit.dll")).unwrap();
            let edit = CreateWindowExW(
                WINDOW_EX_STYLE(0), w!("RICHEDIT50W"), w!(""),
                WS_OVERLAPPEDWINDOW | WINDOW_STYLE(ES_MULTILINE as u32),
                0, 0, 400, 300, None, None, None, None,
            )
            .unwrap();
            SendMessageW(edit, 0x00C5, Some(WPARAM(0)), Some(LPARAM(0))); // EM_LIMITTEXT: no limit
            let wide: Vec<u16> = doc.encode_utf16().chain(std::iter::once(0)).collect();
            SetWindowTextW(edit, windows::core::PCWSTR(wide.as_ptr())).unwrap();
            let end = doc.encode_utf16().count();
            SendMessageW(edit, 0x00B1, Some(WPARAM(end)), Some(LPARAM(end as isize))); // EM_SETSEL: caret at the end

            // UI Automation must not run on the thread that owns the window; this one pumps
            // its messages while another thread asks.
            let raw = edit.0 as isize;
            let worker = std::thread::spawn(move || {
                let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
                let auto: IUIAutomation = CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER).unwrap();
                let el = auto.ElementFromHandle(HWND(raw as *mut std::ffi::c_void)).unwrap();
                let pattern: IUIAutomationTextPattern = el.GetCurrentPatternAs(UIA_TextPatternId).unwrap();
                let caret = pattern.GetSelection().unwrap().GetElement(0).unwrap();
                let near = uia::near_caret(&caret);
                let old = {
                    // What the code used to read: the document from its start, first 4096 characters.
                    let head = pattern.DocumentRange().unwrap();
                    use windows::Win32::UI::Accessibility::{TextPatternRangeEndpoint_End, TextPatternRangeEndpoint_Start};
                    head.MoveEndpointByRange(TextPatternRangeEndpoint_End, &caret, TextPatternRangeEndpoint_Start).unwrap();
                    head.GetText(4096).unwrap().to_string()
                };
                (near, old)
            });
            let mut msg = MSG::default();
            while !worker.is_finished() {
                while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            let (near, old) = worker.join().unwrap();
            let _ = DestroyWindow(edit);

            assert!(!old.ends_with("right here"), "the old read never reached the caret: {:?}", &old[old.len() - 40..]);
            let near = near.expect("read near the caret");
            let got = before_caret_text(|| Some(near), |_| None).unwrap();
            assert!(got.ends_with("and the caret is right here"), "{got:?}");
            assert_eq!(got.chars().count(), BEFORE_CARET_CHARS);
        }
    }

    #[test]
    fn an_empty_or_unreadable_selection_is_none() {
        assert_eq!(complete_selection(|_| Some(String::new())), None);
        assert_eq!(complete_selection(|_| None), None);
    }
}
