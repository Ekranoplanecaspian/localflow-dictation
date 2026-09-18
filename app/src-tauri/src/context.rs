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

/// How long UI Automation gets before we dictate without it.
const UIA_BUDGET: Duration = Duration::from_millis(120);
/// Enough text before the caret to decide spacing and capitalisation.
const BEFORE_CARET_CHARS: usize = 160;

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

/// Fill in what UI Automation can tell us: the selected text, the text just before the caret,
/// and the address of the page in a browser. Best-effort and time-boxed.
pub fn enrich(ctx: &mut Context) {
    let deadline = std::time::Instant::now() + UIA_BUDGET;
    if let Some(found) = uia::focused_text(deadline) {
        ctx.selection = found.selection.filter(|s| !s.is_empty());
        ctx.before_caret = found.before_caret.filter(|s| !s.is_empty());
    }
    if is_browser(&ctx.app) {
        ctx.url = uia::browser_url(ctx.hwnd, deadline);
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
    *LAST.lock().unwrap() = Some(ctx.clone());
}

pub fn last() -> Option<Context> {
    LAST.lock().unwrap().clone()
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
        IUIAutomationValuePattern, TreeScope_Descendants, UIA_ControlTypePropertyId,
        UIA_EditControlTypeId, UIA_TextPatternId, UIA_ValuePatternId, UIA_ValueValuePropertyId,
    };

    pub struct Found {
        pub selection: Option<String>,
        pub before_caret: Option<String>,
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
                });
            };

            let mut selection = None;
            if let Ok(ranges) = pattern.GetSelection() {
                if ranges.Length().unwrap_or(0) > 0 {
                    if let Ok(range) = ranges.GetElement(0) {
                        if let Ok(text) = range.GetText(4096) {
                            let s = text.to_string();
                            if !s.is_empty() {
                                selection = Some(s);
                            }
                        }
                    }
                }
            }
            if expired(deadline) {
                return Some(Found { selection, before_caret: None });
            }

            // Text before the caret: the document from its start to the caret, clipped.
            let mut before = None;
            if let (Ok(ranges), Ok(doc)) = (pattern.GetSelection(), pattern.DocumentRange()) {
                if ranges.Length().unwrap_or(0) > 0 {
                    if let Ok(caret) = ranges.GetElement(0) {
                        if let Ok(head) = doc.Clone() {
                            use windows::Win32::UI::Accessibility::{
                                TextPatternRangeEndpoint_End, TextPatternRangeEndpoint_Start,
                            };
                            if head
                                .MoveEndpointByRange(
                                    TextPatternRangeEndpoint_End,
                                    &caret,
                                    TextPatternRangeEndpoint_Start,
                                )
                                .is_ok()
                            {
                                if let Ok(text) = head.GetText(4096) {
                                    let s = text.to_string();
                                    if !s.is_empty() {
                                        before = Some(tail(&s));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Some(Found { selection, before_caret: before })
        }
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
