//! The flow bar: the small window that appears while you dictate.
//!
//! It is the only part of LocalFlow most people will look at, and it has three hard
//! requirements that ordinary windows do not meet:
//!
//!   * **It must never take focus.** The caret has to stay exactly where the user left it, so
//!     the window carries `WS_EX_NOACTIVATE`, and `WS_EX_TOOLWINDOW` keeps it out of Alt-Tab
//!     and the taskbar. A bar that steals focus would inject text into itself.
//!   * **It must not swallow clicks.** It sits over whatever the user is working in, so the
//!     mouse passes straight through it.
//!   * **It must appear on the monitor being worked on**, above the taskbar, whichever monitor
//!     that is and whatever its scaling - so its position is computed from the work area of the
//!     monitor holding the focused window, in physical pixels.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::guard::LockExt;
use tauri::{AppHandle, Manager, PhysicalPosition, PhysicalSize, WebviewUrl, WebviewWindowBuilder};
use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Gdi::{GetMonitorInfoW, MonitorFromWindow, MONITORINFO, MONITOR_DEFAULTTONEAREST};
use windows::Win32::UI::WindowsAndMessaging::{
    GetForegroundWindow, GetWindowLongPtrW, SetWindowLongPtrW, GWL_EXSTYLE, WS_EX_NOACTIVATE,
    WS_EX_TOOLWINDOW, WS_EX_TRANSPARENT,
};

pub const LABEL: &str = "flowbar";
/// The window is a transparent canvas, not the bar. The pill inside sizes itself to its
/// content and centres, so it can grow to fit a sentence without the window being resized and
/// re-centred on every partial - which would fight the compositor several times a second.
/// The window is click-through, so the empty area around the pill costs nothing.
const WIDTH: f64 = 680.0;
const HEIGHT: f64 = 84.0;
/// Gap between the bar and the bottom of the work area (above the taskbar).
const BOTTOM_GAP: f64 = 28.0;
/// How long a notice - "warming up" - stays. Long enough to read a short line and understand
/// that the take did not happen, short enough not to sit over the user's work.
const NOTICE: Duration = Duration::from_millis(2200);
/// How long the bar stays after a dictation ends, so the last word is readable.
const LINGER: Duration = Duration::from_millis(600);

/// Bumped on every phase change; a scheduled hide only fires if it is still the newest.
static GENERATION: AtomicU64 = AtomicU64::new(0);

pub fn create(app: &AppHandle) -> tauri::Result<()> {
    let window = WebviewWindowBuilder::new(app, LABEL, WebviewUrl::App("index.html#flowbar".into()))
        // Distinct from the Hub window's title: a second launch looks the main window up
        // by name to bring it forward, and this one is invisible and must not be found.
        .title("LocalFlow Flow Bar")
        .inner_size(WIDTH, HEIGHT)
        .decorations(false)
        .transparent(true)
        .always_on_top(true)
        .skip_taskbar(true)
        .resizable(false)
        .shadow(false)
        .focused(false)
        .visible(false)
        .build()?;

    // Clicks belong to the app underneath, not to us.
    let _ = window.set_ignore_cursor_events(true);
    if let Ok(handle) = window.hwnd() {
        // Tauri hands back an HWND from its own copy of the windows crate; the raw pointer is
        // the only thing the two versions agree on.
        let hwnd = HWND(handle.0 as *mut std::ffi::c_void);
        unsafe {
            let style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
            SetWindowLongPtrW(
                hwnd,
                GWL_EXSTYLE,
                style | (WS_EX_NOACTIVATE.0 as isize) | (WS_EX_TOOLWINDOW.0 as isize),
            );
            // Read the styles back rather than assume they took: a bar that can take focus
            // would swallow the caret, and that is not visible until it happens.
            let now = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
            let has = |flag: u32| now & (flag as isize) != 0;
            crate::shell_log!(
                "flow bar ready (exstyle {:#010x}: no-activate {}, tool-window {}, click-through {})",
                now,
                has(WS_EX_NOACTIVATE.0),
                has(WS_EX_TOOLWINDOW.0),
                has(WS_EX_TRANSPARENT.0),
            );
        }
    }
    Ok(())
}

/// The work area of the monitor holding the focused window, in physical pixels.
fn work_area() -> Option<RECT> {
    unsafe {
        let focused = GetForegroundWindow();
        let monitor = MonitorFromWindow(
            if focused.0.is_null() { HWND(std::ptr::null_mut()) } else { focused },
            MONITOR_DEFAULTTONEAREST,
        );
        let mut info = MONITORINFO {
            cbSize: std::mem::size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if GetMonitorInfoW(monitor, &mut info).as_bool() {
            Some(info.rcWork)
        } else {
            None
        }
    }
}

/// Put the bar at the bottom centre of the monitor being worked on.
fn place(app: &AppHandle) {
    let Some(window) = app.get_webview_window(LABEL) else { return };
    let scale = window.scale_factor().unwrap_or(1.0);
    let (w, h) = ((WIDTH * scale).round() as i32, (HEIGHT * scale).round() as i32);
    let _ = window.set_size(PhysicalSize::new(w as u32, h as u32));
    if let Some(area) = work_area() {
        let x = area.left + (area.right - area.left - w) / 2;
        let y = area.bottom - h - (BOTTOM_GAP * scale).round() as i32;
        let _ = window.set_position(PhysicalPosition::new(x, y));
    }
}

/// A notice: shown for `hold` (the default when None), then taken away.
pub fn show_notice(app: &AppHandle, hold: Option<Duration>) {
    *NOTICE_HOLD.locked() = hold.unwrap_or(NOTICE);
    on_phase(app, "notice");
}

/// How long the notice being shown stays.
static NOTICE_HOLD: std::sync::Mutex<Duration> = std::sync::Mutex::new(NOTICE);

/// Show or hide the bar as the dictation phase changes.
pub fn on_phase(app: &AppHandle, phase: &str) {
    let generation = GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
    let Some(window) = app.get_webview_window(LABEL) else { return };
    // "Show the flow bar" off: no bar at all. The switch was saved and never read, so turning it
    // off changed nothing. Read on every phase change, so it applies from the next take.
    if !crate::settings::load().flow_bar {
        let _ = window.hide();
        return;
    }
    match phase {
        "recording" => {
            // Re-place on every take: the user may have moved to another monitor since the last.
            place(app);
            let _ = window.show();
            crate::shell_log!("flow bar shown at {:?}", window.outer_position().ok());
            // Showing a window can raise it above other topmost windows only once; re-assert.
            let _ = window.set_always_on_top(true);
        }
        // Not a take, just a message: show the bar, hold it long enough to read, take it away.
        "notice" => {
            place(app);
            let _ = window.show();
            let _ = window.set_always_on_top(true);
            let app = app.clone();
            std::thread::Builder::new()
                .name("flowbar-notice".into())
                .spawn(move || {
                    let hold = *NOTICE_HOLD.locked();
                    std::thread::sleep(hold);
                    if GENERATION.load(Ordering::SeqCst) != generation {
                        return;
                    }
                    if let Some(window) = app.get_webview_window(LABEL) {
                        let _ = window.hide();
                    }
                })
                .ok();
        }
        "finishing" => {}
        _ => {
            let app = app.clone();
            std::thread::Builder::new()
                .name("flowbar-linger".into())
                .spawn(move || {
                    std::thread::sleep(LINGER);
                    // A new dictation may have started while we waited.
                    if GENERATION.load(Ordering::SeqCst) != generation {
                        return;
                    }
                    if let Some(window) = app.get_webview_window(LABEL) {
                        let _ = window.hide();
                    }
                })
                .ok();
        }
    }
}
