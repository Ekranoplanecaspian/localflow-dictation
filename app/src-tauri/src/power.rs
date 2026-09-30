//! The lock screen, sleep, and the secure desktop: where the keyboard hook cannot follow.
//!
//! A low-level keyboard hook sees nothing on the Winlogon desktop (the lock screen,
//! Ctrl+Alt+Del) or the secure desktop a UAC prompt runs on, and nothing while the machine
//! sleeps. A chord held when the screen locks is released where the hook cannot see it, so the
//! take went on recording, and the chord still read as held. So when input goes away - the
//! session locks or disconnects, the machine suspends, or the input desktop changes - the take
//! being spoken ends as a release would end it (its words are transcribed and kept: with nothing
//! in front they wait for Paste last dictation), and the hook forgets every key it thought held.
//! When input comes back, the keys are forgotten again, and after sleep the microphone stream
//! is rebuilt and the hook put in again: Windows may have dropped either without a word.
//!
//! One hidden top-level window receives `WM_WTSSESSION_CHANGE` and `WM_POWERBROADCAST`
//! (message-only windows get no broadcasts), and the same thread holds a WinEvent hook for
//! `EVENT_SYSTEM_DESKTOPSWITCH`.

use std::sync::OnceLock;

use tauri::{AppHandle, Manager};
use windows::core::PCWSTR;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::UI::Accessibility::{SetWinEventHook, HWINEVENTHOOK};
use windows::Win32::UI::WindowsAndMessaging::*;

static APP: OnceLock<AppHandle> = OnceLock::new();
/// The hidden window, so the end-to-end harness can send it what Windows sends at shutdown.
static WINDOW: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);

const WTS_CONSOLE_CONNECT: u32 = 1;
const WTS_CONSOLE_DISCONNECT: u32 = 2;
const WTS_REMOTE_CONNECT: u32 = 3;
const WTS_REMOTE_DISCONNECT: u32 = 4;
const WTS_SESSION_UNLOCK: u32 = 8;
const PBT_APMRESUMESUSPEND: u32 = 7;

/// Where input went, or came back from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    Locked,
    Unlocked,
    Suspending,
    Resumed,
    Disconnected,
    Connected,
    /// Input moved to another desktop (a UAC prompt, the lock screen, Ctrl+Alt+Del).
    DesktopAway,
    /// Input is back on this session's own desktop.
    DesktopBack,
    /// Windows is shutting down, restarting (an update) or signing out.
    SessionEnding,
}

impl Event {
    fn away(self) -> bool {
        matches!(
            self,
            Event::Locked | Event::Suspending | Event::Disconnected | Event::DesktopAway | Event::SessionEnding
        )
    }
}

/// Start watching. Best effort: without it everything works as before, minus this.
pub fn watch(app: AppHandle) {
    let _ = APP.set(app);
    let spawned = std::thread::Builder::new().name("session-watch".into()).spawn(|| unsafe { run() });
    if let Err(e) = spawned {
        crate::shell_log!("could not watch for lock and sleep: {e}");
    }
}

unsafe fn run() {
    let class: Vec<u16> = "LocalFlowSessionWatch\0".encode_utf16().collect();
    let wc = WNDCLASSW { lpfnWndProc: Some(wndproc), lpszClassName: PCWSTR(class.as_ptr()), ..Default::default() };
    RegisterClassW(&wc);
    // Top-level and never shown: only top-level windows receive power broadcasts.
    let hwnd = match CreateWindowExW(
        WINDOW_EX_STYLE(0),
        PCWSTR(class.as_ptr()),
        PCWSTR(class.as_ptr()),
        WINDOW_STYLE(0),
        0,
        0,
        0,
        0,
        None,
        None,
        None,
        None,
    ) {
        Ok(h) => h,
        Err(e) => {
            crate::shell_log!("could not watch for lock and sleep: {e}");
            return;
        }
    };
    WINDOW.store(hwnd.0 as isize, std::sync::atomic::Ordering::SeqCst);
    if let Err(e) = windows::Win32::System::RemoteDesktop::WTSRegisterSessionNotification(
        hwnd,
        windows::Win32::System::RemoteDesktop::NOTIFY_FOR_THIS_SESSION,
    ) {
        crate::shell_log!("could not watch for the session locking: {e}");
    }
    let hook = SetWinEventHook(
        EVENT_SYSTEM_DESKTOPSWITCH,
        EVENT_SYSTEM_DESKTOPSWITCH,
        None,
        Some(desktop_switched),
        0,
        0,
        WINEVENT_OUTOFCONTEXT,
    );
    if hook.is_invalid() {
        crate::shell_log!("could not watch for desktop switches (UAC prompts)");
    }
    let mut msg = MSG::default();
    while GetMessageW(&mut msg, None, 0, 0).as_bool() {
        let _ = TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let event = match msg {
        WM_WTSSESSION_CHANGE => match wparam.0 as u32 {
            WTS_SESSION_LOCK => Some(Event::Locked),
            WTS_SESSION_UNLOCK => Some(Event::Unlocked),
            WTS_CONSOLE_DISCONNECT | WTS_REMOTE_DISCONNECT => Some(Event::Disconnected),
            WTS_CONSOLE_CONNECT | WTS_REMOTE_CONNECT => Some(Event::Connected),
            _ => None,
        },
        WM_POWERBROADCAST => match wparam.0 as u32 {
            PBT_APMSUSPEND => Some(Event::Suspending),
            PBT_APMRESUMEAUTOMATIC | PBT_APMRESUMESUSPEND => Some(Event::Resumed),
            _ => None,
        },
        // Nothing here needs to hold a shutdown up: say yes, and act when it is really happening.
        WM_QUERYENDSESSION => return LRESULT(1),
        WM_ENDSESSION if wparam.0 != 0 => Some(Event::SessionEnding),
        _ => None,
    };
    if let Some(event) = event {
        // A panic must not unwind into Windows.
        crate::guard::catch("a lock or sleep event", || on_event(event));
    }
    DefWindowProcW(hwnd, msg, wparam, lparam)
}

unsafe extern "system" fn desktop_switched(
    _hook: HWINEVENTHOOK,
    _event: u32,
    _hwnd: HWND,
    _object: i32,
    _child: i32,
    _thread: u32,
    _time: u32,
) {
    let event = if on_own_desktop() { Event::DesktopBack } else { Event::DesktopAway };
    crate::guard::catch("a desktop switch", || on_event(event));
}

/// The hidden window receiving session and power messages (0 before `watch` made it).
pub fn window() -> isize {
    WINDOW.load(std::sync::atomic::Ordering::SeqCst)
}

/// Whether input is on this session's own desktop ("Default"). Winlogon's and the secure
/// desktop cannot even be opened from here, which answers the question too.
pub fn on_own_desktop() -> bool {
    use windows::Win32::System::StationsAndDesktops::*;
    unsafe {
        let Ok(desk) = OpenInputDesktop(DESKTOP_CONTROL_FLAGS(0), false, DESKTOP_ACCESS_FLAGS(DESKTOP_READOBJECTS.0))
        else {
            return false;
        };
        let mut name = [0u16; 64];
        let mut needed = 0u32;
        let ok = GetUserObjectInformationW(
            windows::Win32::Foundation::HANDLE(desk.0),
            UOI_NAME,
            Some(name.as_mut_ptr() as *mut _),
            (name.len() * 2) as u32,
            Some(&mut needed),
        )
        .is_ok();
        let _ = CloseDesktop(desk);
        let len = name.iter().position(|c| *c == 0).unwrap_or(name.len());
        ok && String::from_utf16_lossy(&name[..len]).eq_ignore_ascii_case("Default")
    }
}

pub fn on_event(event: Event) {
    let Some(app) = APP.get() else { return };
    let recording = app
        .try_state::<std::sync::Arc<crate::session::SessionManager>>()
        .is_some_and(|s| s.phase() == crate::session::Phase::Recording);
    crate::shell_log!(
        "[session] {event:?}{}",
        if event.away() && recording { ": ending the take being spoken; its words are kept" } else { "" }
    );
    if event.away() {
        crate::hotkey::interrupted();
        if event == Event::SessionEnding {
            // Windows ends the process when this message returns. The engine is told to stop
            // (it writes down its model timings), rather than being cut off by the job object.
            if let Some(engine) = app.try_state::<crate::engine::Engine>() {
                engine.shutdown();
            }
            std::thread::sleep(std::time::Duration::from_millis(300));
        }
        return;
    }
    crate::hotkey::forget_held_keys();
    if event == Event::Resumed {
        // Sleep can take the microphone stream and the hook away without a word.
        crate::hotkey::reinstall();
        if let Some(shell) = app.try_state::<crate::Shell>() {
            shell.capture.rebuild();
        }
    }
}
