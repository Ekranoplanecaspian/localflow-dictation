//! Small Windows primitives shared by the rest of the shell.

use std::sync::OnceLock;

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows::Win32::System::Threading::{
    CreateMutexW, GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
};

const STILL_ACTIVE: u32 = 259;
/// Name of the single-instance mutex. `Local\` scopes it to this logon session, which is what
/// is wanted: two people signed in at once should each get their own LocalFlow, and a `Global\`
/// mutex would let the first of them lock out the second.
const INSTANCE_MUTEX: &str = r"Local\LocalFlow.SingleInstance";
/// Auto-reset event the running copy waits on. A second launch sets it, which means "the user
/// wants the window": the running copy then shows it the same way the tray does.
const SHOW_EVENT: &str = r"Local\LocalFlow.ShowWindow";
/// No console window for child processes, even when the shell itself has one.
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Claim this session's single-instance mutex, or report that another copy already holds it.
///
/// Two shells would mean two keyboard hooks on the same chord, so every dictation would be
/// started twice, transcribed twice and injected twice. The Python app guarded against this
/// with exactly this mutex; the Rust shell inherited its job but not its guard, and nothing
/// noticed because nobody launches the same app twice on purpose - they double-click a
/// shortcut while it is already in the tray.
///
/// The handle is deliberately leaked into a `OnceLock`: the mutex must be held for as long as
/// the process lives, and Windows releases it when the process exits.
pub fn claim_single_instance() -> bool {
    static HELD: OnceLock<bool> = OnceLock::new();
    *HELD.get_or_init(|| unsafe {
        let name: Vec<u16> = INSTANCE_MUTEX.encode_utf16().chain(std::iter::once(0)).collect();
        match CreateMutexW(None, true, windows::core::PCWSTR(name.as_ptr())) {
            Ok(handle) => {
                // ERROR_ALREADY_EXISTS means the mutex was there and somebody else owns it.
                let already = windows::Win32::Foundation::GetLastError()
                    == windows::Win32::Foundation::ERROR_ALREADY_EXISTS;
                if already {
                    let _ = CloseHandle(handle);
                    false
                } else {
                    // Deliberately never closed. The mutex has to be held for the life of the
                    // process, and Windows releases it when the process ends, however it ends.
                    true
                }
            }
            // If the mutex cannot be created at all, let the app start: refusing to run is a
            // worse failure than the duplicate this is meant to prevent.
            Err(_) => true,
        }
    })
}

/// In the running copy: call `show` whenever a second launch asks for the window.
///
/// This replaced finding the window by its title from the second process, which failed in two
/// ways: another top-level window can carry the same title, and Windows does not let a process
/// that has only just started put somebody else's window in front. The running copy showing
/// its own window, through Tauri, has neither problem.
pub fn on_show_request(show: impl Fn() + Send + 'static) {
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};
    use windows::Win32::Foundation::WAIT_OBJECT_0;

    let name: Vec<u16> = SHOW_EVENT.encode_utf16().chain(std::iter::once(0)).collect();
    let event = match unsafe { CreateEventW(None, false, false, windows::core::PCWSTR(name.as_ptr())) } {
        Ok(event) => event,
        Err(e) => {
            shell_log!("could not create the show-window event: {e}");
            return;
        }
    };
    let raw = event.0 as isize; // HANDLE is not Send; the event lives as long as the process
    std::thread::Builder::new()
        .name("show-requests".into())
        .spawn(move || loop {
            let event = HANDLE(raw as *mut core::ffi::c_void);
            if unsafe { WaitForSingleObject(event, INFINITE) } != WAIT_OBJECT_0 {
                return;
            }
            crate::guard::catch("showing the window", &show);
        })
        .ok();
}

/// In a second copy: ask the running one to show its window. False if there is no running copy
/// listening (an older version), in which case the caller falls back to finding the window.
pub fn request_show() -> bool {
    use windows::Win32::System::Threading::{OpenEventW, SetEvent, EVENT_MODIFY_STATE};
    use windows::Win32::UI::WindowsAndMessaging::{AllowSetForegroundWindow, ASFW_ANY};

    let name: Vec<u16> = SHOW_EVENT.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        let Ok(event) = OpenEventW(EVENT_MODIFY_STATE, false, windows::core::PCWSTR(name.as_ptr())) else {
            return false;
        };
        // This process was just started by the user, so it may bring a window to the front;
        // pass that right on, or the running copy's window would only flash in the taskbar.
        let _ = AllowSetForegroundWindow(ASFW_ANY);
        let ok = SetEvent(event).is_ok();
        let _ = CloseHandle(event);
        ok
    }
}

/// Bring the already-running copy's window to the front, for a running copy too old to listen
/// for `request_show`.
///
/// Without this, launching a second time appears to do nothing at all, which is worse than the
/// duplicate it is preventing: the user double-clicks again, harder.
pub fn show_existing_window() {
    use windows::Win32::UI::WindowsAndMessaging::{
        FindWindowW, SetForegroundWindow, ShowWindow, SW_RESTORE,
    };
    let title: Vec<u16> = "LocalFlow".encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        if let Ok(hwnd) = FindWindowW(None, windows::core::PCWSTR(title.as_ptr())) {
            if !hwnd.0.is_null() {
                let _ = ShowWindow(hwnd, SW_RESTORE);
                let _ = SetForegroundWindow(hwnd);
            }
        }
    }
}

pub fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    unsafe {
        let Ok(handle) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else {
            return false;
        };
        let mut code = 0u32;
        let alive = GetExitCodeProcess(handle, &mut code).is_ok() && code == STILL_ACTIVE;
        let _ = CloseHandle(handle);
        alive
    }
}

/// The executable file name of a running process, lower-cased ("localflow-engine.exe").
pub fn process_image_name(pid: u32) -> Option<String> {
    use windows::Win32::System::Threading::{QueryFullProcessImageNameW, PROCESS_NAME_WIN32};
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut buf = [0u16; 1024];
        let mut len = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(handle, PROCESS_NAME_WIN32, windows::core::PWSTR(buf.as_mut_ptr()), &mut len).is_ok();
        let _ = CloseHandle(handle);
        if !ok {
            return None;
        }
        let path = String::from_utf16_lossy(&buf[..len as usize]);
        std::path::Path::new(&path).file_name().map(|n| n.to_string_lossy().to_ascii_lowercase())
    }
}

/// The top-level window a window belongs to: a dialog or a control counts as its owner's.
pub fn root_window(hwnd: isize) -> isize {
    use windows::Win32::UI::WindowsAndMessaging::{GetAncestor, GA_ROOTOWNER};
    if hwnd == 0 {
        return 0;
    }
    let root = unsafe { GetAncestor(windows::Win32::Foundation::HWND(hwnd as *mut _), GA_ROOTOWNER) };
    if root.0.is_null() { hwnd } else { root.0 as isize }
}

pub fn foreground_window() -> isize {
    unsafe { windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow() }.0 as isize
}

/// The desktop or the taskbar: in front, but nowhere text can go.
pub fn is_shell_window(hwnd: isize) -> bool {
    use windows::Win32::UI::WindowsAndMessaging::GetClassNameW;
    if hwnd == 0 {
        return true;
    }
    let mut buf = [0u16; 64];
    let n = unsafe { GetClassNameW(windows::Win32::Foundation::HWND(hwnd as *mut _), &mut buf) } as usize;
    let class = String::from_utf16_lossy(&buf[..n.min(buf.len())]);
    matches!(class.as_str(), "Progman" | "WorkerW" | "Shell_TrayWnd" | "Shell_SecondaryTrayWnd")
}

/// The app window the user was last in: the topmost window in the Z order that text could go
/// to. Clicking the tray puts the taskbar (and then LocalFlow's own menu window) in front, so
/// the tray's Paste last dictation looks here for the window it is for. 0 when there is none.
pub fn last_app_window() -> isize {
    use windows::Win32::Foundation::{HWND, LPARAM, RECT};
    use windows::Win32::Graphics::Dwm::{DwmGetWindowAttribute, DWMWA_CLOAKED};
    use windows::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetWindowLongW, GetWindowRect, GetWindowThreadProcessId, IsIconic, IsWindowVisible,
        GWL_EXSTYLE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST,
    };
    unsafe extern "system" fn each(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        unsafe {
            let ex = GetWindowLongW(hwnd, GWL_EXSTYLE) as u32;
            let mut pid = 0u32;
            GetWindowThreadProcessId(hwnd, Some(&mut pid));
            let mut cloaked = 0u32;
            let _ = DwmGetWindowAttribute(
                hwnd,
                DWMWA_CLOAKED,
                &mut cloaked as *mut u32 as *mut _,
                std::mem::size_of::<u32>() as u32,
            );
            let mut r = RECT::default();
            let _ = GetWindowRect(hwnd, &mut r);
            let usable = IsWindowVisible(hwnd).as_bool()
                && !IsIconic(hwnd).as_bool()
                && cloaked == 0
                && ex & (WS_EX_TOOLWINDOW.0 | WS_EX_TOPMOST.0) == 0
                && (pid != std::process::id() || crate::e2e::is_target(hwnd.0 as isize))
                && r.right > r.left
                && r.bottom > r.top
                && !is_shell_window(hwnd.0 as isize);
            if usable {
                *(found.0 as *mut isize) = hwnd.0 as isize;
                return false.into(); // the first is the one most recently in front
            }
            true.into()
        }
    }
    let mut found: isize = 0;
    unsafe {
        let _ = EnumWindows(Some(each), LPARAM(&mut found as *mut isize as isize));
    }
    found
}

/// The names of the connected microphones that belong to Bluetooth devices. A microphone
/// endpoint is a device node `SWD\MMDEVAPI\{0.0.1.…}` (1: capture) whose parent is the device
/// behind it - `BTHHFENUM\…` for a headset's hands-free profile, `BTHLE…` for LE Audio,
/// `HDAUDIO\…` or `USB\…` for everything else. The names are the ones the Hub's picker shows.
pub fn bluetooth_microphones() -> Vec<String> {
    bluetooth_microphones_in(true)
}

/// `connected`: only those plugged in or paired and in range now.
fn bluetooth_microphones_in(connected: bool) -> Vec<String> {
    use windows::Win32::Devices::DeviceAndDriverInstallation::*;
    use windows::Win32::Devices::Properties::{DEVPKEY_Device_FriendlyName, DEVPKEY_Device_Parent, DEVPROPTYPE};
    let prop = |node: u32, key: &windows::Win32::Foundation::DEVPROPKEY| -> Option<String> {
        let mut kind = DEVPROPTYPE::default();
        let mut buf = [0u16; 512];
        let mut size = (buf.len() * 2) as u32;
        let r = unsafe {
            CM_Get_DevNode_PropertyW(node, key, &mut kind, Some(buf.as_mut_ptr() as *mut u8), &mut size, 0)
        };
        (r == CR_SUCCESS).then(|| {
            let n = (size as usize / 2).min(buf.len());
            String::from_utf16_lossy(&buf[..n]).trim_end_matches('\0').to_owned()
        })
    };
    let filter: Vec<u16> = "SWD\\MMDEVAPI".encode_utf16().chain(Some(0)).collect();
    let flags = CM_GETIDLIST_FILTER_ENUMERATOR | if connected { CM_GETIDLIST_FILTER_PRESENT } else { 0 };
    let mut len = 0u32;
    if unsafe { CM_Get_Device_ID_List_SizeW(&mut len, windows::core::PCWSTR(filter.as_ptr()), flags) } != CR_SUCCESS {
        return Vec::new();
    }
    let mut list = vec![0u16; len as usize];
    if unsafe { CM_Get_Device_ID_ListW(windows::core::PCWSTR(filter.as_ptr()), &mut list, flags) } != CR_SUCCESS {
        return Vec::new();
    }
    list.split(|c| *c == 0)
        .filter(|id| !id.is_empty())
        .filter_map(|id| {
            let text = String::from_utf16_lossy(id);
            if !text.to_ascii_uppercase().contains("{0.0.1.") {
                return None; // a speaker, not a microphone
            }
            let wide: Vec<u16> = id.iter().copied().chain(Some(0)).collect();
            let mut node = 0u32;
            // A device not connected now is found only as a "phantom".
            let how = if connected { CM_LOCATE_DEVNODE_NORMAL } else { CM_LOCATE_DEVNODE_PHANTOM };
            let located = unsafe { CM_Locate_DevNodeW(&mut node, windows::core::PCWSTR(wide.as_ptr()), how) };
            if located != CR_SUCCESS {
                return None;
            }
            let parent = prop(node, &DEVPKEY_Device_Parent)?;
            parent.to_ascii_uppercase().starts_with("BTH").then(|| prop(node, &DEVPKEY_Device_FriendlyName)).flatten()
        })
        .collect()
}

/// A window's class and title, for the log.
pub fn describe(hwnd: isize) -> String {
    use windows::Win32::UI::WindowsAndMessaging::{GetClassNameW, GetWindowTextW};
    if hwnd == 0 {
        return "no window".into();
    }
    let h = windows::Win32::Foundation::HWND(hwnd as *mut _);
    let mut class = [0u16; 64];
    let mut title = [0u16; 96];
    let c = unsafe { GetClassNameW(h, &mut class) } as usize;
    let t = unsafe { GetWindowTextW(h, &mut title) } as usize;
    format!(
        "{} {:?}",
        String::from_utf16_lossy(&class[..c.min(class.len())]),
        String::from_utf16_lossy(&title[..t.min(title.len())])
    )
}

/// Bring a window to the front and say whether it got there.
pub fn bring_to_front(hwnd: isize) -> bool {
    use windows::Win32::UI::WindowsAndMessaging::SetForegroundWindow;
    let h = windows::Win32::Foundation::HWND(hwnd as *mut _);
    // Windows only lets the process with the most recent input take the foreground; an
    // injected key nobody listens for makes that this one.
    crate::inject::tap_unassigned();
    let _ = unsafe { SetForegroundWindow(h) };
    std::thread::sleep(std::time::Duration::from_millis(150));
    root_window(foreground_window()) == root_window(hwnd)
}

/// Whether a process runs elevated. None when that cannot be told (a process LocalFlow may not
/// even query).
fn process_elevated(pid: u32) -> Option<bool> {
    use windows::Win32::Security::{GetTokenInformation, TokenElevation, TOKEN_ELEVATION, TOKEN_QUERY};
    use windows::Win32::System::Threading::OpenProcessToken;
    unsafe {
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut token = HANDLE::default();
        let opened = OpenProcessToken(process, TOKEN_QUERY, &mut token);
        let _ = CloseHandle(process);
        if opened.is_err() {
            // The process is there but its token is closed to us: an elevated process owned by
            // this same user is exactly what refuses that.
            return Some(true);
        }
        let mut elevation = TOKEN_ELEVATION::default();
        let mut len = 0u32;
        let ok = GetTokenInformation(
            token,
            TokenElevation,
            Some(&mut elevation as *mut _ as *mut std::ffi::c_void),
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut len,
        )
        .is_ok();
        let _ = CloseHandle(token);
        ok.then_some(elevation.TokenIsElevated != 0)
    }
}

#[cfg(test)]
pub fn process_elevated_for_tests(pid: u32) -> Option<bool> {
    process_elevated(pid)
}

/// Whether Windows will drop LocalFlow's keystrokes to this window: it belongs to an app running
/// as administrator and LocalFlow does not. Windows says nothing when it drops them - SendInput
/// reports success - so this has to be known beforehand.
pub fn keystrokes_blocked(hwnd: isize) -> bool {
    use windows::Win32::UI::WindowsAndMessaging::GetWindowThreadProcessId;
    if hwnd == 0 {
        return false;
    }
    let mut pid = 0u32;
    unsafe { GetWindowThreadProcessId(windows::Win32::Foundation::HWND(hwnd as *mut _), Some(&mut pid)) };
    if pid == 0 || pid == std::process::id() {
        return false;
    }
    static SELF_ELEVATED: OnceLock<bool> = OnceLock::new();
    let me = *SELF_ELEVATED.get_or_init(|| process_elevated(std::process::id()).unwrap_or(false));
    !me && process_elevated(pid).unwrap_or(false)
}

pub fn no_window(cmd: &mut tokio::process::Command) {
    cmd.creation_flags(CREATE_NO_WINDOW);
}

/// A process-wide job object that the kernel empties when this process ends, however it ends.
/// Children put in it (the engine, and through it `llama-server`) cannot outlive us holding
/// video memory, which is exactly what used to happen when the engine was killed.
fn kill_on_close_job() -> Option<HANDLE> {
    static JOB: OnceLock<Option<usize>> = OnceLock::new();
    let raw = (*JOB.get_or_init(|| unsafe {
        let job = CreateJobObjectW(None, None).ok()?;
        let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
        .is_ok();
        if !ok {
            let _ = CloseHandle(job);
            return None;
        }
        // The handle is deliberately never closed: it must live as long as the process.
        Some(job.0 as usize)
    }))?;
    Some(HANDLE(raw as *mut std::ffi::c_void))
}

pub fn assign_to_kill_job(child: &tokio::process::Child) {
    let (Some(job), Some(handle)) = (kill_on_close_job(), child.raw_handle()) else {
        return;
    };
    unsafe {
        let _ = AssignProcessToJobObject(job, HANDLE(handle));
    }
}

// ---------------------------------------------------------------------------------------------
// start at sign-in

const RUN_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Run";
const RUN_VALUE: &str = "LocalFlow";

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// The command Windows should run at sign-in: this executable, quoted.
fn autostart_command() -> Option<String> {
    let exe = std::env::current_exe().ok()?;
    Some(format!("\"{}\"", exe.display()))
}

pub fn autostart_enabled() -> bool {
    use windows::Win32::System::Registry::{
        RegGetValueW, HKEY_CURRENT_USER, RRF_RT_REG_SZ,
    };
    unsafe {
        let mut size = 0u32;
        let status = RegGetValueW(
            HKEY_CURRENT_USER,
            windows::core::PCWSTR(wide(RUN_KEY).as_ptr()),
            windows::core::PCWSTR(wide(RUN_VALUE).as_ptr()),
            RRF_RT_REG_SZ,
            None,
            None,
            Some(&mut size),
        );
        status.is_ok() && size > 0
    }
}

/// A string value from the registry, if the key and value exist.
pub fn reg_string(hive: windows::Win32::System::Registry::HKEY, key: &str, value: &str) -> Option<String> {
    use windows::Win32::System::Registry::{RegGetValueW, RRF_RT_REG_SZ};
    let key = wide(key);
    let value = wide(value);
    unsafe {
        let mut size = 0u32;
        let (k, v) = (windows::core::PCWSTR(key.as_ptr()), windows::core::PCWSTR(value.as_ptr()));
        if RegGetValueW(hive, k, v, RRF_RT_REG_SZ, None, None, Some(&mut size)).is_err() || size == 0 {
            return None;
        }
        let mut buf = vec![0u16; (size as usize / 2) + 1];
        let mut len = size;
        RegGetValueW(hive, k, v, RRF_RT_REG_SZ, None, Some(buf.as_mut_ptr() as *mut std::ffi::c_void), Some(&mut len))
            .ok()
            .ok()?;
        let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        Some(String::from_utf16_lossy(&buf[..end]))
    }
}

/// The command currently registered to run at sign-in, if any.
fn autostart_command_stored() -> Option<String> {
    use windows::Win32::System::Registry::{RegGetValueW, HKEY_CURRENT_USER, RRF_RT_REG_SZ};
    unsafe {
        let mut size = 0u32;
        let key = wide(RUN_KEY);
        let value = wide(RUN_VALUE);
        if RegGetValueW(
            HKEY_CURRENT_USER,
            windows::core::PCWSTR(key.as_ptr()),
            windows::core::PCWSTR(value.as_ptr()),
            RRF_RT_REG_SZ,
            None,
            None,
            Some(&mut size),
        )
        .is_err()
            || size == 0
        {
            return None;
        }
        let mut buf = vec![0u16; (size as usize / 2) + 1];
        let mut len = size;
        if RegGetValueW(
            HKEY_CURRENT_USER,
            windows::core::PCWSTR(key.as_ptr()),
            windows::core::PCWSTR(value.as_ptr()),
            RRF_RT_REG_SZ,
            None,
            Some(buf.as_mut_ptr() as *mut std::ffi::c_void),
            Some(&mut len),
        )
        .is_err()
        {
            return None;
        }
        let end = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        Some(String::from_utf16_lossy(&buf[..end]))
    }
}

/// Point a stale sign-in entry at this executable.
///
/// The entry outlives the thing it points at. Moving the program, or installing a build over a
/// copy that was run from a source tree, leaves Windows launching something that is no longer
/// there - and it does so silently, once per sign-in, with nothing to tell the user why their
/// dictation no longer starts itself. This machine had one left over from the Python tray app
/// that phase 4.4 deleted: `pythonw.exe -m localflow run`, a command that no longer exists.
///
/// Only ever rewrites an entry that is already there. Switching autostart on is the user's
/// decision, and repairing one is not the same as making one.
///
/// And only an entry that is actually broken. The first version repaired any entry that was not
/// *this* executable, which is the same thing on a machine with one copy of LocalFlow and a
/// quiet hijack on a machine with two: the installed build that runs every day, and a
/// development build started from the repository. The development build would have repointed
/// sign-in at itself the moment it ran, and the next sign-in would have launched half-finished
/// code instead of the release.
pub fn repair_autostart() {
    let (Some(stored), Some(want)) = (autostart_command_stored(), autostart_command()) else {
        return;
    };
    if stored.trim().eq_ignore_ascii_case(want.trim()) {
        return;
    }
    if !autostart_is_stale(&stored, |p| p.is_file()) {
        // Another working copy of LocalFlow owns sign-in. That is a choice somebody made.
        return;
    }
    crate::shell_log!("start-at-sign-in pointed at {stored}, which is broken; repointing it at this build");
    set_autostart(true);
}

/// The executable a sign-in command runs: the quoted first token, or the first word.
fn autostart_target(command: &str) -> &str {
    let command = command.trim();
    if let Some(rest) = command.strip_prefix('"') {
        rest.split('"').next().unwrap_or("")
    } else {
        command.split_whitespace().next().unwrap_or("")
    }
}

/// Whether a sign-in entry can no longer start LocalFlow: its program has gone, or it was never
/// LocalFlow's shell to begin with - the retired Python tray app, say, whose interpreter still
/// exists but whose `-m localflow run` does not.
fn autostart_is_stale(command: &str, exists: impl Fn(&std::path::Path) -> bool) -> bool {
    let target = autostart_target(command);
    if target.is_empty() {
        return true;
    }
    let path = std::path::Path::new(target);
    if !exists(path) {
        return true;
    }
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
    !matches!(name.as_str(), "app.exe" | "localflow.exe")
}

#[cfg(test)]
mod tests {
    use super::*;

    const INSTALLED: &str = r#""C:\Users\u\AppData\Local\LocalFlow\app.exe""#;
    const DEV: &str = r#""C:\src\WisprFlowClone\app\src-tauri\target\release\app.exe""#;
    const PYTHON: &str = r#""C:\src\WisprFlowClone\.venv\Scripts\pythonw.exe" -m localflow run"#;

    /// Run by hand on a machine that has paired a Bluetooth headset (connected or not): lists
    /// its microphone and none of the others. `cargo test -- --ignored bluetooth --nocapture`
    #[test]
    #[ignore]
    fn bluetooth_headsets_are_told_apart_from_other_microphones() {
        let all = bluetooth_microphones_in(false);
        println!("Bluetooth microphones, paired: {all:?}; connected now: {:?}", bluetooth_microphones());
        assert!(!all.is_empty(), "no Bluetooth headset has ever been paired here");
        assert!(all.iter().all(|m| !m.contains("Realtek")), "{all:?}");
    }

    /// Paste last dictation from the tray goes back to an app window: never the taskbar or the
    /// desktop, and never one of LocalFlow's own.
    #[test]
    fn the_last_app_window_is_somewhere_text_can_go() {
        use windows::Win32::UI::WindowsAndMessaging::GetWindowThreadProcessId;
        let w = last_app_window();
        if w == 0 {
            return; // a desktop with no windows open (a CI runner)
        }
        assert!(!is_shell_window(w), "{}", describe(w));
        let mut pid = 0u32;
        unsafe { GetWindowThreadProcessId(windows::Win32::Foundation::HWND(w as *mut _), Some(&mut pid)) };
        assert_ne!(pid, std::process::id());
    }

    #[test]
    fn the_target_is_read_from_a_quoted_or_bare_command() {
        assert_eq!(autostart_target(PYTHON), r"C:\src\WisprFlowClone\.venv\Scripts\pythonw.exe");
        assert_eq!(autostart_target(r"C:\bin\app.exe --flag"), r"C:\bin\app.exe");
        assert_eq!(autostart_target(""), "");
    }

    /// The reason this rule exists: a development build must not take sign-in away from the
    /// installed copy just by being started.
    #[test]
    fn another_working_copy_of_localflow_is_left_alone() {
        assert!(!autostart_is_stale(INSTALLED, |_| true));
        assert!(!autostart_is_stale(DEV, |_| true));
    }

    #[test]
    fn a_copy_that_has_been_deleted_is_repaired() {
        assert!(autostart_is_stale(INSTALLED, |_| false));
    }

    /// What this machine actually had: the interpreter still exists, so "does the file exist"
    /// alone would have left it broken for ever.
    #[test]
    fn the_retired_python_app_is_repaired_even_though_python_still_exists() {
        assert!(autostart_is_stale(PYTHON, |_| true));
    }

    #[test]
    fn an_empty_entry_is_repaired() {
        assert!(autostart_is_stale("   ", |_| true));
    }
}

/// Flip the sign-in entry and report the new state.
pub fn toggle_autostart() -> bool {
    let want = !autostart_enabled();
    set_autostart(want);
    autostart_enabled()
}

pub fn set_autostart(on: bool) {
    use windows::Win32::System::Registry::{
        RegCloseKey, RegDeleteValueW, RegOpenKeyExW, RegSetValueExW, HKEY, HKEY_CURRENT_USER,
        KEY_SET_VALUE, REG_SZ,
    };
    let Some(command) = autostart_command() else { return };
    unsafe {
        let mut key = HKEY::default();
        if RegOpenKeyExW(
            HKEY_CURRENT_USER,
            windows::core::PCWSTR(wide(RUN_KEY).as_ptr()),
            None,
            KEY_SET_VALUE,
            &mut key,
        )
        .is_err()
        {
            return;
        }
        if on {
            let data = wide(&command);
            let bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2);
            let _ = RegSetValueExW(
                key,
                windows::core::PCWSTR(wide(RUN_VALUE).as_ptr()),
                None,
                REG_SZ,
                Some(bytes),
            );
        } else {
            let _ = RegDeleteValueW(key, windows::core::PCWSTR(wide(RUN_VALUE).as_ptr()));
        }
        let _ = RegCloseKey(key);
    }
}

