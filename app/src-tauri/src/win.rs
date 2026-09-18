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

/// Bring the already-running copy's window to the front.
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
pub fn repair_autostart() {
    let (Some(stored), Some(want)) = (autostart_command_stored(), autostart_command()) else {
        return;
    };
    if stored.trim().eq_ignore_ascii_case(want.trim()) {
        return;
    }
    crate::shell_log!("start-at-sign-in pointed at {stored}; repointing it at this build");
    set_autostart(true);
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
