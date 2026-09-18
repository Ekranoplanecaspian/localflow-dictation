//! A log file for the shell, next to the engine's.
//!
//! The Python tray app wrote a line per dictation and it was the only reason several bugs were
//! ever found. The Rust shell needs the same: without it a report of "it did not type anything"
//! has nothing behind it. One line per event, `%APPDATA%\LocalFlow\shell.log`, rolled at 2 MB.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
#[cfg(not(windows))]
use std::time::SystemTime;

const MAX_BYTES: u64 = 2 * 1024 * 1024;

fn path() -> Option<PathBuf> {
    let dir = crate::paths::config_dir()?;
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("shell.log"))
}

fn file() -> &'static Mutex<Option<std::fs::File>> {
    static FILE: OnceLock<Mutex<Option<std::fs::File>>> = OnceLock::new();
    FILE.get_or_init(|| {
        let handle = path().and_then(|p| {
            // Roll rather than grow without bound; one previous file is enough history.
            if std::fs::metadata(&p).map(|m| m.len() > MAX_BYTES).unwrap_or(false) {
                let _ = std::fs::rename(&p, p.with_extension("log.1"));
            }
            OpenOptions::new().create(true).append(true).open(&p).ok()
        });
        Mutex::new(handle)
    })
}

/// Local wall-clock time as HH:MM:SS.mmm, matching the engine's log format closely enough to
/// read the two side by side.
fn stamp() -> String {
    // The engine logs local time; ask Windows rather than carry a date library for one line.
    #[cfg(windows)]
    {
        use windows::Win32::System::SystemInformation::GetLocalTime;
        let t = unsafe { GetLocalTime() };
        format!("{:02}:{:02}:{:02}.{:03}", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds)
    }
    #[cfg(not(windows))]
    {
        let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
        format!("{}.{:03}", now.as_secs(), now.subsec_millis())
    }
}

pub fn write(line: &str) {
    let text = format!("{} {}\n", stamp(), line);
    // Not `eprint!`: that panics when stderr cannot be written, and a windowed process started
    // from Explorer - a shortcut, the Start menu, autostart, which is every way a user
    // actually launches this - has no stderr at all. The first thread to log took the panic
    // and died, and because the panic happened before the file write there was no log to say
    // so. Launched from a terminal it all worked, which is what made it so hard to see.
    let _ = std::io::stderr().write_all(text.as_bytes());
    if let Ok(mut slot) = file().lock() {
        if let Some(f) = slot.as_mut() {
            let _ = f.write_all(text.as_bytes());
            let _ = f.flush();
        }
    }
}

/// `log!("recording in {}", app)` - same shape as the engine's log calls.
#[macro_export]
macro_rules! shell_log {
    ($($arg:tt)*) => {
        $crate::log::write(&format!($($arg)*))
    };
}

/// The path shown in the UI and in a diagnostics export.
pub fn location() -> String {
    path().map(|p| p.display().to_string()).unwrap_or_default()
}
