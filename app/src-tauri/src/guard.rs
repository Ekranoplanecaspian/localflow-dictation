//! Keeping the shell alive, and knowing what happened when it did not stay alive.
//!
//! Release builds used `panic = "abort"` with no panic hook, so any panic anywhere - and there
//! were forty `lock().unwrap()`s - closed the whole app and left nothing in the log. Now there
//! are four layers, from the smallest fault to the largest:
//!
//! 1. Locks never propagate a panic ([`LockExt`]): a thread that died holding one leaves data
//!    that is still usable here - flags, strings, a queue - so the next user carries on.
//! 2. Long-lived worker threads restart ([`spawn_supervised`]) instead of silently dying and
//!    taking the microphone or the hotkey with them.
//! 3. Every panic is written to `shell.log`, with its thread, place and backtrace, before
//!    anything else happens ([`install_panic_hook`]).
//! 4. A crash cannot be survived - a panic on the main thread, or a fault in native code - so
//!    LocalFlow starts a fresh copy of itself and ends ([`install_fault_handler`]).

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// Passed on the command line to a copy started after a crash.
pub const RESTARTED_ARG: &str = "--restarted";
/// `--after <pid>`: the crashed copy, which the new one waits for before it starts.
const AFTER_ARG: &str = "--after";
/// A copy that crashes sooner than this after starting is not restarted, so a fault at start-up
/// cannot become a loop.
const MIN_UPTIME: Duration = Duration::from_secs(60);

/// A worker that ran this long before panicking was healthy; its next failure starts the
/// backoff from the beginning again.
const HEALTHY_AFTER: Duration = Duration::from_secs(60);
const BACKOFF_MIN: Duration = Duration::from_millis(250);
const BACKOFF_MAX: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------------------------
// locks

/// `lock()` without the panic: a poisoned mutex hands back its data anyway.
///
/// Poisoning exists for data that a panic might have left half-updated. Nothing the shell
/// keeps behind a mutex has an invariant spanning more than one field that a half-finished
/// update could break, and refusing to go on - the `unwrap()` - turned one fault into every
/// later caller panicking as well, the hotkey and the microphone included.
pub trait LockExt<T> {
    fn locked(&self) -> MutexGuard<'_, T>;
}

impl<T> LockExt<T> for Mutex<T> {
    fn locked(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

// ---------------------------------------------------------------------------------------------
// panics

/// The text of a panic payload, whichever of the two usual types it is.
pub fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "(no message)".to_owned()
    }
}

fn started() -> Instant {
    static STARTED: OnceLock<Instant> = OnceLock::new();
    *STARTED.get_or_init(Instant::now)
}

/// Write every panic to the log; on the main thread, also start a new copy and end this one.
///
/// The main thread runs the window and tray event loop. A panic there cannot be caught and
/// resumed from - the event loop is gone - so the only recovery is a fresh process. Everywhere
/// else the panic unwinds as normal, into whatever catches it: a supervised worker restarts, a
/// one-off thread just ends.
pub fn install_panic_hook() {
    started();
    panic::set_hook(Box::new(|info| {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("unnamed").to_owned();
        let place = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_else(|| "unknown place".to_owned());
        let backtrace = std::backtrace::Backtrace::force_capture();
        crate::log::write(&format!(
            "PANIC in thread '{name}' at {place}: {}\n{backtrace}",
            panic_message(info.payload())
        ));
        if name == "main" {
            relaunch_after_crash("the main thread panicked");
            std::process::abort();
        }
    }));
}

/// Run `f`, turning a panic into `None`. The panic itself has already been logged by the hook.
///
/// For work done once per request - an injection, a hotkey action - where losing that one
/// request is bad but losing the thread that serves every later one is much worse.
pub fn catch<R>(what: &str, f: impl FnOnce() -> R) -> Option<R> {
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => Some(r),
        Err(_) => {
            crate::shell_log!("{what} failed with a panic; carrying on");
            None
        }
    }
}

/// Start a named thread that is restarted whenever `body` panics. `body` returning normally
/// ends the thread for good; that is how a worker shuts down.
///
/// Restarts back off from a quarter of a second to half a minute, so a fault that recurs at
/// once does not spin, and a worker that ran for a minute before failing starts again at once.
pub fn spawn_supervised<F>(name: &str, body: F) -> std::io::Result<std::thread::JoinHandle<()>>
where
    F: Fn() + Send + 'static,
{
    let label = name.to_owned();
    std::thread::Builder::new().name(label.clone()).spawn(move || {
        let mut backoff = Duration::ZERO;
        let mut restarts = 0u32;
        loop {
            let started = Instant::now();
            if panic::catch_unwind(AssertUnwindSafe(&body)).is_ok() {
                return;
            }
            restarts += 1;
            backoff = if started.elapsed() >= HEALTHY_AFTER {
                BACKOFF_MIN
            } else {
                (backoff * 2).clamp(BACKOFF_MIN, BACKOFF_MAX)
            };
            crate::shell_log!("{label} stopped with a panic; restart {restarts} in {backoff:?}");
            std::thread::sleep(backoff);
        }
    })
}

// ---------------------------------------------------------------------------------------------
// restarting

/// Start a new copy of LocalFlow that takes over once this one has gone.
///
/// Not left to Windows: `RegisterApplicationRestart` was tried first and, on the development
/// machine, Windows recorded both a panic (a "fail fast" abort) and a native fault in its
/// error reports but restarted neither.
fn relaunch_after_crash(what: &str) {
    let uptime = started().elapsed();
    if uptime < MIN_UPTIME {
        crate::shell_log!(
            "{what} {:.0}s after start; not restarting, so a fault at start-up cannot loop",
            uptime.as_secs_f32()
        );
        return;
    }
    let started = std::env::current_exe().and_then(|exe| {
        std::process::Command::new(exe)
            .args([RESTARTED_ARG, AFTER_ARG, &std::process::id().to_string()])
            .spawn()
    });
    match started {
        Ok(child) => crate::shell_log!("{what}; started a new copy (pid {})", child.id()),
        Err(e) => crate::shell_log!("{what}, and starting a new copy failed: {e}"),
    }
}

/// In a copy started by [`relaunch_after_crash`]: wait for the crashed one to be gone, so its
/// single-instance claim, keyboard hook and engine are released before this one needs them.
pub fn wait_for_predecessor(args: &[String]) {
    let Some(pid) = args
        .iter()
        .position(|a| a == AFTER_ARG)
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse::<u32>().ok())
    else {
        return;
    };
    #[cfg(windows)]
    unsafe {
        use windows::Win32::Foundation::CloseHandle;
        use windows::Win32::System::Threading::{OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE};
        // Already gone is the usual case by the time this runs, and fine.
        if let Ok(handle) = OpenProcess(PROCESS_SYNCHRONIZE, false, pid) {
            let _ = WaitForSingleObject(handle, 10_000);
            let _ = CloseHandle(handle);
        }
    }
}

/// Log a native fault - an access violation, an illegal instruction - and start a new copy.
///
/// Such a fault never becomes a panic, so the panic hook does not see it: it goes straight to
/// Windows, which ended the process without a line in the log. This filter runs first. It
/// hands the fault back to Windows afterwards, so the crash is still recorded there.
pub fn install_fault_handler() {
    #[cfg(windows)]
    unsafe {
        use windows::Win32::System::Diagnostics::Debug::SetUnhandledExceptionFilter;
        SetUnhandledExceptionFilter(Some(on_fault));
    }
}

#[cfg(windows)]
unsafe extern "system" fn on_fault(
    info: *const windows::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> i32 {
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
    // A panic cannot be allowed out of here, and the process may be in a bad way: best effort.
    let _ = panic::catch_unwind(|| {
        let record = if info.is_null() { None } else { (*info).ExceptionRecord.as_ref() };
        let (code, at) = record
            .map(|r| (r.ExceptionCode.0 as u32, r.ExceptionAddress as usize))
            .unwrap_or_default();
        let thread = std::thread::current().name().unwrap_or("unnamed").to_owned();
        crate::log::write(&format!(
            "FAULT 0x{code:08X} at 0x{at:X} in thread '{thread}' (outside Rust's control)"
        ));
        relaunch_after_crash("a native fault ended LocalFlow");
    });
    EXCEPTION_CONTINUE_SEARCH
}

/// `--crash-test [seconds] [panic|fault]`: crash on purpose after a while, to prove the log and
/// the restart work in a real build. `panic` panics on the main thread; `fault` executes an
/// illegal instruction on a worker, a crash no panic hook sees. The delay
/// must be over 60 seconds for either to restart.
pub fn schedule_crash_test(app: &tauri::AppHandle, after: Duration, kind: &str) {
    let fault = kind == "fault";
    crate::shell_log!("crash test: a {} in {after:?}", if fault { "native fault" } else { "main-thread panic" });
    let app = app.clone();
    let _ = std::thread::Builder::new().name("crash-test".into()).spawn(move || {
        std::thread::sleep(after);
        if fault {
            crate::shell_log!("crash test: executing an illegal instruction");
            // Deliberately fatal: the fault is the test.
            #[cfg(target_arch = "x86_64")]
            unsafe {
                std::arch::asm!("ud2");
            }
        } else {
            let _ = app.run_on_main_thread(|| panic!("crash test requested on the command line"));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn a_poisoned_lock_still_hands_over_its_data() {
        let m = Arc::new(Mutex::new(7));
        let held = m.clone();
        let _ = std::thread::spawn(move || {
            let _g = held.lock().unwrap();
            panic!("die holding the lock");
        })
        .join();
        assert!(m.is_poisoned());
        *m.locked() += 1;
        assert_eq!(*m.locked(), 8);
    }

    #[test]
    fn a_supervised_worker_comes_back_after_panicking() {
        let runs = Arc::new(AtomicU32::new(0));
        let seen = runs.clone();
        let worker = spawn_supervised("test-worker", move || {
            // Panic on the first two runs, then finish normally, which ends the thread.
            if seen.fetch_add(1, Ordering::SeqCst) < 2 {
                panic!("worker fault");
            }
        })
        .unwrap();
        worker.join().expect("the supervisor itself never panics");
        assert_eq!(runs.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn catch_turns_a_panic_into_none_and_leaves_results_alone() {
        assert_eq!(catch("test", || 5), Some(5));
        assert_eq!(catch("test", || -> i32 { panic!("boom") }), None);
    }

    #[test]
    fn panic_messages_are_read_from_both_payload_types() {
        let s: Box<dyn Any + Send> = Box::new("static");
        let o: Box<dyn Any + Send> = Box::new(String::from("owned"));
        let n: Box<dyn Any + Send> = Box::new(3u8);
        assert_eq!(panic_message(s.as_ref()), "static");
        assert_eq!(panic_message(o.as_ref()), "owned");
        assert_eq!(panic_message(n.as_ref()), "(no message)");
    }
}
