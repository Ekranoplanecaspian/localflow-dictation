//! The soak test: hours of dictation through the real shell, watching for slow growth.
//!
//! `app.exe --e2e soak [minutes]` (default 120). A leak does not show in a test that lasts a
//! minute; it shows as memory, handles or threads that keep climbing over hours of ordinary use.
//! So this runs takes in bursts - about five minutes of dictation, then a rest longer than the
//! engine's idle release, so the models are let go and loaded again each cycle, which is where
//! leaks like to live - and samples the shell, its webviews, the engine and the clean-up server
//! every minute into `soak.csv`. At the end each measure's growth per hour is fitted over
//! everything after the first cycle and compared with a limit.
//!
//! Nothing is typed: the injector is switched off for the run, so the harness never takes the
//! focus and the machine stays usable. Typing is covered by the other scenarios. Command mode is
//! left out for the same reason - reading a selection presses Ctrl+C in whatever is in front.
//! Takes are held back while the graphics card is hot.

use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use tauri::{AppHandle, Listener, Manager};

use crate::audio::Tape;
use crate::engine::Engine;
use crate::guard::LockExt;
use crate::hotkey::Raw;

const BURST: Duration = Duration::from_secs(5 * 60);
/// Longer than the engine's default idle release (10 minutes), so each cycle reloads models.
const REST: Duration = Duration::from_secs(12 * 60);
const BETWEEN_TAKES: Duration = Duration::from_secs(15);
const SAMPLE_EVERY: Duration = Duration::from_secs(60);
const HOT_C: u32 = 75;
const COOL_C: u32 = 65;

/// Growth per hour that counts as a leak, per measure.
const LIMITS: &[(&str, f64)] = &[
    ("shell_private_mb", 20.0),
    ("shell_handles", 100.0),
    ("shell_threads", 5.0),
    ("shell_gdi", 50.0),
    ("shell_user", 50.0),
    ("webview_private_mb", 40.0),
    ("engine_private_mb", 60.0),
    ("engine_handles", 200.0),
    ("engine_threads", 10.0),
];

pub fn run(app: &AppHandle, keys: Sender<Raw>, tape: Tape, audio: PathBuf, minutes: u64) -> i32 {
    crate::e2e::set_typing(false);
    let events = Arc::new(Mutex::new(Vec::<(&'static str, Value)>::new()));
    for kind in ["final", "injected"] {
        let events = events.clone();
        app.listen(kind, move |e| {
            events.locked().push((kind, serde_json::from_str(e.payload()).unwrap_or(Value::Null)));
        });
    }
    let csv_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target").join("soak.csv");
    let Ok(mut csv) = std::fs::File::create(&csv_path) else {
        report(&format!("cannot write {}", csv_path.display()));
        return 2;
    };
    let _ = writeln!(csv, "minute,phase,takes_ok,takes_failed,{},llama_private_mb,gpu_mem_mb,gpu_c", COLUMNS.join(","));

    let clips: Vec<Vec<i16>> = ["fox", "report", "meeting", "long"]
        .iter()
        .filter_map(|n| crate::read_wav_16k_mono(&audio.join(format!("{n}.wav")).to_string_lossy()).ok())
        .collect();
    if clips.len() < 4 {
        report("the test speech is missing: run scripts\\make-e2e-audio.ps1");
        return 2;
    }

    report(&format!(
        "soak for {minutes} min: {}-min bursts of takes every {}s, {}-min rests; nothing is typed; samples to {}",
        BURST.as_secs() / 60,
        BETWEEN_TAKES.as_secs(),
        REST.as_secs() / 60,
        csv_path.display()
    ));
    let started = Instant::now();
    let total = Duration::from_secs(minutes * 60);
    let mut next_sample = Instant::now();
    let (mut ok, mut failed, mut n) = (0u32, 0u32, 0usize);
    let mut rows: Vec<(f64, Vec<f64>)> = Vec::new();
    let engine = app.state::<Engine>();

    while started.elapsed() < total {
        let cycle = started.elapsed().as_secs() % (BURST + REST).as_secs();
        let bursting = cycle < BURST.as_secs();
        if Instant::now() >= next_sample {
            next_sample += SAMPLE_EVERY;
            let minute = started.elapsed().as_secs_f64() / 60.0;
            let m = measure(engine.link().pid);
            let (gpu_mem, gpu_c) = gpu();
            let _ = writeln!(
                csv,
                "{minute:.1},{},{ok},{failed},{},{:.0},{gpu_mem},{gpu_c}",
                if bursting { "burst" } else { "rest" },
                m.iter().map(|v| format!("{v:.1}")).collect::<Vec<_>>().join(","),
                m_llama(engine.link().pid)
            );
            let _ = csv.flush();
            rows.push((minute, m));
        }
        if !bursting {
            std::thread::sleep(Duration::from_secs(1));
            continue;
        }
        // Gentle on a laptop that runs hot: no takes until the card has cooled.
        if gpu().1 >= HOT_C {
            report(&format!("graphics card at {HOT_C} C or more; pausing until it is under {COOL_C} C"));
            while gpu().1 > COOL_C {
                std::thread::sleep(Duration::from_secs(10));
            }
        }
        // Every fifth take is the long one; one in seven is hands-free.
        let clip = if n % 5 == 4 { &clips[3] } else { &clips[n % 3] };
        let hands_free = n % 7 == 6;
        let mark = events.locked().len();
        take(&keys, &tape, clip, hands_free);
        let deadline = Instant::now() + Duration::from_secs(60);
        let done = loop {
            let got = events.locked()[mark..].iter().filter(|(k, _)| *k == "injected").count();
            if got >= 1 {
                break true;
            }
            if Instant::now() > deadline || !engine.is_connected() {
                break false;
            }
            std::thread::sleep(Duration::from_millis(100));
        };
        if done { ok += 1 } else { failed += 1 }
        n += 1;
        std::thread::sleep(BETWEEN_TAKES);
    }
    crate::e2e::set_typing(true);

    report(&format!("{ok} takes, {failed} without text, over {:.0} min", started.elapsed().as_secs_f32() / 60.0));
    let verdict = judge(&rows);
    let leaks = verdict.iter().filter(|(_, _, _, bad)| *bad).count();
    for (name, per_hour, limit, bad) in verdict {
        report(&format!("{} {name}: {per_hour:+.1}/h (limit {limit})", if bad { "GROWS" } else { "steady" }));
    }
    let bad = leaks > 0 || failed > ok / 20;
    report(if bad { "soak FAILED" } else { "soak passed" });
    if bad { 1 } else { 0 }
}

fn report(line: &str) {
    println!("{line}");
    crate::shell_log!("[soak] {line}");
}

/// One push-to-talk take, or a hands-free one: double tap, speak, press to stop.
fn take(keys: &Sender<Raw>, tape: &Tape, clip: &[i16], hands_free: bool) {
    let key = |r: Raw, ms: u64| {
        let _ = keys.send(r);
        std::thread::sleep(Duration::from_millis(ms));
    };
    if hands_free {
        key(Raw::ChordDown, 80);
        key(Raw::ChordUp, 80);
        key(Raw::ChordDown, 40);
        key(Raw::ChordUp, 40);
    } else {
        key(Raw::ChordDown, 40);
    }
    tape.play(clip);
    while tape.playing() {
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(250));
    if hands_free {
        key(Raw::ChordDown, 40);
    }
    key(Raw::ChordUp, 40);
}

// ---------------------------------------------------------------------------------------------
// measuring

const COLUMNS: [&str; 9] = [
    "shell_private_mb",
    "shell_handles",
    "shell_threads",
    "shell_gdi",
    "shell_user",
    "webview_private_mb",
    "engine_private_mb",
    "engine_handles",
    "engine_threads",
];

#[derive(Default)]
struct Proc {
    private_mb: f64,
    handles: f64,
    threads: f64,
    gdi: f64,
    user: f64,
}

fn proc_stats(pid: u32, threads: f64) -> Proc {
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX};
    use windows::Win32::System::Threading::{
        GetGuiResources, GetProcessHandleCount, OpenProcess, GR_GDIOBJECTS, GR_USEROBJECTS,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };
    let mut out = Proc { threads, ..Default::default() };
    unsafe {
        let Ok(h) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else { return out };
        let mut mem = PROCESS_MEMORY_COUNTERS_EX {
            cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
            ..Default::default()
        };
        if GetProcessMemoryInfo(h, &mut mem as *mut _ as *mut PROCESS_MEMORY_COUNTERS, mem.cb).is_ok() {
            out.private_mb = mem.PrivateUsage as f64 / (1 << 20) as f64;
        }
        let mut handles = 0u32;
        if GetProcessHandleCount(h, &mut handles).is_ok() {
            out.handles = handles as f64;
        }
        out.gdi = GetGuiResources(h, GR_GDIOBJECTS) as f64;
        out.user = GetGuiResources(h, GR_USEROBJECTS) as f64;
        let _ = CloseHandle(h);
    }
    out
}

/// (pid, parent, exe, threads) for every process.
fn processes() -> Vec<(u32, u32, String, u32)> {
    use windows::Win32::Foundation::CloseHandle;
    use windows::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
    };
    let mut all = Vec::new();
    unsafe {
        let Ok(snap) = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) else { return all };
        let mut e = PROCESSENTRY32W { dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32, ..Default::default() };
        let mut more = Process32FirstW(snap, &mut e).is_ok();
        while more {
            let len = e.szExeFile.iter().position(|c| *c == 0).unwrap_or(e.szExeFile.len());
            all.push((e.th32ProcessID, e.th32ParentProcessID, String::from_utf16_lossy(&e.szExeFile[..len]), e.cntThreads));
            more = Process32NextW(snap, &mut e).is_ok();
        }
        let _ = CloseHandle(snap);
    }
    all
}

fn descendants(all: &[(u32, u32, String, u32)], root: u32) -> Vec<u32> {
    let mut out = vec![root];
    let mut i = 0;
    while i < out.len() {
        let parent = out[i];
        out.extend(all.iter().filter(|p| p.1 == parent && p.0 != parent).map(|p| p.0));
        i += 1;
    }
    out.remove(0);
    out
}

/// This shell, LocalFlow's webviews together, and the engine.
fn measure(engine_pid: Option<u32>) -> Vec<f64> {
    let all = processes();
    let threads_of = |pid: u32| all.iter().find(|p| p.0 == pid).map_or(0.0, |p| p.3 as f64);
    let me = std::process::id();
    let shell = proc_stats(me, threads_of(me));
    // Every copy of LocalFlow shares one WebView2 browser, started by whichever came first - in
    // a harness run that is the user's own copy - so the webviews are counted under all of them.
    let exe = std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default();
    let mut webviews: Vec<u32> = all
        .iter()
        .filter(|p| p.2.eq_ignore_ascii_case(&exe))
        .flat_map(|p| descendants(&all, p.0))
        .filter(|pid| all.iter().any(|p| p.0 == *pid && p.2.eq_ignore_ascii_case("msedgewebview2.exe")))
        .collect();
    webviews.sort_unstable();
    webviews.dedup();
    let webview: f64 = webviews.into_iter().map(|pid| proc_stats(pid, 0.0).private_mb).sum();
    let engine = engine_pid.map(|p| proc_stats(p, threads_of(p))).unwrap_or_default();
    vec![
        shell.private_mb,
        shell.handles,
        shell.threads,
        shell.gdi,
        shell.user,
        webview,
        engine.private_mb,
        engine.handles,
        engine.threads,
    ]
}

/// The clean-up server's memory, when there is one: it comes and goes with the models, so it is
/// recorded but not judged.
fn m_llama(engine_pid: Option<u32>) -> f64 {
    let Some(pid) = engine_pid else { return 0.0 };
    let all = processes();
    all.iter()
        .filter(|p| p.1 == pid && p.2.eq_ignore_ascii_case("llama-server.exe"))
        .map(|p| proc_stats(p.0, 0.0).private_mb)
        .sum()
}

/// (memory used in MB, temperature in C) of the graphics card, or zeros without one.
fn gpu() -> (u32, u32) {
    use std::os::windows::process::CommandExt;
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"])
        .creation_flags(0x0800_0000) // no console window
        .output();
    let Ok(out) = out else { return (0, 0) };
    let text = String::from_utf8_lossy(&out.stdout);
    let mut parts = text.lines().next().unwrap_or("").split(',').map(|s| s.trim().parse::<u32>().unwrap_or(0));
    (parts.next().unwrap_or(0), parts.next().unwrap_or(0))
}

// ---------------------------------------------------------------------------------------------
// judging

/// Least-squares growth per hour of each measure after the first cycle, which is warm-up:
/// caches filling, models loading for the first time.
fn judge(rows: &[(f64, Vec<f64>)]) -> Vec<(&'static str, f64, f64, bool)> {
    let warm = (BURST + REST).as_secs_f64() / 60.0;
    let steady: Vec<_> = rows.iter().filter(|(m, _)| *m >= warm).collect();
    COLUMNS
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            let limit = LIMITS.iter().find(|(n, _)| n == name).map(|l| l.1)?;
            let points: Vec<(f64, f64)> = steady.iter().map(|(m, v)| (*m / 60.0, v[i])).collect();
            let per_hour = slope(&points);
            Some((*name, per_hour, limit, per_hour > limit))
        })
        .collect()
}

fn slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    if n < 3.0 {
        return 0.0;
    }
    let (sx, sy) = points.iter().fold((0.0, 0.0), |(a, b), (x, y)| (a + x, b + y));
    let (mx, my) = (sx / n, sy / n);
    let (num, den) = points.iter().fold((0.0, 0.0), |(a, b), (x, y)| (a + (x - mx) * (y - my), b + (x - mx).powi(2)));
    if den == 0.0 { 0.0 } else { num / den }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_is_fitted_per_hour() {
        // 10 MB every 30 minutes is 20 MB an hour.
        let pts: Vec<(f64, f64)> = (0..7).map(|i| (i as f64 * 0.5, 100.0 + i as f64 * 10.0)).collect();
        assert!((slope(&pts) - 20.0).abs() < 1e-9);
        assert_eq!(slope(&[(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]), 0.0);
    }

    #[test]
    fn a_climbing_measure_is_called_a_leak_and_a_flat_one_is_not() {
        let rows: Vec<(f64, Vec<f64>)> = (0..120)
            .map(|m| {
                let mut v = vec![50.0; COLUMNS.len()];
                v[1] = 400.0 + m as f64 * 5.0; // shell handles: +300 an hour
                (m as f64, v)
            })
            .collect();
        let verdict = judge(&rows);
        let handles = verdict.iter().find(|v| v.0 == "shell_handles").unwrap();
        assert!(handles.3, "{handles:?}");
        let memory = verdict.iter().find(|v| v.0 == "shell_private_mb").unwrap();
        assert!(!memory.3, "{memory:?}");
    }
}
