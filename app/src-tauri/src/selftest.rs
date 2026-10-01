//! Headless checks for the parts of the shell that do not need a window.
//!
//! `localflow-shell --selftest <wav>` runs the whole engine link — spawn or attach, handshake,
//! hello, a session, audio frames, the final text — and prints what came back. It is how the
//! sidecar lifecycle is verified in CI and after a change, without a human watching a window.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::engine::{Engine, Sink};
use crate::guard::LockExt;

struct ChannelSink(Mutex<Sender<(String, Value)>>);

impl Sink for ChannelSink {
    fn emit(&self, event: &str, payload: Value) {
        let _ = self.0.locked().send((event.to_owned(), payload));
    }
}

/// Stream a WAV through the engine exactly as a dictation would go.
pub fn dictate(wav: &str, realtime: bool) -> i32 {
    let pcm = match crate::read_wav_16k_mono(wav) {
        Ok(pcm) => pcm,
        Err(e) => {
            eprintln!("cannot read {wav}: {e}");
            return 2;
        }
    };
    let seconds = pcm.len() as f64 / crate::audio::SAMPLE_RATE as f64;
    eprintln!("[selftest] {wav}: {:.1} s of audio", seconds);

    let (tx, rx) = channel();
    let engine = Engine::start(Arc::new(ChannelSink(Mutex::new(tx))));

    if !wait_for_ready(&engine, &rx, Duration::from_secs(240)) {
        eprintln!("[selftest] the engine never became ready: {:?}", engine.link());
        engine.shutdown();
        return 1;
    }

    let id = "selftest";
    engine.session_start(id, json!({"app": "selftest.exe", "title": "LocalFlow self-test"}), None);
    let frame = crate::audio::SAMPLE_RATE as usize / 50; // 20 ms
    let started = Instant::now();
    for chunk in pcm.chunks(frame) {
        engine.send_audio(chunk);
        if realtime {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    engine.session_end(id);
    let released = Instant::now();

    let deadline = Instant::now() + Duration::from_secs(120);
    let mut code = 1;
    while Instant::now() < deadline {
        let Ok((event, payload)) = rx.recv_timeout(Duration::from_millis(500)) else { continue };
        match event.as_str() {
            "partial" => {
                if let Some(text) = payload.get("text").and_then(Value::as_str) {
                    eprintln!("[selftest] partial: {text}");
                }
            }
            "final" => {
                let ms = released.elapsed().as_millis();
                eprintln!(
                    "[selftest] streamed in {:.1} s, final {} ms after release",
                    started.elapsed().as_secs_f64(),
                    ms
                );
                println!("{}", serde_json::to_string_pretty(&payload).unwrap_or_default());
                code = 0;
                break;
            }
            "engine-error" => {
                eprintln!("[selftest] engine error: {payload}");
                break;
            }
            _ => {}
        }
    }
    if code != 0 {
        eprintln!("[selftest] no final text arrived");
    }
    engine.shutdown();
    std::thread::sleep(Duration::from_millis(300));
    code
}

/// Prove that injection works, end to end, with no human in the loop: open Notepad, type into
/// it, then read the text back out through UI Automation and compare.
///
/// This exists because "no text appeared" is otherwise unfalsifiable - it could be the hook,
/// the engine, the injector, or the target app, and guessing between them wastes a day.
pub fn injection(forced: Option<crate::inject::Method>) -> i32 {
    let text = "LocalFlow injection self-test 1 2 3.".to_owned();
    // By default use the strategy the shell would really choose for this app and this text.
    let method = forced.unwrap_or_else(|| crate::inject::method_for("notepad.exe", &text));

    let mut child = match std::process::Command::new("notepad.exe").spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[selftest] could not start notepad: {e}");
            return 2;
        }
    };
    // Notepad takes a moment to create its window and claim the foreground.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut ctx = crate::context::foreground();
    while ctx.app != "notepad.exe" && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(200));
        ctx = crate::context::foreground();
    }
    if ctx.app != "notepad.exe" {
        eprintln!("[selftest] notepad never came to the foreground (saw {:?})", ctx.app);
        let _ = child.kill();
        return 1;
    }
    std::thread::sleep(Duration::from_millis(400)); // let the edit control take focus

    // Start from an empty document so the check sees only what this run typed.
    use windows::Win32::UI::Input::KeyboardAndMouse::VK_A;
    let _ = crate::inject::chord(crate::inject::CTRL, VK_A);
    std::thread::sleep(Duration::from_millis(100));

    let started = Instant::now();
    let result = crate::inject::inject(&text, method, &ctx.app, 0);
    let elapsed = started.elapsed();
    std::thread::sleep(Duration::from_millis(400)); // let Notepad process the input queue

    let mut after = crate::context::foreground();
    crate::context::enrich(&mut after);
    let via_uia = after.before_caret.clone().unwrap_or_default();

    // Read the document back through the clipboard as well. UI Automation can paraphrase what
    // a control holds, and a disagreement between the two says the read-back is the problem
    // rather than the typing.
    use windows::Win32::UI::Input::KeyboardAndMouse::VK_C;
    let _ = crate::inject::chord(crate::inject::CTRL, VK_A);
    std::thread::sleep(Duration::from_millis(120));
    let _ = crate::inject::chord(crate::inject::CTRL, VK_C);
    std::thread::sleep(Duration::from_millis(250));
    let via_clipboard = crate::inject::clipboard_text().unwrap_or_default();

    let seen = if via_clipboard.is_empty() { via_uia.clone() } else { via_clipboard.clone() };
    let landed = seen.contains(text.trim_end());
    let out = json!({
        "method": format!("{:?}", result.as_ref().copied().unwrap_or(method)).to_lowercase(),
        "sent": text,
        "read_back": seen,
        "via_uia": via_uia,
        "via_clipboard": via_clipboard,
        "landed": landed,
        "ms": elapsed.as_millis() as u64,
        "send_error": result.as_ref().err().map(|e| e.to_string()),
        "target": after.app,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());

    // Kill rather than close, so Notepad cannot stop to ask about saving.
    let _ = std::process::Command::new("taskkill")
        .args(["/PID", &child.id().to_string(), "/F"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    if landed { 0 } else { 1 }
}

/// Try every way of handing unicode key events to Windows and report which ones survive.
pub fn batching_probe() -> i32 {
    use crate::inject::Batching;
    use windows::Win32::UI::Input::KeyboardAndMouse::{VK_A, VK_C};

    let text = "LocalFlow injection self-test 1 2 3.";
    let mut child = match std::process::Command::new("notepad.exe").spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("could not start notepad: {e}");
            return 2;
        }
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while crate::context::foreground().app != "notepad.exe" && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(200));
    }
    std::thread::sleep(Duration::from_millis(600));

    let mut worst = 1;
    for mode in [Batching::All, Batching::PerChar, Batching::PerCharSlow, Batching::PerEvent] {
        // Start from an empty document each time.
        let _ = crate::inject::chord(crate::inject::CTRL, VK_A);
        std::thread::sleep(Duration::from_millis(80));
        crate::inject::set_batching(mode);
        let started = Instant::now();
        let _ = crate::inject::inject(text, crate::inject::Method::Type, "notepad.exe", 0);
        let ms = started.elapsed().as_millis();
        std::thread::sleep(Duration::from_millis(300));

        let _ = crate::inject::chord(crate::inject::CTRL, VK_A);
        std::thread::sleep(Duration::from_millis(80));
        let _ = crate::inject::chord(crate::inject::CTRL, VK_C);
        std::thread::sleep(Duration::from_millis(250));
        let seen = crate::inject::clipboard_text().unwrap_or_default();
        let ok = seen.trim() == text;
        if ok {
            worst = 0;
        }
        println!("{:<12} {:>5} ms  {}  {:?}", format!("{mode:?}"), ms, if ok { "OK  " } else { "WRONG" }, seen.trim());
    }
    crate::inject::set_batching(Batching::All);
    let _ = std::process::Command::new("taskkill")
        .args(["/PID", &child.id().to_string(), "/F"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    let _ = child.wait();
    worst
}

/// Type several strings and print exactly what came back, so the failure can be pinned to
/// particular characters rather than guessed at.
pub fn type_probe(samples: &[String]) -> i32 {
    use windows::Win32::UI::Input::KeyboardAndMouse::{VK_A, VK_C};

    let mut child = match std::process::Command::new("notepad.exe").spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("could not start notepad: {e}");
            return 2;
        }
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    while crate::context::foreground().app != "notepad.exe" && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(200));
    }
    std::thread::sleep(Duration::from_millis(600));

    let mut bad = 0;
    for text in samples {
        let _ = crate::inject::chord(crate::inject::CTRL, VK_A);
        std::thread::sleep(Duration::from_millis(80));
        let _ = crate::inject::inject(text, crate::inject::Method::Type, "notepad.exe", 0);
        std::thread::sleep(Duration::from_millis(300));
        let _ = crate::inject::chord(crate::inject::CTRL, VK_A);
        std::thread::sleep(Duration::from_millis(80));
        let _ = crate::inject::chord(crate::inject::CTRL, VK_C);
        std::thread::sleep(Duration::from_millis(250));
        let seen = crate::inject::clipboard_text().unwrap_or_default();
        let seen = seen.trim_end_matches(|c| c == '\r' || c == '\n');
        let ok = seen == text;
        if !ok {
            bad += 1;
        }
        println!("{}  sent {:?}", if ok { "OK   " } else { "WRONG" }, text);
        if !ok {
            println!("       got {seen:?}");
        }
    }
    let _ = std::process::Command::new("taskkill")
        .args(["/PID", &child.id().to_string(), "/F"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    let _ = child.wait();
    bad
}

/// Record what Windows actually delivers for our own injected keystrokes.
///
/// Text arriving mangled could be our `SendInput` array, Windows, or the target application.
/// A low-level hook sits between the first two and the third, so comparing what was sent with
/// what the hook saw says which side of the boundary the bug is on.
pub fn hook_probe(text: &str) -> i32 {
    use std::sync::Mutex as StdMutex;
    use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
    use windows::Win32::UI::WindowsAndMessaging::{
        CallNextHookEx, DispatchMessageW, PeekMessageW, SetWindowsHookExW, UnhookWindowsHookEx,
        HHOOK, KBDLLHOOKSTRUCT, MSG, PM_REMOVE, TranslateMessage, WH_KEYBOARD_LL, WM_KEYDOWN,
    };

    static SEEN: StdMutex<Vec<u16>> = StdMutex::new(Vec::new());

    unsafe extern "system" fn proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        if code >= 0 && wparam.0 as u32 == WM_KEYDOWN {
            let info = &*(lparam.0 as *const KBDLLHOOKSTRUCT);
            if let Ok(mut seen) = SEEN.lock() {
                seen.push(info.scanCode as u16);
            }
        }
        CallNextHookEx(None, code, wparam, lparam)
    }

    let hook: HHOOK = match unsafe { SetWindowsHookExW(WH_KEYBOARD_LL, Some(proc), None, 0) } {
        Ok(h) => h,
        Err(e) => {
            eprintln!("could not install the probe hook: {e}");
            return 2;
        }
    };

    // The hook only fires on a thread that pumps messages, and SendInput is synchronous, so
    // inject from another thread and pump here until the events have all come through.
    let sent = text.to_owned();
    let typist = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(200));
        crate::inject::inject(&sent, crate::inject::Method::Type, "probe.exe", 0)
    });

    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        unsafe {
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
        if typist.is_finished() && SEEN.lock().map(|s| s.len()).unwrap_or(0) >= text.chars().count() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let send_result = typist.join().ok().and_then(|r| r.err());
    unsafe {
        let _ = UnhookWindowsHookEx(hook);
    }

    let seen: Vec<u16> = SEEN.lock().map(|s| s.clone()).unwrap_or_default();
    let delivered: String = seen.iter().filter_map(|u| char::from_u32(*u as u32)).collect();
    let expected: Vec<u16> = text.encode_utf16().collect();
    println!("sent      {text:?}");
    println!("delivered {delivered:?}");
    println!("sent scan codes      {expected:?}");
    println!("delivered scan codes {seen:?}");
    if let Some(e) = send_result {
        println!("SendInput error: {e}");
    }
    if seen == expected {
        println!("=> Windows delivered exactly what was sent; any mangling is in the target app");
        0
    } else {
        println!("=> Windows did NOT deliver what was sent; the fault is on our side of the hook");
        1
    }
}

/// Phase 3's acceptance test: keep the GPU busy for a while and see whether the running app's
/// keyboard hook survives it.
///
/// Windows removes a low-level hook whose callback is slow and says nothing, and heavy GPU load
/// is when that is most likely. This attaches to the engine the app is already using - starting
/// a second one would not fit on an 8 GB card - and streams dictations through it back to back.
///
/// It also taps an unassigned key between rounds. The app's watchdog can only notice a dead
/// hook by comparing system input against what its hook saw, so without input there is nothing
/// to compare; these taps give it something, and are ignored by every application.
pub fn stress(minutes: u64, wav: &str) -> i32 {
    let pcm = match crate::read_wav_16k_mono(wav) {
        Ok(pcm) => pcm,
        Err(e) => {
            eprintln!("cannot read {wav}: {e}");
            return 2;
        }
    };
    let (tx, rx) = channel();
    let engine = Engine::start(Arc::new(ChannelSink(Mutex::new(tx))));
    if !wait_for_ready(&engine, &rx, Duration::from_secs(240)) {
        eprintln!("[stress] the engine never became ready");
        engine.shutdown();
        return 1;
    }
    let attached = engine.link().attached;
    eprintln!(
        "[stress] {} engine (pid {:?}); {minutes} minutes of back-to-back dictation",
        if attached { "attached to the running" } else { "started my own" },
        engine.link().pid
    );

    let end = Instant::now() + Duration::from_secs(minutes * 60);
    let (mut rounds, mut finals, mut failures) = (0u32, 0u32, 0u32);
    let mut worst = Duration::ZERO;
    while Instant::now() < end {
        crate::inject::tap_unassigned();
        rounds += 1;
        let id = format!("stress{rounds}");
        engine.session_start(&id, json!({"app": "stress.exe", "title": "GPU stress"}), None);
        for chunk in pcm.chunks(crate::audio::SAMPLE_RATE as usize / 50) {
            engine.send_audio(chunk);
        }
        engine.session_end(&id);

        let started = Instant::now();
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut got = false;
        while Instant::now() < deadline {
            match rx.recv_timeout(Duration::from_millis(500)) {
                Ok((event, _)) if event == "final" => {
                    got = true;
                    break;
                }
                Ok((event, payload)) if event == "engine-error" => {
                    eprintln!("[stress] round {rounds}: engine error {payload}");
                    break;
                }
                _ => {}
            }
        }
        if got {
            finals += 1;
            worst = worst.max(started.elapsed());
        } else {
            failures += 1;
            eprintln!("[stress] round {rounds}: no final text");
        }
        if rounds % 20 == 0 {
            let left = end.saturating_duration_since(Instant::now());
            eprintln!("[stress] {rounds} rounds, {finals} finals, {failures} failures, {}s left", left.as_secs());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    println!("rounds {rounds}, finals {finals}, failures {failures}, slowest final {} ms", worst.as_millis());
    engine.shutdown();
    std::thread::sleep(Duration::from_millis(300));
    if failures == 0 { 0 } else { 1 }
}

/// Report what the shell can see and do, without dictating: the engine, the focused window,
/// and which injection strategy that window would get.
pub fn report() -> i32 {
    let (tx, rx) = channel();
    let engine = Engine::start(Arc::new(ChannelSink(Mutex::new(tx))));
    let ready = wait_for_ready(&engine, &rx, Duration::from_secs(240));
    let mut ctx = crate::context::foreground();
    crate::context::enrich(&mut ctx);
    crate::context::add_url(&mut ctx);
    let sample = "LocalFlow injection self-test 1 2 3.";
    let out = json!({
        "ready": ready,
        "link": engine.link(),
        "engine": engine.status(),
        "focused": ctx.to_json(),
        "injection": {
            "method": format!("{:?}", crate::inject::method_for(&ctx.app, sample)).to_lowercase(),
            "modifiers_down": crate::inject::modifiers_down(),
        },
        "hotkey": crate::hotkey::Config::load().describe(),
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    engine.shutdown();
    std::thread::sleep(Duration::from_millis(300));
    if ready { 0 } else { 1 }
}

/// Drain events until the speech model reports ready, narrating what happens on the way.
fn wait_for_ready(engine: &Engine, rx: &Receiver<(String, Value)>, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    let mut last = String::new();
    while Instant::now() < deadline {
        if engine.stt_ready() {
            return true;
        }
        if let Ok((event, payload)) = rx.recv_timeout(Duration::from_millis(250)) {
            let note = match event.as_str() {
                "engine-link" => {
                    let link = payload.get("link").and_then(Value::as_str).unwrap_or("?");
                    let detail = payload.get("detail").and_then(Value::as_str).unwrap_or("");
                    format!("link {link} {detail}")
                }
                "engine-status" => {
                    let stt = payload.get("stt").and_then(|s| s.get("state")).and_then(Value::as_str);
                    let llm = payload.get("llm").and_then(|s| s.get("state")).and_then(Value::as_str);
                    format!("stt {} / llm {}", stt.unwrap_or("?"), llm.unwrap_or("?"))
                }
                _ => continue,
            };
            if note != last {
                eprintln!("[selftest] {note}");
                last = note;
            }
        }
    }
    engine.stt_ready()
}
