//! The engine sidecar: spawn `localflow serve`, speak the session protocol, keep it alive.
//!
//! The Python engine owns the models; this shell owns the OS. They talk over a WebSocket on
//! 127.0.0.1 with a per-launch token (see `engine/src/localflow/service/protocol.py`).
//!
//! One supervisor task owns the whole lifecycle: find or spawn the engine, read its
//! `{port, token, pid}` handshake from stdout, connect, say hello, then pump messages until
//! something breaks and start again with backoff. Everything else in the app talks to it
//! through `Engine`, which is a cheap clonable handle over an unbounded outbound channel, so
//! the keyboard hook and the audio callback never block on the socket.

use std::collections::VecDeque;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use futures_util::{FutureExt, SinkExt, StreamExt};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use tokio_tungstenite::tungstenite::Message;

use crate::guard::LockExt;

/// How long we wait for the engine to print its handshake. Cold start loads no models before
/// printing, but a first run may still be unpacking a frozen bundle.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(60);
/// Liveness probe. `status.get` always answers, so silence means a dead link.
const HEARTBEAT: Duration = Duration::from_secs(10);
const SILENCE_LIMIT: Duration = Duration::from_secs(30);
/// A connection that lived this long counts as healthy, so the next failure starts from
/// the shortest backoff again.
const HEALTHY_AFTER: Duration = Duration::from_secs(30);
const BACKOFF_MAX: Duration = Duration::from_secs(10);
/// This many engine crashes inside `CRASH_WINDOW` start safe mode.
const CRASH_LIMIT: usize = 3;
const CRASH_WINDOW: Duration = Duration::from_secs(120);
/// Tells the engine to run in safe mode (`localflow.config.SAFE_MODE_ENV`).
const SAFE_MODE_ENV: &str = "LOCALFLOW_SAFE_MODE";
/// The largest message taken from the engine.
const ENGINE_MESSAGE_MAX: usize = 16 << 20;
/// The longest selection a command sends, as the engine accepts it
/// (`localflow.service.protocol.MAX_SELECTION_CHARS`).
pub const COMMAND_SELECTION_MAX: usize = 100_000;

// ---------------------------------------------------------------------------------------------
// public handle

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Link {
    /// No engine yet: starting one or waiting for its handshake.
    Starting,
    /// Handshake done, opening the socket.
    Connecting,
    /// Connected and authenticated. Dictation is possible as soon as `status.stt` is ready.
    Ready,
    /// Lost the engine; waiting out the backoff before trying again.
    Reconnecting,
    /// Gave up (or never started). `detail` says why.
    Failed,
    /// Shutting down on purpose.
    Stopped,
}

#[derive(Debug, Clone, Serialize)]
pub struct LinkState {
    pub link: Link,
    pub detail: Option<String>,
    /// True when we joined an engine that was already running (the tray app's, say), in which
    /// case we must not shut it down when we exit.
    pub attached: bool,
    pub pid: Option<u32>,
    pub restarts: u64,
    /// The engine kept crashing, so it now runs on the processor with no AI clean-up. A restart
    /// asked for by the user ends it.
    pub safe_mode: bool,
}

impl Default for LinkState {
    fn default() -> Self {
        Self { link: Link::Starting, detail: None, attached: false, pid: None, restarts: 0, safe_mode: false }
    }
}

/// Recent engine crashes, to tell a crash loop from bad luck.
#[derive(Default)]
struct CrashWindow(VecDeque<std::time::Instant>);

impl CrashWindow {
    /// Count a crash at `now`. True when it makes `CRASH_LIMIT` inside `CRASH_WINDOW`.
    fn record(&mut self, now: std::time::Instant) -> bool {
        self.0.push_back(now);
        while self.0.front().is_some_and(|t| now.duration_since(*t) > CRASH_WINDOW) {
            self.0.pop_front();
        }
        self.0.len() >= CRASH_LIMIT
    }
}

enum Out {
    Text(String),
    Audio(Vec<u8>),
}

/// Where engine events go. The app implements this over Tauri's event system; the headless
/// self-test implements it over a channel, which is what makes the whole link testable without
/// a window.
pub trait Sink: Send + Sync + 'static {
    fn emit(&self, event: &str, payload: Value);
}

struct Shared {
    link: Mutex<LinkState>,
    /// Last `status` message from the engine, verbatim, so the UI can render model state.
    status: Mutex<Option<Value>>,
    connected: AtomicBool,
    stopping: AtomicBool,
    restarts: AtomicU64,
    /// Tail of the engine's stderr, to explain a failure without opening the log file.
    stderr_tail: Mutex<VecDeque<String>>,
    safe_mode: AtomicBool,
    crashes: Mutex<CrashWindow>,
}

impl Shared {
    fn new() -> Self {
        Shared {
            link: Mutex::new(LinkState::default()),
            status: Mutex::new(None),
            connected: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            restarts: AtomicU64::new(0),
            stderr_tail: Mutex::new(VecDeque::new()),
            safe_mode: AtomicBool::new(false),
            crashes: Mutex::new(CrashWindow::default()),
        }
    }
}

#[derive(Clone)]
pub struct Engine {
    tx: mpsc::UnboundedSender<Out>,
    shared: Arc<Shared>,
}

impl Engine {
    /// Start the supervisor. Returns immediately; watch the `engine-link` event for progress.
    pub fn start(sink: Arc<dyn Sink>) -> Engine {
        let (tx, rx) = mpsc::unbounded_channel();
        let shared = Arc::new(Shared::new());
        let engine = Engine { tx, shared: shared.clone() };
        tauri::async_runtime::spawn(supervise_forever(sink, shared, rx));
        engine
    }

    pub fn is_connected(&self) -> bool {
        self.shared.connected.load(Ordering::Relaxed)
    }

    pub fn link(&self) -> LinkState {
        self.shared.link.locked().clone()
    }

    pub fn status(&self) -> Option<Value> {
        self.shared.status.locked().clone()
    }

    /// True once the speech model is loaded, i.e. dictation would work right now.
    pub fn stt_ready(&self) -> bool {
        self.status()
            .and_then(|s| s.get("stt").and_then(|v| v.get("state")).and_then(|v| v.as_str().map(str::to_owned)))
            .map(|s| s == "ready")
            .unwrap_or(false)
    }

    pub fn send(&self, msg: Value) {
        if let Ok(text) = serde_json::to_string(&msg) {
            let _ = self.tx.send(Out::Text(text));
        }
    }

    /// One 20 ms frame (or any multiple), int16 little-endian, 16 kHz mono.
    pub fn send_audio(&self, pcm: &[i16]) {
        if !self.is_connected() {
            return; // nowhere to put it; the pre-roll buffer keeps the recent past instead
        }
        let mut bytes = Vec::with_capacity(pcm.len() * 2);
        for s in pcm {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let _ = self.tx.send(Out::Audio(bytes));
    }

    pub fn session_start(&self, id: &str, context: Value, language: Option<String>) {
        let mut msg = json!({"type": "session.start", "id": id, "context": context});
        if let Some(lang) = language {
            msg["language"] = Value::String(lang);
        }
        self.send(msg);
    }

    pub fn session_end(&self, id: &str) {
        self.send(json!({"type": "session.end", "id": id}));
    }

    /// Ask the engine to apply a spoken instruction to the user's selection.
    pub fn run_command(&self, id: &str, selection: &str, instruction: &str) {
        self.send(json!({
            "type": "command.run",
            "id": id,
            "selection": selection,
            "instruction": instruction,
        }));
    }

    pub fn session_cancel(&self, id: &str) {
        self.send(json!({"type": "session.cancel", "id": id}));
    }

    /// Ask the supervisor to drop the current engine and start a fresh one - a normal one: the
    /// user asking for a restart is also how safe mode is left.
    pub fn restart(&self) {
        self.shared.safe_mode.store(false, Ordering::SeqCst);
        *self.shared.crashes.locked() = CrashWindow::default();
        let _ = self.tx.send(Out::Text("\u{0}restart".into()));
    }

    /// Stop supervising. Shuts the engine down too, unless we merely attached to someone else's.
    pub fn shutdown(&self) {
        self.shared.stopping.store(true, Ordering::SeqCst);
        let _ = self.tx.send(Out::Text("\u{0}stop".into()));
    }
}

// ---------------------------------------------------------------------------------------------
// where the engine lives

/// `%APPDATA%\LocalFlow`, matching `localflow.config.CONFIG_DIR`.
fn config_dir() -> Option<PathBuf> {
    crate::paths::config_dir()
}

/// Port and token of an engine that is already running. The info file outlives a killed
/// engine, so the pid in it is checked before it is believed - and checked to be an engine,
/// because Windows reuses process ids: a stale file once named a pid that had since gone to
/// some unrelated program, and the shell sat on "Starting..." for good, attaching to nothing.
fn discover() -> Option<(u16, String, u32)> {
    let path = config_dir()?.join("engine.json");
    let text = std::fs::read_to_string(&path).ok()?;
    let info: Value = serde_json::from_str(&text).ok()?;
    let port = info.get("port")?.as_u64()? as u16;
    let token = info.get("token")?.as_str()?.to_owned();
    let pid = info.get("pid")?.as_u64()? as u32;
    if !crate::win::pid_alive(pid) || !is_engine_process(pid) {
        let _ = std::fs::remove_file(&path);
        return None;
    }
    Some((port, token, pid))
}

/// The frozen engine, or Python running it from a virtualenv in development.
fn is_engine_process(pid: u32) -> bool {
    crate::win::process_image_name(pid)
        .map(|name| name.starts_with("localflow") || name.starts_with("python"))
        .unwrap_or(false)
}

/// Remove the info file if it names `pid` (an engine of ours we have just stopped) or a
/// process that is gone.
fn forget_engine_info_of(pid: u32) {
    let Some(path) = config_dir().map(|d| d.join("engine.json")) else { return };
    let listed = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.get("pid").and_then(Value::as_u64))
        .map(|p| p as u32);
    if let Some(listed) = listed {
        if listed == pid || !crate::win::pid_alive(listed) {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Forget an engine that the info file names but that refuses connections.
fn forget_engine_info() {
    if let Some(dir) = config_dir() {
        let _ = std::fs::remove_file(dir.join("engine.json"));
    }
}

/// How to launch the engine. A packaged build ships a frozen exe next to the shell; a dev
/// build runs the repo's virtualenv.
fn engine_command() -> Result<Command> {
    if let Some(exe) = std::env::var_os("LOCALFLOW_ENGINE_EXE") {
        let mut c = Command::new(exe);
        c.args(["--log-level", "INFO", "serve", "--handshake"]);
        return Ok(c);
    }
    if let Ok(dir) = std::env::current_exe().and_then(|p| p.parent().map(PathBuf::from).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "no parent")
    })) {
        // Two layouts. Beside the shell is the simple one, used when the frozen engine is
        // copied into a dev build's target directory. Inside `localflow-engine/` is what a
        // PyInstaller one-directory bundle actually looks like, and where Tauri puts it when
        // it is shipped as a resource - the engine is a folder of a thousand files, not a
        // single exe, because unpacking a gigabyte of CUDA libraries on every launch would
        // add seconds to every start-up.
        for candidate in [
            dir.join("localflow-engine.exe"),
            dir.join("localflow-engine").join("localflow-engine.exe"),
            dir.join("resources").join("localflow-engine").join("localflow-engine.exe"),
        ] {
            if candidate.is_file() {
                let mut c = Command::new(candidate);
                c.args(["--log-level", "INFO", "serve", "--handshake"]);
                return Ok(c);
            }
        }
    }
    // Dev: the virtualenv at the repo root, found relative to this crate.
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("cannot locate the repository root"))?;
    let py = repo.join(".venv").join("Scripts").join("python.exe");
    if !py.is_file() {
        return Err(anyhow!(
            "no engine found: set LOCALFLOW_ENGINE_EXE, or create the virtualenv at {}",
            py.display()
        ));
    }
    let mut c = Command::new(py);
    c.args(["-m", "localflow", "--log-level", "INFO", "serve", "--handshake"]);
    c.current_dir(&repo);
    Ok(c)
}

struct Spawned {
    child: Child,
    port: u16,
    token: String,
    pid: u32,
}

async fn spawn_engine(shared: &Arc<Shared>) -> Result<Spawned> {
    let mut cmd = engine_command()?;
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
    cmd.kill_on_drop(true);
    if shared.safe_mode.load(Ordering::SeqCst) {
        cmd.env(SAFE_MODE_ENV, "1");
    } else {
        cmd.env_remove(SAFE_MODE_ENV);
    }
    crate::win::no_window(&mut cmd);
    let mut child = cmd.spawn().context("could not start the engine process")?;
    let pid = child.id().unwrap_or(0);

    // The engine must die with us. `kill()` does not reach grandchildren (llama-server), and a
    // crashed shell would leave a model server holding VRAM, so put it in a job object that the
    // kernel empties when this process ends.
    crate::win::assign_to_kill_job(&child);

    // Keep the tail of stderr so a failed start can say something better than "exited".
    if let Some(err) = child.stderr.take() {
        let shared = shared.clone();
        tauri::async_runtime::spawn(async move {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let mut tail = shared.stderr_tail.locked();
                if tail.len() >= 40 {
                    tail.pop_front();
                }
                tail.push_back(line);
            }
        });
    }

    let stdout = child.stdout.take().ok_or_else(|| anyhow!("engine stdout is not a pipe"))?;
    let mut lines = BufReader::new(stdout).lines();
    let deadline = tokio::time::Instant::now() + HANDSHAKE_TIMEOUT;
    let handshake = loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break Err(anyhow!("the engine did not report a port within {:?}", HANDSHAKE_TIMEOUT));
        }
        match tokio::time::timeout(remaining, lines.next_line()).await {
            Err(_) => break Err(anyhow!("the engine did not report a port within {:?}", HANDSHAKE_TIMEOUT)),
            Ok(Err(e)) => break Err(anyhow!("reading the engine handshake failed: {e}")),
            Ok(Ok(None)) => {
                break Err(anyhow!("the engine exited during start{}", shared.stderr_summary()));
            }
            Ok(Ok(Some(line))) => {
                if let Ok(info) = serde_json::from_str::<Value>(&line) {
                    if let (Some(port), Some(token)) = (
                        info.get("port").and_then(Value::as_u64),
                        info.get("token").and_then(Value::as_str),
                    ) {
                        // The engine's own pid: in development a launcher sits in between, so
                        // the process we spawned is not the engine.
                        let engine_pid = info.get("pid").and_then(Value::as_u64).map(|p| p as u32);
                        break Ok((port as u16, token.to_owned(), engine_pid));
                    }
                }
                // anything else on stdout is noise; keep looking
            }
        }
    };

    // Whatever the engine writes to stdout afterwards must still be drained, or a full pipe
    // buffer would eventually block it.
    tauri::async_runtime::spawn(async move { while let Ok(Some(_)) = lines.next_line().await {} });

    match handshake {
        Ok((port, token, engine_pid)) => Ok(Spawned { child, port, token, pid: engine_pid.unwrap_or(pid) }),
        Err(e) => {
            let _ = child.start_kill();
            Err(e)
        }
    }
}

impl Shared {
    /// Count an engine crash, and turn on safe mode for the next start if it makes a loop.
    fn note_crash(&self) {
        if self.crashes.locked().record(std::time::Instant::now())
            && !self.safe_mode.swap(true, Ordering::SeqCst)
        {
            crate::shell_log!(
                "the engine stopped {CRASH_LIMIT} times in {}s; starting it in safe mode",
                CRASH_WINDOW.as_secs()
            );
        }
    }

    fn stderr_summary(&self) -> String {
        let tail = self.stderr_tail.locked();
        let last: Vec<&str> = tail.iter().rev().take(3).map(String::as_str).rev().collect();
        if last.is_empty() {
            String::new()
        } else {
            format!(": {}", last.join(" / "))
        }
    }

    fn set_link(&self, sink: &Arc<dyn Sink>, link: Link, detail: Option<String>, attached: bool, pid: Option<u32>) {
        let state = {
            let mut cur = self.link.locked();
            cur.link = link;
            cur.detail = detail;
            cur.attached = attached;
            cur.pid = pid;
            cur.restarts = self.restarts.load(Ordering::Relaxed);
            cur.safe_mode = self.safe_mode.load(Ordering::SeqCst);
            cur.clone()
        };
        self.connected.store(link == Link::Ready, Ordering::SeqCst);
        crate::shell_log!(
            "engine {:?}{}{}",
            link,
            state.detail.as_deref().map(|d| format!(": {d}")).unwrap_or_default(),
            state.pid.map(|p| format!(" (pid {p}{})", if state.attached { ", attached" } else { "" })).unwrap_or_default()
        );
        sink.emit("engine-link", serde_json::to_value(&state).unwrap_or(Value::Null));
    }
}

// ---------------------------------------------------------------------------------------------
// the supervisor

/// The supervisor, restarted if it panics.
///
/// A panicking task is quietly dropped by the runtime: the link would freeze in whatever state
/// it was in, no engine would ever be started again, and nothing would say so. Unwinding drops
/// the engine process too (`kill_on_drop`), so the next round starts from nothing.
async fn supervise_forever(sink: Arc<dyn Sink>, shared: Arc<Shared>, mut rx: mpsc::UnboundedReceiver<Out>) {
    loop {
        let round = supervise(sink.clone(), shared.clone(), &mut rx);
        if AssertUnwindSafe(round).catch_unwind().await.is_ok() {
            return;
        }
        shared.connected.store(false, Ordering::SeqCst);
        shared.status.locked().take();
        shared.set_link(&sink, Link::Reconnecting, Some("recovering from an internal fault".into()), false, None);
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

async fn supervise(sink: Arc<dyn Sink>, shared: Arc<Shared>, rx: &mut mpsc::UnboundedReceiver<Out>) {
    let mut backoff = Duration::from_millis(500);
    loop {
        if shared.stopping.load(Ordering::SeqCst) {
            shared.set_link(&sink, Link::Stopped, None, false, None);
            return;
        }

        // Join an engine that is already running (the Python tray app's, in development) rather
        // than starting a second set of models: two of them would not fit in 8 GB of VRAM.
        let attached = discover();
        let mut owned: Option<Child> = None;
        let (port, token, pid) = match attached {
            Some(found) => {
                shared.set_link(&sink, Link::Connecting, Some("attaching to a running engine".into()), true, Some(found.2));
                found
            }
            None => {
                shared.set_link(&sink, Link::Starting, None, false, None);
                match spawn_engine(&shared).await {
                    Ok(s) => {
                        let pid = s.pid;
                        owned = Some(s.child);
                        shared.set_link(&sink, Link::Connecting, None, false, Some(pid));
                        (s.port, s.token, pid)
                    }
                    Err(e) => {
                        let msg = format!("{e:#}");
                        shared.note_crash();
                        shared.set_link(&sink, Link::Failed, Some(msg), false, None);
                        if !wait_backoff(&mut backoff, rx, &shared).await {
                            shared.set_link(&sink, Link::Stopped, None, false, None);
                            return;
                        }
                        continue;
                    }
                }
            }
        };
        let is_attached = owned.is_none();

        let started = tokio::time::Instant::now();
        let reason = run_link(&sink, &shared, rx, port, &token, pid, is_attached, owned.as_mut()).await;
        // Only an engine of our own that went away by itself counts: not a restart or a quit,
        // and not someone else's engine we had merely attached to.
        if !is_attached && matches!(reason, Stop::Lost(_) | Stop::Unreachable(_)) {
            shared.note_crash();
        }

        shared.connected.store(false, Ordering::SeqCst);
        shared.status.locked().take();

        // Whether we own the engine decides how it ends: kill ours, leave someone else's alone.
        if let Some(mut child) = owned {
            if shared.stopping.load(Ordering::SeqCst) {
                let _ = child.start_kill();
            } else {
                let _ = child.start_kill();
                let _ = child.wait().await;
            }
            // Killed, it cannot remove its own info file, and the file then names a dead or
            // dying process: the next round tried to join it before starting a fresh engine.
            forget_engine_info_of(pid);
        }

        match reason {
            Stop::Requested => {
                shared.set_link(&sink, Link::Stopped, None, false, None);
                return;
            }
            Stop::Restart => {
                backoff = Duration::from_millis(500);
                shared.restarts.fetch_add(1, Ordering::Relaxed);
                shared.set_link(&sink, Link::Starting, Some("restarting".into()), false, None);
                continue;
            }
            Stop::Unreachable(why) if is_attached => {
                // An engine we only knew of from its info file, which will not take a
                // connection: the file is stale. Drop it and start our own straight away.
                forget_engine_info();
                shared.set_link(&sink, Link::Reconnecting, Some(why), false, None);
                backoff = Duration::from_millis(500);
            }
            Stop::Lost(why) | Stop::Unreachable(why) => {
                if started.elapsed() >= HEALTHY_AFTER {
                    backoff = Duration::from_millis(500);
                }
                shared.restarts.fetch_add(1, Ordering::Relaxed);
                shared.set_link(&sink, Link::Reconnecting, Some(why), false, None);
                if !wait_backoff(&mut backoff, rx, &shared).await {
                    shared.set_link(&sink, Link::Stopped, None, false, None);
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod crash_window_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn three_crashes_in_two_minutes_is_a_loop_and_three_in_an_hour_is_not() {
        let t0 = Instant::now();
        let mut w = CrashWindow::default();
        assert!(!w.record(t0));
        assert!(!w.record(t0 + Duration::from_secs(50)));
        assert!(w.record(t0 + Duration::from_secs(100)), "third inside the window");

        let mut spread = CrashWindow::default();
        assert!(!spread.record(t0));
        assert!(!spread.record(t0 + Duration::from_secs(1200)));
        assert!(!spread.record(t0 + Duration::from_secs(2400)), "old crashes age out");
    }
}

enum Stop {
    /// The app is closing.
    Requested,
    /// Someone asked for a fresh engine.
    Restart,
    /// The link broke by itself.
    Lost(String),
    /// The engine never answered at all.
    Unreachable(String),
}

/// Sentinel messages the handle sends through the ordinary outbound channel, so that a restart
/// or a shutdown cannot overtake audio that is still queued.
fn sentinel(text: &str) -> Option<Stop> {
    match text {
        "\u{0}restart" => Some(Stop::Restart),
        "\u{0}stop" => Some(Stop::Requested),
        _ => None,
    }
}

/// One connection, from hello to whatever ends it.
async fn run_link(
    sink: &Arc<dyn Sink>,
    shared: &Arc<Shared>,
    rx: &mut mpsc::UnboundedReceiver<Out>,
    port: u16,
    token: &str,
    pid: u32,
    attached: bool,
    mut child: Option<&mut Child>,
) -> Stop {
    let url = format!("ws://127.0.0.1:{port}");
    // The engine's largest message is a status of a few kilobytes; tungstenite would otherwise
    // buffer up to 64 MB for whatever answers on this port. No Nagle: audio goes out in 20 ms
    // frames and each should leave at once.
    let config = WebSocketConfig::default()
        .max_message_size(Some(ENGINE_MESSAGE_MAX))
        .max_frame_size(Some(ENGINE_MESSAGE_MAX));
    let connecting = tokio_tungstenite::connect_async_with_config(&url, Some(config), true);
    let ws = match tokio::time::timeout(Duration::from_secs(10), connecting).await {
        Ok(Ok((ws, _))) => ws,
        Ok(Err(e)) => return Stop::Unreachable(format!("could not connect to the engine: {e}")),
        Err(_) => return Stop::Lost("timed out connecting to the engine".into()),
    };
    let (mut ws_tx, mut stream) = ws.split();

    let hello = json!({"type": "hello", "token": token, "client": "shell"});
    if let Err(e) = ws_tx.send(Message::Text(hello.to_string().into())).await {
        return Stop::Lost(format!("could not greet the engine: {e}"));
    }
    match tokio::time::timeout(Duration::from_secs(10), stream.next()).await {
        Ok(Some(Ok(Message::Text(text)))) => {
            let msg: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            if msg.get("type").and_then(Value::as_str) != Some("hello.ok") {
                return Stop::Lost(format!("the engine refused the handshake: {text}"));
            }
            if let Some(status) = msg.get("status") {
                *shared.status.locked() = Some(status.clone());
                sink.emit("engine-status", status.clone());
            }
        }
        Ok(Some(Ok(other))) => return Stop::Lost(format!("unexpected reply to hello: {other:?}")),
        Ok(Some(Err(e))) => return Stop::Lost(format!("the engine closed during hello: {e}")),
        Ok(None) | Err(_) => return Stop::Lost("the engine did not answer hello".into()),
    }

    // Anything queued while we were disconnected is stale audio from a dictation nobody will
    // finish; start this connection with an empty queue.
    while rx.try_recv().is_ok() {}
    shared.set_link(sink, Link::Ready, None, attached, Some(pid));

    let mut beat = tokio::time::interval(HEARTBEAT);
    beat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut last_seen = tokio::time::Instant::now();

    loop {
        tokio::select! {
            biased;

            out = rx.recv() => match out {
                None => return Stop::Requested,
                Some(Out::Text(text)) => {
                    if let Some(stop) = sentinel(&text) {
                        let _ = ws_tx.send(Message::Close(None)).await;
                        return stop;
                    }
                    if let Err(e) = ws_tx.send(Message::Text(text.into())).await {
                        return Stop::Lost(format!("send failed: {e}"));
                    }
                }
                Some(Out::Audio(bytes)) => {
                    if let Err(e) = ws_tx.send(Message::Binary(bytes.into())).await {
                        return Stop::Lost(format!("audio send failed: {e}"));
                    }
                }
            },

            incoming = stream.next() => match incoming {
                None => return Stop::Lost("the engine closed the connection".into()),
                Some(Err(e)) => return Stop::Lost(format!("connection error: {e}")),
                Some(Ok(Message::Text(text))) => {
                    last_seen = tokio::time::Instant::now();
                    dispatch(sink, shared, &text);
                }
                Some(Ok(Message::Close(_))) => return Stop::Lost("the engine closed the connection".into()),
                Some(Ok(_)) => { last_seen = tokio::time::Instant::now(); }
            },

            _ = beat.tick() => {
                if last_seen.elapsed() > SILENCE_LIMIT {
                    return Stop::Lost(format!("the engine stopped answering for {:?}", last_seen.elapsed()));
                }
                if let Some(c) = child.as_deref_mut() {
                    match c.try_wait() {
                        Ok(Some(status)) => return Stop::Lost(format!("the engine exited ({status}){}", shared.stderr_summary())),
                        Err(e) => return Stop::Lost(format!("lost track of the engine process: {e}")),
                        Ok(None) => {}
                    }
                } else if !crate::win::pid_alive(pid) {
                    return Stop::Lost("the engine we attached to went away".into());
                }
                if let Err(e) = ws_tx.send(Message::Text(json!({"type": "status.get"}).to_string().into())).await {
                    return Stop::Lost(format!("heartbeat failed: {e}"));
                }
            }
        }
    }
}

/// Engine events, forwarded to the frontend under their own names so the UI can subscribe to
/// exactly what it needs.
fn dispatch(sink: &Arc<dyn Sink>, shared: &Arc<Shared>, text: &str) {
    let Ok(msg) = serde_json::from_str::<Value>(text) else { return };
    match msg.get("type").and_then(Value::as_str).unwrap_or("") {
        "status" => {
            *shared.status.locked() = Some(msg.clone());
            sink.emit("engine-status", msg);
        }
        "partial" => sink.emit("partial", msg),
        "final" => sink.emit("final", msg),
        "command.result" => sink.emit("command-result", msg),
        "selfcheck.result" => sink.emit("selfcheck-result", msg),
        "selfcheck.repaired" => sink.emit("selfcheck-repaired", msg),
        "error" => sink.emit("engine-error", msg),
        _ => {}
    }
}

/// Sleep out the backoff, but stay responsive to a shutdown or a restart request.
/// Returns false when the app is closing.
async fn wait_backoff(backoff: &mut Duration, rx: &mut mpsc::UnboundedReceiver<Out>, shared: &Arc<Shared>) -> bool {
    let delay = *backoff;
    *backoff = (*backoff * 2).min(BACKOFF_MAX);
    let deadline = tokio::time::Instant::now() + delay;
    loop {
        tokio::select! {
            _ = tokio::time::sleep_until(deadline) => return !shared.stopping.load(Ordering::SeqCst),
            out = rx.recv() => match out {
                None => return false,
                Some(Out::Text(text)) => match sentinel(&text) {
                    Some(Stop::Requested) => return false,
                    Some(_) => return true, // restart now, skip the rest of the wait
                    None => {}
                },
                Some(Out::Audio(_)) => {}
            },
        }
    }
}

#[cfg(test)]
mod message_tests {
    use super::*;

    #[derive(Default)]
    struct Recorded(Mutex<Vec<(String, Value)>>);

    impl Sink for Recorded {
        fn emit(&self, event: &str, payload: Value) {
            self.0.locked().push((event.to_owned(), payload));
        }
    }

    #[test]
    fn malformed_engine_messages_are_ignored() {
        let recorded = Arc::new(Recorded::default());
        let sink: Arc<dyn Sink> = recorded.clone();
        let shared = Arc::new(Shared::new());
        let deep = "[".repeat(100_000);
        let junk = ["", "null", "[]", "{}", r#""status""#, r#"{"type":5}"#, r#"{"type":null}"#,
                    r#"{"type":"nonsense"}"#, r#"{"type":"status""#, "\u{0}", &deep];
        for text in junk {
            dispatch(&sink, &shared, text);
        }
        assert!(recorded.0.locked().is_empty(), "nothing is forwarded");
        assert!(shared.status.locked().is_none(), "no status is taken from junk");

        // Known types with missing or wrong fields are forwarded as they are: every handler
        // reads its fields with defaults.
        dispatch(&sink, &shared, r#"{"type":"final","id":7,"text":null}"#);
        dispatch(&sink, &shared, r#"{"type":"status","stt":"not an object"}"#);
        let events: Vec<String> = recorded.0.locked().iter().map(|(e, _)| e.clone()).collect();
        assert_eq!(events, ["final", "engine-status"]);
    }

    #[test]
    fn a_status_of_the_wrong_shape_means_not_ready() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let engine = Engine { tx, shared: Arc::new(Shared::new()) };
        for status in [json!({"stt": "ready"}), json!({"stt": {"state": 1}}), json!([1]), json!("ready")] {
            *engine.shared.status.locked() = Some(status);
            assert!(!engine.stt_ready());
        }
        *engine.shared.status.locked() = Some(json!({"stt": {"state": "ready"}}));
        assert!(engine.stt_ready());
    }
}
