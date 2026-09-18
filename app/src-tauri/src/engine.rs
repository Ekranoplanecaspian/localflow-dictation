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
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;

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
}

impl Default for LinkState {
    fn default() -> Self {
        Self { link: Link::Starting, detail: None, attached: false, pid: None, restarts: 0 }
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
        let shared = Arc::new(Shared {
            link: Mutex::new(LinkState::default()),
            status: Mutex::new(None),
            connected: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            restarts: AtomicU64::new(0),
            stderr_tail: Mutex::new(VecDeque::new()),
        });
        let engine = Engine { tx, shared: shared.clone() };
        tauri::async_runtime::spawn(supervise(sink, shared, rx));
        engine
    }

    pub fn is_connected(&self) -> bool {
        self.shared.connected.load(Ordering::Relaxed)
    }

    pub fn link(&self) -> LinkState {
        self.shared.link.lock().unwrap().clone()
    }

    pub fn status(&self) -> Option<Value> {
        self.shared.status.lock().unwrap().clone()
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

    /// Ask the supervisor to drop the current engine and start a fresh one.
    pub fn restart(&self) {
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
/// engine, so the pid in it is checked before it is believed.
fn discover() -> Option<(u16, String, u32)> {
    let path = config_dir()?.join("engine.json");
    let text = std::fs::read_to_string(&path).ok()?;
    let info: Value = serde_json::from_str(&text).ok()?;
    let port = info.get("port")?.as_u64()? as u16;
    let token = info.get("token")?.as_str()?.to_owned();
    let pid = info.get("pid")?.as_u64()? as u32;
    if !crate::win::pid_alive(pid) {
        let _ = std::fs::remove_file(&path);
        return None;
    }
    Some((port, token, pid))
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
                let mut tail = shared.stderr_tail.lock().unwrap();
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
                        break Ok((port as u16, token.to_owned()));
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
        Ok((port, token)) => Ok(Spawned { child, port, token, pid }),
        Err(e) => {
            let _ = child.start_kill();
            Err(e)
        }
    }
}

impl Shared {
    fn stderr_summary(&self) -> String {
        let tail = self.stderr_tail.lock().unwrap();
        let last: Vec<&str> = tail.iter().rev().take(3).map(String::as_str).rev().collect();
        if last.is_empty() {
            String::new()
        } else {
            format!(": {}", last.join(" / "))
        }
    }

    fn set_link(&self, sink: &Arc<dyn Sink>, link: Link, detail: Option<String>, attached: bool, pid: Option<u32>) {
        let state = {
            let mut cur = self.link.lock().unwrap();
            cur.link = link;
            cur.detail = detail;
            cur.attached = attached;
            cur.pid = pid;
            cur.restarts = self.restarts.load(Ordering::Relaxed);
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

async fn supervise(sink: Arc<dyn Sink>, shared: Arc<Shared>, mut rx: mpsc::UnboundedReceiver<Out>) {
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
                        shared.set_link(&sink, Link::Failed, Some(msg), false, None);
                        if !wait_backoff(&mut backoff, &mut rx, &shared).await {
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
        let reason = run_link(&sink, &shared, &mut rx, port, &token, pid, is_attached, owned.as_mut()).await;

        shared.connected.store(false, Ordering::SeqCst);
        shared.status.lock().unwrap().take();

        // Whether we own the engine decides how it ends: kill ours, leave someone else's alone.
        if let Some(mut child) = owned {
            if shared.stopping.load(Ordering::SeqCst) {
                let _ = child.start_kill();
            } else {
                let _ = child.start_kill();
                let _ = child.wait().await;
            }
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
            Stop::Lost(why) => {
                if started.elapsed() >= HEALTHY_AFTER {
                    backoff = Duration::from_millis(500);
                }
                shared.restarts.fetch_add(1, Ordering::Relaxed);
                shared.set_link(&sink, Link::Reconnecting, Some(why), false, None);
                if !wait_backoff(&mut backoff, &mut rx, &shared).await {
                    shared.set_link(&sink, Link::Stopped, None, false, None);
                    return;
                }
            }
        }
    }
}

enum Stop {
    /// The app is closing.
    Requested,
    /// Someone asked for a fresh engine.
    Restart,
    /// The link broke by itself.
    Lost(String),
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
    let ws = match tokio::time::timeout(Duration::from_secs(10), tokio_tungstenite::connect_async(&url)).await {
        Ok(Ok((ws, _))) => ws,
        Ok(Err(e)) => return Stop::Lost(format!("could not connect to the engine: {e}")),
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
                *shared.status.lock().unwrap() = Some(status.clone());
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
            *shared.status.lock().unwrap() = Some(msg.clone());
            sink.emit("engine-status", msg);
        }
        "partial" => sink.emit("partial", msg),
        "final" => sink.emit("final", msg),
        "command.result" => sink.emit("command-result", msg),
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
