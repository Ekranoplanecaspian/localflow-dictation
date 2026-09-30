//! One answer to "is LocalFlow working?": every part is ok, degraded or failed, with a reason in
//! plain words and, where there is one, the action that fixes it.
//!
//! The parts used to be read separately by each place that showed them: the tray from the
//! engine link alone, the Hub from the engine's raw status, the notifications from two events.
//! A clean-up model that failed to load was visible in one of them. Here they are assessed once
//! (`assess`, pure, so it is tested without a window) and every surface shows the same result at
//! its own volume: the tray always, the Hub's Status card in full, a notification once per
//! change, the flow bar only when the take in hand is affected.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager};

use crate::engine::{Engine, Link, LinkState};
use crate::guard::LockExt;
use crate::problems;

/// A problem must last this long before it is announced: an engine that restarts in two
/// seconds, or a model moving between devices, is not news.
const NOTIFY_AFTER: Duration = Duration::from_secs(10);
/// Everything must have been working this long before "working normally again" is said.
const RECOVERED_AFTER: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    /// Working.
    Ok,
    /// Turned off by the user: not a problem.
    Off,
    /// Depends on the engine, which is not there to ask.
    Waiting,
    /// Loading or connecting; it will be ready shortly.
    Starting,
    /// Working in a lesser way (clean-up unavailable, so rules only).
    Degraded,
    /// Not working.
    Failed,
}

impl Level {
    fn bad(self) -> bool {
        matches!(self, Level::Degraded | Level::Failed)
    }
}

/// What a Fix button does. The Hub maps the id to a command or a page; the tray and the
/// notifications only mention that there is one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Action {
    pub id: &'static str,
    pub label: &'static str,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Part {
    pub id: &'static str,
    pub name: &'static str,
    pub level: Level,
    /// A word or two for the row: "Ready", "Loading", "Unavailable", the model's name.
    pub summary: String,
    /// The whole situation as one sentence, for a notification title or the tray tooltip.
    pub headline: String,
    /// Why, when there is something to explain.
    pub reason: Option<String>,
    pub action: Option<Action>,
    /// Needed for dictation at all. The others only make it better.
    pub critical: bool,
    /// The problem it is in (`shared/problems.json`), for the Hub's details and the log.
    pub code: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Health {
    pub overall: Level,
    pub headline: String,
    pub parts: Vec<Part>,
}

/// What the microphone supervisor last said.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mic {
    /// The device a stream is open on.
    pub device: Option<String>,
    /// Why no stream could be opened.
    pub error: Option<String>,
    /// The microphone chosen in the Hub; empty for Windows' default.
    pub chosen: String,
    /// The error is another app holding the microphone in exclusive mode.
    pub busy: bool,
    /// The device a stream is open on belongs to a Bluetooth headset.
    pub bluetooth: bool,
}

pub struct Inputs<'a> {
    pub link: &'a LinkState,
    pub status: Option<&'a Value>,
    pub mic: &'a Mic,
    pub hook_installed: bool,
    pub hotkey: &'a str,
    /// Windows privacy settings deny desktop apps the microphone (selfcheck.rs).
    pub mic_blocked: bool,
    /// The settings folder cannot be written: (folder, why).
    pub settings_unwritable: Option<(String, String)>,
}

/// A part that is working, or on its way: its words are its own, not a problem's.
fn part(id: &'static str, name: &'static str, critical: bool, level: Level, summary: impl Into<String>,
        headline: impl Into<String>) -> Part {
    Part { id, name, level, summary: summary.into(), headline: headline.into(), reason: None, action: None,
           critical, code: None }
}

/// A part in one of the catalogue's problems, worded by the catalogue.
fn trouble(id: &'static str, name: &'static str, critical: bool, code: problems::Code, vars: &[(&str, &str)]) -> Part {
    let shown = problems::show(code, vars);
    let level = match problems::get(code).level.as_str() {
        "failed" => Level::Failed,
        _ => Level::Degraded,
    };
    Part {
        id,
        name,
        level,
        summary: shown.summary.unwrap_or_else(|| shown.title.clone()),
        headline: shown.title,
        reason: Some(shown.message).filter(|m| !m.is_empty()),
        action: shown.action.map(|(id, label)| Action { id, label }),
        critical,
        code: Some(shown.code),
    }
}

impl Part {
    fn because(mut self, reason: impl Into<String>) -> Part {
        let r: String = reason.into();
        self.reason = (!r.trim().is_empty()).then_some(r);
        self
    }
}

fn text<'v>(v: Option<&'v Value>, path: &[&str]) -> Option<&'v str> {
    let mut cur = v?;
    for key in path {
        cur = cur.get(key)?;
    }
    cur.as_str().filter(|s| !s.is_empty())
}

fn where_(device: Option<&str>) -> &'static str {
    match device {
        Some("cuda") => "on the graphics card",
        Some("cpu") => "on the processor",
        _ => "",
    }
}

pub fn assess(i: &Inputs) -> Health {
    let ready = i.link.link == Link::Ready;
    let status = if ready { i.status } else { None };
    let mut parts = vec![engine_part(i.link)];
    if status.is_none() {
        for (id, name) in [("speech", "Speech model"), ("cleanup", "AI clean-up"), ("gpu", "Graphics card")] {
            parts.push(part(id, name, id == "speech", Level::Waiting, "Waiting for the engine",
                            format!("{name}: waiting for the engine")));
        }
    } else {
        parts.push(speech_part(status));
        parts.push(cleanup_part(status, i.link.safe_mode));
        parts.push(gpu_part(status));
    }
    parts.push(if i.mic_blocked {
        // The stream opens and delivers silence, so the microphone looks fine but hears nothing.
        trouble("microphone", "Microphone", true, problems::MIC_BLOCKED, &[])
    } else {
        mic_part(i.mic)
    });
    parts.push(hotkey_part(i.hook_installed, i.hotkey));
    parts.push(storage_part(status, i.settings_unwritable.as_ref()));
    let overall = overall(&parts);
    let headline = match overall {
        Level::Ok | Level::Off | Level::Waiting => "Everything is working".to_owned(),
        Level::Starting => "Starting up".to_owned(),
        _ => worst(&parts).map(|p| p.headline.clone()).unwrap_or_default(),
    };
    Health { overall, headline, parts }
}

/// Dictation-critical parts decide between failed and starting; anything else that is wrong,
/// or a critical part working in a lesser way, is degraded.
fn overall(parts: &[Part]) -> Level {
    let critical = |l: Level| parts.iter().any(|p| p.critical && p.level == l);
    if critical(Level::Failed) {
        Level::Failed
    } else if critical(Level::Starting) || critical(Level::Waiting) {
        Level::Starting
    } else if parts.iter().any(|p| p.level.bad()) {
        Level::Degraded
    } else {
        Level::Ok
    }
}

fn worst(parts: &[Part]) -> Option<&Part> {
    parts.iter().filter(|p| p.level.bad()).max_by_key(|p| (p.level, p.critical))
}

/// An engine that cannot be started because its program is not there, as the shell words it
/// (engine.rs) or as Windows does ("os error 2": file not found; 3: path not found).
fn engine_is_missing(detail: &str) -> bool {
    detail.contains("no engine found") || detail.contains("(os error 2)") || detail.contains("(os error 3)")
}

/// Windows refused to run it: a policy (AppLocker, Software Restriction Policies: 1260), Smart
/// App Control or WDAC (4551), antivirus (225: infected, 226: removed).
fn engine_is_blocked(detail: &str) -> bool {
    ["(os error 1260)", "(os error 4551)", "(os error 225)", "(os error 226)"].iter().any(|c| detail.contains(c))
}

fn engine_part(link: &LinkState) -> Part {
    let detail = sentence(link.detail.as_deref().unwrap_or(""));
    let vars = [("detail", detail.as_str())];
    match link.link {
        Link::Ready if link.safe_mode => trouble("engine", "Engine", true, problems::ENGINE_SAFE_MODE, &vars),
        Link::Ready => part("engine", "Engine", true, Level::Ok, "Running", "The engine is running"),
        Link::Starting | Link::Connecting => {
            part("engine", "Engine", true, Level::Starting, "Starting", "The engine is starting")
        }
        Link::Reconnecting => trouble("engine", "Engine", true, problems::ENGINE_RESTARTING, &vars),
        Link::Failed if engine_is_missing(&detail) => trouble("engine", "Engine", true, problems::ENGINE_MISSING, &vars),
        Link::Failed if engine_is_blocked(&detail) => trouble("engine", "Engine", true, problems::ENGINE_BLOCKED, &vars),
        Link::Failed => trouble("engine", "Engine", true, problems::ENGINE_WONT_START, &vars),
        // LocalFlow is quitting.
        Link::Stopped => part("engine", "Engine", true, Level::Off, "Stopped", "The engine is stopped"),
    }
}

/// The catalogue code a model section of the engine's status reports, or `fallback` for an
/// engine too old to send one.
fn code_of(section: Option<&Value>, fallback: problems::Code) -> problems::Code {
    problems::from_engine(text(section, &["error_code"]), fallback)
}

fn speech_part(status: Option<&Value>) -> Part {
    let stt = status.and_then(|s| s.get("stt"));
    let label = text(stt, &["label"]).or(text(stt, &["model"])).unwrap_or("The speech model").to_owned();
    match text(stt, &["state"]).unwrap_or("") {
        "ready" => {
            let place = where_(text(stt, &["device"]));
            let ok = part("speech", "Speech model", true, Level::Ok, label.clone(), format!("{label} is ready"))
                .because(if place.is_empty() { String::new() } else { format!("Running {place}.") });
            switched(ok, stt, "speech", "Speech model", true, problems::SPEECH_SWITCH_FAILED, &label)
        }
        "error" => {
            let detail = sentence(text(stt, &["error"]).unwrap_or(""));
            let code = code_of(stt, problems::SPEECH_LOAD_FAILED);
            trouble("speech", "Speech model", true, code, &[("detail", &detail), ("model", &label)])
        }
        _ => match stt.and_then(|s| s.get("download")).filter(|d| d.is_object()) {
            // A first run: the model is downloading, which can take minutes, not seconds.
            Some(d) => {
                let pct = d.get("progress").and_then(Value::as_f64).unwrap_or(0.0) * 100.0;
                let size = d.get("size_gb").and_then(Value::as_f64).map(|g| format!(" ({g:.1} GB)")).unwrap_or_default();
                part("speech", "Speech model", true, Level::Starting, format!("Downloading {pct:.0} %"),
                     format!("Downloading {label}{size}"))
                    .because("A first run downloads it once. Dictation starts as soon as it is here.")
            }
            None => part("speech", "Speech model", true, Level::Starting, "Loading", format!("Loading {label}")),
        },
    }
}

/// A working part, or the same part with a model switch that failed or is under way.
fn switched(ok: Part, section: Option<&Value>, id: &'static str, name: &'static str, critical: bool,
            failed_code: problems::Code, label: &str) -> Part {
    let Some(sw) = section.and_then(|s| s.get("switch")).filter(|v| v.is_object()) else { return ok };
    let to = text(Some(sw), &["label"]).or(text(Some(sw), &["to"])).unwrap_or("the new model").to_owned();
    match text(Some(sw), &["state"]).unwrap_or("") {
        "error" => {
            let detail = sentence(text(Some(sw), &["error"]).unwrap_or("it failed"));
            let code = code_of(Some(sw), failed_code);
            trouble(id, name, critical, code, &[("detail", &detail), ("to", &to), ("model", label)])
        }
        "downloading" => {
            let pct = sw.get("progress").and_then(Value::as_f64).map(|p| if p <= 1.0 { p * 100.0 } else { p });
            let doing = match pct {
                Some(pct) => format!("Downloading {pct:.0} % of"),
                None => "Downloading".to_owned(),
            };
            ok.because(format!("{doing} {to}; {label} works meanwhile."))
        }
        _ => ok.because(format!("Loading {to}; {label} works meanwhile.")),
    }
}

fn cleanup_part(status: Option<&Value>, safe_mode: bool) -> Part {
    let llm = status.and_then(|s| s.get("llm"));
    let label = text(llm, &["label"]).or(text(llm, &["model"])).unwrap_or("The clean-up model").to_owned();
    let enabled = llm.and_then(|l| l.get("enabled")).and_then(Value::as_bool).unwrap_or(false);
    let cloud = text(llm, &["provider"]).is_some_and(|p| p != "bundled");
    let device = status.and_then(|s| text(Some(s), &["compute", "cleanup"]));
    let p = |level, summary: String, headline: String| part("cleanup", "AI clean-up", false, level, summary, headline);
    if safe_mode {
        return p(Level::Off, "Off in safe mode".into(), "AI clean-up is off in safe mode".into());
    }
    if !enabled {
        return p(Level::Off, "Off".into(), "AI clean-up is off".into()).because("Turned off in Models.");
    }
    match text(llm, &["state"]).unwrap_or("") {
        "ready" => {
            let place = if cloud { "through your cloud provider" } else { where_(device) };
            let ok = p(Level::Ok, label.clone(), format!("{label} is ready"))
                .because(if place.is_empty() { String::new() } else { format!("Running {place}.") });
            switched(ok, llm, "cleanup", "AI clean-up", false, problems::CLEANUP_SWITCH_FAILED, &label)
        }
        "asleep" => p(Level::Ok, label.clone(), format!("{label} is resting"))
            .because("Unloaded while LocalFlow is idle, to free memory; the next dictation wakes it."),
        "error" => {
            let detail = sentence(text(llm, &["error"]).unwrap_or("The clean-up model didn't start."));
            let code = code_of(llm, problems::CLEANUP_LOAD_FAILED);
            trouble("cleanup", "AI clean-up", false, code, &[("detail", &detail), ("model", &label)])
        }
        // "loading", or "off" for the moment between turning it on and the load starting
        _ => p(Level::Starting, "Loading".into(), format!("Loading {label}")),
    }
}

fn gpu_part(status: Option<&Value>) -> Part {
    let compute = status.and_then(|s| s.get("compute")).filter(|c| c.is_object());
    let p = |level, summary: String, headline: String| part("gpu", "Graphics card", false, level, summary, headline);
    let Some(compute) = compute else {
        return p(Level::Waiting, "Waiting for the engine".into(), "Graphics card: waiting for the engine".into());
    };
    if let Some(c) = engine_check(status, "driver").filter(|c| text(Some(c), &["status"]) == Some("warn")) {
        return trouble("gpu", "Graphics card", false, problems::DRIVER_TOO_OLD, &check_vars(c));
    }
    let name = text(Some(compute), &["gpu", "name"]).unwrap_or("The graphics card").to_owned();
    if text(Some(compute), &["mode"]) == Some("cpu") {
        return p(Level::Ok, "Not used".into(), "Everything runs on the processor".into())
            .because("Processor only, as set in Models.");
    }
    // An NVIDIA PC's first run: the libraries speech needs on the card are downloading (B2).
    if let Some(libs) = compute.get("cuda_libs").filter(|v| v.is_object()) {
        match text(Some(libs), &["state"]) {
            Some("downloading") => {
                let mb = |k: &str| libs.get(k).and_then(Value::as_f64).unwrap_or(0.0) / 1_048_576.0;
                return p(Level::Ok, "Getting ready".into(), format!("Getting {name} ready for speech")).because(format!(
                    "Downloading what speech needs to run on it: {:.0} of {:.0} MB. Until then speech runs on the processor.",
                    mb("done"),
                    mb("total")
                ));
            }
            Some("error") => {
                let detail = text(Some(libs), &["error"]).unwrap_or("").to_owned();
                return trouble("gpu", "Graphics card", false, problems::GPU_LIBS_DOWNLOAD_FAILED, &[("detail", &detail)]);
            }
            _ => {}
        }
    }
    if compute.get("gpu").is_none_or(Value::is_null) {
        return p(Level::Ok, "Not used".into(), "Everything runs on the processor".into())
            .because("No NVIDIA graphics card was found, so everything runs on the processor.");
    }
    // Where the placement wants speech, against where it actually runs: a mismatch that is not
    // a move in progress means the graphics card could not be used.
    let wanted = text(Some(compute), &["speech"]);
    let actual = status.and_then(|s| text(Some(s), &["stt", "device"]));
    let moving = compute.get("moving").is_some_and(|m| !m.is_null() && m != &Value::Bool(false));
    let speech_ready = status.and_then(|s| text(Some(s), &["stt", "state"])) == Some("ready");
    if speech_ready && !moving && wanted == Some("cuda") && actual == Some("cpu") {
        return trouble("gpu", "Graphics card", false, problems::GPU_UNAVAILABLE, &[]);
    }
    let level_name = text(Some(compute), &["level"]).unwrap_or("full");
    let reason = text(Some(compute), &["reason"]).map(sentence);
    match level_name {
        "full" | "gentle" => p(Level::Ok, "In use".into(), format!("{name} is in use")).because(
            if level_name == "gentle" { reason.unwrap_or_default() } else { String::new() }),
        // Moved off on purpose (heat, another app, idle): working as designed, said as a note.
        _ => p(Level::Ok, "Resting".into(), format!("{name} is resting")).because(reason.unwrap_or_default()),
    }
}

/// One of the engine's quick self-checks (selfcheck.py), as its status reports them.
fn engine_check<'v>(status: Option<&'v Value>, id: &str) -> Option<&'v Value> {
    status?.get("checks")?.as_array()?.iter().find(|c| c.get("id").and_then(Value::as_str) == Some(id))
}

/// A check's placeholders, its detail among them.
fn check_vars(check: &Value) -> Vec<(&str, &str)> {
    let mut vars: Vec<(&str, &str)> = check
        .get("vars")
        .and_then(Value::as_object)
        .map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|v| (k.as_str(), v))).collect())
        .unwrap_or_default();
    if let Some(d) = check.get("detail").and_then(Value::as_str) {
        vars.push(("detail", d));
    }
    vars
}

/// Where LocalFlow keeps things: its settings folder (the shell's check) and its models folder
/// and the free space beside it (the engine's). Nothing about dictation itself, so never
/// critical: a full disk stops the next download, not the next take.
fn storage_part(status: Option<&Value>, settings_unwritable: Option<&(String, String)>) -> Part {
    if let Some((folder, why)) = settings_unwritable {
        return trouble("storage", "Storage", false, problems::FOLDER_NOT_WRITABLE, &[("folder", folder), ("detail", why)]);
    }
    // Settings a newer LocalFlow wrote (after a downgrade): read, never saved over.
    let engine_newer = status.and_then(|s| s.get("settings_newer")).and_then(Value::as_u64);
    if crate::settings::newer_version().is_some() || engine_newer.is_some() {
        return trouble("storage", "Storage", false, problems::SETTINGS_NEWER, &[]);
    }
    for id in ["models_folder", "disk"] {
        if let Some(c) = engine_check(status, id).filter(|c| matches!(text(Some(c), &["status"]), Some("warn" | "fail"))) {
            let code = problems::from_engine(text(Some(c), &["code"]), problems::FOLDER_NOT_WRITABLE);
            return trouble("storage", "Storage", false, code, &check_vars(c));
        }
    }
    let free = engine_check(status, "disk").and_then(|c| text(Some(c), &["detail"])).unwrap_or("");
    part("storage", "Storage", false, Level::Ok, "Enough space", "LocalFlow's folders are fine").because(free.to_owned())
}

fn mic_part(mic: &Mic) -> Part {
    match (&mic.device, &mic.error) {
        (tried, Some(_)) if mic.busy => {
            let device = tried.as_deref().unwrap_or("The microphone");
            trouble("microphone", "Microphone", true, problems::MIC_IN_USE, &[("device", device)])
        }
        (_, Some(error)) => {
            let detail = sentence(error);
            trouble("microphone", "Microphone", true, problems::MIC_UNAVAILABLE, &[("detail", &detail)])
        }
        (Some(device), None) => {
            let chosen = mic.chosen.trim();
            if !chosen.is_empty() && !device.to_lowercase().contains(&chosen.to_lowercase()) {
                trouble("microphone", "Microphone", true, problems::MIC_FALLBACK, &[("device", device), ("chosen", chosen)])
            } else if mic.bluetooth {
                // Works, so not a problem - but worth knowing: the headset is in call quality.
                part("microphone", "Microphone", true, Level::Ok, device.clone(), format!("Listening with {device}"))
                    .because(problems::show(problems::MIC_BLUETOOTH, &[]).message)
            } else {
                part("microphone", "Microphone", true, Level::Ok, device.clone(), format!("Listening with {device}"))
            }
        }
        (None, None) => part("microphone", "Microphone", true, Level::Starting, "Opening", "Opening the microphone"),
    }
}

fn hotkey_part(installed: bool, chord: &str) -> Part {
    if installed {
        part("hotkey", "Hotkey", true, Level::Ok, chord.to_owned(), format!("Hold {chord} to dictate"))
    } else {
        trouble("hotkey", "Hotkey", true, problems::HOTKEY_HOOK_FAILED, &[])
    }
}

/// Engine and library messages arrive with or without a full stop and a capital; the sentences
/// built around them need both.
fn sentence(s: &str) -> String {
    let s = s.trim();
    if s.is_empty() {
        return String::new();
    }
    let mut out: String = s.chars().take(300).collect();
    if let Some(first) = out.chars().next() {
        out.replace_range(..first.len_utf8(), &first.to_uppercase().to_string());
    }
    if !out.ends_with(['.', '!', '?']) {
        out.push('.');
    }
    out
}

// ---------------------------------------------------------------------------------------------
// the flow bar

/// Why a press of the hotkey cannot start a take right now: the problem, and its words for the
/// flow bar. None when it can. The link and the speech model are read live, because the
/// assessment is only refreshed on events and a press must not be refused on a stale one.
pub fn refusal(link: &LinkState, speech_ready: bool, health: Option<&Health>) -> Option<(&'static str, String)> {
    // the code of a failed part, as the assessment found it
    let failed = |id: &str| {
        health
            .and_then(|h| h.parts.iter().find(|p| p.id == id && p.level == Level::Failed))
            .and_then(|p| p.code)
            .map(|c| problems::from_engine(Some(c), problems::UNKNOWN_MESSAGE))
    };
    let say = |code: problems::Code| Some((code.as_str(), problems::bar(code, &[])));
    match link.link {
        Link::Ready => {}
        Link::Failed if engine_is_missing(link.detail.as_deref().unwrap_or("")) => return say(problems::ENGINE_MISSING),
        Link::Failed if engine_is_blocked(link.detail.as_deref().unwrap_or("")) => return say(problems::ENGINE_BLOCKED),
        Link::Failed => return say(problems::ENGINE_WONT_START),
        _ => return say(problems::ENGINE_RESTARTING),
    }
    if !speech_ready {
        return say(failed("speech").unwrap_or(problems::SPEECH_LOADING));
    }
    if let Some(code) = failed("microphone") {
        return say(code);
    }
    None
}

/// `refusal` for the running app.
pub fn refusal_now(app: &AppHandle) -> Option<(&'static str, String)> {
    let engine = app.try_state::<Engine>()?;
    let current = app.try_state::<Arc<Monitor>>().and_then(|m| m.current.locked().clone());
    refusal(&engine.link(), engine.stt_ready(), current.as_ref())
}

/// A problem with the take being spoken: a tag on the flow bar beside the words, which go on.
pub fn take_notice(app: &AppHandle, code: problems::Code) {
    let text = problems::bar(code, &[]);
    crate::shell_log!("take notice [{}]: {text}", code.as_str());
    let _ = app.emit("take-notice", json!({ "code": code.as_str(), "text": text }));
}

fn recording(app: &AppHandle) -> bool {
    app.try_state::<Arc<crate::session::SessionManager>>()
        .is_some_and(|s| matches!(s.phase(), crate::session::Phase::Recording))
}

// ---------------------------------------------------------------------------------------------
// notifications

/// Decides which changes are announced: a part that goes wrong and stays wrong for
/// `NOTIFY_AFTER`, once per change of level, and "working normally again" once everything has
/// been fine for `RECOVERED_AFTER` after something was announced.
#[derive(Default)]
pub struct Notifier {
    told: HashMap<&'static str, Level>,
    seen: HashMap<&'static str, (Level, Instant)>,
    fine_since: Option<Instant>,
    /// A problem was announced and "working normally again" has not been said since.
    announced: bool,
}

impl Notifier {
    /// Messages (title, body) due at `now`.
    pub fn tick(&mut self, health: &Health, now: Instant) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for p in &health.parts {
            let seen = self.seen.entry(p.id).or_insert((p.level, now));
            if seen.0 != p.level {
                *seen = (p.level, now);
            }
            let held = now.duration_since(seen.1);
            if p.level.bad() {
                if held >= NOTIFY_AFTER && self.told.get(p.id) != Some(&p.level) {
                    self.told.insert(p.id, p.level);
                    let mut body = p.reason.clone().unwrap_or_default();
                    if p.action.is_some() {
                        body = format!("{body} The fix is on the Overview page in LocalFlow.").trim().to_owned();
                    }
                    out.push((p.headline.clone(), body));
                    self.announced = true;
                }
            } else if !matches!(p.level, Level::Starting | Level::Waiting) {
                self.told.remove(p.id);
            }
        }
        let fine = matches!(health.overall, Level::Ok);
        self.fine_since = match (fine, self.fine_since) {
            (true, None) => Some(now),
            (true, since) => since,
            (false, _) => None,
        };
        if self.told.is_empty() {
            if let Some(since) = self.fine_since {
                if self.announced && now.duration_since(since) >= RECOVERED_AFTER {
                    self.announced = false;
                    out.push(("LocalFlow is working normally again".into(), "Dictation is available.".into()));
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------
// the running app

struct Monitor {
    /// Why the settings folder cannot be written, from the last check (at start, and each
    /// "Check LocalFlow").
    settings_unwritable: Mutex<Option<(String, String)>>,
    current: Mutex<Option<Health>>,
    mic: Mutex<Mic>,
    notifier: Mutex<Notifier>,
    /// The end-to-end harness: keys come from a script, not a hook, and nobody is there to
    /// read a notification.
    scripted: bool,
}

/// Set up the monitor and, unless `scripted`, the thread that turns lasting changes into
/// notifications.
pub fn start(app: &AppHandle, scripted: bool) {
    app.manage(Arc::new(Monitor {
        current: Mutex::new(None),
        mic: Mutex::new(Mic::default()),
        notifier: Mutex::new(Notifier::default()),
        scripted,
        settings_unwritable: Mutex::new(crate::selfcheck::settings_folder_writable().err()),
    }));
    refresh(app);
    if scripted {
        return;
    }
    let handle = app.clone();
    crate::guard::spawn_supervised("health-notify", move || loop {
        std::thread::sleep(Duration::from_secs(1));
        let Some(monitor) = handle.try_state::<Arc<Monitor>>() else { continue };
        let Some(health) = monitor.current.locked().clone() else { continue };
        let due = monitor.notifier.locked().tick(&health, Instant::now());
        for (title, body) in due {
            crate::shell_log!("status: {title} - {body}");
            crate::notify(&handle, &title, &body);
        }
    })
    .ok();
}

/// Check the settings folder again (after "Check LocalFlow") and reassess.
pub fn recheck_settings_folder(app: &AppHandle) {
    if let Some(m) = app.try_state::<Arc<Monitor>>() {
        *m.settings_unwritable.locked() = crate::selfcheck::settings_folder_writable().err();
    }
    refresh(app);
}

/// What the microphone supervisor reported (the `audio-device` event).
pub fn set_mic(app: &AppHandle, device: Option<String>, error: Option<String>, busy: bool, bluetooth: bool) {
    if error.is_some() && recording(app) {
        // What was heard before it went is still transcribed (e2e `mic-unplugged`).
        take_notice(app, problems::MIC_LOST_MID_TAKE);
    }
    if let Some(m) = app.try_state::<Arc<Monitor>>() {
        *m.mic.locked() = Mic { device, error, chosen: String::new(), busy, bluetooth };
    }
    refresh(app);
}

/// Assess again and, if anything changed, tell the Hub and the tray.
pub fn refresh(app: &AppHandle) {
    let Some(monitor) = app.try_state::<Arc<Monitor>>() else { return };
    let Some(engine) = app.try_state::<Engine>() else { return };
    let link = engine.link();
    let status = engine.status();
    let mut mic = monitor.mic.locked().clone();
    if mic.device.is_none() && mic.error.is_none() {
        // The supervisor's first report can come before this monitor exists.
        if let Some(shell) = app.try_state::<crate::Shell>() {
            let name = shell.capture.device_name();
            mic.device = (!name.is_empty()).then_some(name);
        }
    }
    mic.chosen = crate::settings::load().microphone;
    let hook_installed = monitor.scripted || crate::hotkey::hook_installed();
    let chord = crate::hotkey::Config::load().describe();
    let settings_unwritable = monitor.settings_unwritable.locked().clone();
    let health = assess(&Inputs {
        link: &link,
        status: status.as_ref(),
        mic: &mic,
        hook_installed,
        hotkey: &chord,
        mic_blocked: crate::selfcheck::mic_blocked(),
        settings_unwritable,
    });
    let mut current = monitor.current.locked();
    if current.as_ref() == Some(&health) {
        return;
    }
    *current = Some(health.clone());
    drop(current);
    let _ = app.emit("health", &health);
    show_in_tray(app);
}

/// The tray shows the status whenever no take is running (a take shows its own colours).
pub fn show_in_tray(app: &AppHandle) {
    let idle = app
        .try_state::<Arc<crate::session::SessionManager>>()
        .is_none_or(|s| matches!(s.phase(), crate::session::Phase::Idle));
    let Some(health) = app.try_state::<Arc<Monitor>>().and_then(|m| m.current.locked().clone()) else {
        return;
    };
    if !idle {
        return;
    }
    let state = match health.overall {
        Level::Ok | Level::Off | Level::Waiting => crate::tray::State::Idle,
        Level::Starting => crate::tray::State::Loading,
        Level::Degraded => crate::tray::State::Degraded,
        Level::Failed => crate::tray::State::Error,
    };
    let detail = if health.overall == Level::Ok { String::new() } else { health.headline.clone() };
    crate::tray::set_state(app, state, Some(&detail));
}

pub fn current(app: &AppHandle) -> Value {
    app.try_state::<Arc<Monitor>>()
        .and_then(|m| m.current.locked().clone())
        .map(|h| serde_json::to_value(h).unwrap_or(Value::Null))
        .unwrap_or_else(|| json!(null))
}

#[cfg(test)]
#[path = "health_tests.rs"]
mod tests;
