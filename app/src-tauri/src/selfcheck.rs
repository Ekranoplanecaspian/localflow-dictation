//! "Check LocalFlow": the shell's checks, and the engine's (`engine/src/localflow/selfcheck.py`)
//! asked for over the connection, as one list the Hub shows with a fix beside each problem.
//!
//! The shell sees what the engine cannot: whether Windows lets desktop apps use the microphone
//! (when it does not, the stream opens and delivers silence, which looked like a quiet room),
//! whether the microphone sends anything at all, the settings folder, and WebView2. The quick
//! ones also feed the status model; the full list runs when the Hub asks.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use tauri::{AppHandle, Manager};
use tokio::sync::oneshot;

use crate::engine::Engine;
use crate::guard::LockExt;
use crate::problems::{self, Code};

/// How long a full check may take: it hashes gigabytes of model files.
const ENGINE_TIMEOUT: Duration = Duration::from_secs(180);
/// How long the microphone is listened to for any sign of sound.
const LISTEN: Duration = Duration::from_millis(1500);

const CONSENT_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone";

/// One check as the Hub shows it.
#[derive(Debug, Clone)]
pub struct Item {
    pub id: String,
    pub name: String,
    /// ok | warn | fail | skip
    pub status: &'static str,
    /// What was found, as a sentence.
    pub detail: String,
    pub code: Option<Code>,
    pub vars: Vec<(String, String)>,
}

impl Item {
    fn ok(id: &str, name: &str, detail: impl Into<String>) -> Item {
        Item { id: id.to_owned(), name: name.to_owned(), status: "ok", detail: detail.into(), code: None, vars: Vec::new() }
    }

    fn problem(id: &str, name: &str, status: &'static str, code: Code, detail: impl Into<String>) -> Item {
        Item { id: id.to_owned(), name: name.to_owned(), status, detail: detail.into(), code: Some(code), vars: Vec::new() }
    }

    fn with(mut self, key: &str, value: impl Into<String>) -> Item {
        self.vars.push((key.to_owned(), value.into()));
        self
    }

    /// With the catalogue's words for its problem, filled in.
    pub fn to_json(&self) -> Value {
        let mut out = json!({"id": self.id, "name": self.name, "status": self.status, "detail": self.detail});
        if let Some(code) = self.code {
            let mut vars: Vec<(&str, &str)> = self.vars.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
            vars.push(("detail", self.detail.as_str()));
            let shown = problems::show(code, &vars);
            out["code"] = json!(shown.code);
            out["title"] = json!(shown.title);
            out["message"] = json!(shown.message);
            if let Some((id, label)) = shown.action {
                out["action"] = json!({"id": id, "label": label});
            }
        }
        out
    }
}

// --- the microphone -------------------------------------------------------------------------

/// Whether Windows privacy settings stop desktop apps from using the microphone: the switch
/// for every app, or the one for desktop apps, in the user's settings or set for the machine.
pub fn mic_blocked() -> bool {
    use windows::Win32::System::Registry::{HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};
    let deny = |hive, key: &str| crate::win::reg_string(hive, key, "Value").is_some_and(|v| v.eq_ignore_ascii_case("Deny"));
    let desktop = format!(r"{CONSENT_KEY}\NonPackaged");
    deny(HKEY_CURRENT_USER, CONSENT_KEY)
        || deny(HKEY_CURRENT_USER, &desktop)
        || deny(HKEY_LOCAL_MACHINE, CONSENT_KEY)
        || deny(HKEY_LOCAL_MACHINE, &desktop)
}

fn check_mic_privacy() -> Item {
    if mic_blocked() {
        Item::problem("mic_privacy", "Microphone permission", "fail", problems::MIC_BLOCKED,
                      "Microphone access for desktop apps is off.")
    } else {
        Item::ok("mic_privacy", "Microphone permission", "Desktop apps may use the microphone.")
    }
}

/// Listen for a moment: a microphone that sends only zeros is muted, switched off, or blocked.
async fn check_mic_signal(app: &AppHandle) -> Item {
    let Some(shell) = app.try_state::<crate::Shell>() else {
        return Item { status: "skip", ..Item::ok("mic_signal", "Microphone sound", "Not available.") };
    };
    let device = shell.capture.device_name();
    if device.is_empty() {
        return Item::problem("mic_signal", "Microphone sound", "fail", problems::MIC_UNAVAILABLE,
                             "No microphone stream is open.");
    }
    let start = shell.capture.produced();
    let mut loudest = 0.0f32;
    let until = Instant::now() + LISTEN;
    while Instant::now() < until {
        loudest = loudest.max(shell.capture.level());
        tokio::time::sleep(Duration::from_millis(40)).await;
    }
    if shell.capture.produced() == start {
        return Item::problem("mic_signal", "Microphone sound", "fail", problems::MIC_UNAVAILABLE,
                             format!("{device} sent no audio at all while LocalFlow listened."));
    }
    if loudest <= 0.0 {
        return Item::problem("mic_signal", "Microphone sound", "warn", problems::MIC_SILENT,
                             "Every sample was zero.").with("device", device);
    }
    Item::ok("mic_signal", "Microphone sound", format!("{device} is sending sound."))
}

// --- files and the web view ---------------------------------------------------------------

/// Can the settings folder be written? `Err` holds why not.
pub fn settings_folder_writable() -> Result<String, (String, String)> {
    let Some(dir) = crate::paths::config_dir() else {
        return Err(("the settings folder".into(), "Windows gives this process no user folder.".into()));
    };
    let shown = dir.display().to_string();
    let probe = dir.join(format!(".write-test-{}", std::process::id()));
    let result = std::fs::create_dir_all(&dir)
        .and_then(|_| std::fs::write(&probe, b"ok"))
        .and_then(|_| std::fs::remove_file(&probe));
    match result {
        Ok(()) => Ok(shown),
        Err(e) => Err((shown, format!("{e}."))),
    }
}

fn check_settings_folder() -> Item {
    match settings_folder_writable() {
        Ok(dir) => Item::ok("settings_folder", "Settings folder", format!("{dir} can be written.")),
        Err((dir, why)) => Item::problem("settings_folder", "Settings folder", "fail", problems::FOLDER_NOT_WRITABLE, why)
            .with("folder", dir),
    }
}

fn check_webview() -> Item {
    match tauri::webview_version() {
        Ok(v) => Item::ok("webview", "WebView2", format!("Version {v}.")),
        // The window this list is shown in is WebView2, so this only fails oddly.
        Err(e) => Item { status: "skip", ..Item::ok("webview", "WebView2", format!("Its version is unknown ({e}).")) },
    }
}

// --- the engine's checks --------------------------------------------------------------------

static WAITERS: Mutex<Option<HashMap<String, oneshot::Sender<Value>>>> = Mutex::new(None);

/// A `selfcheck.result` or `selfcheck.repaired` from the engine, for whoever asked.
pub fn deliver(payload: &Value) {
    let Some(id) = payload.get("id").and_then(Value::as_str) else { return };
    if let Some(tx) = WAITERS.locked().get_or_insert_with(HashMap::new).remove(id) {
        let _ = tx.send(payload.clone());
    }
}

async fn ask_engine(app: &AppHandle, msg: Value, timeout: Duration) -> Option<Value> {
    let engine = app.try_state::<Engine>()?;
    if !engine.is_connected() {
        return None;
    }
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let id = format!("check{}", NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    let (tx, rx) = oneshot::channel();
    WAITERS.locked().get_or_insert_with(HashMap::new).insert(id.clone(), tx);
    let mut msg = msg;
    msg["id"] = json!(id);
    engine.send(msg);
    let answer = tokio::time::timeout(timeout, rx).await.ok().and_then(Result::ok);
    WAITERS.locked().get_or_insert_with(HashMap::new).remove(&id);
    answer
}

/// An engine check, from its JSON, as an `Item`.
fn engine_item(c: &Value) -> Item {
    let s = |k: &str| c.get(k).and_then(Value::as_str).unwrap_or("").to_owned();
    let status = match s("status").as_str() {
        "ok" => "ok",
        "warn" => "warn",
        "fail" => "fail",
        _ => "skip",
    };
    let code = c.get("code").and_then(Value::as_str).map(|code| problems::from_engine(Some(code), problems::UNKNOWN_MESSAGE));
    let vars = c
        .get("vars")
        .and_then(Value::as_object)
        .map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_owned()))).collect())
        .unwrap_or_default();
    Item { id: s("id"), name: s("name"), status, detail: s("detail"), code, vars }
}

/// Every check, the slow ones only when `full`. The engine's may be missing: a stopped
/// engine is already the Status card's first row.
pub async fn run(app: &AppHandle, full: bool) -> Vec<Value> {
    let mut items = vec![check_mic_privacy()];
    if full {
        items.push(check_mic_signal(app).await);
    }
    items.push(check_settings_folder());
    items.push(check_webview());
    let engine_checks = ask_engine(app, json!({"type": "selfcheck.run", "full": full}), ENGINE_TIMEOUT).await;
    match engine_checks.as_ref().and_then(|a| a.get("checks")).and_then(Value::as_array) {
        Some(checks) => items.extend(checks.iter().map(engine_item)),
        None => items.push(Item {
            status: "skip",
            ..Item::ok("engine_checks", "Engine checks", "The engine isn't running, so its checks couldn't run.")
        }),
    }
    let log: Vec<String> = items.iter().filter(|i| i.status != "ok").map(|i| format!("{}={}", i.id, i.status)).collect();
    crate::shell_log!("self-check ({}): {}", if full { "full" } else { "quick" },
                      if log.is_empty() { "all ok".to_owned() } else { log.join(", ") });
    items.iter().map(Item::to_json).collect()
}

/// "Download again": the engine deletes the damaged model files its last full check found,
/// then restarts, and fetches them afresh as it loads. The number of files removed.
pub async fn repair(app: &AppHandle) -> Result<usize, String> {
    let answer = ask_engine(app, json!({"type": "selfcheck.repair"}), Duration::from_secs(30))
        .await
        .ok_or("the engine isn't running")?;
    let removed = answer.get("removed").and_then(Value::as_array).map(Vec::len).unwrap_or(0);
    crate::shell_log!("self-check repair: {removed} damaged file(s) removed; restarting the engine");
    if let Some(engine) = app.try_state::<Engine>() {
        engine.restart();
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_engine_check_is_worded_by_the_catalogue() {
        let c = json!({"id": "disk", "name": "Free disk space", "status": "warn", "detail": "1.2 GB free on C:.",
                       "code": "disk-low", "vars": {"free": "1.2 GB", "drive": "C:", "needed": "5 GB"}});
        let shown = engine_item(&c).to_json();
        assert_eq!(shown["title"], "The disk is nearly full");
        assert_eq!(
            shown["message"],
            "There is 1.2 GB free on C:; downloading or switching a model needs about 5 GB. Dictation with the models already there is unaffected."
        );
        let ok = engine_item(&json!({"id": "driver", "name": "NVIDIA driver", "status": "ok", "detail": "Version 610.62."})).to_json();
        assert!(ok.get("title").is_none() && ok["detail"] == "Version 610.62.");
    }

    #[test]
    fn an_unknown_status_is_a_skip_and_an_unknown_code_is_not_trusted() {
        let item = engine_item(&json!({"id": "x", "name": "X", "status": "weird", "code": "made-up"}));
        assert_eq!(item.status, "skip");
        assert_eq!(item.code, Some(problems::UNKNOWN_MESSAGE));
    }

    #[test]
    fn a_damaged_model_offers_download_again() {
        let c = json!({"id": "speech_files", "name": "Speech model files", "status": "fail",
                       "detail": "Damaged: encoder-model.onnx.", "code": "speech-files-damaged",
                       "vars": {"model": "Parakeet v3"}});
        let shown = engine_item(&c).to_json();
        assert_eq!(shown["action"]["id"], "redownload_models");
        assert!(shown["message"].as_str().unwrap().starts_with("Parakeet v3 couldn't be read: Damaged: encoder-model.onnx."));
    }

    #[test]
    fn this_machine_answers_the_privacy_question_and_the_settings_folder_is_writable() {
        let _ = mic_blocked(); // reads the registry without panicking, whatever it says
        assert!(settings_folder_writable().is_ok());
    }
}
