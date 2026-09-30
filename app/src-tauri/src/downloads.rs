//! Downloads, where they are seen without opening LocalFlow (v0.2.1, M3): the tray icon's ring
//! fills as a download goes and its tooltip says what and how long; a notification says when a
//! model has arrived, and when AI clean-up - having downloaded its model - is ready to use.
//!
//! The engine lists every download in its status (`downloads`); this follows the list from one
//! status to the next. What to show and what to say is worked out in `Watch::update`, apart from
//! the tray and the toasts, so it can be tested.

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::Value;
use tauri::AppHandle;

use crate::guard::LockExt;

/// The download the tray shows: the one running, else the next in line.
#[derive(Debug, Clone, PartialEq)]
pub struct Showing {
    pub label: String,
    /// 0-1, or None while it waits its turn
    pub progress: Option<f64>,
    pub eta_s: Option<f64>,
    /// how many more are waiting behind it
    pub more: usize,
}

impl Showing {
    /// "Downloading Parakeet v2: 42 %, about 2 min left (1 more waiting)"
    pub fn tooltip(&self) -> String {
        let mut line = match self.progress {
            Some(p) => format!("Downloading {}: {:.0} %", self.label, p * 100.0),
            None => format!("Waiting to download {}", self.label),
        };
        if let Some(eta) = self.eta_s {
            line.push_str(&format!(", {}", eta_words(eta)));
        }
        if self.more > 0 {
            line.push_str(&format!(" ({} more waiting)", self.more));
        }
        line
    }
}

pub fn eta_words(seconds: f64) -> String {
    let min = (seconds / 60.0).round() as u64;
    if seconds < 60.0 {
        "less than a minute left".into()
    } else if min < 60 {
        format!("about {min} min left")
    } else {
        format!("about {} h {} min left", min / 60, min % 60)
    }
}

#[derive(Default)]
pub struct Watch {
    /// downloads queued or running at the last status: id -> (kind, label, reason)
    active: HashMap<String, (String, String, String)>,
    /// clean-up's model or runtime downloaded while LocalFlow ran: say so when it is ready
    cleanup_arrived: bool,
    /// Setup was finished before speech was ready ("Finish - tell me when it's ready", M5):
    /// the hotkey to name in the one notification that says it is.
    tell_ready: Option<String>,
}

impl Watch {
    /// What the tray should show, and the notifications (title, body) this status calls for.
    pub fn update(&mut self, status: &Value) -> (Option<Showing>, Vec<(String, String)>) {
        let jobs = status.get("downloads").and_then(Value::as_array).cloned().unwrap_or_default();
        let field = |j: &Value, k: &str| j.get(k).and_then(Value::as_str).unwrap_or("").to_owned();
        let mut said = Vec::new();
        let mut still = HashMap::new();
        for j in &jobs {
            let (id, state) = (field(j, "id"), field(j, "state"));
            if state == "queued" || state == "downloading" {
                still.insert(id, (field(j, "kind"), field(j, "label"), field(j, "reason")));
            } else if let Some((kind, label, reason)) = self.active.get(&id) {
                if state == "done" {
                    // "LocalFlow is ready" follows for someone waiting on it: not two in a row
                    let waiting = self.tell_ready.is_some() && kind == "speech" && reason == "first-run";
                    if let Some(note) = arrived(kind, label, reason).filter(|_| !waiting) {
                        said.push(note);
                    }
                    if kind == "cleanup" || kind == "runtime" {
                        self.cleanup_arrived = true;
                    }
                }
            }
        }
        self.active = still;
        let speech_ready = status.get("stt").and_then(|s| s.get("state")).and_then(Value::as_str) == Some("ready");
        if speech_ready {
            if let Some(chord) = self.tell_ready.take() {
                said.push((
                    "LocalFlow is ready".into(),
                    format!("Hold {chord} and speak: your words appear wherever your cursor is."),
                ));
            }
        }
        let llm = status.get("llm");
        let ready = llm.and_then(|l| l.get("state")).and_then(Value::as_str) == Some("ready");
        if self.cleanup_arrived && ready && !self.active.values().any(|(k, ..)| k == "cleanup" || k == "runtime") {
            self.cleanup_arrived = false;
            let label = llm.and_then(|l| l.get("label")).and_then(Value::as_str).unwrap_or("The clean-up model");
            said.push((
                "Auto-edits are ready".into(),
                format!("{label} now tidies what you dictate: fillers out, your corrections applied."),
            ));
        }
        let running = jobs.iter().find(|j| field(j, "state") == "downloading");
        let waiting = jobs.iter().filter(|j| field(j, "state") == "queued").count();
        let first = running.or_else(|| jobs.iter().find(|j| field(j, "state") == "queued"));
        let showing = first.map(|j| Showing {
            label: field(j, "label"),
            progress: running.map(|r| r.get("progress").and_then(Value::as_f64).unwrap_or(0.0)),
            eta_s: j.get("eta_s").and_then(Value::as_f64),
            more: if running.is_some() { waiting } else { waiting.saturating_sub(1) },
        });
        (showing, said)
    }
}

/// The notification for a finished download, if it deserves one. LocalFlow's own parts (the
/// clean-up runtime, the graphics card's libraries) arrive without a word; clean-up's own model
/// is announced once it is ready to use, not when its file is.
fn arrived(kind: &str, label: &str, reason: &str) -> Option<(String, String)> {
    match (kind, reason) {
        ("speech", "first-run") => Some((
            format!("{label} is downloaded"),
            "It is loading now: you can dictate in a few seconds.".into(),
        )),
        ("speech", "switch") | ("cleanup", "switch") => {
            Some((format!("{label} is downloaded"), "LocalFlow is switching to it now.".into()))
        }
        ("speech", _) | ("cleanup", "library") => Some((
            format!("{label} is downloaded"),
            "It is ready in Models whenever you want to use it.".into(),
        )),
        _ => None,
    }
}

static WATCH: Mutex<Option<Watch>> = Mutex::new(None);

/// Setup finished before speech was ready: one notification when it is, naming `chord`.
pub fn tell_when_ready(chord: String) {
    WATCH.locked().get_or_insert_with(Watch::default).tell_ready = Some(chord);
}

/// A new engine status: update the tray, and say what arrived.
pub fn on_status(app: &AppHandle, status: &Value) {
    let (showing, said) = {
        let mut slot = WATCH.locked();
        slot.get_or_insert_with(Watch::default).update(status)
    };
    crate::tray::set_download(app, showing);
    for (title, body) in said {
        crate::shell_log!("notification: {title}");
        crate::notify(app, &title, &body);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn job(id: &str, kind: &str, label: &str, state: &str, reason: &str, progress: f64) -> Value {
        json!({"id": id, "kind": kind, "label": label, "state": state, "reason": reason,
               "progress": progress, "eta_s": if state == "downloading" { json!(130) } else { Value::Null }})
    }

    fn status(jobs: Vec<Value>, llm_state: &str) -> Value {
        json!({"downloads": jobs, "llm": {"state": llm_state, "label": "Qwen3 4B"}})
    }

    #[test]
    fn the_tray_shows_the_running_download_and_how_many_wait() {
        let mut w = Watch::default();
        let (showing, said) = w.update(&status(
            vec![
                job("d2", "speech", "Parakeet v2", "downloading", "switch", 0.42),
                job("d3", "cleanup", "Gemma 4 E2B", "queued", "library", 0.0),
            ],
            "ready",
        ));
        let s = showing.unwrap();
        assert_eq!(s.tooltip(), "Downloading Parakeet v2: 42 %, about 2 min left (1 more waiting)");
        assert!(said.is_empty());
        let (showing, _) = w.update(&status(vec![job("d3", "cleanup", "Gemma 4 E2B", "queued", "library", 0.0)], "ready"));
        assert_eq!(showing.unwrap().tooltip(), "Waiting to download Gemma 4 E2B");
        assert_eq!(w.update(&status(vec![], "ready")).0, None);
    }

    #[test]
    fn a_finished_model_is_announced_once_and_localflows_own_parts_are_not() {
        let mut w = Watch::default();
        w.update(&status(
            vec![
                job("d1", "gpu-libs", "Graphics card libraries", "downloading", "automatic", 0.5),
                job("d2", "speech", "Whisper Large v3 Turbo", "queued", "library", 0.0),
            ],
            "ready",
        ));
        let (_, said) = w.update(&status(
            vec![
                job("d2", "speech", "Whisper Large v3 Turbo", "done", "library", 1.0),
                job("d1", "gpu-libs", "Graphics card libraries", "done", "automatic", 1.0),
            ],
            "ready",
        ));
        assert_eq!(said, vec![("Whisper Large v3 Turbo is downloaded".to_owned(),
                               "It is ready in Models whenever you want to use it.".to_owned())]);
        let (_, again) = w.update(&status(vec![job("d2", "speech", "Whisper Large v3 Turbo", "done", "library", 1.0)], "ready"));
        assert!(again.is_empty(), "a finished download stays in the list for a while: said once");
    }

    #[test]
    fn clean_up_is_announced_when_it_is_ready_after_downloading_not_when_its_file_is() {
        let mut w = Watch::default();
        w.update(&status(vec![job("d1", "cleanup", "Qwen3 4B", "downloading", "first-run", 0.9)], "loading"));
        let (_, said) = w.update(&status(vec![job("d1", "cleanup", "Qwen3 4B", "done", "first-run", 1.0)], "loading"));
        assert!(said.is_empty(), "downloaded, not yet loaded");
        let (_, said) = w.update(&status(vec![job("d1", "cleanup", "Qwen3 4B", "done", "first-run", 1.0)], "ready"));
        assert_eq!(said.len(), 1);
        assert_eq!(said[0].0, "Auto-edits are ready");
        assert!(said[0].1.starts_with("Qwen3 4B now tidies what you dictate"));
        assert!(w.update(&status(vec![], "ready")).1.is_empty(), "once");
    }

    #[test]
    fn setup_finished_early_hears_once_that_localflow_is_ready() {
        let mut w = Watch { tell_ready: Some("Ctrl + Win".into()), ..Watch::default() };
        let loading = |jobs| json!({"downloads": jobs, "stt": {"state": "loading"}});
        w.update(&loading(vec![job("d1", "speech", "Parakeet v3", "downloading", "first-run", 0.9)]));
        let (_, said) = w.update(&loading(vec![job("d1", "speech", "Parakeet v3", "done", "first-run", 1.0)]));
        assert!(said.is_empty(), "not \"downloaded\" and then \"ready\" a few seconds apart");
        let (_, said) = w.update(&json!({"downloads": [], "stt": {"state": "ready"}}));
        assert_eq!(said, vec![("LocalFlow is ready".to_owned(),
                               "Hold Ctrl + Win and speak: your words appear wherever your cursor is.".to_owned())]);
        assert!(w.update(&json!({"stt": {"state": "ready"}})).1.is_empty(), "once");
    }

    #[test]
    fn a_clean_up_model_that_was_already_here_is_not_announced() {
        let mut w = Watch::default();
        assert!(w.update(&status(vec![], "loading")).1.is_empty());
        assert!(w.update(&status(vec![], "ready")).1.is_empty());
    }
}
