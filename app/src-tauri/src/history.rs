//! What you dictated, kept locally so the Hub can show it back to you.
//!
//! One JSON object per line in `%APPDATA%\LocalFlow\history.jsonl`. A line-per-entry file is
//! the right shape here: appending is a single write that cannot corrupt what came before, the
//! file is readable without this app, and deleting history is deleting a file rather than
//! trusting a database to forget.
//!
//! This is the most private thing LocalFlow stores, so it is easy to turn off and easy to
//! erase: `retention_days` prunes on load, and clearing removes the file outright.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Entries older than this are dropped when the file is next read. 0 means keep everything.
const DEFAULT_RETENTION_DAYS: u64 = 90;
/// A dictation this long is almost certainly a mistake, and storing it helps nobody.
const MAX_TEXT: usize = 8000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    /// Unix seconds.
    pub at: u64,
    /// What the speech model heard, before clean-up.
    pub raw: String,
    /// What was actually typed.
    pub text: String,
    /// Executable that received it.
    pub app: String,
    pub words: u32,
    /// Seconds of audio.
    pub audio_s: f64,
    /// Key release to final text.
    pub ms: u64,
    pub used_llm: bool,
}

fn path() -> Option<PathBuf> {
    let dir = crate::paths::config_dir()?;
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("history.jsonl"))
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Serialises appends so two dictations finishing together cannot interleave a line.
static WRITE_LOCK: Mutex<()> = Mutex::new(());

/// Record one dictation. Does nothing when history is switched off.
pub fn record(msg: &Value, app: &str, enabled: bool) {
    if !enabled {
        return;
    }
    let text = msg.get("text").and_then(Value::as_str).unwrap_or("");
    if text.trim().is_empty() {
        return;
    }
    let timings = msg.get("timings");
    let num = |k: &str| timings.and_then(|t| t.get(k)).and_then(Value::as_f64).unwrap_or(0.0);
    let entry = Entry {
        at: now(),
        raw: clip(msg.get("raw").and_then(Value::as_str).unwrap_or(text)),
        text: clip(text),
        app: app.to_owned(),
        words: text.split_whitespace().count() as u32,
        audio_s: num("audio_s"),
        ms: num("release_to_final_ms") as u64,
        used_llm: timings
            .and_then(|t| t.get("used_llm"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
    };

    let Some(path) = path() else { return };
    let Ok(line) = serde_json::to_string(&entry) else { return };
    let _guard = WRITE_LOCK.lock();
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(file, "{line}");
    }
}

fn clip(s: &str) -> String {
    if s.chars().count() <= MAX_TEXT {
        s.to_owned()
    } else {
        s.chars().take(MAX_TEXT).collect()
    }
}

/// Every entry still within the retention window, newest first.
pub fn load(retention_days: u64) -> Vec<Entry> {
    let Some(path) = path() else { return Vec::new() };
    let Ok(text) = std::fs::read_to_string(&path) else { return Vec::new() };
    let cutoff = if retention_days == 0 {
        0
    } else {
        now().saturating_sub(retention_days * 86_400)
    };
    let mut entries: Vec<Entry> = text
        .lines()
        .filter_map(|line| serde_json::from_str::<Entry>(line).ok())
        .filter(|e| e.at >= cutoff)
        .collect();
    entries.reverse();
    entries
}

/// Drop anything past the retention window by rewriting the file.
pub fn prune(retention_days: u64) {
    if retention_days == 0 {
        return;
    }
    let Some(path) = path() else { return };
    let kept = load(retention_days);
    let _guard = WRITE_LOCK.lock();
    let mut out = String::new();
    // `load` returns newest first; the file stays oldest first so appending is still correct.
    for entry in kept.iter().rev() {
        if let Ok(line) = serde_json::to_string(entry) {
            out.push_str(&line);
            out.push('\n');
        }
    }
    let _ = std::fs::write(&path, out);
}

pub fn clear() -> bool {
    let Some(path) = path() else { return false };
    let _guard = WRITE_LOCK.lock();
    !path.exists() || std::fs::remove_file(&path).is_ok()
}

pub fn location() -> String {
    path().map(|p| p.display().to_string()).unwrap_or_default()
}

#[derive(Debug, Default, Serialize)]
pub struct Stats {
    pub dictations: u64,
    pub words: u64,
    pub audio_s: f64,
    /// Median release-to-final, which is the number the roadmap tracks.
    pub p50_ms: u64,
    pub p95_ms: u64,
    /// Rough words per minute while actually speaking, for the "is this faster than typing"
    /// question. Deliberately not called "time saved": that would need a typing speed we do
    /// not know.
    pub words_per_minute: u32,
    pub with_auto_edits: u64,
    /// Dictations per day, newest last, for a sparkline.
    pub daily: Vec<u64>,
    pub top_apps: Vec<(String, u64)>,
}

pub fn stats(entries: &[Entry]) -> Stats {
    if entries.is_empty() {
        return Stats::default();
    }
    let mut s = Stats { dictations: entries.len() as u64, ..Default::default() };
    let mut latencies: Vec<u64> = Vec::with_capacity(entries.len());
    let mut per_app: HashMap<String, u64> = HashMap::new();
    for e in entries {
        s.words += e.words as u64;
        s.audio_s += e.audio_s;
        if e.used_llm {
            s.with_auto_edits += 1;
        }
        if e.ms > 0 {
            latencies.push(e.ms);
        }
        if !e.app.is_empty() {
            *per_app.entry(e.app.clone()).or_default() += 1;
        }
    }
    latencies.sort_unstable();
    if !latencies.is_empty() {
        s.p50_ms = latencies[latencies.len() / 2];
        s.p95_ms = latencies[(latencies.len() * 95 / 100).min(latencies.len() - 1)];
    }
    if s.audio_s > 1.0 {
        s.words_per_minute = (s.words as f64 / (s.audio_s / 60.0)) as u32;
    }

    // Fourteen days of counts, oldest first, with empty days included so the shape is honest.
    let today = now() / 86_400;
    let mut days = vec![0u64; 14];
    for e in entries {
        let day = e.at / 86_400;
        if today >= day && today - day < 14 {
            let idx = 13 - (today - day) as usize;
            days[idx] += 1;
        }
    }
    s.daily = days;

    let mut apps: Vec<(String, u64)> = per_app.into_iter().collect();
    apps.sort_by(|a, b| b.1.cmp(&a.1));
    apps.truncate(6);
    s.top_apps = apps;
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(at: u64, words: u32, ms: u64, app: &str, llm: bool) -> Entry {
        Entry {
            at,
            raw: "raw".into(),
            text: "some words here".into(),
            app: app.into(),
            words,
            audio_s: 3.0,
            ms,
            used_llm: llm,
        }
    }

    #[test]
    fn summarises_a_run_of_dictations() {
        let now = now();
        let entries = vec![
            entry(now, 10, 100, "code.exe", true),
            entry(now, 20, 300, "code.exe", false),
            entry(now, 30, 200, "slack.exe", true),
        ];
        let s = stats(&entries);
        assert_eq!(s.dictations, 3);
        assert_eq!(s.words, 60);
        assert_eq!(s.with_auto_edits, 2);
        assert_eq!(s.p50_ms, 200, "the median of 100/200/300");
        assert_eq!(s.top_apps[0], ("code.exe".to_owned(), 2));
        assert_eq!(s.daily.len(), 14);
        assert_eq!(s.daily[13], 3, "all three were today");
    }

    /// Latency is only recorded for entries that have it; a zero would drag the median down
    /// and make the app look faster than it is.
    #[test]
    fn ignores_missing_latencies_rather_than_counting_them_as_zero() {
        let now = now();
        let entries = vec![
            entry(now, 5, 0, "a.exe", false),
            entry(now, 5, 400, "a.exe", false),
            entry(now, 5, 500, "a.exe", false),
        ];
        assert_eq!(stats(&entries).p50_ms, 500);
    }

    #[test]
    fn empty_history_does_not_panic() {
        let s = stats(&[]);
        assert_eq!(s.dictations, 0);
        assert_eq!(s.p50_ms, 0);
    }
}
