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
pub fn record(msg: &Value, app: &str, enabled: bool, retention_days: u64) {
    if !enabled {
        return;
    }
    // An app left running for weeks still sheds old entries, not only one just started.
    prune_if_due(retention_days);
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

fn cutoff(retention_days: u64) -> u64 {
    if retention_days == 0 {
        0
    } else {
        now().saturating_sub(retention_days * 86_400)
    }
}

/// The entries in `text` at or after `cutoff`, oldest first, as the file keeps them.
fn parse_since(text: &str, cutoff: u64) -> Vec<Entry> {
    text.lines()
        .filter_map(|line| serde_json::from_str::<Entry>(line).ok())
        .filter(|e| e.at >= cutoff)
        .collect()
}

/// Every entry still within the retention window, newest first.
pub fn load(retention_days: u64) -> Vec<Entry> {
    let Some(path) = path() else { return Vec::new() };
    let Ok(text) = std::fs::read_to_string(&path) else { return Vec::new() };
    let mut entries = parse_since(&text, cutoff(retention_days));
    entries.reverse();
    entries
}

/// Delete everything past the retention window from the file itself.
///
/// Reading used to hide old entries while the file kept them for good: pruning only ran when
/// the retention setting was changed, so a dictation from six months ago was still on disk in
/// plain text under a 90-day setting. It now runs at start-up and once a day (`prune_if_due`).
pub fn prune(retention_days: u64) {
    if let Some(path) = path() {
        if let Err(e) = prune_file(&path, cutoff(retention_days)) {
            crate::shell_log!("could not prune the history: {e}");
        }
    }
}

/// Rewrite `path` without the entries older than `cutoff`. Leaves the file untouched when
/// nothing is due to go. Unreadable lines go too: nothing can show them.
fn prune_file(path: &std::path::Path, cutoff: u64) -> std::io::Result<()> {
    if cutoff == 0 {
        return Ok(()); // keep everything
    }
    // Read under the lock as well as write: a dictation recorded in between would otherwise
    // be erased by the rewrite.
    let _guard = WRITE_LOCK.lock();
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    let kept = parse_since(&text, cutoff);
    if kept.len() == text.lines().filter(|l| !l.trim().is_empty()).count() {
        return Ok(());
    }
    let mut out = String::new();
    for entry in &kept {
        if let Ok(line) = serde_json::to_string(entry) {
            out.push_str(&line);
            out.push('\n');
        }
    }
    // Written aside and swapped in, so a crash part-way loses nothing.
    let tmp = path.with_extension("jsonl.tmp");
    std::fs::write(&tmp, out)?;
    std::fs::rename(&tmp, path)
}

/// The day (since the epoch) this process last pruned the history.
static PRUNED_ON: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Prune at most once a day, on a thread of its own: it rewrites the file, and callers include
/// the path that delivers a dictation.
pub fn prune_if_due(retention_days: u64) {
    if retention_days == 0 {
        return;
    }
    let today = now() / 86_400;
    if PRUNED_ON.swap(today, std::sync::atomic::Ordering::Relaxed) == today {
        return;
    }
    let _ = std::thread::Builder::new()
        .name("history-prune".into())
        .spawn(move || prune(retention_days));
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

    fn history_file(name: &str, entries: &[Entry], extra: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("localflow-{name}-{}.jsonl", std::process::id()));
        let mut text: String =
            entries.iter().map(|e| serde_json::to_string(e).unwrap() + "\n").collect();
        text.push_str(extra);
        std::fs::write(&path, text).unwrap();
        path
    }

    /// The bug: entries past the retention window were hidden but never deleted from disk.
    #[test]
    fn pruning_deletes_old_entries_from_the_file_itself() {
        let now = now();
        let old = entry(now - 200 * 86_400, 3, 100, "slack.exe", false);
        let recent = entry(now - 86_400, 5, 200, "code.exe", true);
        let path = history_file("prune", &[old, recent], "not json at all\n");

        prune_file(&path, now - 90 * 86_400).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let left = parse_since(&text, 0);
        assert_eq!(left.len(), 1, "only the recent entry is left: {text:?}");
        assert_eq!(left[0].app, "code.exe");
        assert_eq!(text.lines().count(), 1, "the unreadable line is gone too");
    }

    #[test]
    fn nothing_to_prune_leaves_the_file_alone_and_zero_keeps_everything() {
        let now = now();
        let path = history_file("keep", &[entry(now - 400 * 86_400, 1, 1, "a.exe", false)], "");
        let before = std::fs::metadata(&path).unwrap().modified().unwrap();
        prune_file(&path, 0).unwrap(); // retention 0: keep everything
        prune_file(&path, now - 500 * 86_400).unwrap(); // nothing that old
        let after = std::fs::metadata(&path).unwrap().modified().unwrap();
        let left = parse_since(&std::fs::read_to_string(&path).unwrap(), 0).len();
        let _ = std::fs::remove_file(&path);
        assert_eq!(left, 1);
        assert_eq!(before, after, "not rewritten when there is nothing to remove");
    }

    #[test]
    fn pruning_a_history_that_does_not_exist_is_fine() {
        let missing = std::env::temp_dir().join("localflow-no-such-history.jsonl");
        assert!(prune_file(&missing, now()).is_ok());
    }

    #[test]
    fn empty_history_does_not_panic() {
        let s = stats(&[]);
        assert_eq!(s.dictations, 0);
        assert_eq!(s.p50_ms, 0);
    }
}
