//! The problem catalogue (`shared/problems.json`): every failure LocalFlow knows how to detect,
//! with its code, what the user is told and what fixes it. Compiled in, so the shell can never
//! run with a catalogue that does not match it.
//!
//! Every message the status model, the flow bar and the notifications show for a problem comes
//! from here, filled in with the specifics (`{detail}`, `{model}`...). The engine sends codes
//! only. Tests on both sides check that every code they use has an entry.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Problem {
    pub code: String,
    pub level: String,
    /// engine, shell or both; read by the tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub source: String,
    /// how the problem is noticed; read by the tests (and E7's troubleshooting pages)
    #[cfg_attr(not(test), allow(dead_code))]
    pub detect: String,
    pub title: String,
    pub message: String,
    /// A word or two for the Status card's row, for problems a part can be in.
    pub summary: Option<String>,
    /// A few words for the flow bar, for problems that stop or touch a take.
    pub bar: Option<String>,
    pub action: Option<Fix>,
}

#[derive(Debug, Deserialize)]
pub struct Fix {
    pub id: String,
    pub label: String,
}

/// A problem code. Only the constants below, or a code the engine sent that is in the
/// catalogue (`from_engine`): a code with no entry cannot be written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Code(&'static str);

impl Code {
    pub fn as_str(self) -> &'static str {
        self.0
    }
}

macro_rules! codes {
    ($($name:ident = $code:literal,)*) => {
        $(pub const $name: Code = Code($code);)*
        /// Every code the shell reports itself.
        #[cfg(test)]
        const SHELL_CODES: &[Code] = &[$($name,)*];
    };
}

codes! {
    ENGINE_MISSING = "engine-missing",
    ENGINE_WONT_START = "engine-wont-start",
    ENGINE_BLOCKED = "engine-blocked",
    ENGINE_RESTARTING = "engine-restarting",
    ENGINE_SAFE_MODE = "engine-safe-mode",
    ENGINE_LOST_MID_TAKE = "engine-lost-mid-take",
    TAKE_LOST = "take-lost",
    SPEECH_LOADING = "speech-loading",
    SPEECH_LOAD_FAILED = "speech-load-failed",
    SPEECH_SWITCH_FAILED = "speech-switch-failed",
    CLEANUP_LOAD_FAILED = "cleanup-load-failed",
    CLEANUP_SWITCH_FAILED = "cleanup-switch-failed",
    GPU_UNAVAILABLE = "gpu-unavailable",
    GPU_LIBS_DOWNLOAD_FAILED = "gpu-libs-download-failed",
    MIC_UNAVAILABLE = "mic-unavailable",
    MIC_FALLBACK = "mic-fallback",
    MIC_LOST_MID_TAKE = "mic-lost-mid-take",
    MIC_GLITCH = "mic-glitch",
    MIC_BLOCKED = "mic-blocked",
    MIC_SILENT = "mic-silent",
    MIC_IN_USE = "mic-in-use",
    TAKE_SILENT = "take-silent",
    MIC_BLUETOOTH = "mic-bluetooth",
    FOLDER_NOT_WRITABLE = "folder-not-writable",
    DRIVER_TOO_OLD = "driver-too-old",
    HOTKEY_HOOK_FAILED = "hotkey-hook-failed",
    HOTKEY_HOOK_REINSTALLED = "hotkey-hook-reinstalled",
    DICTATION_OFF_IN_APP = "dictation-off-in-app",
    TEXT_KEPT_WINDOW_CHANGED = "text-kept-window-changed",
    TEXT_KEPT_NO_WINDOW = "text-kept-no-window",
    TEXT_COPIED_ADMIN = "text-copied-admin",
    PASTE_LAST_EMPTY = "paste-last-empty",
    PASSWORD_FIELD = "password-field",
    COMMAND_NO_SELECTION = "command-no-selection",
    COMMAND_NOTHING_SAID = "command-nothing-said",
    COMMAND_SELECTION_TOO_LONG = "command-selection-too-long",
    SETTINGS_UNREADABLE = "settings-unreadable",
    SETTINGS_NEWER = "settings-newer",
    APP_RESTARTED = "app-restarted",
    UNKNOWN_MESSAGE = "unknown-message",
}

/// A code from the engine, if the catalogue has it; `fallback` otherwise (an older engine, or
/// one that sent nothing).
pub fn from_engine(code: Option<&str>, fallback: Code) -> Code {
    code.and_then(|c| catalogue().get_key_value(c)).map(|(k, _)| Code(k.as_str())).unwrap_or(fallback)
}

#[derive(Deserialize)]
struct File {
    problems: Vec<Problem>,
}

const SOURCE: &str = include_str!("../../../shared/problems.json");

fn catalogue() -> &'static HashMap<String, Problem> {
    static CATALOGUE: OnceLock<HashMap<String, Problem>> = OnceLock::new();
    CATALOGUE.get_or_init(|| {
        let file: File = serde_json::from_str(SOURCE).expect("shared/problems.json is checked by the tests");
        file.problems.into_iter().map(|p| (p.code.clone(), p)).collect()
    })
}

/// The entry for `code`: there always is one (the constants are checked by a test, and
/// `from_engine` only lets through codes the catalogue has).
pub fn get(code: Code) -> &'static Problem {
    catalogue().get(code.0).unwrap_or_else(|| catalogue().get(UNKNOWN_MESSAGE.0).expect("in the catalogue"))
}

/// `text` with its `{placeholders}` filled in. A placeholder with no value is dropped, with the
/// space before it, so a missing detail never shows as "{detail}".
pub fn fill(text: &str, vars: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(text.len() + 64);
    let mut rest = text;
    while let Some(open) = rest.find('{') {
        let Some(close) = rest[open..].find('}').map(|c| open + c) else { break };
        out.push_str(&rest[..open]);
        let name = &rest[open + 1..close];
        match vars.iter().find(|(k, _)| *k == name).map(|(_, v)| v.trim()) {
            Some(value) if !value.is_empty() => out.push_str(value),
            _ => {
                let trimmed = out.trim_end().len();
                out.truncate(trimmed);
            }
        }
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    out.trim().to_owned()
}

/// A problem as it is shown: every text filled in.
#[derive(Debug, Clone, PartialEq)]
pub struct Shown {
    pub code: &'static str,
    pub title: String,
    pub message: String,
    pub summary: Option<String>,
    pub bar: Option<String>,
    pub action: Option<(&'static str, &'static str)>,
}

pub fn show(code: Code, vars: &[(&str, &str)]) -> Shown {
    let p = get(code);
    Shown {
        code: p.code.as_str(),
        title: fill(&p.title, vars),
        message: fill(&p.message, vars),
        summary: p.summary.as_deref().map(|s| fill(s, vars)),
        bar: p.bar.as_deref().map(|s| fill(s, vars)),
        action: p.action.as_ref().map(|a| (a.id.as_str(), a.label.as_str())),
    }
}

/// The flow bar's words for `code`, filled in; its title when it has none.
pub fn bar(code: Code, vars: &[(&str, &str)]) -> String {
    let s = show(code, vars);
    s.bar.unwrap_or(s.title)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_catalogue_parses_and_every_entry_is_whole() {
        assert!(catalogue().len() >= 40);
        for p in catalogue().values() {
            assert!(!p.title.is_empty() && !p.message.is_empty() && !p.detect.is_empty(), "{}", p.code);
            if p.level == "failed" || p.level == "degraded" {
                assert!(p.summary.is_some(), "{} needs a summary", p.code);
            }
        }
    }

    #[test]
    fn placeholders_are_filled_and_missing_ones_leave_no_trace() {
        assert_eq!(fill("Couldn't switch to {to}", &[("to", "Whisper")]), "Couldn't switch to Whisper");
        assert_eq!(fill("{detail} Dictation goes on.", &[]), "Dictation goes on.");
        assert_eq!(fill("It stopped: {detail} Restart it.", &[("detail", "exit code 1.")]), "It stopped: exit code 1. Restart it.");
        assert_eq!(fill("Try {missing}.", &[]), "Try.");
        let s = show(MIC_FALLBACK, &[("device", "Realtek"), ("chosen", "Blue Yeti")]);
        assert_eq!(s.title, "Using Realtek instead of your microphone");
        assert_eq!(s.summary.as_deref(), Some("Realtek"));
        assert_eq!(s.action, Some(("open_voice", "Choose microphone")));
    }

    #[test]
    fn every_code_the_shell_uses_is_in_the_catalogue_and_every_shell_problem_is_used() {
        for code in SHELL_CODES {
            let p = catalogue().get(code.0).unwrap_or_else(|| panic!("{} is not in the catalogue", code.0));
            assert!(p.source == "shell" || p.source == "both" || p.source == "engine", "{}", p.code);
        }
        // and the other way: a problem the catalogue says the shell detects has a constant here
        for p in catalogue().values().filter(|p| p.source == "shell" || p.source == "both") {
            assert!(SHELL_CODES.iter().any(|c| c.0 == p.code), "{} is in the catalogue but never reported", p.code);
        }
    }

    #[test]
    fn engine_codes_are_taken_only_when_the_catalogue_has_them() {
        assert_eq!(from_engine(Some("speech-out-of-memory"), SPEECH_LOAD_FAILED).as_str(), "speech-out-of-memory");
        assert_eq!(from_engine(Some("made-up"), SPEECH_LOAD_FAILED), SPEECH_LOAD_FAILED);
        assert_eq!(from_engine(None, CLEANUP_LOAD_FAILED), CLEANUP_LOAD_FAILED);
    }
}
