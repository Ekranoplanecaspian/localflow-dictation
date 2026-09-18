//! Settings that belong to the shell rather than to the models.
//!
//! Two files, split by who owns the behaviour: `shell.json` holds the hotkey, what is kept in
//! history and how injection behaves; `config.json` belongs to the engine and holds the
//! dictionary, snippets and clean-up model. The Hub reads the engine's file directly (the
//! engine saves it) and writes through the session protocol, so the running engine applies a
//! change immediately instead of at the next restart.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// What to do differently in one particular application.
///
/// Every field means "leave it alone" by default, so a rule only ever describes the part the
/// user actually wanted changed and an empty rule behaves exactly like no rule at all.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppRule {
    /// Ignore the hotkey here entirely. For the places where a stray dictation does damage:
    /// a game, a remote desktop, a terminal running something interactive.
    pub disabled: bool,
    /// "auto" (the default), "type" or "paste". The automatic choice is right almost always;
    /// this is the escape hatch for the app where it is not.
    pub method: String,
    /// Press Enter after the text lands, so a dictated chat message sends itself.
    pub auto_send: bool,
    /// Force a clean-up style instead of letting the engine guess from the app name: "chat",
    /// "email", "docs", "code", "terminal", or "" / "auto" to keep guessing.
    pub profile: String,
}

impl AppRule {
    fn is_empty(&self) -> bool {
        *self == AppRule::default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Chord key names, as a person would write them: ["ctrl", "win"].
    pub hotkey: Vec<String>,
    pub double_tap_hands_free: bool,
    pub double_tap_ms: u64,
    pub escape_cancels: bool,
    /// Keep what was dictated, so the Hub can show it back. The most private setting here.
    pub history: bool,
    /// Entries older than this are pruned. 0 keeps everything.
    pub retention_days: u64,
    /// Show the flow bar while dictating.
    pub flow_bar: bool,
    /// Microphone to use, matched as a case-insensitive substring of the device name.
    /// Empty means whatever Windows calls the default, and follows it when it changes.
    pub microphone: String,
    /// End each dictation with a space, so the next one does not run into it.
    pub trailing_space: bool,
    /// Command mode: select text, hold the command chord, say what to change.
    pub command_mode: bool,
    /// Chord for command mode. Must not nest with `hotkey` - see `command_chord`.
    pub command_hotkey: Vec<String>,
    /// Stop a latched hands-free take after this many seconds of silence. 0 never stops it.
    pub hands_free_timeout_s: u64,
    /// Per-application rules, keyed by executable name in lower case ("slack.exe").
    pub app_rules: BTreeMap<String, AppRule>,
    /// Whether the first-run wizard has been completed. Absent in a file written before the
    /// wizard existed, which deserialises to false - so an existing install sees it once too,
    /// which is the right answer: nobody has been shown it yet.
    pub onboarded: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            hotkey: vec!["ctrl".into(), "win".into()],
            double_tap_hands_free: true,
            double_tap_ms: 400,
            escape_cancels: true,
            history: true,
            retention_days: 90,
            flow_bar: true,
            microphone: String::new(),
            trailing_space: true,
            // Win+Alt rather than something with Ctrl in it: Ctrl+Alt is AltGr on
            // international layouts, and taking that over would break ordinary typing.
            command_mode: true,
            command_hotkey: vec!["win".into(), "alt".into()],
            // Long enough to think mid-sentence, short enough that a take forgotten about
            // does not sit there recording the room.
            hands_free_timeout_s: 8,
            app_rules: BTreeMap::new(),
            onboarded: false,
        }
    }
}

pub fn path() -> Option<PathBuf> {
    let dir = crate::paths::config_dir()?;
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("shell.json"))
}

pub fn load() -> Settings {
    let Some(path) = path() else { return Settings::default() };
    std::fs::read_to_string(path)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

pub fn save(settings: &Settings) -> Result<(), String> {
    let path = path().ok_or_else(|| "no settings directory".to_owned())?;
    let text = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| e.to_string())
}

impl Settings {
    /// The chord as virtual-key codes, ignoring names that do not parse. An empty result means
    /// the caller should keep the chord it has: no hotkey at all is never what was wanted.
    pub fn chord(&self) -> BTreeSet<u16> {
        self.hotkey.iter().filter_map(|k| crate::hotkey::parse_key(k)).collect()
    }

    /// The rule for an application, or the default (which changes nothing) when it has none.
    ///
    /// Matched on the executable name, case-insensitively, because that is the one thing the
    /// shell can always see; window titles move around too much to key behaviour on.
    pub fn rule_for(&self, app: &str) -> AppRule {
        self.app_rules.get(&app.to_ascii_lowercase()).cloned().unwrap_or_default()
    }

    /// Rules that do nothing are dropped on save, so the file does not fill up with entries
    /// left behind by a toggle that was switched on and off again.
    pub fn prune_rules(&mut self) {
        self.app_rules.retain(|_, rule| !rule.is_empty());
    }

    /// The command-mode chord, or `None` when it is off or unusable.
    ///
    /// The two chords must not nest. The hook fires a chord the moment every one of its keys is
    /// held, so if command mode were Ctrl+Win+Shift and dictation were Ctrl+Win, pressing the
    /// first two keys would already have started a dictation and the third would arrive too
    /// late to mean anything. Rather than delay dictation to wait for a key that usually never
    /// comes - which would make the app feel slow for the sake of the rarer feature - a nested
    /// chord is refused and command mode stays off until it is set to something disjoint.
    pub fn command_chord(&self) -> Option<BTreeSet<u16>> {
        if !self.command_mode {
            return None;
        }
        let chord: BTreeSet<u16> =
            self.command_hotkey.iter().filter_map(|k| crate::hotkey::parse_key(k)).collect();
        if chord.len() < 2 {
            return None;
        }
        let dictation = self.chord();
        if chord.is_subset(&dictation) || chord.is_superset(&dictation) {
            crate::shell_log!(
                "command hotkey {:?} overlaps the dictation hotkey {:?}; command mode is off",
                self.command_hotkey,
                self.hotkey
            );
            return None;
        }
        Some(chord)
    }
}

/// The engine's own config file, read for display. Writes go through the protocol instead, so
/// the running engine applies them at once.
pub fn engine_config() -> Value {
    let Some(dir) = crate::paths::config_dir() else {
        return Value::Null;
    };
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_chord_and_survives_nonsense_in_it() {
        let s = Settings {
            hotkey: vec!["ctrl".into(), "win".into(), "not-a-key".into()],
            ..Settings::default()
        };
        let chord = s.chord();
        assert_eq!(chord.len(), 2);
        assert!(chord.contains(&0x11) && chord.contains(&0x5B));
    }

    #[test]
    fn a_rule_is_found_whatever_case_the_application_reports() {
        let mut s = Settings::default();
        s.app_rules.insert(
            "slack.exe".into(),
            AppRule { auto_send: true, profile: "chat".into(), ..AppRule::default() },
        );
        assert!(s.rule_for("Slack.exe").auto_send);
        assert!(s.rule_for("SLACK.EXE").auto_send);
        assert_eq!(s.rule_for("slack.exe").profile, "chat");
    }

    #[test]
    fn an_application_with_no_rule_gets_one_that_changes_nothing() {
        let rule = Settings::default().rule_for("notepad.exe");
        assert!(!rule.disabled && !rule.auto_send);
        assert!(rule.method.is_empty() && rule.profile.is_empty());
    }

    /// A toggle switched on and then off again would otherwise leave an entry behind for ever.
    #[test]
    fn rules_that_do_nothing_are_dropped_on_save() {
        let mut s = Settings::default();
        s.app_rules.insert("notepad.exe".into(), AppRule::default());
        s.app_rules.insert("slack.exe".into(), AppRule { auto_send: true, ..AppRule::default() });
        s.prune_rules();
        assert_eq!(s.app_rules.len(), 1);
        assert!(s.app_rules.contains_key("slack.exe"));
    }

    /// The hook fires a chord as soon as all of its keys are held, so a command chord that
    /// contains the dictation chord could never win the race: the dictation would already have
    /// started. Rather than slow dictation down to wait, such a pairing turns command mode off.
    #[test]
    fn a_command_chord_that_nests_with_the_dictation_chord_is_refused() {
        let nested = Settings {
            hotkey: vec!["ctrl".into(), "win".into()],
            command_hotkey: vec!["ctrl".into(), "win".into(), "shift".into()],
            ..Settings::default()
        };
        assert_eq!(nested.command_chord(), None, "a superset would never fire");

        let inside = Settings {
            hotkey: vec!["ctrl".into(), "win".into(), "shift".into()],
            command_hotkey: vec!["ctrl".into(), "win".into()],
            ..Settings::default()
        };
        assert_eq!(inside.command_chord(), None, "a subset would swallow the other chord");
    }

    #[test]
    fn a_disjoint_command_chord_is_accepted_and_can_be_turned_off() {
        let s = Settings::default();
        assert_eq!(s.command_chord(), Some([0x5B, 0x12].into_iter().collect()), "Win+Alt");

        let off = Settings { command_mode: false, ..Settings::default() };
        assert_eq!(off.command_chord(), None);

        let single = Settings { command_hotkey: vec!["win".into()], ..Settings::default() };
        assert_eq!(single.command_chord(), None, "one key is not a chord");
    }

    /// Missing fields must take their defaults rather than failing the whole file, or one
    /// hand-edited line would reset every setting.
    #[test]
    fn an_older_settings_file_still_loads() {
        let partial: Settings = serde_json::from_str(r#"{"hotkey": ["alt", "space"]}"#).unwrap();
        assert_eq!(partial.hotkey, vec!["alt", "space"]);
        assert!(partial.history, "unspecified settings keep their default");
        assert_eq!(partial.retention_days, 90);
    }
}
