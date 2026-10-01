//! Settings that belong to the shell rather than to the models.
//!
//! Two files, split by who owns the behaviour: `shell.json` holds the hotkey, what is kept in
//! history and how injection behaves; `config.json` belongs to the engine and holds the
//! dictionary, snippets and clean-up model. The Hub reads the engine's file directly (the
//! engine saves it) and writes through the session protocol, so the running engine applies a
//! change immediately instead of at the next restart.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

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
    /// Started by Windows at sign-in: open the LocalFlow window as well. Off by default -
    /// sign-in starts LocalFlow quietly in the tray, ready to dictate.
    pub open_window_at_sign_in: bool,
    /// The version of this file's layout. Raise it whenever a setting is added: an older
    /// LocalFlow then leaves the file alone instead of saving it without the setting it did not
    /// know (after a downgrade). Absent (0) in files written before it existed.
    pub version: u32,
}

/// See `Settings::version`.
pub const SETTINGS_VERSION: u32 = 2; // 2: open_window_at_sign_in

/// The version of a settings file written by a newer LocalFlow, once one has been read: from
/// then on nothing is saved over it.
static NEWER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn newer_version() -> Option<u32> {
    Some(NEWER.load(std::sync::atomic::Ordering::Relaxed)).filter(|v| *v > 0)
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
            open_window_at_sign_in: false,
            version: SETTINGS_VERSION,
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
    let settings = load_from(&path);
    if settings.version > SETTINGS_VERSION
        && NEWER.swap(settings.version, std::sync::atomic::Ordering::Relaxed) != settings.version
    {
        crate::shell_log!(
            "[{}] shell.json was written by a newer LocalFlow (version {}; this one knows {}): reading what              is understood, and not saving changes over it",
            crate::problems::SETTINGS_NEWER.as_str(),
            settings.version,
            SETTINGS_VERSION
        );
    }
    settings
}

pub fn save(settings: &Settings) -> Result<(), String> {
    let path = path().ok_or_else(|| "no settings directory".to_owned())?;
    if let Some(v) = newer_version() {
        return Err(refused(v));
    }
    save_to(&path, settings)
}

fn refused(version: u32) -> String {
    format!("these settings belong to a newer LocalFlow (version {version}), so changes are not saved over them")
}

/// JSON text, without the byte-order mark some editors put in front of UTF-8.
fn json_text(text: &str) -> &str {
    text.strip_prefix('\u{feff}').unwrap_or(text)
}

/// The settings in `path`; the defaults when there is none. A file that cannot be read is set
/// aside, not ignored: it used to count as "no settings" silently, and the next change in the
/// Hub saved the defaults over it - the hotkey, the app rules and everything else gone. A file
/// starting with a byte-order mark (Notepad's older UTF-8) was one such file.
///
/// Unreadable means anything but "not there": text that is not UTF-8 (an editor saving in the
/// ANSI code page) and a failed read counted as no settings too, set aside by nothing.
fn load_from(path: &Path) -> Settings {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Settings::default(),
        Err(e) => return set_aside(path, &e.to_string()),
    };
    let Ok(text) = String::from_utf8(bytes) else {
        return set_aside(path, "it is not UTF-8 text");
    };
    match serde_json::from_str::<Settings>(json_text(&text)) {
        Ok(settings) => settings,
        Err(e) => set_aside(path, &e.to_string()),
    }
}

/// Move an unreadable settings file out of the way, so nothing is saved over it, and start from
/// the defaults.
fn set_aside(path: &Path, why: &str) -> Settings {
    let aside = path.with_extension(format!("json.broken-{}", unix_time()));
    let kept = std::fs::rename(path, &aside).is_ok();
    crate::shell_log!(
        "[{}] the settings file could not be read ({why}); starting from the defaults{}",
        crate::problems::SETTINGS_UNREADABLE.as_str(),
        if kept { format!(", and the unreadable file was kept as {}", aside.display()) } else { String::new() }
    );
    Settings::default()
}

/// Written whole or not at all: to a temporary file, then moved over the old one. Writing in
/// place truncated first, and a crash or a full disk in between left half a file.
fn save_to(path: &Path, settings: &Settings) -> Result<(), String> {
    let settings = &Settings { version: SETTINGS_VERSION, ..settings.clone() };
    let text = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        e.to_string()
    })
}

fn unix_time() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Keys a hotkey may be made of, besides letters, digits and F1-F24.
const MODIFIERS: [u16; 4] = [0x11, 0x5B, 0x12, 0x10]; // Ctrl, Win, Alt, Shift
const CTRL: u16 = 0x11;
const WIN: u16 = 0x5B;
const ALT: u16 = 0x12;
const SHIFT: u16 = 0x10;

/// Chords Windows or nearly every application already owns. The hook passes every key on, so
/// taking one of these would do both: lock the PC and start a dictation, copy and dictate.
const RESERVED: &[(&[u16], &str)] = &[
    (&[WIN, 0x4C], "locks the PC (Win+L)"),
    (&[WIN, 0x44], "shows the desktop (Win+D)"),
    (&[WIN, 0x45], "opens File Explorer (Win+E)"),
    (&[WIN, 0x52], "opens Run (Win+R)"),
    (&[WIN, 0x48], "starts Windows' own dictation (Win+H)"),
    (&[WIN, 0x56], "opens clipboard history (Win+V)"),
    (&[WIN, 0x49], "opens Settings (Win+I)"),
    (&[WIN, 0x41], "opens quick settings (Win+A)"),
    (&[WIN, 0x53], "opens search (Win+S)"),
    (&[WIN, 0x58], "opens the power menu (Win+X)"),
    (&[WIN, 0x09], "opens Task View (Win+Tab)"),
    (&[ALT, 0x09], "switches windows (Alt+Tab)"),
    (&[ALT, 0x73], "closes the window (Alt+F4)"),
    (&[CTRL, 0x41], "selects all (Ctrl+A)"),
    (&[CTRL, 0x43], "copies (Ctrl+C)"),
    (&[CTRL, 0x56], "pastes (Ctrl+V)"),
    (&[CTRL, 0x58], "cuts (Ctrl+X)"),
    (&[CTRL, 0x5A], "undoes (Ctrl+Z)"),
    (&[CTRL, 0x59], "redoes (Ctrl+Y)"),
    (&[CTRL, 0x53], "saves (Ctrl+S)"),
    (&[CTRL, 0x46], "finds (Ctrl+F)"),
    (&[CTRL, 0x50], "prints (Ctrl+P)"),
    (&[CTRL, 0x57], "closes the tab (Ctrl+W)"),
    (&[CTRL, 0x54], "opens a new tab (Ctrl+T)"),
    (&[CTRL, 0x4E], "opens a new window (Ctrl+N)"),
];

const METHODS: [&str; 4] = ["", "auto", "type", "paste"]; // "" is the default: automatic
const PROFILES: [&str; 7] = ["", "auto", "chat", "email", "docs", "code", "terminal"];

/// Why `keys` cannot be a hotkey, or Ok. `what` names it in the message.
pub fn check_chord(keys: &[String], what: &str) -> Result<(), String> {
    let mut vks = BTreeSet::new();
    for k in keys {
        match crate::hotkey::parse_key(k) {
            Some(vk) => {
                vks.insert(vk);
            }
            None => return Err(format!("The {what} can't use \"{k}\": LocalFlow doesn't know that key.")),
        }
    }
    if vks.is_empty() {
        return Err(format!("The {what} needs at least one key."));
    }
    if vks.contains(&0x1B) {
        return Err(format!("Escape cancels a dictation, so it can't be part of the {what}."));
    }
    let mods: BTreeSet<u16> = vks.iter().copied().filter(|v| MODIFIERS.contains(v)).collect();
    let others: Vec<u16> = vks.iter().copied().filter(|v| !MODIFIERS.contains(v)).collect();
    let function_key = |v: u16| (0x70..=0x87).contains(&v);
    if others.is_empty() {
        if mods.len() < 2 {
            return Err(format!(
                "A modifier on its own would start a dictation with every shortcut that uses it: \
                 add a second key to the {what}."
            ));
        }
    } else if !(others.len() == 1 && mods.is_empty() && function_key(others[0])) {
        // Anything but a lone function key needs Ctrl, Win or Alt: without one it is typing -
        // Shift with a letter is a capital.
        if !mods.iter().any(|m| *m != SHIFT) {
            return Err(format!(
                "Without Ctrl, Win or Alt the {what} would go off while you type. \
                 A function key such as F8 on its own is fine."
            ));
        }
    }
    for (chord, does) in RESERVED {
        if vks.len() == chord.len() && chord.iter().all(|k| vks.contains(k)) {
            return Err(format!("That shortcut already {does}; choose another for the {what}."));
        }
    }
    Ok(())
}

impl Settings {
    /// These settings, tidied, or why they cannot be saved. Everything the Hub (or a hand-edited
    /// file) can put here passes through this before it is written: a lone "A" as the hotkey
    /// started a dictation with every A typed.
    pub fn validated(mut self) -> Result<Settings, String> {
        check_chord(&self.hotkey, "dictation hotkey")?;
        if self.command_mode && !self.command_hotkey.is_empty() {
            check_chord(&self.command_hotkey, "command hotkey")?;
            let command: BTreeSet<u16> =
                self.command_hotkey.iter().filter_map(|k| crate::hotkey::parse_key(k)).collect();
            let dictation = self.chord();
            if command.is_subset(&dictation) || command.is_superset(&dictation) {
                return Err("The command hotkey can't contain the dictation hotkey, or be part of it: \
                            one would always start before the other."
                    .into());
            }
        }
        if self.retention_days > 3650 {
            return Err("History can be kept for up to ten years (3650 days), or forever (0).".into());
        }
        if !(150..=1200).contains(&self.double_tap_ms) {
            return Err("The double-tap window must be between 150 and 1200 ms.".into());
        }
        if self.hands_free_timeout_s > 3600 {
            return Err("Hands-free stops after at most an hour of silence (3600 s), or never (0).".into());
        }
        self.microphone = self.microphone.trim().to_owned();
        let mut rules = BTreeMap::new();
        for (app, rule) in std::mem::take(&mut self.app_rules) {
            let app = app.trim().to_ascii_lowercase();
            if app.is_empty() || app.contains(['\\', '/']) {
                return Err(format!("\"{app}\" is not an application name, like notepad.exe."));
            }
            if !METHODS.contains(&rule.method.as_str()) {
                return Err(format!("\"{}\" is not a way of typing: auto, type or paste.", rule.method));
            }
            if !PROFILES.contains(&rule.profile.as_str()) {
                return Err(format!("\"{}\" is not a clean-up style.", rule.profile));
            }
            rules.insert(app, rule);
        }
        self.app_rules = rules;
        Ok(self)
    }

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
        // The same rule saving used: a lone function key such as F8 passed it and was saved, then
        // was refused here for being one key, and command mode was silently off.
        if check_chord(&self.command_hotkey, "command hotkey").is_err() {
            return None;
        }
        let chord: BTreeSet<u16> =
            self.command_hotkey.iter().filter_map(|k| crate::hotkey::parse_key(k)).collect();
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
        .and_then(|text| serde_json::from_str(json_text(&text)).ok())
        .unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("localflow-settings-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("shell.json")
    }

    fn keys(k: &[&str]) -> Vec<String> {
        k.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn hotkeys_that_would_go_off_while_typing_are_refused() {
        for bad in [&["a"][..], &["space"], &["ctrl"], &["shift", "a"], &["a", "s"], &["ctrl", "esc"], &["ctrl", "nope"]] {
            assert!(check_chord(&keys(bad), "hotkey").is_err(), "{bad:?} should be refused");
        }
        for good in [&["ctrl", "win"][..], &["f8"], &["ctrl", "alt", "d"], &["win", "alt"], &["ctrl", "shift", "space"]] {
            assert!(check_chord(&keys(good), "hotkey").is_ok(), "{good:?} should be fine");
        }
    }

    #[test]
    fn shortcuts_windows_or_every_app_owns_are_refused_with_the_reason() {
        let e = check_chord(&keys(&["win", "l"]), "dictation hotkey").unwrap_err();
        assert!(e.contains("locks the PC"), "{e}");
        assert!(check_chord(&keys(&["ctrl", "c"]), "hotkey").unwrap_err().contains("copies"));
        assert!(check_chord(&keys(&["ctrl", "shift", "c"]), "hotkey").is_ok(), "only the exact chord");
    }

    #[test]
    fn settings_are_tidied_or_refused_before_they_are_saved() {
        let mut s = Settings::default();
        s.microphone = "  Headset  ".into();
        s.app_rules.insert(" Slack.EXE ".into(), AppRule { auto_send: true, ..AppRule::default() });
        let ok = s.clone().validated().unwrap();
        assert_eq!(ok.microphone, "Headset");
        assert!(ok.app_rules.contains_key("slack.exe"));

        let mut nested = Settings::default();
        nested.command_hotkey = keys(&["ctrl", "win", "shift"]);
        assert!(nested.validated().is_err(), "contains the dictation chord");

        let mut rule = Settings::default();
        rule.app_rules.insert("code.exe".into(), AppRule { method: "teleport".into(), ..AppRule::default() });
        assert!(rule.validated().is_err());

        let mut days = Settings::default();
        days.retention_days = 100_000;
        assert!(days.validated().is_err());
    }

    #[test]
    fn a_file_saved_with_a_byte_order_mark_is_read() {
        let path = temp("bom");
        let mut s = Settings::default();
        s.flow_bar = false;
        let text = serde_json::to_string(&s).unwrap();
        std::fs::write(&path, format!("\u{feff}{text}")).unwrap();
        assert!(!load_from(&path).flow_bar, "read, not replaced by the defaults");
    }

    #[test]
    fn an_unreadable_file_is_kept_aside_rather_than_saved_over() {
        let path = temp("broken");
        std::fs::write(&path, r#"{"hotkey": ["f8"], "flow_bar": fal"#).unwrap();
        let s = load_from(&path);
        assert_eq!(s.hotkey, Settings::default().hotkey, "the defaults, for now");
        assert!(!path.exists());
        let kept: Vec<_> = std::fs::read_dir(path.parent().unwrap()).unwrap().flatten().collect();
        assert_eq!(kept.len(), 1);
        assert!(std::fs::read_to_string(kept[0].path()).unwrap().contains("f8"), "the user's own file survives");
    }

    #[test]
    fn a_file_that_is_not_utf8_is_kept_aside_too() {
        let path = temp("ansi");
        // "café.exe" saved in Windows-1252: the é is one byte, 0xE9, not UTF-8.
        let mut bytes = br#"{"hotkey": ["f7"], "app_rules": {"caf"#.to_vec();
        bytes.push(0xE9);
        bytes.extend_from_slice(br#".exe": {}}}"#);
        std::fs::write(&path, &bytes).unwrap();
        let s = load_from(&path);
        assert_eq!(s.hotkey, Settings::default().hotkey, "the defaults, for now");
        assert!(!path.exists(), "so the next save cannot go over it");
        let kept: Vec<_> = std::fs::read_dir(path.parent().unwrap()).unwrap().flatten().collect();
        assert_eq!(kept.len(), 1);
        assert_eq!(std::fs::read(kept[0].path()).unwrap(), bytes, "the user's own file survives");
    }

    #[test]
    fn saving_replaces_the_file_whole() {
        let path = temp("save");
        let mut s = Settings::default();
        s.hotkey = vec!["f9".into()];
        save_to(&path, &s).unwrap();
        s.hotkey = vec!["f10".into()];
        save_to(&path, &s).unwrap();
        assert_eq!(load_from(&path).hotkey, ["f10"]);
        let files = std::fs::read_dir(path.parent().unwrap()).unwrap().count();
        assert_eq!(files, 1, "no temporary file left behind");
    }

    /// After a downgrade: a file a newer LocalFlow wrote is read for what this one knows, and
    /// every save says why it is refused rather than dropping the settings it does not know.
    #[test]
    fn settings_from_a_newer_version_are_read_and_never_saved_over() {
        let path = temp("newer");
        std::fs::write(&path, r#"{"hotkey": ["f8"], "version": 7, "something_new": {"on": true}}"#).unwrap();
        let s = load_from(&path);
        assert_eq!((s.hotkey.as_slice(), s.version), (["f8".to_owned()].as_slice(), 7));
        assert!(refused(7).contains("newer LocalFlow (version 7)"));
        // Our own saves carry the version, so a newer one can tell the file apart.
        let mine = temp("mine");
        save_to(&mine, &Settings { version: 0, ..Settings::default() }).unwrap();
        assert_eq!(load_from(&mine).version, SETTINGS_VERSION);
    }

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
        assert_eq!(single.command_chord(), None, "a lone modifier is not a chord");
    }

    #[test]
    fn a_command_hotkey_that_saves_is_the_one_the_hook_listens_for() {
        // A lone F8 passed validation and was saved, but never reached the hook (found by an
        // outside review of 0.2.3).
        let f8 = Settings { command_hotkey: vec!["f8".into()], ..Settings::default() };
        let saved = f8.clone().validated().expect("a lone function key may be saved");
        let f8_vk = crate::hotkey::parse_key("f8").unwrap();
        assert_eq!(saved.command_chord(), Some([f8_vk].into_iter().collect()));
        assert_eq!(crate::hotkey::Config::from_settings(&saved).command_chord, [f8_vk].into_iter().collect());

        // And the other way round: whatever the hook would refuse, saving refuses too.
        for keys in [vec!["win"], vec!["a"], vec!["ctrl", "nonsense"]] {
            let s = Settings { command_hotkey: keys.iter().map(|k| k.to_string()).collect(), ..Settings::default() };
            assert!(s.clone().validated().is_err(), "{keys:?} must not save");
            assert_eq!(s.command_chord(), None, "{keys:?}");
        }
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
