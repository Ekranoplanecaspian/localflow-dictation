//! The global hotkey: a low-level keyboard hook, a chord engine, and push-to-talk.
//!
//! `WH_KEYBOARD_LL` is the only way to see key *releases* globally, which push-to-talk needs
//! (`RegisterHotKey` only reports presses). The hook callback runs on the thread that installed
//! it, inside every keystroke on the machine, and Windows silently removes a hook whose callback
//! takes longer than `LowLevelHooksTimeout`. So the callback here only updates a bitset and
//! sends on a channel; everything else happens on the controller thread.
//!
//! Three things that are easy to get wrong and are handled here:
//!   * **Injected input is ignored.** Our own `SendInput` typing must not retrigger the chord.
//!   * **The Start menu.** Tapping Win on its own opens it. When a chord containing Win fires,
//!     an unassigned key is injected so the eventual Win release is not a lone tap.
//!   * **The hook dying.** A stalled callback gets the hook removed with no notification, so a
//!     watchdog compares the last event we saw against the system's last input time and
//!     reinstalls when the machine clearly had input we never saw.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetLastInputInfo, SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS,
    KEYEVENTF_KEYUP, LASTINPUTINFO, VIRTUAL_KEY,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, DispatchMessageW, GetMessageW, PostThreadMessageW, SetWindowsHookExW,
    TranslateMessage, UnhookWindowsHookEx, HHOOK, KBDLLHOOKSTRUCT, LLKHF_INJECTED, MSG,
    KBDLLHOOKSTRUCT_FLAGS, WH_KEYBOARD_LL, WM_KEYDOWN, WM_KEYUP, WM_QUIT, WM_SYSKEYDOWN, WM_SYSKEYUP,
};

/// A key Windows does not use, injected to break up a lone Win tap.
const VK_UNASSIGNED: u16 = 0x07;
const WM_REHOOK: u32 = windows::Win32::UI::WindowsAndMessaging::WM_APP + 1;
/// If the system saw input this much more recently than our hook did, the hook is gone.
const HOOK_STALE: Duration = Duration::from_millis(1500);
const WATCHDOG_PERIOD: Duration = Duration::from_secs(3);
/// How often the liveness probe may actually inject a key.
const PROBE_EVERY: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------------------------
// key names

/// Canonical virtual-key code, with left/right variants collapsed.
fn canonical(vk: u16) -> u16 {
    match vk {
        0xA0 | 0xA1 => 0x10, // shift
        0xA2 | 0xA3 => 0x11, // ctrl
        0xA4 | 0xA5 => 0x12, // alt
        0x5B | 0x5C => 0x5B, // win
        other => other,
    }
}

/// Parse a key name from the config into a canonical virtual-key code.
pub fn parse_key(name: &str) -> Option<u16> {
    let n = name.trim().to_lowercase();
    let vk = match n.as_str() {
        "ctrl" | "control" => 0x11,
        "win" | "windows" | "super" | "meta" | "cmd" => 0x5B,
        "alt" | "option" => 0x12,
        "shift" => 0x10,
        "space" => 0x20,
        "enter" | "return" => 0x0D,
        "tab" => 0x09,
        "esc" | "escape" => 0x1B,
        "capslock" => 0x14,
        "backspace" => 0x08,
        _ => {
            if let Some(rest) = n.strip_prefix('f') {
                if let Ok(i) = rest.parse::<u16>() {
                    if (1..=24).contains(&i) {
                        return Some(0x70 + i - 1);
                    }
                }
            }
            let mut chars = n.chars();
            let (c, extra) = (chars.next()?, chars.next());
            if extra.is_some() {
                return None;
            }
            match c {
                'a'..='z' => c.to_ascii_uppercase() as u16,
                '0'..='9' => c as u16,
                _ => return None,
            }
        }
    };
    Some(vk)
}

// ---------------------------------------------------------------------------------------------
// what the hook tells the controller

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Raw {
    ChordDown,
    ChordUp,
    /// The command-mode chord, which is a different chord entirely rather than a modifier on
    /// the dictation one - see `Settings::command_chord` for why they may not nest.
    CommandDown,
    CommandUp,
    Escape,
}

/// What the controller decides the user meant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Begin a dictation.
    Start,
    /// The key was released (or hands-free was toggled off): finish and inject.
    Stop,
    /// Throw the take away.
    Cancel,
    /// A double tap latched hands-free on; the previous accidental take was cancelled.
    StartHandsFree,
    /// Begin a command-mode take: the words spoken are an instruction for the selected text,
    /// not something to type.
    StartCommand,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub chord: BTreeSet<u16>,
    /// Command mode's chord. Empty means command mode is off.
    pub command_chord: BTreeSet<u16>,
    pub double_tap: bool,
    pub double_tap_ms: u64,
    /// Escape cancels a dictation in progress.
    pub escape_cancels: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chord: [0x11u16, 0x5B].into_iter().collect(), // Ctrl+Win, like Wispr Flow
            command_chord: [0x5Bu16, 0x12].into_iter().collect(), // Win+Alt
            double_tap: true,
            double_tap_ms: 400,
            escape_cancels: true,
        }
    }
}

impl Config {
    /// Build from the shell's settings file. An unparseable chord keeps the default, because
    /// no hotkey at all is never what was wanted.
    pub fn load() -> Config {
        Config::from_settings(&crate::settings::load())
    }

    pub fn from_settings(s: &crate::settings::Settings) -> Config {
        let mut cfg = Config::default();
        let chord = s.chord();
        if !chord.is_empty() {
            cfg.chord = chord;
        }
        cfg.command_chord = s.command_chord().unwrap_or_default();
        cfg.double_tap = s.double_tap_hands_free;
        cfg.double_tap_ms = s.double_tap_ms.clamp(150, 1200);
        cfg.escape_cancels = s.escape_cancels;
        cfg
    }

    /// Human-readable chord, for the tray tooltip and the window.
    pub fn describe(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for vk in &self.chord {
            parts.push(match *vk {
                0x11 => "Ctrl".into(),
                0x5B => "Win".into(),
                0x12 => "Alt".into(),
                0x10 => "Shift".into(),
                0x20 => "Space".into(),
                0x0D => "Enter".into(),
                0x1B => "Esc".into(),
                0x09 => "Tab".into(),
                0x14 => "Caps Lock".into(),
                v @ 0x70..=0x87 => format!("F{}", v - 0x70 + 1),
                v @ (0x30..=0x39 | 0x41..=0x5A) => (v as u8 as char).to_string(),
                v => format!("VK{v:#04X}"),
            });
        }
        parts.join(" + ")
    }
}

// ---------------------------------------------------------------------------------------------
// hook state, reachable from the callback

/// Held keys and the chord as 256-bit masks, one bit per virtual-key code.
///
/// The callback runs inside every keystroke on the machine and Windows removes a hook whose
/// callback is slow. The first version locked two mutexes and cloned a `BTreeSet` - a heap
/// allocation per keypress - and the hook was observed dropping twice in the first thirty
/// seconds of a GPU stress run. Atomics only now: no locks, no allocation, no waiting.
type KeyMask = [AtomicU64; 4];

fn bit_of(vk: u16) -> (usize, u64) {
    ((vk as usize) >> 6, 1u64 << (vk & 63))
}

fn mask_set(mask: &KeyMask, vk: u16, on: bool) {
    let (word, bit) = bit_of(vk);
    if on {
        mask[word].fetch_or(bit, Ordering::Relaxed);
    } else {
        mask[word].fetch_and(!bit, Ordering::Relaxed);
    }
}

fn mask_has(mask: &KeyMask, vk: u16) -> bool {
    let (word, bit) = bit_of(vk);
    mask[word].load(Ordering::Relaxed) & bit != 0
}

/// Is every key of the chord currently held?
fn chord_complete(chord: &KeyMask, held: &KeyMask) -> bool {
    let mut any = false;
    for i in 0..4 {
        let c = chord[i].load(Ordering::Relaxed);
        if c != 0 {
            any = true;
        }
        if c & !held[i].load(Ordering::Relaxed) != 0 {
            return false;
        }
    }
    any // an empty chord is never "complete", or every keystroke would start a dictation
}

fn mask_clear(mask: &KeyMask) {
    for word in mask.iter() {
        word.store(0, Ordering::Relaxed);
    }
}

struct HookState {
    chord: KeyMask,
    command_chord: KeyMask,
    held: KeyMask,
    active: AtomicBool,
    command_active: AtomicBool,
    /// Milliseconds (`GetTickCount`) of the last event the hook saw, for the watchdog.
    last_event_tick: AtomicU32,
    tx: Mutex<Option<Sender<Raw>>>,
    /// Thread id of the hook's message loop, so the watchdog can ask it to reinstall.
    thread_id: AtomicU32,
    hook: AtomicI64,
    recording: AtomicBool,
    /// Whether a take is latched on. Owned by the controller, but readable and clearable from
    /// outside so the hands-free watcher can end a take the user has walked away from.
    hands_free: AtomicBool,
    /// Every event the callback saw, injected ones included. The watchdog's liveness probe
    /// works by making this number move.
    events_seen: AtomicU64,
}

fn state() -> &'static HookState {
    static STATE: OnceLock<HookState> = OnceLock::new();
    STATE.get_or_init(|| {
        let st = HookState {
        chord: Default::default(),
        command_chord: Default::default(),
        held: Default::default(),
        active: AtomicBool::new(false),
        command_active: AtomicBool::new(false),
        last_event_tick: AtomicU32::new(0),
        tx: Mutex::new(None),
        thread_id: AtomicU32::new(0),
        hook: AtomicI64::new(0),
        recording: AtomicBool::new(false),
        hands_free: AtomicBool::new(false),
        events_seen: AtomicU64::new(0),
        };
        for vk in Config::default().chord {
            mask_set(&st.chord, vk, true);
        }
        st
    })
}

/// Replace the chord. Cheap and lock free, so it can be changed while the hook is running.
pub fn set_chord(keys: &BTreeSet<u16>) {
    let st = state();
    mask_clear(&st.chord);
    for vk in keys {
        mask_set(&st.chord, *vk, true);
    }
}

/// Replace the command-mode chord. An empty set turns command mode off.
pub fn set_command_chord(keys: &BTreeSet<u16>) {
    let st = state();
    mask_clear(&st.command_chord);
    for vk in keys {
        mask_set(&st.command_chord, *vk, true);
    }
}

/// The session tells us whether a take is running, so Escape only fires when it would mean
/// something and the chord can latch correctly.
pub fn set_recording(on: bool) {
    state().recording.store(on, Ordering::Relaxed);
}

/// Forget that a take was latched. Called when something other than the user's chord ended it,
/// so the next tap starts a new take rather than being read as "stop".
pub fn clear_hands_free() {
    state().hands_free.store(false, Ordering::Relaxed);
}

unsafe extern "system" fn keyboard_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let st = state();
    if code >= 0 {
        let info = &*(lparam.0 as *const KBDLLHOOKSTRUCT);
        st.last_event_tick.store(info.time, Ordering::Relaxed);
        st.events_seen.fetch_add(1, Ordering::Relaxed);
        // Our own injected typing must never look like the user pressing the chord.
        if info.flags & LLKHF_INJECTED != KBDLLHOOKSTRUCT_FLAGS(0) {
            return CallNextHookEx(None, code, wparam, lparam);
        }
        let vk = canonical(info.vkCode as u16);
        let msg = wparam.0 as u32;
        let down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
        let up = msg == WM_KEYUP || msg == WM_SYSKEYUP;

        if down || up {
            mask_set(&st.held, vk, down);
            // Command mode is tested first. The two chords are guaranteed not to nest (see
            // `Settings::command_chord`), so at most one of them can be complete, but testing
            // in a fixed order keeps that guarantee from mattering to the hook.
            if down
                && chord_complete(&st.command_chord, &st.held)
                && !st.command_active.swap(true, Ordering::SeqCst)
            {
                send(Raw::CommandDown);
            } else if up
                && mask_has(&st.command_chord, vk)
                && st.command_active.swap(false, Ordering::SeqCst)
            {
                send(Raw::CommandUp);
            } else if down && chord_complete(&st.chord, &st.held) && !st.active.swap(true, Ordering::SeqCst) {
                send(Raw::ChordDown);
            } else if up && mask_has(&st.chord, vk) && st.active.swap(false, Ordering::SeqCst) {
                send(Raw::ChordUp);
            } else if down && vk == 0x1B && st.recording.load(Ordering::Relaxed) {
                send(Raw::Escape);
            }
        }
    }
    CallNextHookEx(None, code, wparam, lparam)
}

fn send(raw: Raw) {
    if let Some(tx) = state().tx.lock().unwrap().as_ref() {
        let _ = tx.send(raw);
    }
}

/// A lone Win press-and-release opens the Start menu. Injecting a key Windows ignores makes the
/// eventual release part of a sequence instead of a tap.
///
/// This must not run on the hook thread. `SendInput` from inside a low-level hook callback
/// re-enters the hook and takes long enough that Windows can decide the callback has timed out
/// and remove the hook entirely - which looks exactly like "the hotkey worked once and then
/// stopped".
fn suppress_start_menu() {
    let vk = VIRTUAL_KEY(VK_UNASSIGNED);
    let make = |flags: KEYBD_EVENT_FLAGS| INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT { wVk: vk, wScan: 0, dwFlags: flags, time: 0, dwExtraInfo: 0 },
        },
    };
    let inputs = [make(KEYBD_EVENT_FLAGS(0)), make(KEYEVENTF_KEYUP)];
    unsafe {
        SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
    }
}

// ---------------------------------------------------------------------------------------------
// install

pub struct Hotkeys {
    thread_id: u32,
}

impl Hotkeys {
    /// Install the hook on its own thread and run the chord policy. `on_action` is called on
    /// the controller thread, never on the hook thread.
    pub fn install<F>(cfg: Config, on_action: F) -> Hotkeys
    where
        F: Fn(Action) + Send + 'static,
    {
        set_chord(&cfg.chord);
        set_command_chord(&cfg.command_chord);
        let (tx, rx) = mpsc::channel::<Raw>();
        *state().tx.lock().unwrap() = Some(tx);

        let ready = std::sync::Arc::new((Mutex::new(0u32), std::sync::Condvar::new()));
        let signal = ready.clone();
        std::thread::Builder::new()
            .name("keyboard-hook".into())
            .spawn(move || hook_thread(signal))
            .expect("hook thread");

        let (lock, cv) = &*ready;
        let mut id = lock.lock().unwrap();
        while *id == 0 {
            let (guard, timeout) = cv.wait_timeout(id, Duration::from_secs(5)).unwrap();
            id = guard;
            if timeout.timed_out() {
                break;
            }
        }
        let thread_id = *id;
        drop(id);

        std::thread::Builder::new()
            .name("hotkey-controller".into())
            .spawn(move || controller(cfg, rx, on_action))
            .expect("controller thread");
        std::thread::Builder::new()
            .name("hook-watchdog".into())
            .spawn(watchdog)
            .expect("watchdog thread");

        Hotkeys { thread_id }
    }

    pub fn stop(&self) {
        unsafe {
            let _ = PostThreadMessageW(self.thread_id, WM_QUIT, WPARAM(0), LPARAM(0));
        }
    }
}

fn install_hook() -> bool {
    unsafe {
        let old = state().hook.swap(0, Ordering::SeqCst);
        if old != 0 {
            let _ = UnhookWindowsHookEx(HHOOK(old as *mut std::ffi::c_void));
        }
        match SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_proc), None, 0) {
            Ok(h) => {
                state().hook.store(h.0 as i64, Ordering::SeqCst);
                mask_clear(&state().held);
                state().active.store(false, Ordering::SeqCst);
                true
            }
            Err(_) => false,
        }
    }
}

fn hook_thread(ready: std::sync::Arc<(Mutex<u32>, std::sync::Condvar)>) {
    let tid = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
    state().thread_id.store(tid, Ordering::SeqCst);
    install_hook();
    {
        let (lock, cv) = &*ready;
        *lock.lock().unwrap() = tid;
        cv.notify_all();
    }
    unsafe {
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            if msg.message == WM_REHOOK {
                let was_active = state().active.load(Ordering::SeqCst);
                install_hook();
                // The key release happened while we were deaf, or will happen while the chord
                // state says nothing is held. Either way the take has to be ended by hand.
                if was_active {
                    send(Raw::ChordUp);
                }
                continue;
            }
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        let old = state().hook.swap(0, Ordering::SeqCst);
        if old != 0 {
            let _ = UnhookWindowsHookEx(HHOOK(old as *mut std::ffi::c_void));
        }
    }
}

/// Windows removes a low-level hook whose callback timed out, without telling anyone, so the
/// hook has to be checked rather than trusted.
///
/// Suspicion is cheap: `GetLastInputInfo` moving while our hook saw nothing. It is also wrong
/// most of the time, because it counts mouse movement and our hook only sees the keyboard -
/// that false positive reinstalled the hook 46 times during ten dictations, and a reinstall
/// during a held chord would have lost the key release and left the take running forever.
///
/// So suspicion only triggers a real test: inject a key nothing is bound to and see whether the
/// callback observes it. If it does, the hook is alive and nothing happens.
fn watchdog() {
    let mut last_probe = Instant::now() - PROBE_EVERY;
    loop {
        std::thread::sleep(WATCHDOG_PERIOD);
        if !suspicious() {
            continue;
        }
        // Browsing the web is a long stretch of mouse input and no keys, which looks suspicious
        // for as long as it lasts. Probing on every check would inject a keystroke every three
        // seconds all day for nothing.
        if last_probe.elapsed() < PROBE_EVERY {
            continue;
        }
        last_probe = Instant::now();
        if alive() {
            continue;
        }
        crate::shell_log!("the keyboard hook stopped receiving events; reinstalling");
        let tid = state().thread_id.load(Ordering::SeqCst);
        if tid != 0 {
            unsafe {
                let _ = PostThreadMessageW(tid, WM_REHOOK, WPARAM(0), LPARAM(0));
            }
        }
    }
}

/// Has the machine had input that our hook did not see? Includes the mouse, so this is only a
/// reason to look closer.
fn suspicious() -> bool {
    let mut info = LASTINPUTINFO { cbSize: std::mem::size_of::<LASTINPUTINFO>() as u32, dwTime: 0 };
    if !unsafe { GetLastInputInfo(&mut info) }.as_bool() {
        return false;
    }
    let ours = state().last_event_tick.load(Ordering::Relaxed);
    if ours == 0 {
        return false; // nothing has been typed yet; there is nothing to compare against
    }
    let behind = info.dwTime.wrapping_sub(ours);
    behind > HOOK_STALE.as_millis() as u32 && behind < 60_000
}

/// Definitive: inject a key nobody uses and see whether the callback counts it.
fn alive() -> bool {
    let before = state().events_seen.load(Ordering::Relaxed);
    suppress_start_menu(); // the same unassigned key; harmless wherever it lands
    for _ in 0..20 {
        std::thread::sleep(Duration::from_millis(10));
        if state().events_seen.load(Ordering::Relaxed) != before {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------------------------
// policy: push-to-talk, and double tap to latch

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_names_a_config_file_would_use() {
        assert_eq!(parse_key("ctrl"), Some(0x11));
        assert_eq!(parse_key("Windows"), Some(0x5B));
        assert_eq!(parse_key(" Shift "), Some(0x10));
        assert_eq!(parse_key("f13"), Some(0x7C));
        assert_eq!(parse_key("k"), Some(b'K' as u16));
        assert_eq!(parse_key("space"), Some(0x20));
        assert_eq!(parse_key("f25"), None);
        assert_eq!(parse_key("nonsense"), None);
    }

    /// Left and right modifiers are the same key as far as a chord is concerned, or holding
    /// right-Ctrl would not start a dictation bound to "ctrl".
    #[test]
    fn collapses_left_and_right_modifiers() {
        assert_eq!(canonical(0xA3), canonical(0xA2)); // right ctrl == left ctrl
        assert_eq!(canonical(0x5C), canonical(0x5B)); // right win == left win
        assert_eq!(canonical(0x41), 0x41); // letters are left alone
    }

    /// The callback's whole job, without a keyboard: the chord fires when every one of its
    /// keys is held and not before, and an empty chord never fires.
    #[test]
    fn the_chord_mask_fires_only_when_every_key_is_held() {
        let chord: KeyMask = Default::default();
        let held: KeyMask = Default::default();
        assert!(!chord_complete(&chord, &held), "an empty chord must never fire");

        mask_set(&chord, 0x11, true); // ctrl
        mask_set(&chord, 0x5B, true); // win
        assert!(!chord_complete(&chord, &held));

        mask_set(&held, 0x11, true);
        assert!(!chord_complete(&chord, &held), "half a chord is not a chord");
        mask_set(&held, 0x5B, true);
        assert!(chord_complete(&chord, &held));

        // Other keys held at the same time do not break it, but releasing one of ours does.
        mask_set(&held, b'K' as u16, true);
        assert!(chord_complete(&chord, &held));
        mask_set(&held, 0x11, false);
        assert!(!chord_complete(&chord, &held));

        assert!(mask_has(&chord, 0x5B));
        assert!(!mask_has(&chord, b'K' as u16));
    }

    /// Virtual-key codes run to 0xFF and the mask is four words; the top of the range must
    /// land in the last word rather than wrapping into the first.
    #[test]
    fn the_mask_covers_the_whole_virtual_key_range() {
        let mask: KeyMask = Default::default();
        for vk in [0u16, 63, 64, 127, 128, 191, 192, 255] {
            mask_set(&mask, vk, true);
            assert!(mask_has(&mask, vk), "vk {vk} did not survive a round trip");
        }
        assert!(!mask_has(&mask, 200));
        mask_clear(&mask);
        assert!(!mask_has(&mask, 255));
    }

    #[test]
    fn describes_a_chord_the_way_a_person_would_write_it() {
        let cfg = Config::default();
        assert_eq!(cfg.describe(), "Ctrl + Win");
        let cfg = Config { chord: [0x10, 0x70].into_iter().collect(), ..Config::default() };
        assert_eq!(cfg.describe(), "Shift + F1");
    }
}

fn controller<F: Fn(Action)>(cfg: Config, rx: mpsc::Receiver<Raw>, on_action: F) {
    let double_tap = Duration::from_millis(cfg.double_tap_ms);
    let chord_has_win = cfg.chord.contains(&0x5B);
    let command_has_win = cfg.command_chord.contains(&0x5B);
    let shared = state();
    // Read back through the shared flag on every event: the hands-free watcher may have ended
    // the take from another thread while the user was not touching the keyboard.
    macro_rules! hands_free {
        () => {
            shared.hands_free.load(Ordering::Relaxed)
        };
    }
    macro_rules! set_hands_free {
        ($v:expr) => {
            shared.hands_free.store($v, Ordering::Relaxed)
        };
    }
    let mut pressed_at: Option<Instant> = None;
    let mut released_at: Option<Instant> = None;

    while let Ok(raw) = rx.recv() {
        match raw {
            Raw::ChordDown => {
                let now = Instant::now();
                let quick_second_tap = cfg.double_tap
                    && released_at.map(|r| now.duration_since(r) < double_tap).unwrap_or(false)
                    && pressed_at
                        .zip(released_at)
                        .map(|(p, r)| r.duration_since(p) < double_tap)
                        .unwrap_or(false);
                pressed_at = Some(now);

                if chord_has_win {
                    suppress_start_menu();
                }
                if hands_free!() {
                    // A press while latched means "stop".
                    set_hands_free!(false);
                    on_action(Action::Stop);
                } else if quick_second_tap {
                    set_hands_free!(true);
                    on_action(Action::StartHandsFree);
                } else {
                    on_action(Action::Start);
                }
            }
            Raw::ChordUp => {
                released_at = Some(Instant::now());
                if !hands_free!() {
                    on_action(Action::Stop);
                }
            }
            Raw::CommandDown => {
                // Hands-free never applies here: a command is one instruction about one
                // selection, so latching it on would only leave the bar running.
                if command_has_win {
                    suppress_start_menu();
                }
                on_action(Action::StartCommand);
            }
            Raw::CommandUp => on_action(Action::Stop),
            Raw::Escape => {
                if cfg.escape_cancels {
                    set_hands_free!(false);
                    on_action(Action::Cancel);
                }
            }
        }
    }
}
