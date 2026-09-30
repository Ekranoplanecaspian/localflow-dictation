//! The tray icon: the only permanently visible part of a background app.
//!
//! The icon is drawn rather than loaded, so a state change is a colour change with no assets to
//! keep in sync, and it stays crisp at any tray scaling. Brand artwork replaces this in phase 7.

use std::sync::Mutex;

use tauri::image::Image;
use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, Runtime};

use crate::guard::LockExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// The engine is starting or loading models.
    Loading,
    /// Ready and waiting for the hotkey.
    Idle,
    /// The hotkey is held.
    Recording,
    /// Transcribing and cleaning up.
    Processing,
    /// Working, but something needs attention (clean-up unavailable, safe mode).
    Degraded,
    /// Dictation cannot work right now.
    Error,
}

impl State {
    fn colour(self) -> [u8; 3] {
        match self {
            State::Loading => [0x9a, 0x8c, 0xd8],
            State::Idle => [0x8b, 0x6c, 0xff],   // the LocalFlow accent
            State::Recording => [0x3d, 0xd6, 0x8c],
            State::Processing => [0xff, 0xb3, 0x4d],
            State::Degraded => [0xe0, 0xa6, 0x4a],
            State::Error => [0xff, 0x5c, 0x5c],
        }
    }

    fn label(self) -> &'static str {
        match self {
            State::Loading => "LocalFlow - loading",
            State::Idle => "LocalFlow - ready",
            State::Recording => "LocalFlow - listening",
            State::Processing => "LocalFlow - transcribing",
            State::Degraded => "LocalFlow - needs attention",
            State::Error => "LocalFlow - not working",
        }
    }
}

const SIZE: u32 = 32;

/// A ring with a filled centre, drawn with 4x supersampling so the edges are smooth.
fn icon(state: State) -> Image<'static> {
    let [r, g, b] = state.colour();
    let mut rgba = vec![0u8; (SIZE * SIZE * 4) as usize];
    let c = SIZE as f32 / 2.0 - 0.5;
    let outer = SIZE as f32 * 0.46;
    let inner = SIZE as f32 * 0.30;
    let dot = SIZE as f32 * 0.17;
    for y in 0..SIZE {
        for x in 0..SIZE {
            let mut cover = 0f32;
            for sy in 0..4 {
                for sx in 0..4 {
                    let px = x as f32 + (sx as f32 + 0.5) / 4.0 - 0.5;
                    let py = y as f32 + (sy as f32 + 0.5) / 4.0 - 0.5;
                    let d = ((px - c).powi(2) + (py - c).powi(2)).sqrt();
                    if (d <= outer && d >= inner) || d <= dot {
                        cover += 1.0;
                    }
                }
            }
            let a = (cover / 16.0 * 255.0) as u8;
            let i = ((y * SIZE + x) * 4) as usize;
            rgba[i] = r;
            rgba[i + 1] = g;
            rgba[i + 2] = b;
            rgba[i + 3] = a;
        }
    }
    Image::new_owned(rgba, SIZE, SIZE)
}

pub struct Tray<R: Runtime> {
    state: Mutex<State>,
    /// The one-line explanation shown under the state in the tooltip.
    detail: Mutex<String>,
    /// "Restart engine", which reads "Leave safe mode" while the engine is in safe mode.
    restart: MenuItem<R>,
}

/// Build the tray icon and its menu. Kept in app state so the rest of the shell can update it.
pub fn create<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, "open", "Open LocalFlow", true, None::<&str>)?;
    let paste_last = MenuItem::with_id(app, "paste_last", "Paste last dictation\tWin+Alt+V", true, None::<&str>)?;
    let restart = MenuItem::with_id(app, "restart", "Restart engine", true, None::<&str>)?;
    let autostart = CheckMenuItem::with_id(
        app,
        "autostart",
        "Start at sign-in",
        true,
        crate::win::autostart_enabled(),
        None::<&str>,
    )?;
    let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
    let autostart_item = autostart.clone();
    let restart_item = restart.clone();
    let menu = Menu::with_items(
        app,
        &[
            &open,
            &paste_last,
            &restart,
            &PredefinedMenuItem::separator(app)?,
            &autostart,
            &PredefinedMenuItem::separator(app)?,
            &quit,
        ],
    )?;

    TrayIconBuilder::with_id("localflow")
        .icon(icon(State::Loading))
        .tooltip(State::Loading.label())
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(move |app, event| match event.id.as_ref() {
            "open" => show_window(app),
            // Handled in lib.rs, which waits for the menu to give the focus back first.
            "paste_last" => {
                let _ = tauri::Emitter::emit(app, "paste-last-request", ());
            }
            "restart" => {
                if let Some(engine) = app.try_state::<crate::engine::Engine>() {
                    engine.restart();
                }
            }
            "autostart" => {
                // Reflect what the registry actually says, not what was clicked: the write
                // can fail under a locked-down policy.
                let _ = autostart_item.set_checked(crate::win::toggle_autostart());
            }
            "quit" => crate::shutdown(app),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = event {
                show_window(tray.app_handle());
            }
        })
        .build(app)?;

    app.manage(Tray { state: Mutex::new(State::Loading), detail: Mutex::new(String::new()), restart: restart_item });
    Ok(())
}

pub fn show_window<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

/// Update the icon, its tooltip, and the detail line shown in the window.
pub fn set_state<R: Runtime>(app: &AppHandle<R>, state: State, detail: Option<&str>) {
    if let Some(tray) = app.try_state::<Tray<R>>() {
        let mut current = tray.state.locked();
        let same = *current == state;
        *current = state;
        drop(current);
        if let Some(d) = detail {
            *tray.detail.locked() = d.to_owned();
        }
        if same && detail.is_none() {
            return;
        }
    }
    if let Some(icon_handle) = app.tray_by_id("localflow") {
        let _ = icon_handle.set_icon(Some(icon(state)));
        let tip = match detail {
            Some(d) if !d.is_empty() => format!("{}\n{}", state.label(), d),
            _ => state.label().to_owned(),
        };
        let _ = icon_handle.set_tooltip(Some(&tip));
    }
}

/// Safe mode is left by restarting the engine, so the restart item says that while it is on.
pub fn set_safe_mode<R: Runtime>(app: &AppHandle<R>, on: bool) {
    if let Some(tray) = app.try_state::<Tray<R>>() {
        let _ = tray.restart.set_text(if on { "Leave safe mode" } else { "Restart engine" });
    }
}
