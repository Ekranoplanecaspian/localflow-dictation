//! "Report a problem": everything needed to find a cause, in one zip the user attaches to an
//! issue themselves. Nothing is sent anywhere by LocalFlow.
//!
//! Built by the shell, not the engine: a report matters most when the engine will not start.
//!
//! What goes in: the logs (shell, engine, the clean-up server's tail, crash reports), the status
//! model and the engine's own status, the settings, the model timings, and a page of system
//! details. What does not: dictated words and window titles (replaced by their length - decided
//! 2026-09-28), the history, the dictionary, snippets and house style (their counts only), and
//! any API key (only whether one is set).

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::{json, Value};
use tauri::{AppHandle, Manager};

use crate::guard::LockExt;

/// Where issues are filed: the public repository, not the private one.
pub const ISSUES: &str = "https://github.com/Ekranoplanecaspian/localflow-dictation/issues/new";

/// The last file made, the only one "Show in folder" may point at.
static LAST: Mutex<Option<PathBuf>> = Mutex::new(None);

const README: &str = "LocalFlow diagnostics\r\n\
=====================\r\n\
\r\n\
This file helps find the cause of a problem. LocalFlow made it on your computer and sent it\r\n\
nowhere; attach it to your report yourself.\r\n\
\r\n\
Inside: LocalFlow's logs, its status, its settings, the timings of its models, and a page of\r\n\
system details (Windows version, processor, memory, graphics card).\r\n\
\r\n\
Not inside:\r\n\
  - what you dictated: every dictated sentence and command in the logs is replaced by its\r\n\
    length, like \"<42 chars>\", and so is every window title\r\n\
  - your history\r\n\
  - your dictionary, snippets and house style: only how many there are\r\n\
  - any API key: only whether one is set\r\n\
\r\n\
The files are plain text; open any of them to see exactly what is there.\r\n";

// ---------------------------------------------------------------------------------------------
// redaction

/// Every double-quoted string in `line` as its length: `"Hello there."` -> `"<12 chars>"`.
/// Rust's `{:?}` escapes a quote inside as `\"`, which does not end the string. A string that
/// is already a length is left as it is: shell.log is redacted as it is written, and again when
/// it goes into a report.
pub fn redact_quoted(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '"' {
            out.push(c);
            continue;
        }
        let mut inner = String::new();
        let mut n = 0usize;
        let mut closed = false;
        while let Some(d) = chars.next() {
            match d {
                '\\' => {
                    inner.push(d);
                    if let Some(e) = chars.next() {
                        inner.push(e);
                    }
                    n += 1;
                }
                '"' => {
                    closed = true;
                    break;
                }
                _ => {
                    inner.push(d);
                    n += 1
                }
            }
        }
        if closed && is_length(&inner, "<") {
            out.push_str(&format!("\"{inner}\""));
        } else {
            out.push_str(&format!("\"<{n} chars>\""));
        }
        if !closed {
            break;
        }
    }
    out
}

/// A take's window title as its length: `s3 recording in outlook.exe (Re: the offer - Mail)`
/// -> `s3 recording in outlook.exe (<title, 21 chars>)`. The title runs to the last `)` before
/// a ` | ` (a command's line goes on with the selection) or the end.
pub fn redact_title(line: &str) -> String {
    for marker in [" recording in ", " hands-free in ", " command in "] {
        let Some(at) = line.find(marker) else { continue };
        let after_app = at + marker.len();
        let Some(open) = line[after_app..].find(" (").map(|i| after_app + i + 2) else { continue };
        let end_of_title = line[open..].find(" | ").map(|i| open + i).unwrap_or(line.len());
        let Some(close) = line[open..end_of_title].rfind(')').map(|i| open + i) else { continue };
        let title = &line[open..close];
        if is_length(title, "<title, ") {
            return line.to_owned();
        }
        let n = title.chars().count();
        return format!("{}<title, {n} chars>{}", &line[..open], &line[close..]);
    }
    line.to_owned()
}

/// Whether `s` is already a redaction such as `<12 chars>` or `<title, 12 chars>`.
fn is_length(s: &str, prefix: &str) -> bool {
    s.strip_prefix(prefix)
        .and_then(|rest| rest.strip_suffix(" chars>"))
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// A line of shell.log, without the user's words.
pub fn redact_shell_line(line: &str) -> String {
    let line = redact_title(line);
    // Hotkey lines quote key names ("ctrl", "win"), which are worth keeping. Only when "hotkey"
    // comes before the first quote: a dictation that says "hotkey" is still a dictation.
    let before_quote = &line[..line.find('"').unwrap_or(line.len())];
    if before_quote.contains("hotkey") { line } else { redact_quoted(&line) }
}

/// A line of the engine's log. Version 0.1's tray app logged each dictation as a line of its
/// own under `localflow.app`; such a line is replaced whole.
pub fn redact_engine_line(line: &str) -> String {
    const KNOWN: [&str; 8] = [
        "Recording", "Starting engine", "Loading the auto-edit", "Engine ready", "Auto-edits ready",
        "Mic:", "Stopping", "Hotkey",
    ];
    if let Some(at) = line.find("localflow.app: ") {
        let (head, message) = line.split_at(at + "localflow.app: ".len());
        let first = message.trim_start();
        let audio_line = first.chars().next().is_some_and(|c| c.is_ascii_digit()) && first.contains(" audio |");
        if !audio_line && !KNOWN.iter().any(|k| first.starts_with(k)) {
            return format!("{head}<{} chars>", message.chars().count());
        }
        // Its timing line ends with where the text went: "-> brave.exe - <window title>".
        if audio_line {
            if let Some(arrow) = message.find("-> ") {
                if let Some(dash) = message[arrow..].find(" - ").map(|i| arrow + i + 3) {
                    let n = message[dash..].chars().count();
                    return format!("{head}{}<title, {n} chars>", &message[..dash]);
                }
            }
        }
        if first.starts_with("Recording") {
            if let (Some(open), Some(close)) = (message.find('('), message.rfind(')')) {
                if open < close {
                    let n = message[open + 1..close].chars().count();
                    return format!("{head}{}(<title, {n} chars>){}", &message[..open], &message[close + 1..]);
                }
            }
        }
    }
    redact_quoted(line)
}

/// The engine's settings with the user's own words replaced by how many there are.
pub fn redact_engine_config(mut cfg: Value) -> Value {
    if let Some(pp) = cfg.get_mut("postprocess").and_then(Value::as_object_mut) {
        for key in ["dictionary", "snippets", "dictionary_terms"] {
            if let Some(v) = pp.get_mut(key) {
                let n = v.as_object().map(|o| o.len()).or(v.as_array().map(|a| a.len())).unwrap_or(0);
                *v = json!(format!("<{n} entries>"));
            }
        }
        if let Some(v) = pp.get_mut("custom_instructions") {
            let n = v.as_str().map(|s| s.chars().count()).unwrap_or(0);
            *v = json!(format!("<{n} chars>"));
        }
        if let Some(v) = pp.get_mut("llm_api_key") {
            let set = v.as_str().is_some_and(|s| !s.is_empty());
            *v = json!(if set { "<set>" } else { "" });
        }
    }
    cfg
}

fn redact_file(path: &Path, redact: fn(&str) -> String, last_lines: Option<usize>) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let text = String::from_utf8_lossy(&bytes);
    let lines: Vec<&str> = text.lines().collect();
    let from = last_lines.map(|n| lines.len().saturating_sub(n)).unwrap_or(0);
    Some(lines[from..].iter().map(|l| redact(l)).collect::<Vec<_>>().join("\r\n"))
}

// ---------------------------------------------------------------------------------------------
// system details

fn system(app: &AppHandle) -> Value {
    use windows::Win32::System::Registry::HKEY_LOCAL_MACHINE;
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let nt = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion";
    let build = crate::win::reg_string(HKEY_LOCAL_MACHINE, nt, "CurrentBuild").unwrap_or_default();
    let release = crate::win::reg_string(HKEY_LOCAL_MACHINE, nt, "DisplayVersion").unwrap_or_default();
    let edition = crate::win::reg_string(HKEY_LOCAL_MACHINE, nt, "EditionID").unwrap_or_default();
    // ProductName still says "Windows 10" on Windows 11; the build number tells them apart.
    let windows = if build.parse::<u32>().unwrap_or(0) >= 22000 { "Windows 11" } else { "Windows 10" };
    let cpu = crate::win::reg_string(
        HKEY_LOCAL_MACHINE,
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        "ProcessorNameString",
    )
    .unwrap_or_default();
    let mut mem = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
    let ram = unsafe { GlobalMemoryStatusEx(&mut mem) }.ok().map(|_| {
        json!({"total_gb": (mem.ullTotalPhys as f64 / 1e9 * 10.0).round() / 10.0,
               "free_gb": (mem.ullAvailPhys as f64 / 1e9 * 10.0).round() / 10.0})
    });
    let status = app.try_state::<crate::engine::Engine>().and_then(|e| e.status());
    let compute = status.as_ref().and_then(|s| s.get("compute"));
    json!({
        "localflow": env!("CARGO_PKG_VERSION"),
        "windows": format!("{windows} {edition} {release} (build {build})"),
        "processor": cpu.trim(),
        "memory": ram,
        "graphics_card": compute.and_then(|c| c.get("gpu")).cloned(),
        "hardware": compute.and_then(|c| c.get("hardware")).cloned(),
        "engine": status.as_ref().and_then(|s| s.get("version")).cloned(),
    })
}

/// One line for an issue: version, Windows, graphics card, and the problems showing now.
pub fn summary(app: &AppHandle) -> String {
    let sys = system(app);
    let health = crate::health::current(app);
    let codes: Vec<String> = health
        .get("parts")
        .and_then(Value::as_array)
        .map(|parts| parts.iter().filter_map(|p| p.get("code").and_then(Value::as_str).map(str::to_owned)).collect())
        .unwrap_or_default();
    let gpu = graphics_names(&sys);
    format!(
        "LocalFlow {} · {} · {} · status: {}{}",
        sys["localflow"].as_str().unwrap_or("?"),
        sys["windows"].as_str().unwrap_or("?"),
        gpu,
        health.get("overall").and_then(Value::as_str).unwrap_or("?"),
        if codes.is_empty() { String::new() } else { format!(" ({})", codes.join(", ")) }
    )
}

/// Every graphics adapter the engine's capability report found ("NVIDIA GeForce RTX 4060 Laptop
/// GPU + AMD Radeon(TM) 890M Graphics"); the NVIDIA reading alone for an engine too old to say.
fn graphics_names(sys: &Value) -> String {
    let listed: Vec<&str> = sys["hardware"]["gpus"]
        .as_array()
        .map(|gpus| gpus.iter().filter_map(|g| g["name"].as_str()).collect())
        .unwrap_or_default();
    if !listed.is_empty() {
        return listed.join(" + ");
    }
    sys["graphics_card"]["name"].as_str().unwrap_or("no graphics card found").to_owned()
}

// ---------------------------------------------------------------------------------------------
// the file

/// Downloads, where people look for a file they just made; the Desktop, or LocalFlow's own
/// folder, if there is none.
pub fn destination() -> Option<PathBuf> {
    let profile = std::env::var_os("USERPROFILE").map(PathBuf::from);
    for dir in [profile.as_ref().map(|p| p.join("Downloads")), profile.as_ref().map(|p| p.join("Desktop"))]
        .into_iter()
        .flatten()
    {
        if dir.is_dir() {
            return Some(dir);
        }
    }
    crate::paths::config_dir()
}

fn file_name() -> String {
    use windows::Win32::System::SystemInformation::GetLocalTime;
    let t = unsafe { GetLocalTime() };
    format!(
        "LocalFlow diagnostics {:04}-{:02}-{:02} {:02}{:02}{:02}.zip",
        t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond
    )
}

/// The clean-up server's logs: `bin\llama\<build>\<cpu|cuda>\llama-server.log`.
fn llama_logs() -> Vec<(String, PathBuf)> {
    let Some(local) = std::env::var_os("LOCALAPPDATA") else { return Vec::new() };
    let root = PathBuf::from(local).join("LocalFlow").join("bin").join("llama");
    let mut found = Vec::new();
    for build in std::fs::read_dir(&root).into_iter().flatten().flatten() {
        for kind in std::fs::read_dir(build.path()).into_iter().flatten().flatten() {
            let log = kind.path().join("llama-server.log");
            if log.is_file() {
                found.push((format!("llama-server-{}.log", kind.file_name().to_string_lossy()), log));
            }
        }
    }
    found
}

/// Make the zip. Returns where it went.
pub fn export(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = destination().ok_or("there is no folder to save it in")?;
    let path = dir.join(file_name());
    let config = crate::paths::config_dir().ok_or("LocalFlow's folder is unknown")?;
    let file = std::fs::File::create(&path).map_err(|e| format!("could not create {}: {e}", path.display()))?;
    let mut zip = zip::ZipWriter::new(file);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
    let mut add = |name: &str, text: &str| -> Result<(), String> {
        zip.start_file(name, options).map_err(|e| e.to_string())?;
        zip.write_all(text.as_bytes()).map_err(|e| e.to_string())
    };

    add("README.txt", README)?;
    add("system.json", &serde_json::to_string_pretty(&system(app)).unwrap_or_default())?;
    let engine = app.try_state::<crate::engine::Engine>();
    let status = json!({
        "summary": summary(app),
        "health": crate::health::current(app),
        "link": engine.as_ref().map(|e| e.link()),
        // The engine's status holds no dictated text: model states, placement, errors.
        "engine": engine.as_ref().and_then(|e| e.status()),
    });
    add("status.json", &serde_json::to_string_pretty(&status).unwrap_or_default())?;

    for (name, redact) in [("shell.log", redact_shell_line as fn(&str) -> String), ("localflow.log", redact_engine_line)] {
        for (suffix, file) in [("", name.to_owned()), (".old", format!("{name}.old"))] {
            if let Some(text) = redact_file(&config.join(&file), redact, None) {
                add(&format!("logs/{name}{suffix}"), &text)?;
            }
        }
    }
    for fault in ["localflow-fault.log", "speech-worker-fault.log"] {
        if let Ok(text) = std::fs::read_to_string(config.join(fault)) {
            if !text.trim().is_empty() {
                add(&format!("logs/{fault}"), &redact_quoted(&text))?;
            }
        }
    }
    for (name, log) in llama_logs() {
        if let Some(text) = redact_file(&log, redact_quoted, Some(2000)) {
            add(&format!("logs/{name}"), &text)?;
        }
    }

    if let Ok(text) = std::fs::read_to_string(config.join("config.json")) {
        if let Ok(cfg) = serde_json::from_str::<Value>(text.trim_start_matches('\u{feff}')) {
            add("settings/config.json", &serde_json::to_string_pretty(&redact_engine_config(cfg)).unwrap_or_default())?;
        }
    }
    if let Ok(text) = std::fs::read_to_string(config.join("shell.json")) {
        add("settings/shell.json", &text)?;
    }
    if let Ok(text) = std::fs::read_to_string(config.join("perf.json")) {
        add("perf.json", &text)?;
    }
    drop(add);
    zip.finish().map_err(|e| e.to_string())?;
    crate::shell_log!("diagnostics saved to {}", path.display());
    *LAST.locked() = Some(path.clone());
    Ok(path)
}

/// Remember a file LocalFlow just saved, for "Show in folder".
pub fn remember(path: &Path) {
    *LAST.locked() = Some(path.to_path_buf());
}

/// A local date for file names: 2026-09-28.
pub fn today() -> String {
    use windows::Win32::System::SystemInformation::GetLocalTime;
    let t = unsafe { GetLocalTime() };
    format!("{:04}-{:02}-{:02}", t.wYear, t.wMonth, t.wDay)
}

/// Show the file just made, selected, in File Explorer.
pub fn reveal() -> Result<(), String> {
    let path = LAST.locked().clone().ok_or("no diagnostics file has been made yet")?;
    std::process::Command::new("explorer.exe")
        .arg(format!("/select,{}", path.display()))
        .spawn()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn percent_encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// The new-issue page on GitHub, filled in with what is not personal: the version, Windows,
/// the graphics card, the problems showing now - and where the zip is, to attach it.
pub fn issue_url(summary: &str, file: Option<&str>) -> String {
    let body = format!(
        "**What happened?**\n\n<!-- What were you doing, what did you expect, what happened instead? -->\n\n\
         **Diagnostics**\n\n{}\n\n---\n{summary}\n",
        match file {
            Some(f) => format!("<!-- Drag \"{f}\" from your Downloads folder here. -->"),
            None => "<!-- Help > Report a problem > Create diagnostics file, then drag it here. -->".to_owned(),
        }
    );
    format!("{ISSUES}?title=&body={}", percent_encode(&body))
}

/// Open the issue page in the default browser.
pub fn open_issue(app: &AppHandle) -> Result<(), String> {
    let file = LAST.locked().as_ref().and_then(|p| p.file_name()).map(|f| f.to_string_lossy().into_owned());
    let url = issue_url(&summary(app), file.as_deref());
    // The long-standing way to hand a URL to the default browser without a console flashing up.
    std::process::Command::new("rundll32.exe")
        .args(["url.dll,FileProtocolHandler", &url])
        .spawn()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictated_words_become_their_length() {
        let final_line = r#"12:00:00.000 s4 final: 4.1s audio | stt 0 ms | auto-edits 263 ms | release->final 351 ms | "Meet \"Priya\" at 3."
"#;
        let out = redact_shell_line(final_line.trim_end());
        assert!(out.ends_with(r#"| "<18 chars>""#), "{out}");
        assert!(!out.contains("Priya"));
        let command = r#"12:00:00.000 s5 command: "make it shorter" on 120 chars"#;
        assert_eq!(redact_shell_line(command), r#"12:00:00.000 s5 command: "<15 chars>" on 120 chars"#);
        // Key names are kept: they say which chord was meant.
        let keys = r#"command hotkey ["win", "alt"] overlaps the dictation hotkey"#;
        assert_eq!(redact_shell_line(keys), keys);
        // ...but a dictation that mentions a hotkey is still a dictation.
        let said = r#"s7 final: 2.0s audio | stt 0 ms | rules 1 ms | release->final 90 ms | "change the hotkey""#;
        assert!(redact_shell_line(said).ends_with(r#"| "<17 chars>""#), "{}", redact_shell_line(said));
    }

    #[test]
    fn a_line_redacted_when_written_is_not_changed_by_the_report() {
        // shell.log is redacted as it is written (D1) and again when it goes into a report.
        for line in [
            r#"s4 final: 4.1s audio | stt 0 ms | auto-edits 263 ms | release->final 351 ms | "Meet \"Priya\" at 3.""#,
            "s3 recording in outlook.exe (Re: the offer (draft) - Mail)",
            "s6 command in code.exe (plan.md - x) | selection: 40 chars from UI Automation",
            r#"s5 command: "make it shorter" on 120 chars"#,
        ] {
            let once = redact_shell_line(line);
            assert_eq!(redact_shell_line(&once), once);
        }
        // A dictation that looks like a redaction is still counted.
        assert_eq!(redact_quoted(r#"| "<not a length>""#), r#"| "<14 chars>""#);
    }

    #[test]
    fn window_titles_become_their_length() {
        let l = "12:00:00.000 s3 recording in outlook.exe (Re: the offer (draft) - Mail)";
        assert_eq!(redact_shell_line(l), "12:00:00.000 s3 recording in outlook.exe (<title, 28 chars>)");
        let c = "12:00:00.000 s6 command in code.exe (plan.md - x) | selection: 40 chars from UI Automation";
        assert_eq!(
            redact_shell_line(c),
            "12:00:00.000 s6 command in code.exe (<title, 11 chars>) | selection: 40 chars from UI Automation"
        );
        let untouched = "12:00:00.000 microphone: Microphone Array (Realtek(R) Audio)";
        assert_eq!(redact_shell_line(untouched), untouched);
    }

    #[test]
    fn version_0_1_dictation_lines_are_replaced_whole() {
        let old = "10:00:00 INFO    localflow.app: What is the population of Mumbai?";
        assert_eq!(redact_engine_line(old), "10:00:00 INFO    localflow.app: <33 chars>");
        let rec = "10:00:00 INFO    localflow.app: Recording ... (brave.exe: Inbox - Gmail)";
        assert_eq!(redact_engine_line(rec), "10:00:00 INFO    localflow.app: Recording ... (<title, 24 chars>)");
        let timing = "10:00:00 INFO    localflow.app: 4.1s audio | continuous, 3 live";
        assert_eq!(redact_engine_line(timing), timing);
        let went = "04:51:24 INFO    localflow.app: 2.8s audio | stt 139 ms | llm 0 ms | -> brave.exe - Inbox (3) - Gmail";
        assert_eq!(
            redact_engine_line(went),
            "04:51:24 INFO    localflow.app: 2.8s audio | stt 139 ms | llm 0 ms | -> brave.exe - <title, 17 chars>"
        );
        let engine = "10:00:00 INFO    localflow.service.engine: Engine ready in 3.5s";
        assert_eq!(redact_engine_line(engine), engine);
    }

    #[test]
    fn settings_keep_their_shape_and_lose_the_users_words() {
        let cfg = json!({"postprocess": {"dictionary": {"kwen": "Qwen", "arnub": "Arnab"},
            "dictionary_terms": ["Okonkwo"], "snippets": {"my email": "a@b.c"},
            "custom_instructions": "British spelling.", "llm_api_key": "sk-secret", "llm_model": "qwen3-4b"}});
        let out = redact_engine_config(cfg).to_string();
        for secret in ["Qwen", "Arnab", "Okonkwo", "a@b.c", "British", "sk-secret"] {
            assert!(!out.contains(secret), "{secret} in {out}");
        }
        assert!(out.contains("<2 entries>") && out.contains("<set>") && out.contains("qwen3-4b"));
    }

    #[test]
    fn the_summary_names_every_graphics_adapter() {
        let both = json!({"hardware": {"gpus": [{"name": "NVIDIA GeForce RTX 4060 Laptop GPU"},
            {"name": "AMD Radeon(TM) 890M Graphics"}]}, "graphics_card": {"name": "RTX 4060 Laptop GPU"}});
        assert_eq!(graphics_names(&both), "NVIDIA GeForce RTX 4060 Laptop GPU + AMD Radeon(TM) 890M Graphics");
        let old_engine = json!({"hardware": {"cpu_cores": 12}, "graphics_card": {"name": "RTX 4060 Laptop GPU"}});
        assert_eq!(graphics_names(&old_engine), "RTX 4060 Laptop GPU");
        assert_eq!(graphics_names(&json!({"hardware": null, "graphics_card": null})), "no graphics card found");
    }

    #[test]
    fn the_issue_link_carries_nothing_personal() {
        let url = issue_url("LocalFlow 0.2.0 · Windows 11 Core 25H2 (build 26200) · RTX 4060 · status: ok", Some("x.zip"));
        assert!(url.starts_with(ISSUES));
        assert!(url.contains("LocalFlow%200.2.0"));
        assert!(!url.contains(' '));
    }
}
