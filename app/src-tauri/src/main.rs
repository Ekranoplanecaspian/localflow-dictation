// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // Two headless modes, so the engine link and the OS integration can be checked from a
    // terminal instead of by watching a window.
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--selftest") => {
            let Some(wav) = args.get(1) else {
                eprintln!("usage: --selftest <path.wav> [--fast]");
                std::process::exit(2);
            };
            let realtime = !args.iter().any(|a| a == "--fast");
            std::process::exit(app_lib::selftest_dictate(wav, realtime));
        }
        Some("--stress") => {
            let minutes = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(10);
            let wav = args.get(2).cloned().unwrap_or_default();
            std::process::exit(app_lib::selftest_stress(minutes, &wav));
        }
        Some("--report") => std::process::exit(app_lib::selftest_report()),
        Some("--hook-probe") => {
            let text = args.get(1).cloned().unwrap_or_else(|| "aaaa bbbb cccc".into());
            std::process::exit(app_lib::selftest_hook_probe(&text));
        }
        Some("--type-probe") => std::process::exit(app_lib::selftest_type_probe(&args[1..])),
        Some("--batching-probe") => std::process::exit(app_lib::selftest_batching()),
        Some("--inject-test") => {
            let forced = if args.iter().any(|a| a == "--paste") {
                Some("paste")
            } else if args.iter().any(|a| a == "--type") {
                Some("type")
            } else {
                None
            };
            std::process::exit(app_lib::selftest_injection(forced));
        }
        _ => app_lib::run(),
    }
}
