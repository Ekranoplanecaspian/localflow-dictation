//! Tests for the status model (`health.rs`), kept beside it: `assess` and `Notifier` are pure,
//! so every situation is built from plain values.

use super::*;

fn link(state: Link) -> LinkState {
    LinkState { link: state, detail: None, attached: false, pid: Some(1), restarts: 0, safe_mode: false }
}

fn working() -> Value {
    json!({
        "stt": {"state": "ready", "label": "Parakeet v3", "device": "cuda"},
        "llm": {"state": "ready", "label": "Qwen3 4B", "enabled": true, "provider": "bundled"},
        "compute": {"mode": "adaptive", "level": "full", "reason": "graphics card is cool",
                    "speech": "cuda", "cleanup": "cuda", "moving": null,
                    "gpu": {"name": "RTX 4060 Laptop GPU", "temp_c": 55}},
    })
}

fn mic() -> Mic {
    Mic { device: Some("Microphone Array (Realtek)".into()), ..Mic::default() }
}

fn assess_with(l: &LinkState, status: Option<&Value>, m: &Mic, hook: bool) -> Health {
    assess(&Inputs {
        link: l,
        status,
        mic: m,
        hook_installed: hook,
        hotkey: "Ctrl + Win",
        mic_blocked: false,
        settings_unwritable: None,
    })
}

fn get<'h>(h: &'h Health, id: &str) -> &'h Part {
    h.parts.iter().find(|p| p.id == id).unwrap()
}

#[test]
fn a_working_app_is_ok_throughout() {
    let s = working();
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    assert_eq!(h.overall, Level::Ok);
    assert_eq!(h.headline, "Everything is working");
    assert!(h.parts.iter().all(|p| p.level == Level::Ok), "{h:#?}");
    let ids: Vec<_> = h.parts.iter().map(|p| p.id).collect();
    assert_eq!(ids, ["engine", "speech", "cleanup", "gpu", "microphone", "hotkey", "storage"]);
}

#[test]
fn a_failed_clean_up_model_degrades_with_a_way_back() {
    let mut s = working();
    s["llm"]["state"] = json!("error");
    s["llm"]["error"] = json!("llama-server exited with code 3");
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    assert_eq!(h.overall, Level::Degraded, "dictation still works");
    let c = get(&h, "cleanup");
    assert_eq!(c.level, Level::Degraded);
    assert_eq!(h.headline, "AI clean-up is unavailable");
    assert_eq!(
        c.reason.as_deref(),
        Some("Llama-server exited with code 3. Dictation goes on with the basic clean-up rules.")
    );
    assert_eq!(c.action.as_ref().map(|a| a.id), Some("restart_engine"));
}

#[test]
fn clean_up_turned_off_or_resting_is_not_a_problem() {
    let mut s = working();
    s["llm"]["enabled"] = json!(false);
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    assert_eq!((get(&h, "cleanup").level, h.overall), (Level::Off, Level::Ok));
    s["llm"]["enabled"] = json!(true);
    s["llm"]["state"] = json!("asleep");
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    assert_eq!((get(&h, "cleanup").level, h.overall), (Level::Ok, Level::Ok));
}

#[test]
fn clean_up_downloading_its_model_says_so_rather_than_loading() {
    let mut s = working();
    s["llm"]["state"] = json!("loading");
    s["llm"]["download"] = json!({"id": "d4", "label": "Qwen3 4B", "progress": 0.456, "size_gb": 2.5, "state": "downloading"});
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    let c = get(&h, "cleanup");
    assert_eq!((c.level, c.summary.as_str(), c.headline.as_str()), (Level::Starting, "Downloading 46 %", "Downloading Qwen3 4B (2.5 GB)"));
    assert!(c.reason.as_deref().unwrap_or("").starts_with("Auto-edits start as soon as it is here"));
    s["llm"]["download"]["state"] = json!("queued");
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    assert_eq!(get(&h, "cleanup").summary, "Waiting to download");
}

#[test]
fn an_engine_that_cannot_start_is_one_problem_not_four() {
    let mut l = link(Link::Failed);
    l.detail = Some("the engine exited (exit code: 1)".into());
    let h = assess_with(&l, None, &mic(), true);
    assert_eq!(h.overall, Level::Failed);
    assert_eq!(h.headline, "LocalFlow can't start its engine");
    assert_eq!(get(&h, "engine").action.as_ref().map(|a| a.label), Some("Restart engine"));
    for id in ["speech", "cleanup", "gpu"] {
        assert_eq!(get(&h, id).level, Level::Waiting, "{id} is not a separate problem");
    }
}

#[test]
fn starting_up_is_neither_ok_nor_a_problem() {
    let h = assess_with(&link(Link::Starting), None, &Mic::default(), true);
    assert_eq!(h.overall, Level::Starting);
    let mut s = working();
    s["stt"]["state"] = json!("loading");
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    assert_eq!((h.overall, get(&h, "speech").level), (Level::Starting, Level::Starting));
}

#[test]
fn safe_mode_is_degraded_and_offers_the_way_out() {
    let mut l = link(Link::Ready);
    l.safe_mode = true;
    let s = working();
    let h = assess_with(&l, Some(&s), &mic(), true);
    assert_eq!(h.overall, Level::Degraded);
    assert_eq!(get(&h, "engine").action.as_ref().map(|a| a.label), Some("Leave safe mode"));
    assert_eq!(get(&h, "cleanup").level, Level::Off, "off in safe mode, not a second problem");
}

#[test]
fn the_microphone_failing_stops_dictation_and_points_at_the_privacy_setting() {
    let s = working();
    let m = Mic {
        device: None,
        error: Some("no microphone: is one plugged in and allowed?".into()),
        ..Mic::default()
    };
    let h = assess_with(&link(Link::Ready), Some(&s), &m, true);
    assert_eq!(h.overall, Level::Failed);
    assert_eq!(get(&h, "microphone").action.as_ref().map(|a| a.id), Some("privacy_microphone"));
}

#[test]
fn a_microphone_another_app_holds_is_named_as_such() {
    let s = working();
    let m = Mic {
        device: Some("Microphone Array (Realtek)".into()),
        error: Some("Failed to initialize audio client: OS Error -2004287478".into()),
        busy: true,
        ..Mic::default()
    };
    let h = assess_with(&link(Link::Ready), Some(&s), &m, true);
    let p = get(&h, "microphone");
    assert_eq!((p.level, p.code), (Level::Failed, Some("mic-in-use")));
    assert!(p.reason.as_deref().unwrap().starts_with("Microphone Array (Realtek) is being used by another app"));
    assert!(!p.reason.as_deref().unwrap().contains("OS Error"), "no raw error text");
    assert_eq!(p.action.as_ref().map(|a| a.id), Some("sound"));
}

#[test]
fn a_bluetooth_microphone_works_and_says_what_it_costs() {
    let s = working();
    let m = Mic { device: Some("Headset (HD 450BT)".into()), bluetooth: true, ..Mic::default() };
    let p = get(&assess_with(&link(Link::Ready), Some(&s), &m, true), "microphone").clone();
    assert_eq!(p.level, Level::Ok);
    assert!(p.reason.as_deref().unwrap().contains("call quality"));
}

#[test]
fn a_missing_chosen_microphone_is_degraded_not_failed() {
    let s = working();
    let m = Mic { chosen: "Blue Yeti".into(), ..mic() };
    let h = assess_with(&link(Link::Ready), Some(&s), &m, true);
    let p = get(&h, "microphone");
    assert_eq!((p.level, h.overall), (Level::Degraded, Level::Degraded));
    assert!(p.reason.as_deref().unwrap().starts_with("Blue Yeti isn't connected"));
    let m = Mic { chosen: "realtek".into(), ..mic() };
    assert_eq!(get(&assess_with(&link(Link::Ready), Some(&s), &m, true), "microphone").level, Level::Ok);
}

#[test]
fn work_moved_off_the_graphics_card_on_purpose_is_ok_with_a_note() {
    let mut s = working();
    s["compute"]["level"] = json!("light");
    s["compute"]["cleanup"] = json!("cpu");
    s["compute"]["reason"] = json!("graphics card at 81 °C");
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    let g = get(&h, "gpu");
    assert_eq!((g.level, h.overall), (Level::Ok, Level::Ok));
    assert_eq!(g.reason.as_deref(), Some("Graphics card at 81 °C."));
}

#[test]
fn speech_that_could_not_use_the_graphics_card_is_degraded() {
    let mut s = working();
    s["stt"]["device"] = json!("cpu");
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    assert_eq!(get(&h, "gpu").level, Level::Degraded);
    // ...but not while it is on its way there
    s["compute"]["moving"] = json!("speech");
    assert_eq!(get(&assess_with(&link(Link::Ready), Some(&s), &mic(), true), "gpu").level, Level::Ok);
}

#[test]
fn no_nvidia_card_or_processor_only_is_fine() {
    let mut s = working();
    s["compute"]["gpu"] = Value::Null;
    s["stt"]["device"] = json!("cpu");
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    assert_eq!((get(&h, "gpu").level, h.overall), (Level::Ok, Level::Ok));
}

#[test]
fn a_hook_windows_would_not_install_fails() {
    let s = working();
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), false);
    assert_eq!((get(&h, "hotkey").level, h.overall), (Level::Failed, Level::Failed));
}

#[test]
fn a_failed_model_switch_is_degraded_and_the_old_model_goes_on() {
    let mut s = working();
    s["stt"]["switch"] = json!({"to": "whisper-turbo", "label": "Whisper Large v3 Turbo",
                                "state": "error", "error": "download failed"});
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    let p = get(&h, "speech");
    assert_eq!(p.level, Level::Degraded);
    assert_eq!(p.headline, "Couldn't switch to Whisper Large v3 Turbo");
    assert_eq!(p.reason.as_deref(), Some("Download failed. Dictation goes on with Parakeet v3."));
}

#[test]
fn problems_are_announced_once_after_ten_seconds_and_recovery_once() {
    let t0 = Instant::now();
    let at = |s: u64| t0 + Duration::from_secs(s);
    let mut n = Notifier::default();
    let s = working();
    let ok = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    let mut b = working();
    b["llm"]["state"] = json!("error");
    let broken = assess_with(&link(Link::Ready), Some(&b), &mic(), true);
    assert!(n.tick(&ok, at(0)).is_empty());
    assert!(n.tick(&broken, at(1)).is_empty(), "not yet: it may clear by itself");
    assert!(n.tick(&broken, at(10)).is_empty());
    let said = n.tick(&broken, at(11));
    assert_eq!(said.len(), 1);
    assert_eq!(said[0].0, "AI clean-up is unavailable");
    assert!(said[0].1.ends_with("The fix is on the Overview page in LocalFlow."), "{said:?}");
    assert!(n.tick(&broken, at(60)).is_empty(), "never repeated");
    assert!(n.tick(&ok, at(61)).is_empty(), "recovery waits a moment too");
    let back = n.tick(&ok, at(66));
    assert_eq!(back.len(), 1);
    assert_eq!(back[0].0, "LocalFlow is working normally again");
    assert!(n.tick(&ok, at(200)).is_empty());
}

#[test]
fn a_short_engine_restart_is_never_announced() {
    let t0 = Instant::now();
    let at = |s: u64| t0 + Duration::from_secs(s);
    let mut n = Notifier::default();
    let s = working();
    let ok = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    let down = assess_with(&link(Link::Reconnecting), None, &mic(), true);
    let starting = assess_with(&link(Link::Starting), None, &mic(), true);
    let mut said = n.tick(&ok, at(0));
    said.extend(n.tick(&down, at(1)));
    said.extend(n.tick(&starting, at(4)));
    said.extend(n.tick(&ok, at(6)));
    said.extend(n.tick(&ok, at(30)));
    assert!(said.is_empty(), "{said:?}");
}

#[test]
fn a_problem_that_gets_worse_is_announced_again() {
    let t0 = Instant::now();
    let at = |s: u64| t0 + Duration::from_secs(s);
    let mut n = Notifier::default();
    let mut l = link(Link::Ready);
    l.safe_mode = true;
    let s = working();
    let safe = assess_with(&l, Some(&s), &mic(), true);
    let failed = assess_with(&link(Link::Failed), None, &mic(), true);
    n.tick(&safe, at(0));
    assert_eq!(n.tick(&safe, at(10)).len(), 1);
    n.tick(&failed, at(11));
    assert_eq!(n.tick(&failed, at(21)).len(), 1);
}

#[test]
fn a_press_is_refused_with_the_actual_reason() {
    let s = working();
    let ok = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    assert_eq!(refusal(&link(Link::Ready), true, Some(&ok), None), None);
    assert_eq!(refusal(&link(Link::Ready), true, None, None), None, "no assessment yet is no reason to refuse");

    let said = |r: Option<(&'static str, String)>| r.map(|(code, text)| (code, text));
    for state in [Link::Starting, Link::Connecting, Link::Reconnecting] {
        assert_eq!(
            said(refusal(&link(state), false, None, None)),
            Some(("engine-restarting", "Engine restarting — try again in a moment".into()))
        );
    }
    assert_eq!(
        said(refusal(&link(Link::Failed), false, None, None)),
        Some(("engine-wont-start", "LocalFlow's engine isn't running — see LocalFlow".into()))
    );
    let mut missing = link(Link::Failed);
    missing.detail = Some("could not start the engine process: program not found (os error 2)".into());
    assert_eq!(refusal(&missing, false, None, None).map(|r| r.0), Some("engine-missing"));
    assert_eq!(
        said(refusal(&link(Link::Ready), false, Some(&ok), None)),
        Some(("speech-loading", "Warming up — try again in a moment".into()))
    );

    let mut e = working();
    e["stt"]["state"] = json!("error");
    e["stt"]["error_code"] = json!("speech-download-failed");
    let no_speech = assess_with(&link(Link::Ready), Some(&e), &mic(), true);
    assert_eq!(
        said(refusal(&link(Link::Ready), false, Some(&no_speech), None)),
        Some(("speech-download-failed", "The speech model isn't downloaded yet — see LocalFlow".into()))
    );

    let m = Mic { device: None, error: Some("no microphone".into()), ..Mic::default() };
    let no_mic = assess_with(&link(Link::Ready), Some(&s), &m, true);
    assert_eq!(
        said(refusal(&link(Link::Ready), true, Some(&no_mic), None)),
        Some(("mic-unavailable", "Microphone unavailable — see LocalFlow".into()))
    );

    // A chosen microphone that is missing still records with the default: not refused.
    let fallback = assess_with(&link(Link::Ready), Some(&s), &Mic { chosen: "Blue Yeti".into(), ..mic() }, true);
    assert_eq!(refusal(&link(Link::Ready), true, Some(&fallback), None), None);
}

#[test]
fn a_press_during_the_first_download_says_how_far_it_has_got() {
    let mut s = working();
    s["stt"] = json!({"state": "loading", "label": "Parakeet v3",
                      "download": {"id": "d1", "label": "Parakeet v3", "progress": 0.453, "size_gb": 2.6, "state": "downloading"}});
    s["downloads"] = json!([{"id": "d1", "state": "downloading", "eta_s": 130}]);
    let words = speech_download_words(&s);
    assert_eq!(words.as_deref(), Some("45 %, about 2 min left"));
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    assert_eq!(
        refusal(&link(Link::Ready), false, Some(&h), words.as_deref()),
        Some(("speech-downloading", "Still downloading the speech model — 45 %, about 2 min left".into()))
    );
    s["stt"]["download"]["state"] = json!("queued");
    assert_eq!(speech_download_words(&s).as_deref(), Some("waiting to start"));
    assert_eq!(speech_download_words(&working()), None, "no download: loading, as before");
}

fn with_inputs(status: &Value, mic_blocked: bool, settings_unwritable: Option<(String, String)>) -> Health {
    assess(&Inputs {
        link: &link(Link::Ready),
        status: Some(status),
        mic: &mic(),
        hook_installed: true,
        hotkey: "Ctrl + Win",
        mic_blocked,
        settings_unwritable,
    })
}

#[test]
fn a_microphone_blocked_by_windows_fails_even_though_its_stream_opened() {
    let s = working();
    let h = with_inputs(&s, true, None);
    let p = get(&h, "microphone");
    assert_eq!((p.level, p.code, h.overall), (Level::Failed, Some("mic-blocked"), Level::Failed));
    assert_eq!(p.action.as_ref().map(|a| a.id), Some("privacy_microphone"));
    assert_eq!(
        refusal(&link(Link::Ready), true, Some(&h), None),
        Some(("mic-blocked", "Microphone blocked — see LocalFlow".into()))
    );
}

#[test]
fn storage_problems_are_degraded_never_fatal() {
    let mut s = working();
    assert_eq!(get(&with_inputs(&s, false, None), "storage").level, Level::Ok);
    s["checks"] = json!([{"id": "disk", "status": "warn", "code": "disk-low", "detail": "1.2 GB free on C:.",
                          "vars": {"free": "1.2 GB", "drive": "C:", "needed": "5 GB"}}]);
    let h = with_inputs(&s, false, None);
    let p = get(&h, "storage");
    assert_eq!((p.level, p.code, h.overall), (Level::Degraded, Some("disk-low"), Level::Degraded));
    assert_eq!(p.summary, "1.2 GB free");
    let h = with_inputs(&working(), false, Some(("C:\\Users\\x\\AppData\\Roaming\\LocalFlow".into(), "Access is denied.".into())));
    let p = get(&h, "storage");
    assert_eq!(p.code, Some("folder-not-writable"));
    assert!(p.reason.as_deref().unwrap().contains("Access is denied."), "{:?}", p.reason);
}

#[test]
fn an_old_nvidia_driver_shows_on_the_graphics_card() {
    let mut s = working();
    s["checks"] = json!([{"id": "driver", "status": "warn", "code": "driver-too-old", "detail": "Version 561.09 is installed.",
                          "vars": {"version": "561.09", "needed": "580"}}]);
    let p = get(&with_inputs(&s, false, None), "gpu").clone();
    assert_eq!((p.level, p.code), (Level::Degraded, Some("driver-too-old")));
    assert!(p.reason.unwrap().starts_with("Version 561.09 is installed, and LocalFlow's graphics-card code needs 580"));
}

/// A first run's download takes minutes: the Status card says so, with how far it has got.
#[test]
fn a_first_run_download_shows_its_progress() {
    let mut s = working();
    s["stt"]["state"] = json!("loading");
    s["stt"]["download"] = json!({"label": "Parakeet v3", "progress": 0.34, "size_gb": 2.6});
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    let p = get(&h, "speech");
    assert_eq!((p.level, p.summary.as_str()), (Level::Starting, "Downloading 34 %"));
    assert!(p.headline.ends_with("(2.6 GB)"), "{}", p.headline);
}

/// A company policy or antivirus stopping the engine is named as that - not "it stopped while
/// starting", which sends people restarting something that can never start.
#[test]
fn an_engine_windows_refuses_to_run_is_named_as_blocked() {
    for code in [1260, 4551, 225] {
        let l = LinkState {
            detail: Some(format!(
                "could not start the engine process: This program is blocked by group policy. (os error {code})"
            )),
            ..link(Link::Failed)
        };
        let h = assess_with(&l, None, &mic(), true);
        let p = get(&h, "engine");
        assert_eq!((p.level, p.code), (Level::Failed, Some("engine-blocked")), "os error {code}");
        assert!(p.reason.as_deref().unwrap().contains("AppLocker"));
    }
    let l = LinkState { detail: Some("it exited with code 1".into()), ..link(Link::Failed) };
    assert_eq!(get(&assess_with(&l, None, &mic(), true), "engine").code, Some("engine-wont-start"));
}

/// After a downgrade the Status card says why changes are not being kept.
#[test]
fn settings_from_a_newer_version_are_said() {
    let mut s = working();
    s["settings_newer"] = json!(5);
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    let p = get(&h, "storage");
    assert_eq!((p.level, p.code), (Level::Degraded, Some("settings-newer")));
}

/// B2: an NVIDIA PC's first run downloads speech's CUDA libraries; meanwhile speech is on the
/// processor on purpose, which is not "the graphics card could not be used".
#[test]
fn the_graphics_card_says_it_is_getting_ready_and_what_failed() {
    let mut s = working();
    s["stt"]["device"] = json!("cpu");
    s["compute"]["speech"] = json!("cpu");
    s["compute"]["cuda_libs"] = json!({"state": "downloading", "done": 356_515_840u64, "total": 1_071_985_899u64});
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    let p = get(&h, "gpu");
    assert_eq!((p.level, p.summary.as_str()), (Level::Ok, "Getting ready"));
    assert_eq!(p.headline, "Getting RTX 4060 Laptop GPU ready for speech");
    assert_eq!(
        p.reason.as_deref(),
        Some("Downloading what speech needs to run on it: 340 of 1022 MB. Until then speech runs on the processor.")
    );

    s["compute"]["cuda_libs"] = json!({"state": "error", "error": "No connection to the internet."});
    let h = assess_with(&link(Link::Ready), Some(&s), &mic(), true);
    let p = get(&h, "gpu");
    assert_eq!((p.level, p.code), (Level::Degraded, Some("gpu-libs-download-failed")));
    assert!(p.reason.as_deref().unwrap().contains("No connection to the internet. Until they arrive"), "{:?}", p.reason);
}
