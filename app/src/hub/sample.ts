import type { HubData } from "./types";

/**
 * Stand-in data for `#hub?demo=1`.
 *
 * The Hub reads everything through one `invoke("hub_data")`, which throws outside Tauri, so in
 * a browser it never gets past "Loading..." - and that is precisely why its pages shipped
 * unreviewed. The flow bar got a demo route for the same reason and it immediately turned up
 * two layout bugs.
 *
 * Deliberately not tidy sample data: a history entry whose clean-up changed the words, an app
 * name long enough to test the chart labels, and an engine that is still loading, because those
 * are the states that break a layout.
 */
export const SAMPLE: HubData = {
    settings: {
        hotkey: ["ctrl", "win"],
        double_tap_hands_free: true,
        double_tap_ms: 400,
        escape_cancels: true,
        history: true,
        retention_days: 90,
        flow_bar: true,
        microphone: "",
        trailing_space: true,
        command_mode: true,
        command_hotkey: ["win", "alt"],
        hands_free_timeout_s: 8,
        app_rules: {
            "ms-teams.exe": { disabled: false, method: "", auto_send: true, profile: "chat" },
            "windowsterminal.exe": { disabled: true, method: "", auto_send: false, profile: "" },
        },
        onboarded: true,
    },
    engine_config: {
        postprocess: {
            remove_fillers: true,
            spoken_newlines: true,
            dictionary: { parakeet: "Parakeet", kwen: "Qwen" },
            dictionary_terms: ["LocalFlow", "Parakeet", "Qwen"],
            snippets: { sig: "Best,\nArnab" },
            custom_instructions: "",
            llm_cleanup: true,
            llm_provider: "bundled",
            llm_model: "qwen3-4b",
            llm_min_words: 6,
        },
    },
    engine: {
        stt: { state: "ready", backend: "parakeet", model: "nemo-parakeet-tdt-0.6b-v3", device: "cuda", precision: "fp32" },
        llm: { state: "ready", provider: "bundled", model: "qwen3-4b", enabled: true },
        version: "0.1.0",
    },
    link: { link: "ready", detail: null, attached: false, pid: 4242, restarts: 0 },
    stats: {
        dictations: 148,
        words: 5230,
        audio_s: 2140,
        p50_ms: 312,
        p95_ms: 688,
        words_per_minute: 147,
        with_auto_edits: 96,
        daily: [4, 11, 0, 23, 17, 9, 31, 12, 0, 6, 19, 22, 8, 14],
        top_apps: [
            ["brave.exe", 84],
            ["code.exe", 31],
            ["ms-teams.exe", 18],
            ["windowsterminal.exe", 9],
            ["some-very-long-application-name.exe", 6],
        ],
    },
    history: [
        {
            at: Date.now() / 1000 - 300,
            raw: "um so we should probably ship this on tuesday if the tests pass",
            text: "We should ship this on Tuesday if the tests pass.",
            app: "brave.exe",
            words: 10,
            audio_s: 4.2,
            ms: 318,
            used_llm: true,
        },
        {
            at: Date.now() / 1000 - 4000,
            raw: "hello darkness my old friend",
            text: "Hello darkness, my old friend.",
            app: "code.exe",
            words: 5,
            audio_s: 2.6,
            ms: 104,
            used_llm: false,
        },
        {
            at: Date.now() / 1000 - 90000,
            raw: "add a note that the meeting moved to thursday at half past two",
            text: "Add a note that the meeting moved to Thursday at 2:30.",
            app: "ms-teams.exe",
            words: 11,
            audio_s: 5.1,
            ms: 402,
            used_llm: true,
        },
    ],
    history_total: 148,
    microphones: ["Microphone Array (Realtek(R) Audio)", "Headset (WH-1000XM4)"],
    suggestions: [
        { from: "kwen", to: "Qwen", seen: 7 },
        { from: "parakeet", to: "Parakeet", seen: 5 },
        { from: "okonkwo", to: "Okonkwo", seen: 3 },
    ],
    paths: {
        settings: "C:\\Users\\you\\AppData\\Roaming\\LocalFlow\\shell.json",
        history: "C:\\Users\\you\\AppData\\Roaming\\LocalFlow\\history.jsonl",
        log: "C:\\Users\\you\\AppData\\Roaming\\LocalFlow\\shell.log",
    },
};
