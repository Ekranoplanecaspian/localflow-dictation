import { HEALTH_SAMPLES } from "./sampleHealth";
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
    version: "0.2.0",
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
        stt: {
            state: "ready",
            backend: "auto",
            model: "nemo-parakeet-tdt-0.6b-v3",
            key: "parakeet-v3",
            label: "Parakeet v3",
            device: "cuda",
            precision: "fp32",
            // mid-download, because that is the state with the most to lay out
            switch: { to: "parakeet-v2", state: "downloading", progress: 0.42, error: null },
            choices: [
                {
                    key: "parakeet-v3",
                    label: "Parakeet v3",
                    blurb: "The best all-rounder: the most accurate in our tests and the fastest on a graphics card. Knows 25 European languages and works out which one you are speaking.",
                    languages: 25,
                    size_gb: 2.6,
                    installed: true,
                    current: true,
                    recommended: true,
                    speed: 5,
                    accuracy: 5,
                    fit: { rating: "good", why: "quick on the graphics card (0.04 s per second of speech, estimated)" },
                    rated_on: "cuda",
                    disk_gb: 2.43,
                    removable: false,
                    download: null,
                },
                {
                    key: "parakeet-v2",
                    label: "Parakeet v2",
                    blurb: "English only. Excellent on clear, read-aloud English, but less forgiving of accents than v3.",
                    languages: 1,
                    size_gb: 2.5,
                    installed: false,
                    current: false,
                    recommended: false,
                    speed: 5,
                    accuracy: 4,
                    fit: { rating: "good", why: "quick on the graphics card (0.05 s per second of speech, estimated)" },
                    rated_on: "cuda",
                    disk_gb: 0,
                    removable: false,
                    download: "d2",
                },
                {
                    key: "parakeet-v3-compact",
                    label: "Parakeet v3 Compact",
                    blurb: "A quarter of the download and a third of the memory. For computers without an NVIDIA graphics card, where it is quicker than v3; a little less accurate.",
                    languages: 25,
                    size_gb: 0.7,
                    installed: true,
                    current: false,
                    recommended: false,
                    speed: 1,
                    accuracy: 3,
                    fit: { rating: "good", why: "quick on the processor (0.06 s per second of speech, estimated)" },
                    rated_on: "cpu",
                    disk_gb: 0.64,
                    removable: true,
                    download: null,
                },
                {
                    key: "whisper-turbo",
                    label: "Whisper Large v3 Turbo",
                    blurb: "OpenAI's model, for 99 languages. About half the speed of Parakeet on a graphics card, and far too slow without one.",
                    languages: 99,
                    size_gb: 1.6,
                    installed: false,
                    current: false,
                    recommended: false,
                    speed: 3,
                    accuracy: 4,
                    fit: { rating: "good", why: "quick on the graphics card (0.09 s per second of speech, estimated)" },
                    rated_on: "cuda",
                    disk_gb: 0,
                    removable: false,
                    download: null,
                },
            ],
        },
        llm: {
            state: "ready",
            provider: "bundled",
            model: "qwen3-4b",
            label: "Qwen3 4B",
            enabled: true,
            switch: null,
            choices: [
                {
                    key: "qwen3-4b",
                    label: "Qwen3 4B",
                    blurb: "The most careful editor in our tests: the most exact clean-ups, never undid one of your corrections, and handled every command-mode rewrite.",
                    size_gb: 2.5,
                    installed: true,
                    current: true,
                    recommended: true,
                    speed: 3,
                    accuracy: 5,
                    fit: { rating: "good", why: "quick on the graphics card (0.4 s per clean-up, measured)" },
                    rated_on: "cuda",
                    disk_gb: 2.5,
                    removable: false,
                    download: null,
                },
                {
                    key: "phi-4-mini",
                    label: "Phi-4 mini",
                    blurb: "Twice as fast and just as careful with your corrections, but a plainer editor and weaker at command-mode rewrites.",
                    size_gb: 2.5,
                    installed: true,
                    current: false,
                    recommended: false,
                    speed: 5,
                    accuracy: 4,
                    fit: { rating: "good", why: "quick on the graphics card (0.2 s per clean-up, estimated)" },
                    rated_on: "cuda",
                    disk_gb: 2.49,
                    removable: true,
                    download: null,
                },
                {
                    key: "gemma-4-e2b",
                    label: "Gemma 4 E2B",
                    blurb: "Fast, and good at command-mode rewrites. Sometimes leaves numbers spelled out (\"two point four million\"), and the largest download.",
                    size_gb: 3.4,
                    installed: false,
                    current: false,
                    recommended: false,
                    speed: 5,
                    accuracy: 3,
                    fit: { rating: "slow", why: "1.7 s per clean-up on the processor, estimated" },
                    rated_on: "cpu",
                    disk_gb: 0,
                    removable: false,
                    download: "d3",
                },
            ],
        },
        downloads: [
            { id: "d2", kind: "speech", key: "parakeet-v2", label: "Parakeet v2", state: "downloading", reason: "switch",
              cancellable: true, done: 1_050_000_000, total: 2_500_000_000, progress: 0.42, speed_bps: 12_400_000, eta_s: 117, error: null },
            { id: "d3", kind: "cleanup", key: "gemma-4-e2b", label: "Gemma 4 E2B", state: "queued", reason: "library",
              cancellable: true, done: 0, total: 3_400_000_000, progress: 0, speed_bps: 0, eta_s: null, error: null },
            { id: "d1", kind: "runtime", key: "cuda", label: "Clean-up runtime (llama.cpp)", state: "done", reason: "automatic",
              cancellable: false, done: 610_000_000, total: 610_000_000, progress: 1, speed_bps: 0, eta_s: null, error: null,
              ended_s_ago: 8 },
        ],
        // warm, with clean-up already moved off the GPU: the state with the most to lay out
        compute: {
            mode: "adaptive",
            temp_limit_c: 80,
            idle_release_min: 10,
            level: "light",
            reason: "graphics card at 82 °C",
            speech: "cuda",
            cleanup: "vulkan",
            cleanup_off_card: "vulkan",
            vulkan: "Built-in graphics",
            keep_warm: false,
            moving: null,
            gpu: {
                name: "RTX 4060 Laptop GPU",
                temp_c: 82,
                util_pct: 64,
                mem_used_mb: 4105,
                mem_total_mb: 8188,
                slowdown_c: 91,
            },
            auto: { speech: true, cleanup: true },
            chosen: {
                speech: {
                    key: "parakeet-v3",
                    label: "Parakeet v3",
                    why: "the most accurate, and quick enough on the graphics card (0.04 s per second of speech, measured)",
                },
                cleanup: {
                    key: "qwen3-4b",
                    label: "Qwen3 4B",
                    why: "the most accurate, and quick enough on the processor (0.8 s per clean-up, measured)",
                },
            },
            hardware: {
                cpu_cores: 12,
                ram_gb: 31.1,
                vram_mb: 8188,
                cpu: {
                    name: "AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
                    cores: 12,
                    threads: 24,
                    isa: ["sse4.2", "avx", "avx2", "avx512"],
                },
                gpus: [
                    {
                        name: "NVIDIA GeForce RTX 4060 Laptop GPU",
                        vendor: "nvidia",
                        dedicated_mb: 7956,
                        shared_mb: 15932,
                        integrated: false,
                    },
                    {
                        name: "AMD Radeon(TM) 890M Graphics",
                        vendor: "amd",
                        dedicated_mb: 338,
                        shared_mb: 15932,
                        integrated: true,
                    },
                ],
                ram_free_gb: 8.3,
                disk_free_gb: 191,
            },
            recent: [
                { at: Date.now() / 1000 - 240, what: "clean-up", to: "vulkan", reason: "graphics card at 82 °C" },
                { at: Date.now() / 1000 - 5400, what: "speech", to: "cuda", reason: "graphics card is cool" },
                { at: Date.now() / 1000 - 5400, what: "clean-up", to: "cuda", reason: "graphics card is cool" },
            ],
        },
        version: "0.2.0",
    },
    link: { link: "ready", detail: null, attached: false, pid: 4242, restarts: 0, safe_mode: false },
    health: HEALTH_SAMPLES.ok,
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
    bluetooth_microphones: ["Headset (WH-1000XM4)"],
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

/** `#hub?demo=1&health=gpuprep`: an NVIDIA PC's first run, speech's CUDA libraries downloading. */
export function gettingReady(): HubData {
    const compute = SAMPLE.engine?.compute;
    const llm = SAMPLE.engine?.llm;
    return {
        ...SAMPLE,
        engine: {
            ...SAMPLE.engine,
            // a first run: clean-up's model is downloading, the CUDA libraries wait their turn
            llm: llm && {
                ...llm,
                state: "loading",
                download: { id: "d5", label: "Qwen3 4B", progress: 0.36, size_gb: 2.5, state: "downloading" },
            },
            downloads: [
                { id: "d5", kind: "cleanup", key: "qwen3-4b", label: "Qwen3 4B", state: "downloading", reason: "first-run",
                  cancellable: true, done: 900_000_000, total: 2_500_000_000, progress: 0.36, speed_bps: 18_200_000,
                  eta_s: 88, error: null, ended_s_ago: null },
                { id: "d6", kind: "gpu-libs", key: "cuda", label: "Graphics card libraries (NVIDIA CUDA)", state: "queued",
                  reason: "automatic", cancellable: false, done: 0, total: 1_071_985_899, progress: 0, speed_bps: 0,
                  eta_s: null, error: null, ended_s_ago: null },
            ],
            compute: compute && {
                ...compute,
                level: "full",
                reason: "graphics card is cool",
                speech: "cpu",
                cleanup: "cuda",
                recent: [],
                cuda_libs: { state: "downloading", done: 356_515_840, total: 1_071_985_899 },
            },
        },
    };
}

/** `#onboarding?demo=1&first=1`: a PC's first minutes - speech downloading, clean-up not yet here (M5). */
export function firstRun(): HubData {
    const e = SAMPLE.engine;
    return {
        ...SAMPLE,
        settings: { ...SAMPLE.settings, onboarded: false },
        engine: e && {
            ...e,
            stt: e.stt && {
                ...e.stt,
                state: "loading",
                switch: null,
                download: { id: "d1", label: "Parakeet v3", progress: 0.31, size_gb: 2.6, state: "downloading" },
            },
            llm: e.llm && {
                ...e.llm,
                state: "loading",
                choices: e.llm.choices?.map((c) => (c.key === "qwen3-4b" ? { ...c, installed: false } : c)),
            },
            downloads: [
                { id: "d1", kind: "speech", key: "parakeet-v3", label: "Parakeet v3", state: "downloading", reason: "first-run",
                  cancellable: false, done: 806_000_000, total: 2_600_000_000, progress: 0.31, speed_bps: 16_300_000,
                  eta_s: 110, error: null, ended_s_ago: null },
            ],
            recommended: [],
        },
    };
}

/** `#hub?demo=1&recs=1`: the sample with things to recommend (M4). */
export function withRecommendations(data: HubData): HubData {
    return {
        ...data,
        engine: data.engine && {
            ...data.engine,
            recommended: [
                { kind: "speech", key: "whisper-turbo", label: "Whisper Large v3 Turbo", size_gb: 1.6, installed: false,
                  action: "download",
                  why: "Your language is set to \"hi\", which Parakeet does not know: Whisper turns it into text." },
                { kind: "cleanup", key: "phi-4-mini", label: "Phi-4 mini", size_gb: 2.5, installed: false, action: "use",
                  why: "Quick on this PC, where Qwen3 4B is slow (2.1 s per clean-up on the processor, measured)." },
            ],
        },
    };
}

/** `#hub?demo=1&ram=8`: the sample on a PC with that much memory, where clean-up starts off. */
export function smallPc(ramGb: number): HubData {
    const compute = SAMPLE.engine?.compute;
    return {
        ...SAMPLE,
        engine_config: { postprocess: { ...SAMPLE.engine_config?.postprocess, llm_cleanup: false } },
        engine: {
            ...SAMPLE.engine,
            compute: compute && {
                ...compute,
                hardware: compute.hardware && { ...compute.hardware, ram_gb: ramGb - 0.4, ram_free_gb: 2.1 },
                chosen: {
                    cleanup: null,
                    speech: {
                        key: "parakeet-v3-compact",
                        label: "Parakeet v3 Compact",
                        why: "Parakeet v3 needs about 3.0 GB of free memory, and 2.1 GB is free; this is the most accurate one that fits and keeps up on the processor (0.12 s per second of speech, estimated)",
                    },
                },
            },
        },
    };
}
