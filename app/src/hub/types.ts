export type Phase = "idle" | "recording" | "finishing";
export type Link = "starting" | "connecting" | "ready" | "reconnecting" | "failed" | "stopped";

export type ModelState = {
    state: "loading" | "ready" | "error" | "off";
    error?: string | null;
    backend?: string;
    model?: string;
    device?: string;
    precision?: string;
    provider?: string;
    enabled?: boolean;
};

/** Per-application overrides. Every field defaults to "leave it alone". */
export type AppRule = {
    disabled: boolean;
    /** "" or "auto" lets the injector choose; "type" or "paste" forces it. */
    method: string;
    auto_send: boolean;
    /** "" or "auto" lets the engine guess the style from the app name. */
    profile: string;
};

export type Settings = {
    hotkey: string[];
    double_tap_hands_free: boolean;
    double_tap_ms: number;
    escape_cancels: boolean;
    history: boolean;
    retention_days: number;
    flow_bar: boolean;
    microphone: string;
    trailing_space: boolean;
    command_mode: boolean;
    command_hotkey: string[];
    hands_free_timeout_s: number;
    app_rules: Record<string, AppRule>;
    onboarded: boolean;
};

/** The engine's clean-up settings, as they appear in its own config file. */
export type PostProcess = {
    remove_fillers?: boolean;
    spoken_newlines?: boolean;
    dictionary?: Record<string, string>;
    dictionary_terms?: string[];
    snippets?: Record<string, string>;
    custom_instructions?: string;
    llm_cleanup?: boolean;
    llm_provider?: string;
    llm_model?: string;
    llm_min_words?: number;
    llm_url?: string;
    llm_api_key?: string;
};

export type Entry = {
    at: number;
    raw: string;
    text: string;
    app: string;
    words: number;
    audio_s: number;
    ms: number;
    used_llm: boolean;
};

export type Stats = {
    dictations: number;
    words: number;
    audio_s: number;
    p50_ms: number;
    p95_ms: number;
    words_per_minute: number;
    with_auto_edits: number;
    daily: number[];
    top_apps: [string, number][];
};

/** A correction the clean-up keeps making, offered as a dictionary entry. */
export type Suggestion = {
    from: string;
    to: string;
    seen: number;
};

export type HubData = {
    settings: Settings;
    engine_config: { postprocess?: PostProcess } | null;
    engine: { stt?: ModelState; llm?: ModelState; version?: string } | null;
    link: { link: Link; detail: string | null; attached: boolean; pid: number | null; restarts: number };
    stats: Stats;
    history: Entry[];
    history_total: number;
    microphones: string[];
    suggestions: Suggestion[];
    paths: { settings: string | null; history: string; log: string };
};

export type SectionProps = {
    data: HubData;
    onChange: () => void | Promise<void>;
    say: (message: string) => void;
};
