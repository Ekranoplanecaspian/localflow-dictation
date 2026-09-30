export type Phase = "idle" | "recording" | "finishing";
export type Link = "starting" | "connecting" | "ready" | "reconnecting" | "failed" | "stopped";

export type ModelState = {
    /** "asleep": clean-up unloaded while LocalFlow is idle; the next dictation wakes it. */
    state: "loading" | "ready" | "error" | "off" | "asleep";
    error?: string | null;
    backend?: string;
    model?: string;
    device?: string;
    precision?: string;
    provider?: string;
    enabled?: boolean;
    /** The catalogue entry in use, when the settings match one. */
    key?: string | null;
    label?: string;
    choices?: ModelChoice[];
    switch?: ModelSwitch | null;
    /** What this part waits for before it works at all: its model's (or runtime's) download. */
    download?: {
        id: string;
        label: string;
        progress: number;
        size_gb: number;
        state: "queued" | "downloading";
    } | null;
};

/** How a model suits this PC, on the device it would run on (the engine's model choice). */
export type Fit = { rating: "good" | "slow" | "too-big"; why: string };

/** One model the engine can move to, as its catalogue describes it. */
export type ModelChoice = {
    key: string;
    label: string;
    blurb: string;
    /** Speech models only. */
    languages?: number;
    size_gb: number;
    installed: boolean;
    current: boolean;
    recommended: boolean;
    /** 1 to 5, measured on the benchmark sets rather than guessed. */
    speed?: number;
    accuracy?: number;
    /** The library (M1): how it suits this PC, what it takes on disk, and whether it can go. */
    fit?: Fit | null;
    rated_on?: Device | null;
    disk_gb?: number;
    removable?: boolean;
    /** The id of its download, while one is queued or running. */
    download?: string | null;
};

/** A model that would make dictation better on this PC and is not in use (M4). */
export type Recommendation = {
    kind: "speech" | "cleanup";
    key: string;
    label: string;
    why: string;
    size_gb: number;
    installed: boolean;
    /** use: switch to it (downloading first); download: fetch it for Automatic; enable: turn auto-edits on */
    action: "use" | "download" | "enable";
};

/** One download in the engine's queue (M1). */
export type Download = {
    id: string;
    kind: "speech" | "cleanup" | "runtime" | "gpu-libs";
    key: string;
    label: string;
    state: "queued" | "downloading" | "done" | "error" | "cancelled";
    reason: "first-run" | "switch" | "automatic" | "library";
    cancellable: boolean;
    done: number;
    total: number;
    progress: number;
    speed_bps: number;
    eta_s: number | null;
    error: string | null;
    /** Seconds since it finished; null while it is queued or running. */
    ended_s_ago?: number | null;
};

export type ModelSwitch = {
    to: string;
    state: "downloading" | "loading" | "error";
    progress: number;
    error?: string | null;
};

/** "vulkan": AMD or Intel graphics, which only clean-up uses (B3). */
export type Device = "cuda" | "cpu" | "vulkan";

/** Where the models run, and why - the engine's placement controller. */
export type ComputeStatus = {
    mode: "adaptive" | "gpu" | "cpu";
    temp_limit_c: number;
    idle_release_min: number;
    level: "full" | "gentle" | "light" | "off";
    reason: string;
    speech: Device;
    cleanup: Device;
    keep_warm: boolean;
    /** What is being moved right now, e.g. "clean-up to the processor". */
    moving: string | null;
    gpu: {
        name: string | null;
        temp_c: number;
        util_pct: number;
        mem_used_mb: number;
        mem_total_mb: number;
        slowdown_c: number | null;
    } | null;
    recent: { at: number; what: string; to: Device; model?: string | null; reason: string }[];
    /** Whether LocalFlow picks the model, per kind, and what it picked and why. */
    auto?: { speech: boolean; cleanup: boolean };
    chosen?: { speech: AutoPick | null; cleanup: AutoPick | null };
    hardware?: Hardware | null;
    /** Where clean-up goes when it leaves the NVIDIA card: the processor, or other graphics. */
    cleanup_off_card?: Device;
    /** An NVIDIA PC's first-run download of speech's CUDA libraries (B2); null when none. */
    cuda_libs?: { state: "downloading" | "ready" | "error"; done?: number; total?: number; error?: string } | null;
    /** What "vulkan" is on this machine: "Built-in graphics" or a card's name; null with none. */
    vulkan?: string | null;
};

/** What this computer has (the engine's capability report); the later fields need engine 0.2. */
export type Hardware = {
    cpu_cores: number;
    ram_gb: number;
    /** The NVIDIA card's memory; null with none LocalFlow can use. */
    vram_mb: number | null;
    cpu?: { name: string; cores: number; threads: number; isa: string[] | null };
    gpus?: GraphicsAdapter[];
    ram_free_gb?: number | null;
    disk_free_gb?: number | null;
};

export type GraphicsAdapter = {
    name: string;
    vendor: "nvidia" | "amd" | "intel" | "qualcomm" | "other";
    dedicated_mb: number;
    shared_mb: number;
    integrated: boolean;
};

export type AutoPick = { key: string; label: string; why: string };

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

/** The status model (health.rs): every part ok, degraded or failed, why, and the fix. */
export type Level = "ok" | "off" | "waiting" | "starting" | "degraded" | "failed";

export type HealthPart = {
    id: string;
    name: string;
    level: Level;
    summary: string;
    headline: string;
    reason: string | null;
    action: { id: string; label: string } | null;
    critical: boolean;
    /** The problem it is in (shared/problems.json); null while it works. */
    code: string | null;
};

export type Health = {
    overall: Level;
    headline: string;
    parts: HealthPart[];
};

/** One line of "Check LocalFlow" (selfcheck.rs + the engine's selfcheck.py). */
export type CheckItem = {
    id: string;
    name: string;
    status: "ok" | "warn" | "fail" | "skip";
    detail: string;
    code?: string;
    title?: string;
    message?: string;
    action?: { id: string; label: string };
};

export type HubData = {
    version: string;
    settings: Settings;
    engine_config: { postprocess?: PostProcess } | null;
    engine: {
        stt?: ModelState;
        llm?: ModelState;
        /** Every download: running and queued first, then the last few finished. */
        downloads?: Download[];
        /** Models that would make dictation better on this PC. */
        recommended?: Recommendation[];
        compute?: ComputeStatus;
        /** Where downloads come from: the mirror ("" for huggingface.co) and Windows' proxy. */
        network?: { hf_endpoint: string; proxy: string | null };
        version?: string;
    } | null;
    link: {
        link: Link;
        detail: string | null;
        attached: boolean;
        pid: number | null;
        restarts: number;
        /** The engine kept crashing, so it runs on the processor without AI clean-up. */
        safe_mode?: boolean;
    };
    /** Null only for the moment before the shell's first assessment. */
    health: Health | null;
    stats: Stats;
    history: Entry[];
    history_total: number;
    microphones: string[];
    /** Those of `microphones` that belong to a Bluetooth headset (call-quality audio). */
    bluetooth_microphones?: string[];
    suggestions: Suggestion[];
    paths: { settings: string | null; history: string; log: string };
};

export type SectionProps = {
    data: HubData;
    onChange: () => void | Promise<void>;
    say: (message: string) => void;
};
