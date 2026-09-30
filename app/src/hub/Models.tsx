import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Compute } from "./Compute";
import { ModelPicker } from "./ModelPicker";
import type { ComputeStatus, HubData, PostProcess, SectionProps } from "./types";

type EngineStatus = NonNullable<HubData["engine"]>;

/** Below this much RAM (GB, as Windows reports an "8 GB" PC) clean-up starts off; see hwinfo.py. */
const LOW_MEMORY_GB = 9;

const PROVIDERS = [
    { id: "bundled", label: "On this computer", local: true },
    { id: "ollama", label: "Ollama", local: true },
    { id: "openai", label: "OpenAI-compatible endpoint", local: false },
    { id: "anthropic", label: "Anthropic", local: false },
];

export function Models({ data, onChange, say }: SectionProps) {
    const saved = data.engine_config?.postprocess ?? {};
    const [draft, setDraft] = useState<PostProcess>(saved);
    const [instructions, setInstructions] = useState(saved.custom_instructions ?? "");
    // The Hub refreshes every few seconds; a download wants smoother progress than that, so
    // the engine's own status messages are followed as they arrive.
    const [live, setLive] = useState<EngineStatus | null>(data.engine);

    useEffect(() => setLive(data.engine), [data.engine]);
    const [mirror, setMirror] = useState(data.engine?.network?.hf_endpoint ?? "");
    useEffect(() => setMirror(data.engine?.network?.hf_endpoint ?? ""), [data.engine?.network?.hf_endpoint]);
    const saveMirror = async (hf_endpoint: string) => {
        try {
            await invoke("save_engine_settings", { network: { hf_endpoint } });
            say(hf_endpoint ? `models will download from ${hf_endpoint}` : "models will download from huggingface.co");
        } catch (e) {
            say(String(e));
        }
    };

    useEffect(() => {
        if (!("__TAURI_INTERNALS__" in window)) return; // the browser demo has no engine
        const un = listen<EngineStatus>("engine-status", (e) => setLive(e.payload));
        return () => void un.then((f) => f());
    }, []);

    useEffect(() => {
        const next = data.engine_config?.postprocess ?? {};
        setDraft(next);
        setInstructions(next.custom_instructions ?? "");
    }, [data.engine_config]);

    const push = async (patch: PostProcess) => {
        setDraft({ ...draft, ...patch });
        try {
            await invoke("save_engine_settings", { postprocess: patch });
            say("saved");
            window.setTimeout(onChange, 400);
        } catch (e) {
            say(String(e));
        }
    };

    const provider = draft.llm_provider ?? "bundled";
    const isLocal = PROVIDERS.find((p) => p.id === provider)?.local ?? true;
    const pickSpeech = async (key: string) => {
        try {
            await invoke("save_engine_settings", { stt: { model: key } });
        } catch (e) {
            say(String(e));
        }
    };

    const pickCleanup = async (key: string) => {
        try {
            await invoke("save_engine_settings", { llm: { model: key } });
        } catch (e) {
            say(String(e));
        }
    };

    const saveCompute = async (
        patch: Partial<Pick<ComputeStatus, "mode" | "temp_limit_c" | "idle_release_min">> & {
            auto_speech?: boolean;
            auto_cleanup?: boolean;
        },
    ) => {
        try {
            await invoke("save_engine_settings", { compute: patch });
        } catch (e) {
            say(String(e));
        }
    };

    const stt = live?.stt;
    const llm = live?.llm;
    const ram = live?.compute?.hardware?.ram_gb;
    const connected = data.link.link === "ready";
    // Clean-up the placement controller does not manage: switched off, or not on this computer.
    const cleanupElsewhere =
        draft.llm_cleanup === false
            ? "Off"
            : provider === "ollama"
              ? "Ollama"
              : provider === "openai" || provider === "anthropic"
                ? "Cloud"
                : null;

    return (
        <>
            <header className="pane-head">
                <h1>Models</h1>
                <p>Which models turn your voice into text and tidy it up, and where they run.</p>
            </header>

            <Compute
                compute={live?.compute}
                cleanupElsewhere={cleanupElsewhere}
                enabled={connected}
                onChange={(patch) => void saveCompute(patch)}
            />

            <article className="card">
                <h2>Speech</h2>
                {stt?.choices?.length ? (
                    <ModelPicker
                        label="Speech model"
                        choices={stt.choices}
                        change={stt.switch}
                        enabled={connected}
                        onPick={(key) => void pickSpeech(key)}
                        auto={
                            live?.compute?.auto
                                ? { on: live.compute.auto.speech, pick: live.compute.chosen?.speech ?? null }
                                : undefined
                        }
                        onAuto={() => void saveCompute({ auto_speech: true })}
                    />
                ) : (
                    <div className="row">
                        <span className="k">Model</span>
                        <span className="v">{stt?.label ?? stt?.model ?? "—"}</span>
                    </div>
                )}
                <div className="row">
                    <span className="k">Running on</span>
                    <span className="v">
                        {stt?.device === "cuda" ? "Graphics card" : stt?.device === "cpu" ? "Processor" : "—"}
                        {stt?.precision ? ` · ${stt.precision}` : ""}
                    </span>
                </div>
                <div className="row">
                    <span className="k">State</span>
                    <span className={`v ${stt?.state === "ready" ? "good" : "warn"}`}>
                        {stt?.state ?? "—"}
                        {stt?.error ? ` · ${stt.error}` : ""}
                    </span>
                </div>
                <p className="note">
                    Everything here runs on this computer. Models download once, the first time
                    you choose them, and the one you were using keeps working until the new one
                    is ready. With Parakeet, a language it does not know goes to Whisper
                    automatically.
                </p>
            </article>

            <article className="card">
                <h2>Auto-edits</h2>
                <label className="toggle">
                    <input
                        type="checkbox"
                        checked={draft.llm_cleanup ?? true}
                        onChange={(e) => push({ llm_cleanup: e.target.checked })}
                    />
                    <span className="track" aria-hidden />
                    <span className="text">
                        Clean up what I said
                        <i>
                            Removes fillers, applies your self-corrections, formats lists and
                            numbers. Off means rules only, which is faster but blunter.
                        </i>
                    </span>
                </label>
                {draft.llm_cleanup === false && ram !== undefined && ram < LOW_MEMORY_GB && (
                    <p className="note">
                        This PC has {Math.round(ram)} GB of memory, so clean-up started off: with
                        speech it takes about 6 GB, which leaves too little for your other apps.
                        Turn it on if you would rather have it. LocalFlow skips it whenever free
                        memory runs short.
                    </p>
                )}

                <div className="row">
                    <span className="k">Provider</span>
                    <select value={provider} onChange={(e) => push({ llm_provider: e.target.value })}>
                        {PROVIDERS.map((p) => (
                            <option key={p.id} value={p.id}>
                                {p.label}
                            </option>
                        ))}
                    </select>
                </div>

                {!isLocal && (
                    <p className="warn-box">
                        A cloud provider sends the text of every dictation off this machine. The
                        audio never leaves, but the transcript does.
                    </p>
                )}

                {provider === "bundled" && llm?.choices?.length ? (
                    <ModelPicker
                        label="Clean-up model"
                        choices={llm.choices}
                        change={llm.switch}
                        enabled={connected}
                        onPick={(key) => void pickCleanup(key)}
                        auto={
                            live?.compute?.auto
                                ? { on: live.compute.auto.cleanup, pick: live.compute.chosen?.cleanup ?? null }
                                : undefined
                        }
                        onAuto={() => void saveCompute({ auto_cleanup: true })}
                    />
                ) : (
                    <div className="row">
                        <span className="k">Model</span>
                        <input
                            value={draft.llm_model ?? ""}
                            onChange={(e) => setDraft({ ...draft, llm_model: e.target.value })}
                            onBlur={() => push({ llm_model: draft.llm_model })}
                            placeholder={provider === "bundled" ? "qwen3-4b" : "model name"}
                        />
                    </div>
                )}

                {(provider === "ollama" || provider === "openai") && (
                    <div className="row">
                        <span className="k">URL</span>
                        <input
                            value={draft.llm_url ?? ""}
                            onChange={(e) => setDraft({ ...draft, llm_url: e.target.value })}
                            onBlur={() => push({ llm_url: draft.llm_url })}
                            placeholder="http://127.0.0.1:11434"
                        />
                    </div>
                )}

                {(provider === "openai" || provider === "anthropic") && (
                    <div className="row">
                        <span className="k">API key</span>
                        <input
                            type="password"
                            value={draft.llm_api_key ?? ""}
                            onChange={(e) => setDraft({ ...draft, llm_api_key: e.target.value })}
                            onBlur={() => push({ llm_api_key: draft.llm_api_key })}
                            placeholder="stored on this machine only"
                        />
                    </div>
                )}

                <div className="row">
                    <span className="k">State</span>
                    <span className={`v ${llm?.state === "ready" ? "good" : "warn"}`}>
                        {llm?.state ?? "—"}
                        {llm?.error ? ` · ${llm.error}` : ""}
                    </span>
                </div>
            </article>

            <article className="card">
                <h2>House style</h2>
                <p className="note">
                    Your own notes for the clean-up model, applied to every dictation. Keep them
                    short and concrete: it is editing what you said, not writing for you.
                </p>
                <textarea
                    rows={4}
                    value={instructions}
                    placeholder={"British spelling.\nNever start a sentence with \"So\"."}
                    onChange={(e) => setInstructions(e.target.value)}
                    onBlur={() => {
                        if (instructions !== (draft.custom_instructions ?? "")) {
                            push({ custom_instructions: instructions });
                        }
                    }}
                />
                <div className="row">
                    <span className="k">Skip below</span>
                    <select
                        value={draft.llm_min_words ?? 6}
                        onChange={(e) => push({ llm_min_words: Number(e.target.value) })}
                    >
                        {[0, 3, 6, 10, 15].map((n) => (
                            <option key={n} value={n}>
                                {n === 0 ? "never skip" : `${n} words`}
                            </option>
                        ))}
                    </select>
                </div>
                <p className="note">
                    Short utterances skip the model entirely unless they contain a correction or a
                    number, which is why "Hello." comes back in 70 ms and a paragraph takes 300.
                </p>
            </article>

            <article className="card">
                <h2>Downloads</h2>
                <div className="row">
                    <span className="k">Proxy</span>
                    <span className="v">
                        {live?.network?.proxy ? live.network.proxy : "None (a direct connection)"}
                    </span>
                </div>
                <div className="row">
                    <span className="k">Download from</span>
                    <input
                        value={mirror}
                        onChange={(e) => setMirror(e.target.value)}
                        onBlur={() => {
                            if (mirror.trim() !== (live?.network?.hf_endpoint ?? "")) void saveMirror(mirror.trim());
                        }}
                        placeholder="huggingface.co"
                        disabled={!connected}
                    />
                </div>
                <p className="note">
                    Models download from huggingface.co through the proxy set in Windows&apos;
                    settings, including a company&apos;s automatic configuration. Where
                    huggingface.co is blocked, enter a mirror&apos;s address, such as
                    https://hf-mirror.com; leave it empty for huggingface.co.
                </p>
            </article>
        </>
    );
}
