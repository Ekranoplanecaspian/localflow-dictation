import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { PostProcess, SectionProps } from "./types";

const PROVIDERS = [
    { id: "bundled", label: "Bundled (Qwen3-4B, on this GPU)", local: true },
    { id: "ollama", label: "Ollama", local: true },
    { id: "openai", label: "OpenAI-compatible endpoint", local: false },
    { id: "anthropic", label: "Anthropic", local: false },
];

export function Models({ data, onChange, say }: SectionProps) {
    const saved = data.engine_config?.postprocess ?? {};
    const [draft, setDraft] = useState<PostProcess>(saved);
    const [instructions, setInstructions] = useState(saved.custom_instructions ?? "");

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
    const stt = data.engine?.stt;
    const llm = data.engine?.llm;

    return (
        <>
            <header className="pane-head">
                <h1>Models</h1>
                <p>Speech runs locally always. Clean-up is what you can change.</p>
            </header>

            <article className="card">
                <h2>Speech</h2>
                <div className="row">
                    <span className="k">Model</span>
                    <span className="v">{stt?.model ?? "—"}</span>
                </div>
                <div className="row">
                    <span className="k">Running on</span>
                    <span className="v">
                        {stt?.device ?? "—"} {stt?.precision ? `· ${stt.precision}` : ""}
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
                    Parakeet handles 25 European languages; other languages route to Whisper
                    automatically. Changing this is not exposed yet.
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

                <div className="row">
                    <span className="k">Model</span>
                    <input
                        value={draft.llm_model ?? ""}
                        onChange={(e) => setDraft({ ...draft, llm_model: e.target.value })}
                        onBlur={() => push({ llm_model: draft.llm_model })}
                        placeholder={provider === "bundled" ? "qwen3-4b" : "model name"}
                    />
                </div>

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
        </>
    );
}
