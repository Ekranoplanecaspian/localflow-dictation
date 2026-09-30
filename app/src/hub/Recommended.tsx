import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { formatSize } from "./downloads";
import type { HubData, Recommendation } from "./types";

const DISMISSED = "localflow.recommendations.dismissed";
/** The one prompt after setup (M4): shown once, when the first model is ready. */
const INTRO_SEEN = "localflow.models.introSeen";

function read(key: string): string | null {
    try {
        return window.localStorage.getItem(key);
    } catch {
        return null;
    }
}

function write(key: string, value: string) {
    try {
        window.localStorage.setItem(key, value);
    } catch {
        // private storage off: the prompt may come back, nothing worse
    }
}

const id = (r: Recommendation) => `${r.kind}:${r.key}:${r.action}`;

function dismissed(): string[] {
    try {
        return JSON.parse(read(DISMISSED) ?? "[]");
    } catch {
        return [];
    }
}

function buttonLabel(r: Recommendation): string {
    if (r.action === "enable") return "Turn on auto-edits";
    if (r.action === "download") return `Download (${formatSize(r.size_gb)})`;
    return r.installed ? "Use it" : `Download and use (${formatSize(r.size_gb)})`;
}

/** Take up a recommendation: switch to the model, fetch it for Automatic, or turn auto-edits on. */
async function take(r: Recommendation) {
    if (r.action === "enable") {
        await invoke("save_engine_settings", { postprocess: { llm_cleanup: true } });
    } else if (r.action === "download") {
        await invoke("model_action", { action: "download", kind: r.kind, key: r.key });
    } else {
        await invoke("save_engine_settings", r.kind === "speech" ? { stt: { model: r.key } } : { llm: { model: r.key } });
    }
}

/**
 * What would make dictation better on this PC (v0.2.1, M4): the engine's recommendations, each
 * with why, its size and one button. "Not now" hides one for good on this PC. With nothing to
 * recommend it says so: that is worth knowing too.
 */
export function Recommended({ data, say, compact = false }: { data: HubData; say: (m: string) => void; compact?: boolean }) {
    const [hidden, setHidden] = useState<string[]>(dismissed);
    const list = (data.engine?.recommended ?? []).filter((r) => !hidden.includes(id(r)));
    const connected = data.link.link === "ready";

    const act = async (r: Recommendation) => {
        try {
            await take(r);
            say(r.action === "enable" ? "auto-edits are on" : `${r.label}: on its way`);
        } catch (e) {
            say(String(e));
        }
    };
    const notNow = (r: Recommendation) => {
        const next = [...hidden, id(r)];
        setHidden(next);
        write(DISMISSED, JSON.stringify(next));
    };

    if (!list.length) {
        if (compact) return null;
        const stt = data.engine?.stt;
        const llm = data.engine?.llm;
        return (
            <p className="rec-none">
                <b>You have the best models for this PC.</b> Speech: {stt?.label ?? "—"}
                {llm?.enabled ? ` · Auto-edits: ${llm.label ?? llm.model}` : " · Auto-edits: off"}. The other models
                are below, with how each would suit this PC.
            </p>
        );
    }
    return (
        <ul className="rec-list">
            {list.map((r) => (
                <li key={id(r)} className="rec">
                    <div className="rec-text">
                        <b>{r.action === "enable" ? "Turn on auto-edits" : r.label}</b>
                        <span>{r.why}</span>
                        {r.action === "download" && r.kind === "speech" && !r.why.includes("language") && (
                            <i>Automatic switches to it once it is here.</i>
                        )}
                    </div>
                    <div className="rec-actions">
                        <button type="button" className="primary small" disabled={!connected} onClick={() => void act(r)}>
                            {buttonLabel(r)}
                        </button>
                        <button type="button" className="link" onClick={() => notNow(r)}>
                            Not now
                        </button>
                    </div>
                </li>
            ))}
        </ul>
    );
}

/**
 * The one prompt after setup (M4), on Overview: once the first speech model is ready, what
 * LocalFlow has, what else would help on this PC, and the way to all of it. Shown once.
 */
export function ModelsIntro({ data, say, go }: { data: HubData; say: (m: string) => void; go: () => void }) {
    const [seen, setSeen] = useState(read(INTRO_SEEN) === "1");
    if (seen || data.engine?.stt?.state !== "ready") return null;
    const done = () => {
        write(INTRO_SEEN, "1");
        setSeen(true);
    };
    const any = (data.engine?.recommended ?? []).length > 0;
    return (
        <article className="card models-intro">
            <h2>Your models</h2>
            <p>
                {`${data.engine?.stt?.label ?? "Your speech model"} turns your voice into text`}
                {data.engine?.llm?.enabled ? `, and ${data.engine.llm.label ?? "the clean-up model"} tidies it up` : ""}
                {data.engine?.llm?.enabled ? ". Both run on this PC. " : ". It runs on this PC. "}
                {any ? "A few more would help here:" : "They are the best for this PC."}
            </p>
            <Recommended data={data} say={say} compact />
            <div className="rec-actions">
                <button
                    type="button"
                    className="ghost small"
                    onClick={() => {
                        done();
                        go();
                    }}
                >
                    See all models
                </button>
                <button type="button" className="link" onClick={done}>
                    Got it
                </button>
            </div>
        </article>
    );
}
