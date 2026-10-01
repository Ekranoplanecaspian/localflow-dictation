import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { PostProcess, SectionProps } from "./types";
import { useSaved } from "./useSaved";

/** Key-value editor shared by exact replacements and snippets. */
function Pairs({
    pairs,
    onChange,
    fromLabel,
    toLabel,
    placeholderFrom,
    placeholderTo,
}: {
    pairs: Record<string, string>;
    onChange: (next: Record<string, string>) => void;
    fromLabel: string;
    toLabel: string;
    placeholderFrom: string;
    placeholderTo: string;
}) {
    const [from, setFrom] = useState("");
    const [to, setTo] = useState("");
    const entries = Object.entries(pairs);

    const add = () => {
        const key = from.trim();
        if (!key || !to.trim()) return;
        onChange({ ...pairs, [key]: to.trim() });
        setFrom("");
        setTo("");
    };

    return (
        <>
            <ul className="pairs">
                {entries.map(([k, v]) => (
                    <li key={k}>
                        <span className="from">{k}</span>
                        <span className="arrow">→</span>
                        <span className="to">{v}</span>
                        <button
                            className="link danger"
                            onClick={() => {
                                const next = { ...pairs };
                                delete next[k];
                                onChange(next);
                            }}
                        >
                            remove
                        </button>
                    </li>
                ))}
                {entries.length === 0 && <li className="note">Nothing here yet.</li>}
            </ul>
            <div className="pair-add">
                <input
                    aria-label={fromLabel}
                    placeholder={placeholderFrom}
                    value={from}
                    onChange={(e) => setFrom(e.target.value)}
                />
                <span className="arrow">→</span>
                <input
                    aria-label={toLabel}
                    placeholder={placeholderTo}
                    value={to}
                    onChange={(e) => setTo(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && add()}
                />
                <button className="ghost" onClick={add}>
                    Add
                </button>
            </div>
        </>
    );
}

export function Dictionary({ data, onChange, say }: SectionProps) {
    const saved = data.engine_config?.postprocess ?? {};
    // Adopts what the engine reports when it really changes, keeping an edit in progress.
    const [draft, setDraft] = useSaved<PostProcess>(saved);
    const [term, setTerm] = useState("");


    const push = async (patch: PostProcess) => {
        const next = { ...draft, ...patch };
        setDraft(next);
        try {
            await invoke("save_engine_settings", { postprocess: patch });
            say("saved");
            window.setTimeout(onChange, 300);
        } catch (e) {
            say(String(e));
        }
    };

    const terms = draft.dictionary_terms ?? [];

    return (
        <>
            <header className="pane-head">
                <h1>Dictionary</h1>
                <p>
                    Names and jargon the speech model has no reason to know. Terms are matched by
                    sound, so "Arnub", "Arnob" and "Arnabh" all become the spelling you taught it.
                </p>
            </header>

            {(data.suggestions ?? []).length > 0 && (
                <article className="card">
                    <h2>
                        Learned from your dictations
                        <i className="count">{(data.suggestions ?? []).length}</i>
                    </h2>
                    <p className="note">
                        Words the clean-up keeps fixing for you. Adding one teaches the speech
                        model directly, so it stops needing to be fixed. Nothing is added until
                        you say so.
                    </p>
                    <ul className="pairs">
                        {(data.suggestions ?? []).map((s) => (
                            <li key={`${s.from}->${s.to}`}>
                                <span className="from">{s.from}</span>
                                <span className="to">{s.to}</span>
                                <span className="note">{s.seen}x</span>
                                <button
                                    className="ghost"
                                    disabled={terms.includes(s.to)}
                                    onClick={() => push({ dictionary_terms: [...terms, s.to] })}
                                >
                                    {terms.includes(s.to) ? "Added" : "Add"}
                                </button>
                            </li>
                        ))}
                    </ul>
                </article>
            )}

            <article className="card">
                <h2>Your words</h2>
                <div className="chips">
                    {terms.map((t) => (
                        <span key={t} className="chip">
                            {t}
                            <button
                                aria-label={`Remove ${t}`}
                                onClick={() => push({ dictionary_terms: terms.filter((x) => x !== t) })}
                            >
                                ×
                            </button>
                        </span>
                    ))}
                    {terms.length === 0 && <span className="note">No terms yet.</span>}
                </div>
                <div className="pair-add">
                    <input
                        placeholder="Okonkwo"
                        value={term}
                        onChange={(e) => setTerm(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key !== "Enter") return;
                            const value = term.trim();
                            if (!value || terms.includes(value)) return;
                            push({ dictionary_terms: [...terms, value] });
                            setTerm("");
                        }}
                    />
                    <button
                        className="ghost"
                        onClick={() => {
                            const value = term.trim();
                            if (!value || terms.includes(value)) return;
                            push({ dictionary_terms: [...terms, value] });
                            setTerm("");
                        }}
                    >
                        Add
                    </button>
                </div>
                <p className="note">
                    Common English words are never replaced, so adding "Smith" cannot turn "sit"
                    into a surname.
                </p>
            </article>

            <article className="card">
                <h2>Always replace</h2>
                <p className="note">
                    Exact substitutions, applied before the model sees the text. Use these when a
                    word is heard correctly but you want something else written.
                </p>
                <Pairs
                    pairs={draft.dictionary ?? {}}
                    onChange={(dictionary) => push({ dictionary })}
                    fromLabel="heard"
                    toLabel="written"
                    placeholderFrom="gonna"
                    placeholderTo="going to"
                />
            </article>

            <article className="card">
                <h2>Snippets</h2>
                <p className="note">Say the phrase, get the text.</p>
                <Pairs
                    pairs={draft.snippets ?? {}}
                    onChange={(snippets) => push({ snippets })}
                    fromLabel="phrase"
                    toLabel="expansion"
                    placeholderFrom="my email"
                    placeholderTo="you@example.com"
                />
            </article>
        </>
    );
}
