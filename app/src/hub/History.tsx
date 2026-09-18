import { useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { Entry, SectionProps } from "./types";

function when(at: number) {
    const date = new Date(at * 1000);
    const today = new Date();
    const sameDay = date.toDateString() === today.toDateString();
    const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return sameDay ? time : `${date.toLocaleDateString([], { day: "numeric", month: "short" })} ${time}`;
}

/** Group by day, so a long list reads as a diary rather than a wall. */
function byDay(entries: Entry[]) {
    const groups: { day: string; entries: Entry[] }[] = [];
    for (const entry of entries) {
        const day = new Date(entry.at * 1000).toDateString();
        const last = groups[groups.length - 1];
        if (last && last.day === day) last.entries.push(entry);
        else groups.push({ day, entries: [entry] });
    }
    return groups;
}

function dayLabel(day: string) {
    const date = new Date(day);
    const today = new Date().toDateString();
    const yesterday = new Date(Date.now() - 86400000).toDateString();
    if (day === today) return "Today";
    if (day === yesterday) return "Yesterday";
    return date.toLocaleDateString([], { weekday: "long", day: "numeric", month: "long" });
}

export function History({ data, onChange, say }: SectionProps) {
    const [query, setQuery] = useState("");
    const [showRaw, setShowRaw] = useState<number | null>(null);

    const matches = useMemo(() => {
        const q = query.trim().toLowerCase();
        if (!q) return data.history;
        return data.history.filter(
            (e) => e.text.toLowerCase().includes(q) || e.app.toLowerCase().includes(q),
        );
    }, [data.history, query]);

    if (!data.settings.history) {
        return (
            <>
                <header className="pane-head">
                    <h1>History</h1>
                    <p>History is switched off, so nothing is being kept.</p>
                </header>
                <p className="note">Turn it back on under Voice if you want it.</p>
            </>
        );
    }

    return (
        <>
            <header className="pane-head">
                <h1>History</h1>
                <p>
                    {data.history_total.toLocaleString()} dictations, kept on this machine for{" "}
                    {data.settings.retention_days} days.
                </p>
            </header>

            <div className="toolbar">
                <input
                    type="search"
                    placeholder="Search what you said…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                />
                <button
                    className="ghost danger"
                    onClick={async () => {
                        if (!confirm("Delete every dictation kept on this machine? This cannot be undone.")) {
                            return;
                        }
                        await invoke("clear_history");
                        say("history erased");
                        void onChange();
                    }}
                >
                    Erase history
                </button>
            </div>

            {matches.length === 0 && (
                <p className="note">{query ? "Nothing matches that." : "No dictations yet."}</p>
            )}

            {byDay(matches).map((group) => (
                <section key={group.day} className="day">
                    <h2>{dayLabel(group.day)}</h2>
                    <ul className="entries">
                        {group.entries.map((entry) => (
                            <li key={entry.at + entry.text.slice(0, 12)}>
                                <button
                                    className="entry"
                                    title="Click to copy"
                                    onClick={async () => {
                                        await navigator.clipboard.writeText(entry.text);
                                        say("copied");
                                    }}
                                >
                                    {entry.text}
                                </button>
                                <div className="entry-meta">
                                    <span>{when(entry.at)}</span>
                                    <span>{entry.app.replace(/\.exe$/i, "")}</span>
                                    <span>{entry.words} words</span>
                                    {entry.used_llm && <span>auto-edited</span>}
                                    {entry.raw !== entry.text && (
                                        <button
                                            className="link"
                                            onClick={() => setShowRaw(showRaw === entry.at ? null : entry.at)}
                                        >
                                            {showRaw === entry.at ? "hide original" : "what you said"}
                                        </button>
                                    )}
                                </div>
                                {showRaw === entry.at && <p className="raw">{entry.raw}</p>}
                            </li>
                        ))}
                    </ul>
                </section>
            ))}

            {data.history_total > data.history.length && !query && (
                <p className="note">
                    Showing the most recent {data.history.length} of {data.history_total}. The rest are
                    in {data.paths.history}.
                </p>
            )}
        </>
    );
}
