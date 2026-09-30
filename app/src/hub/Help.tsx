import { useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import catalogue from "../../../shared/problems.json";
import { makeAct, type Go } from "./actions";
import type { SectionProps } from "./types";

/** One problem as the catalogue has it (shared/problems.json, which the shell compiles in). */
type Problem = {
    code: string;
    part: string;
    level: string;
    title: string;
    message: string;
    help?: string;
    help_title?: string;
    action?: { id: string; label: string };
};

const PARTS: [string, string][] = [
    ["microphone", "Microphone"],
    ["hotkey", "Hotkey and typing"],
    ["speech", "Speech model"],
    ["cleanup", "Clean-up"],
    ["engine", "Engine"],
    ["gpu", "Graphics card"],
    ["storage", "Storage"],
    ["network", "Downloads"],
];

// A live problem's message names the model, the microphone, the reason; here there is no live
// problem, so each is said in general.
const GENERAL: Record<string, string> = {
    model: "the model",
    device: "the microphone",
    chosen: "the microphone you chose",
    to: "the new model",
    needed: "the space it needs",
    app: "this app",
    folder: "LocalFlow's folder",
    free: "too little",
    drive: "the drive",
    version: "a newer version",
};

export function general(text: string): string {
    return text
        .replace(/\s*\{detail\}\s*/g, " ")
        .replace(/\{(\w+)\}/g, (_, k: string) => GENERAL[k] ?? k)
        .replace(/:\s*(?=[A-Z]|$)/g, ". ")
        .replace(/\s+/g, " ")
        .trim()
        .replace(/(^|[.!?]\s+)([a-z])/g, (_, a: string, b: string) => a + b.toUpperCase());
}

/** Everything a person may see: all but the connection's own protocol errors. */
const PROBLEMS: Problem[] = (catalogue.problems as Problem[]).filter((p) => p.level !== "internal");

const title = (p: Problem) => p.help_title ?? general(p.title);
const text = (p: Problem) => p.help ?? general(p.message);

const FAQ: [string, string][] = [
    [
        "How do I dictate?",
        "Hold the hotkey (Ctrl + Win unless you changed it in Voice), say what you want to write, and let go: the words appear where the cursor is, in any app. Double-tap the hotkey to keep listening without holding it, and tap it again to stop. Escape throws a dictation away.",
    ],
    [
        "How do I change text by voice?",
        "Select the text, hold the command chord (Win + Alt unless you changed it), and say what to do with it: \"make it shorter\", \"turn this into a list\", \"translate to German\". The selection is replaced; what you said is never typed.",
    ],
    [
        "My words didn't appear. Where did they go?",
        "If another window came to the front, or nothing could take text (the desktop, the taskbar, the lock screen), LocalFlow keeps the words rather than typing them somewhere wrong. Press Win + Alt + V, or choose Paste last dictation in the tray menu, to put them where you are now.",
    ],
    [
        "Why was a password typed exactly as I said it, and shown as dots?",
        "In a password field LocalFlow types exactly what it heard: no clean-up, never pasted through the clipboard, never kept in history or the log, and shown as dots on the flow bar.",
    ],
    [
        "Does anything leave my computer?",
        "Speech recognition and clean-up run on this PC. What does reach the internet: downloading the models (from Hugging Face) and the clean-up server (from GitHub), once each - and, only if you choose a cloud provider for clean-up in Models, the text of each dictation (never the audio). There is no analytics of any kind.",
    ],
    [
        "Why is the first dictation after a break a little slower?",
        "After ten minutes unused, LocalFlow frees the graphics card and its memory for other programs, and unloads the clean-up model. The next dictation wakes them; speech is ready at once, and clean-up within about a second.",
    ],
    [
        "Does it use the graphics card or the processor?",
        "Automatic, by default: the graphics card while it is cool and free, the processor when it runs hot, on battery, or when a game has it busy. Models → Graphics card lets you choose yourself.",
    ],
    [
        "Which languages does it understand?",
        "The standard speech model, Parakeet, recognises 25 European languages by itself, English, German, French and Spanish among them. For another language, put its code (\"hi\", \"ja\", \"ar\"...) as \"language\" under \"stt\" in %APPDATA%\\LocalFlow\\config.json and restart LocalFlow: it then uses Whisper, which knows 99 and is downloaded the first time it is needed. (A language setting in the Hub is on its way.)",
    ],
    [
        "How much disk space and memory does it use?",
        "About 2.6 GB for the speech model and 2.5 GB for the clean-up model on disk. Resting, LocalFlow keeps about 3 GB of memory so a dictation starts instantly, and nothing on the graphics card.",
    ],
    [
        "Where are my settings and history kept?",
        "On this PC only: settings, history and logs in %APPDATA%\\LocalFlow, the models in %LOCALAPPDATA%\\LocalFlow. History keeps what you dictated for as long as Voice says (90 days unless changed), and can be turned off.",
    ],
];

const KNOWN: string[] = [
    "Apps running as administrator (Task Manager, some installers) don't accept typing from other programs; LocalFlow copies the text instead and the flow bar says to press Ctrl + V.",
    "Remote desktop windows and games with anti-cheat may not let LocalFlow see the hotkey.",
    "The lock screen, a UAC prompt or sleep end a dictation in progress; its words are kept for Win + Alt + V.",
    "A Bluetooth headset's microphone switches the headset to call quality, and recognition is less accurate than with a built-in or USB microphone.",
    "LocalFlow isn't code-signed yet, so Windows SmartScreen or antivirus software may warn about it when it is installed.",
    "On a laptop whose processor and graphics card share cooling, moving work to the processor frees the graphics card but not all of the heat.",
    "Speech is ready about three seconds after LocalFlow starts; a dictation before then is refused, and the flow bar says so.",
    "Win + H is Windows' own dictation, and can't be LocalFlow's hotkey.",
];

export function Help({ data, say, onChange, go, topic }: SectionProps & { go: Go; topic?: string }) {
    const [query, setQuery] = useState("");
    const act = makeAct({ go, say, onChange, safeMode: data.link.safe_mode });
    const opened = useRef<HTMLDetailsElement | null>(null);

    // Problems the Status card shows right now, first.
    const now = useMemo(() => {
        const codes = new Set((data.health?.parts ?? []).map((p) => p.code).filter((c): c is string => !!c));
        return PROBLEMS.filter((p) => codes.has(p.code));
    }, [data.health]);

    const q = query.trim().toLowerCase();
    const matches = (p: Problem) =>
        !q || [p.code, title(p), text(p)].some((s) => s.toLowerCase().includes(q));

    useEffect(() => {
        opened.current?.scrollIntoView({ block: "start" });
    }, [topic]);

    // The entry a Status card link asked for is scrolled to where it first appears: under
    // "Happening now" when it is happening now.
    const entry = (p: Problem, first = true) => (
        <details
            key={p.code}
            className={`help-entry ${p.level}`}
            open={p.code === topic || !!q}
            ref={p.code === topic && first ? opened : undefined}
        >
            <summary>
                {title(p)}
                <code className="part-code">{p.code}</code>
            </summary>
            <p>{text(p)}</p>
            {p.action && (
                <button className="ghost small" onClick={() => void act(p.action)}>
                    {p.action.label}
                </button>
            )}
        </details>
    );

    return (
        <>
            <header className="pane-head">
                <h1>Help</h1>
                <p>What each problem means and how to fix it, answers to common questions, and known issues.</p>
            </header>

            {now.length > 0 && !q && (
                <article className="card">
                    <h2>Happening now</h2>
                    {now.map((p) => entry(p))}
                </article>
            )}

            <article className="card">
                <h2>Problems and fixes</h2>
                <input
                    className="help-search"
                    type="search"
                    placeholder="Search: microphone, download, clock…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                />
                {PARTS.map(([part, name]) => {
                    const list = PROBLEMS.filter((p) => p.part === part && matches(p));
                    if (list.length === 0) return null;
                    return (
                        <section key={part} className="help-group">
                            <h3>{name}</h3>
                            {list.map((p) => entry(p, !(now.includes(p) && !q)))}
                        </section>
                    );
                })}
                {q && !PROBLEMS.some(matches) && <p className="note">Nothing matches “{query}”.</p>}
            </article>

            <ReportProblem say={say} />

            <YourSettings say={say} onChange={onChange} />

            <article className="card">
                <h2>Questions</h2>
                {FAQ.map(([question, answer]) => (
                    <details key={question} className="help-entry">
                        <summary>{question}</summary>
                        <p>{answer}</p>
                    </details>
                ))}
            </article>

            <article className="card">
                <h2>Known issues</h2>
                <ul className="help-known">
                    {KNOWN.map((k) => (
                        <li key={k}>{k}</li>
                    ))}
                </ul>
            </article>
        </>
    );
}

/** Report a problem: a diagnostics file the user attaches to a GitHub issue themselves. */
function ReportProblem({ say }: { say: (message: string) => void }) {
    const [file, setFile] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);
    const make = async () => {
        setBusy(true);
        try {
            setFile(await invoke<string>("export_diagnostics"));
        } catch (e) {
            say(String(e));
        } finally {
            setBusy(false);
        }
    };
    return (
        <article className="card">
            <h2>Report a problem</h2>
            <p className="note">
                LocalFlow gathers what is needed to find the cause into one file: its logs, its status,
                its settings and a page of system details. What you dictated and the titles of your
                windows are replaced by their length; your history, dictionary, snippets and any API key
                are left out. Nothing is sent anywhere: you attach the file to your report yourself.
            </p>
            <div className="report-steps">
                <button className="ghost" disabled={busy} onClick={() => void make()}>
                    {busy ? "Gathering…" : file ? "Make it again" : "1. Create diagnostics file"}
                </button>
                <button
                    className="ghost"
                    onClick={() => void invoke("report_issue").catch((e) => say(String(e)))}
                >
                    2. Open a report on GitHub
                </button>
            </div>
            {file && (
                <p className="note">
                    Saved as <code>{file}</code>{" "}
                    <button className="link" onClick={() => void invoke("reveal_diagnostics").catch((e) => say(String(e)))}>
                        Show in folder
                    </button>
                    . Drag it into the report, and say what you were doing when it went wrong.
                </p>
            )}
            <p className="note">A report on GitHub needs a free GitHub account.</p>
        </article>
    );
}

type Merged = { added: number; kept: number; style: "added" | "extended" | "unchanged" };

/** Export and import the user's own words; reset the preferences, and only those. */
function YourSettings({ say, onChange }: { say: (message: string) => void; onChange: () => void | Promise<void> }) {
    const [saved, setSaved] = useState<string | null>(null);
    const [confirming, setConfirming] = useState(false);
    const picker = useRef<HTMLInputElement>(null);

    const exportWords = async () => {
        try {
            setSaved(await invoke<string>("export_words"));
        } catch (e) {
            say(String(e));
        }
    };

    const importWords = async (file: File | undefined) => {
        if (!file) return;
        try {
            const m = await invoke<Merged>("import_words", { text: await file.text() });
            const style = m.style === "unchanged" ? "" : m.style === "added" ? "; house style added" : "; house style extended";
            say(
                m.added === 0 && m.style === "unchanged"
                    ? "nothing new in that file"
                    : `added ${m.added}${m.kept ? `, kept ${m.kept} of yours with the same name` : ""}${style}`,
            );
            window.setTimeout(onChange, 800);
        } catch (e) {
            say(String(e));
        } finally {
            if (picker.current) picker.current.value = "";
        }
    };

    const reset = async () => {
        try {
            await invoke("reset_preferences");
            say("preferences are back to their defaults");
            setConfirming(false);
            window.setTimeout(onChange, 800);
        } catch (e) {
            say(String(e));
        }
    };

    return (
        <article className="card">
            <h2>Your settings</h2>
            <h3 className="help-sub">Your words</h3>
            <p className="note">
                Your dictionary, snippets, app rules and house style in one file, to keep safe or take
                to another PC. Importing adds what is new; where you already have an entry of the same
                name, yours stays. No API key is ever in the file.
            </p>
            <div className="report-steps">
                <button className="ghost" onClick={() => void exportWords()}>
                    Export
                </button>
                <button className="ghost" onClick={() => picker.current?.click()}>
                    Import…
                </button>
                <input
                    ref={picker}
                    type="file"
                    accept=".json,application/json"
                    hidden
                    onChange={(e) => void importWords(e.target.files?.[0])}
                />
            </div>
            {saved && (
                <p className="note">
                    Saved as <code>{saved}</code>{" "}
                    <button className="link" onClick={() => void invoke("reveal_diagnostics").catch((e) => say(String(e)))}>
                        Show in folder
                    </button>
                </p>
            )}

            <h3 className="help-sub">Reset preferences</h3>
            <p className="note">
                Puts the hotkeys, microphone, model choice, clean-up and other options back to how
                LocalFlow comes. Your dictionary, snippets, app rules and house style stay, and so do
                your history, whether it is kept, and the downloaded models.
            </p>
            {confirming ? (
                <div className="report-steps">
                    <button className="danger" onClick={() => void reset()}>
                        Reset preferences
                    </button>
                    <button className="ghost" onClick={() => setConfirming(false)}>
                        Cancel
                    </button>
                </div>
            ) : (
                <div className="report-steps">
                    <button className="ghost" onClick={() => setConfirming(true)}>
                        Reset preferences…
                    </button>
                </div>
            )}
        </article>
    );
}
