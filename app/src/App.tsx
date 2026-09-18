import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import "./App.css";
import { Apps } from "./hub/Apps";
import { Dictionary } from "./hub/Dictionary";
import { History } from "./hub/History";
import { Models } from "./hub/Models";
import { Overview } from "./hub/Overview";
import { Voice } from "./hub/Voice";
import Onboarding from "./onboarding/Onboarding";
import { SAMPLE } from "./hub/sample";
import type { HubData, Phase } from "./hub/types";

const SECTIONS = [
    { id: "overview", label: "Overview" },
    { id: "history", label: "History" },
    { id: "dictionary", label: "Dictionary" },
    { id: "voice", label: "Voice" },
    { id: "apps", label: "Apps" },
    { id: "models", label: "Models" },
] as const;

type SectionId = (typeof SECTIONS)[number]["id"];

/** `#onboarding` forces the wizard for a look at it without clearing the settings file. */
const FORCE_ONBOARDING = window.location.hash.startsWith("#onboarding");
/** `#hub?demo=1` renders the Hub from sample data, so it can be reviewed in a browser. */
const DEMO = new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("demo") !== null;

export default function App() {
    const [section, setSection] = useState<SectionId>("overview");
    const [data, setData] = useState<HubData | null>(null);
    const [phase, setPhase] = useState<Phase>("idle");
    const [partial, setPartial] = useState("");
    const [toast, setToast] = useState<string | null>(null);
    const toastTimer = useRef<number | undefined>(undefined);

    const refresh = useCallback(async () => {
        if (DEMO) {
            setData(SAMPLE);
            return;
        }
        setData(await invoke<HubData>("hub_data"));
    }, []);

    const say = useCallback((message: string) => {
        setToast(message);
        window.clearTimeout(toastTimer.current);
        toastTimer.current = window.setTimeout(() => setToast(null), 2600);
    }, []);

    useEffect(() => {
        void refresh();
        if (DEMO) return;
        const timer = setInterval(refresh, 4000);
        const unlisteners = [
            listen<{ phase: Phase }>("phase", (e) => {
                setPhase(e.payload.phase);
                if (e.payload.phase === "recording") setPartial("");
            }),
            listen<{ text: string }>("partial", (e) => setPartial(e.payload.text ?? "")),
            // A finished dictation changes the stats and the history list.
            listen("final", () => window.setTimeout(refresh, 400)),
            listen<{ message?: string }>("engine-error", (e) =>
                say(e.payload.message ?? "the engine reported a problem"),
            ),
        ];
        return () => {
            clearInterval(timer);
            unlisteners.forEach((p) => void p.then((un) => un()));
        };
    }, [refresh, say]);

    const link = data?.link.link ?? "starting";
    const stt = data?.engine?.stt;
    const ready = link === "ready" && stt?.state === "ready";
    const health = useMemo<"good" | "warn" | "bad">(() => {
        if (link === "failed" || stt?.state === "error") return "bad";
        return ready ? "good" : "warn";
    }, [link, stt, ready]);

    // First run: the wizard owns the window until it is finished or skipped. Waiting for
    // `data` avoids a flash of the Hub before we know which of the two to show.
    if (FORCE_ONBOARDING || (data && !data.settings.onboarded)) {
        return <Onboarding data={data} onDone={refresh} />;
    }

    return (
        <div className="hub">
            <nav className="side">
                <div className="brand">
                    <span className={`orb ${phase}`} aria-hidden />
                    <span className="wordmark">LocalFlow</span>
                </div>
                <ul>
                    {SECTIONS.map((s) => (
                        <li key={s.id}>
                            <button
                                className={section === s.id ? "on" : ""}
                                onClick={() => setSection(s.id)}
                            >
                                {s.label}
                            </button>
                        </li>
                    ))}
                </ul>
                <div className="side-foot">
                    <div className={`pill ${health}`}>
                        <span className="dot" />
                        {ready ? "Ready" : link === "failed" ? "Engine unavailable" : "Starting…"}
                    </div>
                    <span className="version">v{data?.settings ? "0.1.0" : "…"}</span>
                </div>
            </nav>

            <main className="pane">
                {!data && <p className="muted">Loading…</p>}
                {data && section === "overview" && (
                    <Overview data={data} phase={phase} partial={partial} onChange={refresh} say={say} />
                )}
                {data && section === "history" && <History data={data} onChange={refresh} say={say} />}
                {data && section === "dictionary" && (
                    <Dictionary data={data} onChange={refresh} say={say} />
                )}
                {data && section === "voice" && <Voice data={data} onChange={refresh} say={say} />}
                {data && section === "apps" && <Apps data={data} onChange={refresh} say={say} />}
                {data && section === "models" && <Models data={data} onChange={refresh} say={say} />}
            </main>

            {toast && <div className="toast">{toast}</div>}
        </div>
    );
}
