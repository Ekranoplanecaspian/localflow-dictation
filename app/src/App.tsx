import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import "./App.css";
import { Apps } from "./hub/Apps";
import { Dictionary } from "./hub/Dictionary";
import { ErrorBoundary } from "./hub/ErrorBoundary";
import { Help } from "./hub/Help";
import { History } from "./hub/History";
import { Models } from "./hub/Models";
import { Overview } from "./hub/Overview";
import { statusPill } from "./hub/Status";
import { Voice } from "./hub/Voice";
import Onboarding from "./onboarding/Onboarding";
import { SAMPLE, gettingReady, smallPc } from "./hub/sample";
import { HEALTH_SAMPLES } from "./hub/sampleHealth";
import type { Health, HubData, Phase } from "./hub/types";

const SECTIONS = [
    { id: "overview", label: "Overview" },
    { id: "history", label: "History" },
    { id: "dictionary", label: "Dictionary" },
    { id: "voice", label: "Voice" },
    { id: "apps", label: "Apps" },
    { id: "models", label: "Models" },
    { id: "help", label: "Help" },
] as const;

type SectionId = (typeof SECTIONS)[number]["id"];

// The review routes below exist in development builds only (`npm run dev`): a release build
// compiles them out, sample data and all.
/** `#onboarding` forces the wizard for a look at it without clearing the settings file. */
const FORCE_ONBOARDING = import.meta.env.DEV && window.location.hash.startsWith("#onboarding");
const PARAMS = new URLSearchParams(window.location.hash.split("?")[1] ?? "");
/** `#hub?demo=1` renders the Hub from sample data, so it can be reviewed in a browser. */
const DEMO = import.meta.env.DEV && PARAMS.get("demo") !== null;
/** `#hub?demo=1&fault=history`: that page throws while rendering, to see the error page. */
const FAULT = DEMO ? PARAMS.get("fault") : null;

const Fault = import.meta.env.DEV
    ? ({ page }: { page: string }): null => {
          throw new Error(`a test fault on the ${page} page`);
      }
    : (): null => null;

export default function App() {
    const [section, setSection] = useState<SectionId>("overview");
    // The Help entry to open, when a problem's code on the Status card was clicked.
    const [topic, setTopic] = useState<string | undefined>(undefined);
    const go = useCallback((id: SectionId, about?: string) => {
        setSection(id);
        setTopic(about);
        paneRef.current?.scrollTo({ top: 0 });
    }, []);
    const [data, setData] = useState<HubData | null>(null);
    const [phase, setPhase] = useState<Phase>("idle");
    const [partial, setPartial] = useState("");
    // A take into a password field: its words are never shown, here or on the bar.
    const [hidden, setHidden] = useState(false);
    const [toast, setToast] = useState<string | null>(null);
    const toastTimer = useRef<number | undefined>(undefined);
    const paneRef = useRef<HTMLElement>(null);

    const refresh = useCallback(async () => {
        if (DEMO) {
            // `&health=degraded` (failed, engine, safe, starting, lowmem, gpuprep): the Status card in that state
            const pick = HEALTH_SAMPLES[PARAMS.get("health") ?? "ok"] ?? HEALTH_SAMPLES.ok;
            const sample = PARAMS.get("ram")
                ? smallPc(Number(PARAMS.get("ram")))
                : PARAMS.get("health") === "gpuprep"
                  ? gettingReady()
                  : SAMPLE;
            setData({ ...sample, health: pick, link: { ...sample.link, safe_mode: pick === HEALTH_SAMPLES.safe } });
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
        // Only while someone can see it. It went on every 4 s behind a hidden window all day:
        // with nothing to show, that was most of LocalFlow's idle processor use (7 % of a core).
        const win = getCurrentWindow();
        const timer = setInterval(async () => {
            if (document.hidden || !(await win.isVisible().catch(() => true))) return;
            void refresh();
        }, 4000);
        const onShown = () => {
            if (!document.hidden) void refresh();
        };
        document.addEventListener("visibilitychange", onShown);
        const unlisteners = [
            listen<{ phase: Phase }>("phase", (e) => {
                setPhase(e.payload.phase);
                if (e.payload.phase === "recording") {
                    setPartial("");
                    setHidden(false);
                }
            }),
            listen<{ text: string }>("partial", (e) => setPartial(e.payload.text ?? "")),
            listen("take-private", () => setHidden(true)),
            // A finished dictation changes the stats and the history list.
            listen("final", () => window.setTimeout(refresh, 400)),
            // The status model changed: a part failed, recovered, or started.
            listen<Health>("health", (e) => setData((d) => (d ? { ...d, health: e.payload } : d))),
            listen<{ message?: string }>("engine-error", (e) =>
                say(e.payload.message ?? "the engine reported a problem"),
            ),
        ];
        return () => {
            clearInterval(timer);
            document.removeEventListener("visibilitychange", onShown);
            unlisteners.forEach((p) => void p.then((un) => un()));
        };
    }, [refresh, say]);

    const pill = statusPill(data?.health ?? null);

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
                                // each section starts at its top, not where the last one was left
                                onClick={() => go(s.id)}
                            >
                                {s.label}
                            </button>
                        </li>
                    ))}
                </ul>
                <div className="side-foot">
                    <div className={`pill ${pill.tone}`}>
                        <span className="dot" />
                        {pill.label}
                    </div>
                    <span className="version">{data?.version ? `v${data.version}` : "…"}</span>
                </div>
            </nav>

            <main className="pane" ref={paneRef}>
                {!data && <p className="muted">Loading…</p>}
                {data && (
                    // Keyed by page: moving to another page starts it without the last one's fault.
                    <ErrorBoundary
                        key={section}
                        page={SECTIONS.find((s) => s.id === section)?.label ?? section}
                        onReload={refresh}
                    >
                        {FAULT === section && <Fault page={section} />}
                        {section === "overview" && (
                            <Overview
                                data={data}
                                phase={phase}
                                partial={hidden ? "•••" : partial}
                                onChange={refresh}
                                say={say}
                                go={go}
                            />
                        )}
                        {section === "history" && <History data={data} onChange={refresh} say={say} />}
                        {section === "dictionary" && <Dictionary data={data} onChange={refresh} say={say} />}
                        {section === "voice" && <Voice data={data} onChange={refresh} say={say} />}
                        {section === "apps" && <Apps data={data} onChange={refresh} say={say} />}
                        {section === "models" && <Models data={data} onChange={refresh} say={say} />}
                        {section === "help" && (
                            <Help data={data} onChange={refresh} say={say} go={go} topic={topic} />
                        )}
                    </ErrorBoundary>
                )}
            </main>

            {toast && <div className="toast">{toast}</div>}
        </div>
    );
}
