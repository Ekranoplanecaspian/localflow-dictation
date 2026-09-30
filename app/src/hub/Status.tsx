import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { makeAct, type Go } from "./actions";
import type { CheckItem, Health, HubData, Level } from "./types";

export type { Go } from "./actions";

const NEEDS_A_LOOK: Level[] = ["starting", "degraded", "failed"];

/**
 * Every part of LocalFlow, from the shell's status model (health.rs). One calm line while
 * everything works; the parts that don't, with the reason and the fix, when something is wrong.
 */
export function StatusCard({
    health,
    link,
    go,
    say,
    onChange,
}: {
    health: Health | null;
    link: HubData["link"];
    go: Go;
    say: (message: string) => void;
    onChange: () => void | Promise<void>;
}) {
    const [open, setOpen] = useState(false);
    const [checks, setChecks] = useState<CheckItem[] | null>(null);
    const [checking, setChecking] = useState(false);
    if (!health) return null;

    const trouble = health.parts.filter((p) => NEEDS_A_LOOK.includes(p.level));
    const calm = trouble.length === 0;
    const shown = open ? health.parts : trouble;
    // What it means for dictation; the rows below say which part and why. (The tray, with one
    // line to spare, shows the model's own headline instead.)
    const problems = trouble.filter((p) => p.level !== "starting").length;
    const heading =
        health.overall === "failed"
            ? "Dictation isn't working"
            : health.overall === "degraded"
              ? `Dictation works; ${problems === 1 ? "one thing needs" : `${problems} things need`} a look`
              : health.headline;

    const act = makeAct({ go, say, onChange, safeMode: link.safe_mode, onRepaired: () => setChecks(null) });

    return (
        <section className={`status ${health.overall}`} aria-live="polite">
            <div className="status-head">
                <span className="dot" aria-hidden />
                <strong>{heading}</strong>
                <button className="link" onClick={() => setOpen(!open)} aria-expanded={open}>
                    {open ? "Hide details" : calm ? "Details" : "All parts"}
                </button>
                <button
                    className="ghost small"
                    disabled={checking}
                    onClick={async () => {
                        setChecking(true);
                        try {
                            setChecks(await invoke<CheckItem[]>("run_selfcheck", { full: true }));
                        } catch (e) {
                            say(String(e));
                        } finally {
                            setChecking(false);
                        }
                    }}
                >
                    {checking ? "Checking…" : "Check LocalFlow"}
                </button>
            </div>
            {shown.length > 0 && (
                <ul className="parts">
                    {shown.map((p) => (
                        <li key={p.id} className={p.level}>
                            <span className="dot" aria-hidden />
                            <span className="part-name">{p.name}</span>
                            <span className="part-summary">{p.summary}</span>
                            {p.action && NEEDS_A_LOOK.includes(p.level) && (
                                <button className="ghost small" onClick={() => void act(p.action)}>
                                    {p.action.label}
                                </button>
                            )}
                            {p.reason && (
                                <p className="part-reason">
                                    {p.reason}
                                    {/* for a report, and the Help page's entry of the same name */}
                                    {p.code && (
                                        <button
                                            className="link part-code"
                                            title="What this means, on the Help page"
                                            onClick={() => go("help", p.code ?? undefined)}
                                        >
                                            {p.code}
                                        </button>
                                    )}
                                </p>
                            )}
                        </li>
                    ))}
                </ul>
            )}
            {open && (
                <p className="note">
                    Engine {link.pid ? `pid ${link.pid}${link.attached ? " (attached)" : ""}` : "not running"}
                    {link.restarts
                        ? ` · restarted ${link.restarts === 1 ? "once" : `${link.restarts} times`}`
                        : ""}{" "}
                    ·{" "}
                    <button className="link" onClick={() => void act(RESTART)}>
                        {link.safe_mode ? "Leave safe mode" : "Restart engine"}
                    </button>
                </p>
            )}
            {checking && !checks && <p className="note">Checking the microphone, folders, driver and every model file…</p>}
            {checks && <Checklist items={checks} act={act} onClose={() => setChecks(null)} />}
        </section>
    );
}

const MARK: Record<CheckItem["status"], string> = { ok: "✓", warn: "!", fail: "✕", skip: "–" };

/** The result of "Check LocalFlow": every check, the ones that found something first. */
function Checklist({
    items,
    act,
    onClose,
}: {
    items: CheckItem[];
    act: (action: CheckItem["action"]) => Promise<void>;
    onClose: () => void;
}) {
    const rank = { fail: 0, warn: 1, skip: 2, ok: 3 };
    const sorted = [...items].sort((a, b) => rank[a.status] - rank[b.status]);
    const found = items.filter((i) => i.status === "fail" || i.status === "warn").length;
    return (
        <div className="checklist">
            <div className="checklist-head">
                <strong>
                    {found === 0
                        ? `All ${items.filter((i) => i.status === "ok").length} checks passed`
                        : `${found} of ${items.length} checks found something`}
                </strong>
                <button className="link" onClick={onClose}>
                    Close
                </button>
            </div>
            <ul>
                {sorted.map((c) => (
                    <li key={c.id} className={c.status}>
                        <span className="mark" aria-label={c.status}>
                            {MARK[c.status]}
                        </span>
                        <span className="check-name">{c.name}</span>
                        <span className="check-text">
                            {c.title ? (
                                <>
                                    <b>{c.title}</b> {c.message}
                                    {c.code && <code className="part-code">{c.code}</code>}
                                </>
                            ) : (
                                c.detail
                            )}
                        </span>
                        {c.action && (c.status === "fail" || c.status === "warn") && (
                            <button className="ghost small" onClick={() => void act(c.action)}>
                                {c.action.label}
                            </button>
                        )}
                    </li>
                ))}
            </ul>
        </div>
    );
}

const RESTART = { id: "restart_engine", label: "Restart engine" };

/** The sidebar's one-word version. */
export function statusPill(health: Health | null): { tone: "good" | "warn" | "bad"; label: string } {
    switch (health?.overall) {
        case "ok":
        case "off":
            return { tone: "good", label: "Ready" };
        case "degraded":
            return { tone: "warn", label: "Needs attention" };
        case "failed":
            return { tone: "bad", label: "Not working" };
        default:
            return { tone: "warn", label: "Starting…" };
    }
}
