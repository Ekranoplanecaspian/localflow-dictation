import { useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import type { AppRule, SectionProps, Settings } from "./types";

/**
 * Per-application rules.
 *
 * The engine already guesses a writing style from the application name, and the injector
 * already picks how to deliver text. Both guesses are right almost all of the time, which is
 * exactly why the exceptions are so annoying: there is no way to tell it that this one app is
 * different. This page is that way.
 *
 * The list is the applications the user actually dictates in, newest usage first, rather than
 * everything installed on the machine. Anything with a rule is shown whether or not it appears
 * in the history, so a rule can never become invisible and unremovable.
 */

const PROFILES = [
    { id: "", label: "Guess from the app" },
    { id: "chat", label: "Chat: casual, fragments fine" },
    { id: "email", label: "Email: full sentences" },
    { id: "docs", label: "Document: prose and lists" },
    { id: "code", label: "Code: identifiers kept exact" },
    { id: "terminal", label: "Terminal: verbatim, no full stop" },
];

const METHODS = [
    { id: "", label: "Automatic" },
    { id: "paste", label: "Always paste" },
    { id: "type", label: "Always type" },
];

const EMPTY: AppRule = { disabled: false, method: "", auto_send: false, profile: "" };

/** "ms-teams.exe" -> "ms-teams". The .exe adds nothing once they are in a list together. */
const pretty = (app: string) => app.replace(/\.exe$/i, "");

function Toggle({ label, on, onChange }: { label: string; on: boolean; onChange: (v: boolean) => void }) {
    return (
        <label className="toggle">
            <input type="checkbox" checked={on} onChange={(e) => onChange(e.target.checked)} />
            <span className="track" aria-hidden />
            <span className="text">{label}</span>
        </label>
    );
}

export function Apps({ data, onChange, say }: SectionProps) {
    const [settings, setSettings] = useState<Settings>(data.settings);
    useEffect(() => setSettings(data.settings), [data.settings]);

    const apps = useMemo(() => {
        const seen = new Map<string, number>();
        for (const [app, count] of data.stats.top_apps) {
            if (app) seen.set(app.toLowerCase(), count);
        }
        // A rule on an app that has dropped out of the history must still be editable.
        for (const app of Object.keys(settings.app_rules ?? {})) {
            if (!seen.has(app)) seen.set(app, 0);
        }
        return [...seen.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    }, [data.stats.top_apps, settings.app_rules]);

    const save = async (app: string, patch: Partial<AppRule>) => {
        const rules = { ...(settings.app_rules ?? {}) };
        rules[app] = { ...EMPTY, ...rules[app], ...patch };
        const next = { ...settings, app_rules: rules };
        setSettings(next);
        try {
            await invoke("save_settings", { settings: next });
            say("saved");
            void onChange();
        } catch (e) {
            say(String(e));
        }
    };

    return (
        <>
            <header className="pane-head">
                <h1>Apps</h1>
                <p>Where LocalFlow should behave differently from its defaults.</p>
            </header>

            {apps.length === 0 && (
                <p className="muted">
                    Nothing here yet. Dictate somewhere and the app will appear in this list.
                </p>
            )}

            {apps.map(([app, count]) => {
                const rule = { ...EMPTY, ...(settings.app_rules ?? {})[app] };
                const changed = JSON.stringify(rule) !== JSON.stringify(EMPTY);
                return (
                    <article className="card" key={app}>
                        <h2>
                            {pretty(app)}
                            {count > 0 && <i className="count">{count} dictations</i>}
                            {changed && <i className="count on">customised</i>}
                        </h2>

                        <Toggle
                            label="Turn dictation off in this app"
                            on={rule.disabled}
                            onChange={(v) => void save(app, { disabled: v })}
                        />

                        <div className="row">
                            <span className="k">Writing style</span>
                            <select
                                value={rule.profile}
                                disabled={rule.disabled}
                                onChange={(e) => void save(app, { profile: e.target.value })}
                            >
                                {PROFILES.map((p) => (
                                    <option key={p.id} value={p.id}>
                                        {p.label}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className="row">
                            <span className="k">Insert by</span>
                            <select
                                value={rule.method}
                                disabled={rule.disabled}
                                onChange={(e) => void save(app, { method: e.target.value })}
                            >
                                {METHODS.map((m) => (
                                    <option key={m.id} value={m.id}>
                                        {m.label}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <Toggle
                            label="Press Enter after inserting"
                            on={rule.auto_send}
                            onChange={(v) => void save(app, { auto_send: v })}
                        />
                        {rule.auto_send && (
                            <p className="note danger">
                                Dictations here send themselves. There is no undo for a sent message.
                            </p>
                        )}
                    </article>
                );
            })}
        </>
    );
}
