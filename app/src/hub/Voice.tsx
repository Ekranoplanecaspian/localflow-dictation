import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { pretty, useChordCapture } from "./useChordCapture";
import type { SectionProps, Settings } from "./types";

function Toggle({
    label,
    hint,
    on,
    onChange,
}: {
    label: string;
    hint?: string;
    on: boolean;
    onChange: (v: boolean) => void;
}) {
    return (
        <label className="toggle">
            <input type="checkbox" checked={on} onChange={(e) => onChange(e.target.checked)} />
            <span className="track" aria-hidden />
            <span className="text">
                {label}
                {hint && <i>{hint}</i>}
            </span>
        </label>
    );
}

export function Voice({ data, onChange, say }: SectionProps) {
    const [settings, setSettings] = useState<Settings>(data.settings);
    const [autostart, setAutostart] = useState(false);

    useEffect(() => setSettings(data.settings), [data.settings]);
    const bluetooth = new Set(data.bluetooth_microphones ?? []);
    // The one chosen, or with "System default", the one the microphone status says is in use.
    const mic = data.health?.parts.find((p) => p.id === "microphone");
    const listening = settings.microphone
        ? data.microphones.find((m) => m.toLowerCase().includes(settings.microphone.toLowerCase()))
        : mic?.level === "ok"
          ? mic.summary
          : undefined;
    const onBluetooth = listening !== undefined && bluetooth.has(listening);
    useEffect(() => {
        void invoke<{ autostart: boolean }>("shell_status").then((s) => setAutostart(s.autostart));
    }, [data]);

    const save = async (patch: Partial<Settings>) => {
        const next = { ...settings, ...patch };
        setSettings(next);
        try {
            await invoke("save_settings", { settings: next });
            say("saved");
            void onChange();
        } catch (e) {
            say(String(e));
        }
    };

    const capture = useChordCapture((keys) => void save({ hotkey: keys }));
    const commandCapture = useChordCapture((keys) => void save({ command_hotkey: keys }));
    // The shell refuses a nested pair outright; saying so here is better than the chord
    // silently doing nothing.
    const overlapping =
        settings.command_mode &&
        (settings.command_hotkey.every((k) => settings.hotkey.includes(k)) ||
            settings.hotkey.every((k) => settings.command_hotkey.includes(k)));

    return (
        <>
            <header className="pane-head">
                <h1>Voice</h1>
                <p>How dictation starts, what it listens through, and what is kept.</p>
            </header>

            <article className="card">
                <h2>Hotkey</h2>
                <div className="hotkey-row">
                    <div className="chord">
                        {(capture.armed ? capture.held : settings.hotkey).map((k, i) => (
                            <span key={k + i}>
                                {i > 0 && <span className="plus">+</span>}
                                <kbd>{pretty(k)}</kbd>
                            </span>
                        ))}
                        {capture.armed && capture.held.length === 0 && (
                            <span className="note">Press the keys you want…</span>
                        )}
                    </div>
                    <button className="ghost" onClick={capture.arm} disabled={capture.armed}>
                        {capture.armed ? "Listening… (Esc to cancel)" : "Change"}
                    </button>
                </div>
                <Toggle
                    label="Double-tap to keep listening"
                    hint="Tap the chord twice to latch it on, then once to stop."
                    on={settings.double_tap_hands_free}
                    onChange={(v) => save({ double_tap_hands_free: v })}
                />
                {settings.double_tap_hands_free && (
                    <div className="row">
                        <span className="k">Stop after silence</span>
                        <select
                            value={settings.hands_free_timeout_s}
                            onChange={(e) =>
                                save({ hands_free_timeout_s: Number(e.target.value) })
                            }
                        >
                            <option value={5}>5 seconds</option>
                            <option value={8}>8 seconds</option>
                            <option value={15}>15 seconds</option>
                            <option value={30}>30 seconds</option>
                            <option value={0}>never</option>
                        </select>
                    </div>
                )}
                <Toggle
                    label="Escape cancels a dictation"
                    on={settings.escape_cancels}
                    onChange={(v) => save({ escape_cancels: v })}
                />
                <Toggle
                    label="End each dictation with a space"
                    hint="So two sentences in a row do not run into each other."
                    on={settings.trailing_space}
                    onChange={(v) => save({ trailing_space: v })}
                />
                <Toggle
                    label="Show the flow bar"
                    hint="The small waveform above the taskbar while you speak."
                    on={settings.flow_bar}
                    onChange={(v) => save({ flow_bar: v })}
                />
            </article>

            <article className="card">
                <h2>Command mode</h2>
                <p className="note">
                    Select some text anywhere, hold this chord and say what to change - "make this
                    more formal", "turn it into bullet points". The selection is replaced only if
                    the edit looks sound; if anything goes wrong your text is left exactly as it
                    was.
                </p>
                <Toggle
                    label="Enable command mode"
                    on={settings.command_mode}
                    onChange={(v) => save({ command_mode: v })}
                />
                <div className="hotkey-row">
                    <div className="chord">
                        {(commandCapture.armed ? commandCapture.held : settings.command_hotkey).map(
                            (k, i) => (
                                <span key={k + i}>
                                    {i > 0 && <span className="plus">+</span>}
                                    <kbd>{pretty(k)}</kbd>
                                </span>
                            ),
                        )}
                        {commandCapture.armed && commandCapture.held.length === 0 && (
                            <span className="note">Press the keys you want...</span>
                        )}
                    </div>
                    <button
                        className="ghost"
                        onClick={commandCapture.arm}
                        disabled={commandCapture.armed || !settings.command_mode}
                    >
                        {commandCapture.armed ? "Listening... (Esc to cancel)" : "Change"}
                    </button>
                </div>
                {overlapping && (
                    <p className="note danger">
                        This chord overlaps the dictation hotkey, so it could never fire - holding
                        its keys would start a dictation first. Command mode is off until you pick
                        a different chord.
                    </p>
                )}
            </article>

            <article className="card">
                <h2>Microphone</h2>
                <select
                    value={settings.microphone}
                    onChange={(e) => save({ microphone: e.target.value })}
                >
                    <option value="">System default</option>
                    {data.microphones.map((m) => (
                        <option key={m} value={m}>
                            {bluetooth.has(m) ? `${m} — Bluetooth` : m}
                        </option>
                    ))}
                </select>
                {onBluetooth && (
                    <p className="note">
                        This is a Bluetooth headset. Using its microphone switches it to call quality:
                        speech is heard less clearly, and music in the headphones drops to mono. A
                        built-in or USB microphone usually recognises you better.
                    </p>
                )}
                <p className="note">
                    Recording is always on in the background so the half second before you press the
                    hotkey is not lost. Nothing is sent anywhere until you hold the chord.
                </p>
            </article>

            <article className="card">
                <h2>Startup</h2>
                <Toggle
                    label="Start LocalFlow at sign-in"
                    on={autostart}
                    onChange={async (v) => {
                        const now = await invoke<boolean>("set_autostart", { on: v });
                        setAutostart(now);
                        say(now ? "will start at sign-in" : "will not start at sign-in");
                    }}
                />
            </article>

            <article className="card">
                <h2>Privacy</h2>
                <Toggle
                    label="Keep a history of what I dictate"
                    hint="Stored only on this machine, and used for the stats and history pages."
                    on={settings.history}
                    onChange={(v) => save({ history: v })}
                />
                <div className="row">
                    <span className="k">Keep for</span>
                    <select
                        value={settings.retention_days}
                        onChange={(e) => save({ retention_days: Number(e.target.value) })}
                        disabled={!settings.history}
                    >
                        <option value={7}>7 days</option>
                        <option value={30}>30 days</option>
                        <option value={90}>90 days</option>
                        <option value={365}>a year</option>
                        <option value={0}>forever</option>
                    </select>
                </div>
                <p className="note">
                    Everything lives in {data.paths.history}. Speech and clean-up both run on this
                    machine; no audio or text leaves it unless you pick a cloud provider under
                    Models.
                </p>
            </article>
        </>
    );
}
