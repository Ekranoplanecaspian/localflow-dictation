import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { pretty, useChordCapture } from "../hub/useChordCapture";
import type { HubData, Settings } from "../hub/types";
import "./Onboarding.css";

/**
 * The first run.
 *
 * Someone who has just installed this has a working microphone, a hotkey they have not chosen
 * and no idea whether any of it functions. The wizard answers those three in order and then
 * gets out of the way. It is deliberately short: five steps, none of which can fail in a way
 * that traps the user, and a Skip on every one of them.
 *
 * The step that matters is "Try it". It does not simulate anything - the user holds the real
 * chord, the real engine transcribes it and the real injector types it into the box on screen.
 * That exercises the whole path end to end, which is the one thing a first run should prove.
 */

type Props = {
    data: HubData | null;
    onDone: () => void | Promise<void>;
};

const STEPS = ["Welcome", "Microphone", "Hotkey", "Try it", "Done"] as const;

function subscribe<T>(event: string, handler: (payload: T) => void) {
    let off: (() => void) | undefined;
    let cancelled = false;
    listen<T>(event, (e) => handler(e.payload))
        .then((un) => {
            if (cancelled) un();
            else off = un;
        })
        .catch(() => {});
    return () => {
        cancelled = true;
        off?.();
    };
}

/**
 * A live meter, so "is it hearing me?" is answered by talking rather than by a device name.
 *
 * `heard` latches: the point is not the current level but whether this microphone has ever
 * picked the user up, and a meter that only shows the instant reading makes someone speak, look
 * down, and see nothing.
 */
function Level({ onHeard }: { onHeard: () => void }) {
    const fill = useRef<HTMLSpanElement>(null);
    useEffect(
        () =>
            subscribe<{ level: number }>("level", (p) => {
                const v = Math.min(1, Math.max(0, (p.level ?? 0) * 3.2));
                if (fill.current) fill.current.style.transform = `scaleX(${v.toFixed(3)})`;
                if (v > 0.12) onHeard();
            }),
        [onHeard],
    );
    return (
        <div className="meter" aria-hidden>
            <span ref={fill} />
        </div>
    );
}

export default function Onboarding({ data, onDone }: Props) {
    const [step, setStep] = useState(0);
    const [settings, setSettings] = useState<Settings | null>(data?.settings ?? null);
    const [autostart, setAutostart] = useState(true);
    const [heard, setHeard] = useState(false);
    const [said, setSaid] = useState("");
    const [busy, setBusy] = useState(false);
    const box = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        if (data?.settings) setSettings((s) => s ?? data.settings);
    }, [data]);

    const patch = useCallback(async (p: Partial<Settings>) => {
        setSettings((current) => {
            const next = { ...(current as Settings), ...p };
            void invoke("save_settings", { settings: next }).catch(() => {});
            return next;
        });
    }, []);

    const onHeard = useCallback(() => setHeard(true), []);
    const capture = useChordCapture(useCallback((keys: string[]) => void patch({ hotkey: keys }), [patch]));

    // The final of a test dictation. The text itself arrives in the box by injection, the same
    // way it would in any other app; this only lights the confirmation.
    useEffect(() => {
        if (step !== 3) return;
        return subscribe<{ text?: string }>("final", (p) => {
            const text = (p.text ?? "").trim();
            if (text) setSaid(text);
        });
    }, [step]);

    // The box has to be the focused control or the injector will put the text somewhere else:
    // it delivers to whatever had focus when the take started.
    useEffect(() => {
        if (step === 3) box.current?.focus();
    }, [step]);

    const finish = async () => {
        setBusy(true);
        try {
            await invoke("set_autostart", { on: autostart }).catch(() => {});
            await patch({ onboarded: true });
        } finally {
            setBusy(false);
            void onDone();
        }
    };

    const hotkey = settings?.hotkey ?? ["ctrl", "win"];
    const chord = (capture.armed ? capture.held : hotkey).map(pretty).join(" + ");
    const ready = data?.engine?.stt?.state === "ready" && data?.link.link === "ready";
    const next = () => setStep((s) => Math.min(STEPS.length - 1, s + 1));
    const back = () => setStep((s) => Math.max(0, s - 1));

    return (
        <div className="onboarding">
            <div className="wizard">
                <ol className="steps" aria-label="Setup progress">
                    {STEPS.map((label, i) => (
                        <li key={label} className={i === step ? "on" : i < step ? "past" : ""}>
                            <span className="pip" aria-hidden />
                            <span className="label">{label}</span>
                        </li>
                    ))}
                </ol>

                <section className="step">
                    {step === 0 && (
                        <>
                            <h1>Talk, and it types.</h1>
                            <p className="lede">
                                Hold a key, say what you mean, let go. The words appear wherever your
                                cursor already is — any app, no copy and paste.
                            </p>
                            <ul className="points">
                                <li>
                                    <b>Nothing leaves this machine.</b> Your voice and your text are
                                    transcribed and tidied by models running on your own PC.
                                </li>
                                <li>
                                    <b>It cleans up as it goes.</b> Filler words, false starts and
                                    punctuation are handled, so you can think out loud.
                                </li>
                            </ul>
                            <p className="note">Three questions and a test. About a minute.</p>
                        </>
                    )}

                    {step === 1 && (
                        <>
                            <h1>Which microphone?</h1>
                            <p className="lede">Say something — the bar should move.</p>
                            <select
                                value={settings?.microphone ?? ""}
                                onChange={(e) => void patch({ microphone: e.target.value })}
                            >
                                <option value="">System default</option>
                                {(data?.microphones ?? []).map((m) => (
                                    <option key={m} value={m}>
                                        {data?.bluetooth_microphones?.includes(m) ? `${m} — Bluetooth` : m}
                                    </option>
                                ))}
                            </select>
                            <Level onHeard={onHeard} />
                            <p className={`verdict ${heard ? "good" : ""}`}>
                                {heard ? "Heard you." : "Waiting to hear you…"}
                            </p>
                            <p className="note">
                                LocalFlow listens all the time so the half second before you press the
                                key is not lost, but nothing is sent anywhere until you hold the chord.
                            </p>
                        </>
                    )}

                    {step === 2 && (
                        <>
                            <h1>Which keys?</h1>
                            <p className="lede">
                                Hold these to dictate, let go to insert. Pick something you would never
                                press by accident.
                            </p>
                            <div className="chord-show">
                                <kbd>{chord || "…"}</kbd>
                            </div>
                            <button className="ghost" onClick={capture.arm} disabled={capture.armed}>
                                {capture.armed ? "Press the keys… (Esc to cancel)" : "Change"}
                            </button>
                            <p className="note">
                                Tap it twice quickly to latch it on for a long passage, then once more
                                to stop.
                            </p>
                        </>
                    )}

                    {step === 3 && (
                        <>
                            <h1>Try it.</h1>
                            <p className="lede">
                                Hold <kbd>{chord}</kbd> and say a sentence. It will be typed into the
                                box below, exactly as it would into any other app.
                            </p>
                            <textarea
                                ref={box}
                                className="tryout"
                                rows={4}
                                placeholder="Your words will land here…"
                            />
                            <p className={`verdict ${said ? "good" : ""}`}>
                                {said
                                    ? "That worked."
                                    : ready
                                      ? "Ready when you are."
                                      : "Warming up the model — give it a few seconds."}
                            </p>
                        </>
                    )}

                    {step === 4 && (
                        <>
                            <h1>You're set.</h1>
                            <p className="lede">
                                Hold <kbd>{chord}</kbd> in any app. LocalFlow sits in the tray; open it
                                from there whenever you want your history, your dictionary or these
                                settings again.
                            </p>
                            <label className="toggle">
                                <input
                                    type="checkbox"
                                    checked={autostart}
                                    onChange={(e) => setAutostart(e.target.checked)}
                                />
                                <span className="track" aria-hidden />
                                <span className="text">
                                    Start LocalFlow when I sign in
                                    <i>So it is there without you thinking about it.</i>
                                </span>
                            </label>
                        </>
                    )}
                </section>

                <footer className="nav">
                    <button className="quiet" onClick={() => void finish()} disabled={busy}>
                        Skip setup
                    </button>
                    <div className="spacer" />
                    {step > 0 && (
                        <button className="ghost" onClick={back} disabled={busy}>
                            Back
                        </button>
                    )}
                    {step < STEPS.length - 1 ? (
                        <button className="primary" onClick={next}>
                            {step === 0 ? "Get started" : "Next"}
                        </button>
                    ) : (
                        <button className="primary" onClick={() => void finish()} disabled={busy}>
                            Finish
                        </button>
                    )}
                </footer>
            </div>
        </div>
    );
}
