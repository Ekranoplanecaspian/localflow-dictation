import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import "./FlowBar.css";

type Phase = "idle" | "recording" | "finishing";

/**
 * Subscribe to a shell event without letting a failure escape.
 *
 * `listen` rejects when the Tauri bridge is not there - the demo route, or a webview that has
 * not finished wiring itself up - and an unhandled rejection during an effect takes the whole
 * component down with it, animation loop included. The bar going silent is bad; the bar
 * vanishing is worse.
 */
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

/** Bars in the waveform, from design/tokens.json. */
const BARS = 22;
const BAR_MIN = 4;
const BAR_MAX = 30;
/** Spring constants for each bar's height: stiff enough to catch a syllable, damped enough
 *  not to ring afterwards. */
const STIFFNESS = 260;
const DAMPING = 22;
/** The level arrives at 50 Hz and the bars run at 60, so they interpolate between readings. */
const ATTACK = 0.6;
const RELEASE = 0.25;
/**
 * How long each value spends on a bar before moving outward.
 *
 * This is the whole character of the animation and it has to be slower than the spring, not
 * faster. Shifting once per frame - the obvious thing - moves a loud syllable across a bar in
 * sixteen milliseconds while the spring needs two hundred to respond to it, so nothing ever
 * rises and the waveform reads as a row of dots. At this pace the bar holds about a second of
 * speech across its width.
 */
const TRAVEL_MS = 45;

/**
 * The waveform is a row of springs, not a spectrum.
 *
 * The newest level enters at the centre and travels outward, which reads as "listening"
 * rather than as a VU meter, and each bar chases its target through a real spring so a loud
 * syllable overshoots a little and settles instead of snapping.
 *
 * Heights are written straight to the DOM rather than through React state: this runs on every
 * animation frame while the user is speaking, and reconciling twenty-two elements sixty times
 * a second to change one number each is exactly the kind of work that makes a bar stutter.
 */
function useWaveform(active: boolean) {
    const bars = useRef<(HTMLSpanElement | null)[]>([]);
    const level = useRef(0);

    useEffect(
        () =>
            subscribe<{ level: number }>("level", (p) => {
                level.current = Math.min(1, Math.max(0, p.level * 3.2));
            }),
        [],
    );

    useEffect(() => {
        const heights = new Float32Array(BARS);
        const velocity = new Float32Array(BARS);
        const targets = new Float32Array(BARS);
        const half = Math.floor(BARS / 2);
        let smoothed = 0;
        let last = performance.now();
        let sinceShift = 0;
        let raf = 0;

        const frame = (now: number) => {
            const dt = Math.min(0.05, (now - last) / 1000);
            last = now;

            const goal = active ? level.current : 0;
            smoothed += (goal - smoothed) * (goal > smoothed ? ATTACK : RELEASE);

            // Push the newest value into the middle; the previous values travel outward, one
            // step every TRAVEL_MS rather than every frame.
            sinceShift += dt * 1000;
            while (sinceShift >= TRAVEL_MS) {
                sinceShift -= TRAVEL_MS;
                for (let i = 0; i < half; i++) {
                    targets[i] = targets[i + 1];
                }
                for (let i = BARS - 1; i > half; i--) {
                    targets[i] = targets[i - 1];
                }
                targets[half] = smoothed;
            }
            // The centre always shows the live level, so speaking feels immediate even between
            // steps of the travelling wave.
            targets[half] = Math.max(targets[half], smoothed);

            for (let i = 0; i < BARS; i++) {
                // The centre carries the most, so the bar has a shape instead of being a block.
                const edge = 1 - Math.abs(i - half) / BARS;
                const want = targets[i] * (0.45 + 0.55 * edge);
                const accel = (want - heights[i]) * STIFFNESS - velocity[i] * DAMPING;
                velocity[i] += accel * dt;
                heights[i] += velocity[i] * dt;
                if (heights[i] < 0) {
                    heights[i] = 0;
                    velocity[i] = 0;
                }
                const el = bars.current[i];
                if (el) {
                    el.style.height = `${BAR_MIN + Math.min(1, heights[i]) * (BAR_MAX - BAR_MIN)}px`;
                }
            }
            // Once a take is over and the bars have fallen still, stop until the next one. It
            // ran sixty frames a second all day: the bar's window is hidden, not closed, so the
            // page never learns nobody is looking, and that loop was most of LocalFlow's idle
            // processor use.
            const still = smoothed < 1e-3 && heights.every((h, i) => h < 1e-3 && Math.abs(velocity[i]) < 1e-3);
            if (!active && still) {
                raf = 0;
                return;
            }
            raf = requestAnimationFrame(frame);
        };
        raf = requestAnimationFrame(frame);
        return () => {
            if (raf) cancelAnimationFrame(raf);
        };
    }, [active]);

    // The demo route needs to drive the level directly.
    return Object.assign(bars, { level });
}

/**
 * Show only what the decoder has stopped changing its mind about.
 *
 * Every live decode re-transcribes the *whole* take, not just the new audio, so words well
 * behind the end get revised as context arrives: "by T?" becomes "by tune?" becomes "by
 * Tuesday?", and a phrase four words back can be rewritten wholesale. Rendering each partial
 * verbatim puts that deliberation on screen and reads as jitter.
 *
 * So a word is only shown once two consecutive decodes agree on it, and once shown it is never
 * taken back - the caption only ever grows. If the decoder later changes its mind about
 * something already on screen, the bar keeps the older reading and the corrected text arrives
 * whole at the end. The bar is a progress indicator; the text that gets typed is the truth.
 */
export function makeStabiliser() {
    let shown: string[] = [];
    let previous: string[] = [];

    return {
        /** The text to display for this partial. */
        push(partial: string): string {
            const words = partial.trim().split(/\s+/).filter(Boolean);
            let agreed = 0;
            while (
                agreed < words.length &&
                agreed < previous.length &&
                words[agreed] === previous[agreed]
            ) {
                agreed++;
            }
            previous = words;
            if (agreed > shown.length) {
                shown = words.slice(0, agreed);
            }
            return shown.join(" ");
        },
        reset() {
            shown = [];
            previous = [];
        },
    };
}

/** A safety cap only: the caption keeps its end visible and fades the start, so CSS decides
 *  what actually fits. This just keeps a runaway dictation out of the DOM. */
function tail(text: string, max = 300) {
    return text.length <= max ? text : text.slice(text.length - max);
}

/**
 * `#flowbar?demo=recording` renders the bar outside Tauri with made-up input. Development builds
 * only: a release build compiles it out.
 *
 * The bar only ever appears for a second or two on top of another application, which makes it
 * the hardest part of the app to look at properly - and the first version shipped with the
 * caption crushed into seventy pixels because it was never seen. This route makes every state
 * visible in a browser.
 */
function useDemo(
    setPhase: (p: Phase) => void,
    setText: (t: string) => void,
    setNotice: (n: boolean) => void,
    setTag: (t: string) => void,
    level: React.MutableRefObject<number>,
) {
    const demo = import.meta.env.DEV
        ? new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("demo")
        : null;
    useEffect(() => {
        if (!demo) return;
        // A notice is not a take: no waveform, no session, just the line.
        const notices: Record<string, string> = {
            warming: "Warming up — try again in a moment",
            restarting: "Engine restarting — try again in a moment",
            "no-mic": "Microphone unavailable — see LocalFlow",
            off: "Dictation is off in slack",
        };
        if (notices[demo]) {
            setPhase("idle");
            setNotice(true);
            setText(notices[demo]);
            return;
        }
        // `demo=tag`: a take whose engine went away mid-sentence; the words go on.
        if (demo === "tag") setTag("Engine restarting — your words are kept");
        setPhase(demo === "transcribing" ? "finishing" : "recording");
        if (demo !== "listening") {
            setText(
                "So what I want you to do now is show me the work after each feature, " +
                    "because I need to review it before we move on",
            );
        }
        let t = 0;
        const timer = setInterval(() => {
            t += 0.02;
            // Speech-shaped and at the amplitude the real meter produces: syllables riding
            // on a slower breath, reaching the top of the range on stressed vowels.
            level.current = Math.max(
                0,
                0.45 + 0.35 * Math.sin(t * 11) + 0.2 * Math.sin(t * 2.3) + 0.1 * Math.sin(t * 27),
            );
        }, 20);
        return () => clearInterval(timer);
    }, [demo, setPhase, setText, setNotice, setTag, level]);
    return Boolean(demo);
}

export default function FlowBar() {
    const [phase, setPhase] = useState<Phase>("idle");
    const [text, setText] = useState("");
    const [error, setError] = useState(false);
    // A notice ("warming up") is not a transcript: it is short, it fits, and it reads
    // centred rather than pinned to the right edge where the newest words belong.
    const [notice, setNotice] = useState(false);
    // A problem with the take in hand ("Engine restarting — your words are kept").
    const [tag, setTag] = useState("");
    // Spoken into a password field: dots, never the words, on a bar anyone can see.
    const [hidden, setHidden] = useState(false);
    const bars = useWaveform(phase === "recording");
    const demo = useDemo(setPhase, setText, setNotice, setTag, bars.level);
    const caption = useRef<HTMLDivElement>(null);
    const [clipped, setClipped] = useState(false);

    // The bar widens once, when the first words arrive, and holds that width for the rest of
    // the take. Sizing to the text would make it flinch on every partial, and a pill breathing
    // in and out is just the decoder thinking out loud where the user can see it.
    const [wide, setWide] = useState(false);

    // Which dictation the text on screen belongs to. The window is hidden between takes and a
    // hidden webview is throttled, so its events can arrive late: without this the bar
    // reappears still showing the previous dictation until the next partial lands.
    const session = useRef<string | null>(null);
    const stabiliser = useRef(makeStabiliser());

    // Widen as soon as there are words, whatever put them there.
    useEffect(() => setWide((w) => w || text.length > 0), [text]);

    useLayoutEffect(() => {
        const el = caption.current;
        if (!el) return;
        // Keep the newest words in view. A block with hidden overflow stays scrolled to the
        // start whatever `text-align` says, so the end has to be scrolled to by hand.
        el.scrollLeft = el.scrollWidth;
        // Only fade the left edge when something is genuinely cut off there. Fading text that
        // fits makes the first word look like it is hiding under the waveform.
        setClipped(el.scrollWidth > el.clientWidth + 1);
    }, [text]);

    useEffect(() => {
        if (demo) return;
        const offs = [
            subscribe<{ phase: Phase; id: string | null }>("phase", (p) => {
                setPhase(p.phase);
                if (p.phase === "recording") {
                    session.current = p.id ?? null;
                    stabiliser.current.reset();
                    setText("");
                    setWide(false);
                    setError(false);
                    setNotice(false);
                    setTag("");
                    setHidden(false);
                }
            }),
            subscribe("take-private", () => setHidden(true)),
            // A partial from a session we have not seen means the phase event is late (a
            // hidden webview is throttled). Adopt it rather than drop it: the shell only ever
            // runs one dictation at a time, so the newest id is always the right one.
            subscribe<{ id: string; text: string }>("partial", (p) => {
                if (session.current !== p.id) {
                    session.current = p.id;
                    stabiliser.current.reset();
                    setText("");
                }
                const shown = stabiliser.current.push(p.text ?? "");
                if (shown) setText(shown);
            }),
            subscribe<{ id: string; text?: string }>("final", (p) => {
                if (session.current !== p.id) {
                    session.current = p.id;
                    stabiliser.current.reset();
                }
                const shown = (p.text ?? "").trim();
                if (shown) setText(shown);
            }),
            subscribe("engine-error", () => setError(true)),
            // The hotkey was pressed and no take could start: the shell says why (the engine
            // restarting, the speech model loading, no microphone, dictation off in this app).
            // Nothing was recorded, and the bar is the only place that can say so without
            // stealing focus.
            subscribe<{ text: string }>("notice", (p) => {
                session.current = null;
                stabiliser.current.reset();
                setError(false);
                setTag("");
                setPhase("idle");
                setWide(true);
                setNotice(true);
                setText(p.text);
            }),
            // Something went wrong with the take being spoken, which goes on: said beside the
            // words rather than instead of them.
            subscribe<{ text: string }>("take-notice", (p) => {
                setTag(p.text);
                setWide(true);
            }),
            subscribe("cancelled", () => {
                session.current = null;
                stabiliser.current.reset();
                setText("");
                setWide(false);
                setPhase("idle");
            }),
        ];
        return () => offs.forEach((off) => off());
    }, [demo]);

    return (
        <div
            className={`bar ${phase} ${error ? "error" : ""} ${wide ? "wide" : ""} ${
                clipped ? "clipped" : ""
            } ${notice ? "notice" : ""}`}
        >
            <div className="wave" aria-hidden>
                {Array.from({ length: BARS }, (_, i) => (
                    <span
                        key={i}
                        ref={(el) => {
                            bars.current[i] = el;
                        }}
                    />
                ))}
            </div>
            {text && (
                <div className="caption" ref={caption}>
                    {hidden && !notice ? "•".repeat(Math.min(12, Math.max(3, text.length))) : tail(text)}
                </div>
            )}
            {tag && <div className="tag">{tag}</div>}
        </div>
    );
}
