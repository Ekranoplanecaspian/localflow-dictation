import { useEffect, useState } from "react";

const PRETTY: Record<string, string> = {
    ctrl: "Ctrl",
    win: "Win",
    alt: "Alt",
    shift: "Shift",
    space: "Space",
    enter: "Enter",
    tab: "Tab",
    capslock: "Caps Lock",
};

/** "ctrl" -> "Ctrl". Anything unrecognised is shown upper-case, which reads fine for letters. */
export const pretty = (key: string) => PRETTY[key] ?? key.toUpperCase();

const MODIFIERS = ["ctrl", "win", "alt", "shift"];

/**
 * Record a chord by pressing it.
 *
 * The global hook deliberately ignores injected input and does not report to the webview, so
 * this listens to ordinary key events while the button is armed. The Windows key arrives as
 * "Meta", and browsers give no keyup for some combinations, so the chord is taken from what is
 * held at the moment a non-modifier is pressed, or from the modifiers themselves on release.
 *
 * Shared by the Voice page and the first-run wizard: two ways into the same setting should not
 * mean two implementations of the fiddly part.
 */
export function useChordCapture(onCaptured: (keys: string[]) => void) {
    const [armed, setArmed] = useState(false);
    const [held, setHeld] = useState<string[]>([]);

    useEffect(() => {
        if (!armed) return;

        const nameOf = (e: KeyboardEvent): string | null => {
            switch (e.key) {
                case "Control":
                    return "ctrl";
                case "Meta":
                case "OS":
                    return "win";
                case "Alt":
                    return "alt";
                case "Shift":
                    return "shift";
                case " ":
                    return "space";
                case "Escape":
                    return null;
                default:
                    if (/^F\d{1,2}$/.test(e.key)) return e.key.toLowerCase();
                    if (/^[a-zA-Z0-9]$/.test(e.key)) return e.key.toLowerCase();
                    return null;
            }
        };

        const down = (e: KeyboardEvent) => {
            e.preventDefault();
            if (e.key === "Escape") {
                setArmed(false);
                setHeld([]);
                return;
            }
            const name = nameOf(e);
            if (!name) return;
            setHeld((current) => {
                const next = current.includes(name) ? current : [...current, name];
                // A real key (not a modifier) completes the chord immediately.
                if (!MODIFIERS.includes(name)) {
                    onCaptured(next);
                    setArmed(false);
                    return [];
                }
                return next;
            });
        };

        const up = (e: KeyboardEvent) => {
            e.preventDefault();
            setHeld((current) => {
                if (current.length >= 2) {
                    onCaptured(current);
                    setArmed(false);
                    return [];
                }
                return current.filter((k) => k !== nameOf(e));
            });
        };

        window.addEventListener("keydown", down, true);
        window.addEventListener("keyup", up, true);
        return () => {
            window.removeEventListener("keydown", down, true);
            window.removeEventListener("keyup", up, true);
        };
    }, [armed, onCaptured]);

    return { armed, held, arm: () => setArmed(true) };
}
