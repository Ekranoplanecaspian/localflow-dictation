import { type Dispatch, type SetStateAction, useEffect, useRef, useState } from "react";

/**
 * Editable state for a value the Hub keeps refreshing - every four seconds, as a new object each
 * time even when nothing changed.
 *
 * The pages used to copy the refreshed value over their own state on every refresh, so a field
 * being edited (custom instructions, a server address, a hotkey's time-out) lost whatever had
 * been typed since the last save, before its blur could save it. Now the saved value is followed
 * only when it genuinely changes (saved here, by another page, or by the engine), and even then
 * a field edited here and not yet saved keeps the edit.
 */
export function useSaved<T>(saved: T): [T, Dispatch<SetStateAction<T>>] {
    const [value, setValue] = useState<T>(saved);
    const key = JSON.stringify(saved ?? null);
    const last = useRef({ key, saved });
    useEffect(() => {
        if (key === last.current.key) return;
        const before = last.current.saved;
        last.current = { key, saved };
        setValue((local) => reconcile(local, before, saved));
    }, [key, saved]);
    return [value, setValue];
}

/** The newly saved value, except for fields edited locally (different from what was saved
 * before) and not saved yet: those keep the local edit. */
export function reconcile<T>(local: T, before: T, saved: T): T {
    if (!isRecord(local) || !isRecord(before) || !isRecord(saved)) {
        return same(local, before) ? saved : local;
    }
    const out: Record<string, unknown> = { ...saved };
    for (const field of Object.keys(local)) {
        if (!same(local[field], before[field])) out[field] = local[field];
    }
    return out as T;
}

function isRecord(v: unknown): v is Record<string, unknown> {
    return typeof v === "object" && v !== null && !Array.isArray(v);
}

function same(a: unknown, b: unknown): boolean {
    return JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
}
