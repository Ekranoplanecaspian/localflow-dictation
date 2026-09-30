import type { Download, Fit } from "./types";

/** "640 MB", "2.4 GB". */
export function formatSize(gb: number): string {
    if (gb < 1) return `${Math.max(1, Math.round(gb * 1000))} MB`;
    return `${gb.toFixed(1)} GB`;
}

export function formatBytes(bytes: number): string {
    return formatSize(bytes / 1e9);
}

/** "about 2 min left", "less than a minute left". */
export function formatEta(seconds: number | null | undefined): string | null {
    if (seconds == null) return null;
    if (seconds < 60) return "less than a minute left";
    const min = Math.round(seconds / 60);
    if (min < 60) return `about ${min} min left`;
    return `about ${Math.floor(min / 60)} h ${min % 60} min left`;
}

/** One line for a download: "45% · 12 MB/s · about 2 min left", "Waiting its turn", "Done". */
export function downloadLine(d: Download): string {
    switch (d.state) {
        case "queued":
            return "Waiting its turn";
        case "done":
            return "Downloaded";
        case "cancelled":
            return "Cancelled";
        case "error":
            return d.error ? `Failed: ${d.error}` : "Failed";
        default: {
            const parts = [`${Math.floor(d.progress * 100)}%`, `${formatBytes(d.done)} of ${formatBytes(d.total)}`];
            if (d.speed_bps > 0) parts.push(`${(d.speed_bps / 1e6).toFixed(1)} MB/s`);
            const eta = formatEta(d.eta_s);
            if (eta) parts.push(eta);
            return parts.join(" · ");
        }
    }
}

/** How a model suits this PC, as a short verdict. */
export const FIT_LABEL: Record<Fit["rating"], string> = {
    good: "Suits this PC",
    slow: "Works, but slowly here",
    "too-big": "Too big for this PC",
};

export function sentence(text: string): string {
    return text ? text[0].toUpperCase() + text.slice(1) : text;
}
