import { invoke } from "@tauri-apps/api/core";
import { downloadLine, formatBytes } from "./downloads";
import type { Download } from "./types";

const KIND: Record<Download["kind"], string> = {
    speech: "Speech model",
    cleanup: "Clean-up model",
    runtime: "Clean-up runtime",
    "gpu-libs": "Graphics card support",
};

function why(d: Download): string {
    switch (d.reason) {
        case "first-run":
            // only speech holds dictation up; without clean-up it goes on with the basic rules
            return d.kind === "speech" ? "needed before you can dictate" : "for auto-edits; dictation works meanwhile";
        case "switch":
            return "you chose it";
        case "automatic":
            return "LocalFlow needs it";
        default:
            return "downloading to have it ready";
    }
}

/** How long a finished download stays on the strip. */
const DONE_FOR_S = 20;

/**
 * Every download, at the top of every page (v0.2.1, M3): the one running with a large bar, its
 * speed and the time left; what waits behind it; and for a little while, what has just arrived
 * or failed. A download used to be visible only as a small bar inside its row on Models - and
 * the clean-up model's first download not at all.
 */
export function DownloadStrip({
    downloads,
    onOpen,
    say,
}: {
    downloads: Download[] | undefined;
    onOpen: () => void;
    say: (message: string) => void;
}) {
    const list = downloads ?? [];
    const running = list.filter((d) => d.state === "downloading");
    const queued = list.filter((d) => d.state === "queued");
    const recent = list.filter(
        (d) =>
            (d.state === "done" && (d.ended_s_ago ?? 0) <= DONE_FOR_S) ||
            (d.state === "error" && (d.ended_s_ago ?? 0) <= 5 * 60),
    );
    if (!running.length && !queued.length && !recent.length) return null;

    const act = async (args: Record<string, string>) => {
        try {
            await invoke("model_action", args);
        } catch (e) {
            say(String(e));
        }
    };

    return (
        <section className="dl-strip" aria-label="Downloads" aria-live="polite">
            {running.map((d) => (
                <div className="dl" key={d.id}>
                    <div className="dl-head">
                        <button type="button" className="link dl-name" onClick={onOpen}>
                            Downloading {d.label}
                        </button>
                        <span className="dl-kind">
                            {KIND[d.kind]} · {why(d)}
                        </span>
                        {d.cancellable && (
                            <button
                                type="button"
                                className="ghost small"
                                onClick={() => void act({ action: "cancel", id: d.id })}
                            >
                                Cancel
                            </button>
                        )}
                    </div>
                    <div className="dl-bar" role="progressbar" aria-valuenow={Math.round(d.progress * 100)}
                         aria-valuemin={0} aria-valuemax={100} aria-label={d.label}>
                        <span style={{ width: `${Math.max(2, d.progress * 100)}%` }} />
                    </div>
                    <div className="dl-line">{downloadLine(d)}</div>
                </div>
            ))}
            {queued.length > 0 && (
                <div className="dl-next">
                    Next: {queued.map((d) => `${d.label} (${formatBytes(d.total)})`).join(", ")}
                    {queued.length === 1 && queued[0].cancellable && (
                        <button
                            type="button"
                            className="link"
                            onClick={() => void act({ action: "cancel", id: queued[0].id })}
                        >
                            Don&apos;t download it
                        </button>
                    )}
                </div>
            )}
            {recent.map((d) =>
                d.state === "done" ? (
                    <div className="dl-done" key={d.id}>
                        <span aria-hidden>✓</span> {d.label} is downloaded
                    </div>
                ) : (
                    <div className="dl-failed" key={d.id}>
                        {d.label} did not download: {d.error ?? "the download failed"}
                        {(d.kind === "speech" || d.kind === "cleanup") && (
                            <button
                                type="button"
                                className="link"
                                onClick={() => void act({ action: "download", kind: d.kind, key: d.key })}
                            >
                                Try again
                            </button>
                        )}
                    </div>
                ),
            )}
        </section>
    );
}

/** "42 %" beside Models in the sidebar while something downloads; "…" while it waits. */
export function navBadge(downloads: Download[] | undefined): string | null {
    const running = downloads?.find((d) => d.state === "downloading");
    if (running) return `${Math.floor(running.progress * 100)}%`;
    return downloads?.some((d) => d.state === "queued") ? "…" : null;
}
