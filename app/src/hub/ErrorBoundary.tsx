import { Component, type ErrorInfo, type ReactNode } from "react";
import { invoke } from "@tauri-apps/api/core";

/** A fault that repeats - on the Hub's 4-second refresh, say - would otherwise write the same
 *  lines to the log all day. The first few say everything there is to say. */
const MAX_REPORTS = 20;
let reports = 0;

/** Write a fault in the window to the shell's log, which is where every other fault goes.
 *  Best effort: in a browser (demo mode) there is no shell to write to. */
export function logUiError(where: string, error: unknown, stack?: string) {
    console.error(`[${where}]`, error);
    if (++reports > MAX_REPORTS) return;
    try {
        const message = error instanceof Error ? error.message : String(error);
        const detail = stack ?? (error instanceof Error ? error.stack : undefined) ?? "";
        void invoke("log_ui_error", { page: where, message, stack: detail }).catch(() => {});
    } catch {
        // Reporting a fault must never become a fault of its own.
    }
}

/** Faults outside rendering - an event handler, a promise nobody awaited - never reach an
 *  error boundary. They are logged too, so a button that silently does nothing has a trace. */
export function logUnhandledErrors(where: string) {
    window.addEventListener("error", (e) => logUiError(where, e.error ?? e.message));
    window.addEventListener("unhandledrejection", (e) => logUiError(where, e.reason));
}

type Props = {
    /** The page's name, for the message and the log. */
    page: string;
    /** "page": the fault replaces one page and the rest of the window keeps working.
     *  "window": the last line, around everything, when even the frame could not render. */
    scope?: "page" | "window";
    /** Called by Reload after the page is cleared, to fetch fresh data. */
    onReload?: () => void;
    children: ReactNode;
};

type State = { error: Error | null };

/** A render fault used to leave a blank window with nothing to click. Now it leaves a message,
 *  a Reload button and a line in the log. */
export class ErrorBoundary extends Component<Props, State> {
    state: State = { error: null };

    static getDerivedStateFromError(error: Error): State {
        return { error };
    }

    componentDidCatch(error: Error, info: ErrorInfo) {
        logUiError(this.props.page, error, `${error.stack ?? ""}\ncomponents:${info.componentStack ?? ""}`);
    }

    reload = () => {
        if (this.props.scope === "window") {
            window.location.reload();
            return;
        }
        this.setState({ error: null });
        this.props.onReload?.();
    };

    render() {
        const { error } = this.state;
        if (!error) return this.props.children;
        const whole = this.props.scope === "window";
        return (
            <article className={`card fault${whole ? " whole" : ""}`} role="alert">
                <h2>{whole ? "LocalFlow" : this.props.page}</h2>
                <p className="fault-title">
                    {whole ? "This window hit a problem." : "This page hit a problem."}
                </p>
                <p className="muted">
                    Dictation is not affected. Reloading usually fixes it; the details are in
                    LocalFlow's log.
                </p>
                <p className="fault-detail">{error.message}</p>
                <div className="actions">
                    <button onClick={this.reload}>Reload</button>
                </div>
            </article>
        );
    }
}
