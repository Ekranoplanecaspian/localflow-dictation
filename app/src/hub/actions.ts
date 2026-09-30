import { invoke } from "@tauri-apps/api/core";

/** Where a Fix button can send the Hub: a page, and for Help, the problem to open there. */
export type Go = (section: "voice" | "models" | "apps" | "help", topic?: string) => void;

export type Action = { id: string; label: string } | null | undefined;

/**
 * What a Fix button does, by the action id the problem catalogue (shared/problems.json) gives
 * it. The Status card, the self-check list and the Help page all use this one.
 */
export function makeAct({
    go,
    say,
    onChange,
    safeMode = false,
    onRepaired,
}: {
    go: Go;
    say: (message: string) => void;
    onChange: () => void | Promise<void>;
    safeMode?: boolean;
    onRepaired?: () => void;
}) {
    return async (action: Action) => {
        if (!action) return;
        switch (action.id) {
            case "restart_engine":
                await invoke("restart_engine");
                say(safeMode ? "leaving safe mode" : "restarting the engine");
                window.setTimeout(onChange, 1500);
                break;
            case "open_models":
                go("models");
                break;
            case "open_voice":
                go("voice");
                break;
            case "open_apps":
                go("apps");
                break;
            case "redownload_models": {
                const removed = await invoke<number>("repair_models").catch((e) => {
                    say(String(e));
                    return null;
                });
                if (removed !== null) {
                    say(`removed ${removed} damaged file${removed === 1 ? "" : "s"}; downloading again`);
                    onRepaired?.();
                    window.setTimeout(onChange, 1500);
                }
                break;
            }
            default:
                // Windows Settings pages (the microphone privacy switch, Sound, Storage, Date &
                // time): only ones the shell knows by name.
                await invoke("open_windows_settings", { page: action.id }).catch((e) => say(String(e)));
        }
    };
}
