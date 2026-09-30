import type { Health, HealthPart } from "./types";

/**
 * Status samples for `#hub?demo=1&health=<name>`, worded as the shell's status model
 * (health.rs) words them, so every state of the Status card can be looked at in a browser.
 */
/** The codes the sample problems would carry. */
const CODES: Record<string, string> = {
    "cleanup:Unavailable": "cleanup-server-crashed",
    "cleanup:Not enough memory": "cleanup-low-memory",
    "microphone:Unavailable": "mic-unavailable",
    "engine:Not running": "engine-wont-start",
    "engine:Safe mode": "engine-safe-mode",
};

const ok: HealthPart[] = [
    part("engine", "Engine", "ok", "Running", "The engine is running", null, null, true),
    part("speech", "Speech model", "ok", "Parakeet v3", "Parakeet v3 is ready", "Running on the graphics card.", null, true),
    part("cleanup", "AI clean-up", "ok", "Qwen3 4B", "Qwen3 4B is ready", "Running on the graphics card.", null, false),
    part("gpu", "Graphics card", "ok", "In use", "RTX 4060 Laptop GPU is in use", null, null, false),
    part("microphone", "Microphone", "ok", "Microphone Array (Realtek(R) Audio)",
        "Listening with Microphone Array (Realtek(R) Audio)", null, null, true),
    part("hotkey", "Hotkey", "ok", "Ctrl + Win", "Hold Ctrl + Win to dictate", null, null, true),
];

function part(
    id: string,
    name: string,
    level: HealthPart["level"],
    summary: string,
    headline: string,
    reason: string | null,
    action: HealthPart["action"],
    critical: boolean,
): HealthPart {
    return { id, name, level, summary, headline, reason, action, critical, code: CODES[`${id}:${summary}`] ?? null };
}

function replace(parts: HealthPart[], ...changed: HealthPart[]): HealthPart[] {
    return parts.map((p) => changed.find((c) => c.id === p.id) ?? p);
}

export const HEALTH_SAMPLES: Record<string, Health> = {
    ok: { overall: "ok", headline: "Everything is working", parts: ok },
    degraded: {
        overall: "degraded",
        headline: "AI clean-up is unavailable",
        parts: replace(
            ok,
            part("cleanup", "AI clean-up", "degraded", "Unavailable", "AI clean-up is unavailable",
                "Llama-server exited with code 3. Dictation goes on with the basic clean-up rules.",
                { id: "restart_engine", label: "Try again" }, false),
            part("gpu", "Graphics card", "ok", "Resting", "RTX 4060 Laptop GPU is resting",
                "Graphics card at 81 °C.", null, false),
        ),
    },
    gpuprep: {
        overall: "ok",
        headline: "Everything is working",
        parts: replace(
            ok,
            part("speech", "Speech model", "ok", "Parakeet v3", "Parakeet v3 is ready", "Running on the processor.", null, true),
            part("gpu", "Graphics card", "ok", "Getting ready", "Getting RTX 4060 Laptop GPU ready for speech",
                "Downloading what speech needs to run on it: 340 of 1022 MB. Until then speech runs on the processor.", null, false),
        ),
    },
    lowmem: {
        overall: "degraded",
        headline: "AI clean-up is waiting for free memory",
        parts: replace(
            ok,
            part("cleanup", "AI clean-up", "degraded", "Not enough memory", "AI clean-up is waiting for free memory",
                "Qwen3 4B needs about 3.8 GB of free memory, and 1.9 GB is free now. Dictation goes on with the basic clean-up rules, and clean-up starts by itself once there is room. Closing some programs makes room sooner.",
                { id: "open_models", label: "Open Models" }, false),
        ),
    },
    failed: {
        overall: "failed",
        headline: "The microphone isn't available",
        parts: replace(
            ok,
            part("microphone", "Microphone", "failed", "Unavailable", "The microphone isn't available",
                "No microphone: is one plugged in and allowed? If Windows is blocking it, allow microphone access for desktop apps.",
                { id: "privacy_microphone", label: "Microphone settings" }, true),
        ),
    },
    engine: {
        overall: "failed",
        headline: "LocalFlow can't start its engine",
        parts: replace(
            ok,
            part("engine", "Engine", "failed", "Not running", "LocalFlow can't start its engine",
                "The engine exited (exit code: 1): ModuleNotFoundError: No module named 'onnx_asr'",
                { id: "restart_engine", label: "Restart engine" }, true),
            part("speech", "Speech model", "waiting", "Waiting for the engine", "", null, null, true),
            part("cleanup", "AI clean-up", "waiting", "Waiting for the engine", "", null, null, false),
            part("gpu", "Graphics card", "waiting", "Waiting for the engine", "", null, null, false),
        ),
    },
    safe: {
        overall: "degraded",
        headline: "LocalFlow is in safe mode",
        parts: replace(
            ok,
            part("engine", "Engine", "degraded", "Safe mode", "LocalFlow is in safe mode",
                "The engine stopped three times in two minutes, so it now runs on the processor with no AI clean-up.",
                { id: "restart_engine", label: "Leave safe mode" }, true),
            part("cleanup", "AI clean-up", "off", "Off in safe mode", "AI clean-up is off in safe mode", null, null, false),
        ),
    },
    starting: {
        overall: "starting",
        headline: "Starting up",
        parts: replace(
            ok,
            part("speech", "Speech model", "starting", "Loading", "Loading Parakeet v3", null, null, true),
            part("cleanup", "AI clean-up", "starting", "Loading", "Loading Qwen3 4B", null, null, false),
        ),
    },
};
