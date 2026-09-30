import type { ComputeStatus, Device, GraphicsAdapter, Hardware } from "./types";

type Props = {
    compute: ComputeStatus | undefined;
    /** What clean-up is, when it is not a bundled model the engine can place: "Ollama", "off"... */
    cleanupElsewhere: string | null;
    enabled: boolean;
    onChange: (patch: Partial<Pick<ComputeStatus, "mode" | "temp_limit_c" | "idle_release_min">>) => void;
};

const MODES: { id: ComputeStatus["mode"]; label: string; about: string }[] = [
    {
        id: "adaptive",
        label: "Automatic",
        about:
            "LocalFlow uses the graphics card while it is cool and moves work to the processor as it heats up, when another app is using it, or after LocalFlow has been idle a while.",
    },
    {
        id: "gpu",
        label: "Graphics card",
        about: "Always the graphics card: the fastest dictation, whatever its temperature.",
    },
    {
        id: "cpu",
        label: "Processor only",
        about:
            "Never touch the graphics card. Speech and clean-up run on the processor, which is slower: roughly half a second more on each dictation.",
    },
];

/** Where a model is, as the Hub names it. "vulkan" is AMD or Intel graphics: its own name. */
function whereLabel(device: Device, compute: ComputeStatus): string {
    if (device === "vulkan") return compute.vulkan ?? "Built-in graphics";
    return device === "cuda" ? "Graphics card" : "Processor";
}

/** What each level means, as the headline of the explanation; the engine's reason follows. */
function headline(compute: ComputeStatus): string {
    const cleanupOn = whereLabel(compute.cleanup, compute).toLowerCase();
    // Speech kept on the processor at a level that would allow the card: Parakeet Compact,
    // which is quicker there, or the card's libraries still downloading (B2).
    if ((compute.level === "full" || compute.level === "gentle") && compute.speech === "cpu") {
        return compute.cleanup === "cuda"
            ? "Clean-up on the graphics card, speech on the processor"
            : "Speech and clean-up on the processor";
    }
    switch (compute.level) {
        case "full":
            return "Everything on the graphics card";
        case "gentle":
            return "On the graphics card, letting it rest between words";
        case "light":
            return `Clean-up moved to the ${cleanupOn}`;
        default:
            return compute.cleanup === "vulkan"
                ? `Speech on the processor, clean-up on the ${cleanupOn}`
                : "Nothing on the graphics card";
    }
}

/**
 * Where the models run. The live readout is the reason the rest exists: the user can see the
 * temperature, what LocalFlow did about it, and when.
 */
export function Compute({ compute, cleanupElsewhere, enabled, onChange }: Props) {
    if (!compute) {
        return null;
    }
    const gpu = compute.gpu;
    const limit = compute.temp_limit_c;
    const mode = MODES.find((m) => m.id === compute.mode) ?? MODES[0];

    return (
        <article className="card compute">
            <h2>Graphics card</h2>

            {gpu ? (
                <div className="gpu-now">
                    <span className="gpu-name">{gpu.name ?? "NVIDIA GPU"}</span>
                    <div className="gauges">
                        <Gauge
                            label="Temperature"
                            value={`${gpu.temp_c} °C`}
                            fraction={gpu.temp_c / (gpu.slowdown_c ?? 90)}
                            tone={gpu.temp_c >= limit ? "hot" : gpu.temp_c >= limit - 8 ? "warm" : "cool"}
                        />
                        <Gauge label="Busy" value={`${gpu.util_pct} %`} fraction={gpu.util_pct / 100} />
                        <Gauge
                            label="Memory"
                            value={`${(gpu.mem_used_mb / 1024).toFixed(1)} of ${(gpu.mem_total_mb / 1024).toFixed(0)} GB`}
                            fraction={gpu.mem_used_mb / gpu.mem_total_mb}
                        />
                    </div>
                </div>
            ) : (
                <p className="note">
                    {compute.cleanup === "vulkan"
                        ? `No NVIDIA graphics card was found, so speech runs on the processor and clean-up on the ${whereLabel("vulkan", compute).toLowerCase()}, where it is quicker.`
                        : "No NVIDIA graphics card was found, so everything runs on the processor."}
                </p>
            )}

            {compute.cuda_libs?.state === "downloading" && (
                <p className="note">
                    Getting the graphics card ready for speech: downloading{" "}
                    {Math.round((compute.cuda_libs.done ?? 0) / 1048576)} of{" "}
                    {Math.round((compute.cuda_libs.total ?? 0) / 1048576)} MB. Until then speech runs on the
                    processor.
                </p>
            )}

            <div className="placement">
                <Where what="Speech" where={whereLabel(compute.speech, compute)} gpu={compute.speech !== "cpu"} />
                <Where
                    what="Clean-up"
                    where={cleanupElsewhere ?? whereLabel(compute.cleanup, compute)}
                    gpu={!cleanupElsewhere && compute.cleanup !== "cpu"}
                />
            </div>
            <p className={`placement-why${compute.moving ? " moving" : ""}`}>
                {compute.moving ? (
                    `Moving ${compute.moving}…`
                ) : (
                    <>
                        <b>{headline(compute)}</b>
                        {compute.reason ? ` · ${compute.reason}` : ""}
                    </>
                )}
            </p>

            <div className="segmented" role="radiogroup" aria-label="Where the models run">
                {MODES.map((m) => (
                    <button
                        key={m.id}
                        type="button"
                        role="radio"
                        aria-checked={m.id === compute.mode}
                        className={m.id === compute.mode ? "on" : ""}
                        disabled={!enabled}
                        onClick={() => m.id !== compute.mode && onChange({ mode: m.id })}
                    >
                        {m.label}
                        {m.id === "adaptive" && <em>Recommended</em>}
                    </button>
                ))}
            </div>
            <p className="note">{mode.about}</p>

            {compute.mode === "adaptive" && (
                <>
                    <div className="row">
                        <span className="k">Ease off at</span>
                        <select
                            value={limit}
                            disabled={!enabled}
                            onChange={(e) => onChange({ temp_limit_c: Number(e.target.value) })}
                        >
                            {[70, 75, 80, 85].map((t) => (
                                <option key={t} value={t}>
                                    {t} °C{t === 80 ? " (default)" : ""}
                                </option>
                            ))}
                        </select>
                    </div>
                    <div className="row">
                        <span className="k">Free it after</span>
                        <select
                            value={compute.idle_release_min}
                            disabled={!enabled}
                            onChange={(e) => onChange({ idle_release_min: Number(e.target.value) })}
                        >
                            {[0, 5, 10, 30, 60].map((m) => (
                                <option key={m} value={m}>
                                    {m === 0 ? "never" : `${m} minutes unused`}
                                </option>
                            ))}
                        </select>
                    </div>
                    <p className="note">
                        Clean-up moves to the {whereLabel(compute.cleanup_off_card ?? "cpu", compute).toLowerCase()} at{" "}
                        {limit} °C, and speech follows to the processor at {limit + 6} °C.
                        They come back once it has been cool for a while. A freed graphics card is
                        taken back the moment you start dictating.
                    </p>
                </>
            )}

            {compute.recent.length > 0 && (
                <ul className="moves" aria-label="Recent moves">
                    {compute.recent.map((m) => (
                        <li key={`${m.at}-${m.what}`}>
                            <time>{clock(m.at)}</time>
                            <span>
                                {m.model
                                    ? `${capitalise(m.what)} switched to ${m.model} on the ${whereLabel(m.to, compute).toLowerCase()}`
                                    : `${capitalise(m.what)} moved to the ${whereLabel(m.to, compute).toLowerCase()}`}
                            </span>
                            <i>{m.reason}</i>
                        </li>
                    ))}
                </ul>
            )}

            {compute.hardware && <ThisPc hw={compute.hardware} />}
        </article>
    );
}

/** What this computer has, folded away: it explains the choices above, but is rarely read. */
function ThisPc({ hw }: { hw: Hardware }) {
    const gpus = hw.gpus ?? [];
    const cores = hw.cpu?.cores ?? hw.cpu_cores;
    const glance = [`${cores} cores`, `${Math.round(hw.ram_gb)} GB memory`];
    if (hw.gpus) {
        glance.push(gpus.length === 1 ? "1 graphics adapter" : `${gpus.length} graphics adapters`);
    }
    return (
        <details className="this-pc">
            <summary>
                <span>This PC</span>
                <span className="glance">{glance.join(" · ")}</span>
            </summary>
            <div className="row">
                <span className="k">Processor</span>
                <span className="v">
                    {hw.cpu ? `${hw.cpu.name}: ${hw.cpu.cores} cores, ${hw.cpu.threads} threads` : `${cores} cores`}
                    {hw.cpu?.isa && hw.cpu.isa.length > 0 && (
                        <span className="sub">{hw.cpu.isa.map((s) => s.toUpperCase()).join(" · ")}</span>
                    )}
                </span>
            </div>
            <div className="row">
                <span className="k">Memory</span>
                <span className="v">
                    {hw.ram_gb.toFixed(0)} GB
                    {hw.ram_free_gb != null && `, ${hw.ram_free_gb.toFixed(1)} GB free now`}
                </span>
            </div>
            {hw.gpus &&
                (gpus.length === 0 ? (
                    <div className="row">
                        <span className="k">Graphics</span>
                        <span className="v">none found</span>
                    </div>
                ) : (
                    gpus.map((g, i) => (
                        <div className="row" key={`${g.name}-${i}`}>
                            <span className="k">{i === 0 ? "Graphics" : ""}</span>
                            <span className="v">
                                {g.name}
                                <span className="sub">{adapterNote(g)}</span>
                            </span>
                        </div>
                    ))
                ))}
            {hw.disk_free_gb != null && (
                <div className="row">
                    <span className="k">Disk</span>
                    <span className="v">{hw.disk_free_gb} GB free where models are kept</span>
                </div>
            )}
        </details>
    );
}

function adapterNote(g: GraphicsAdapter): string {
    const gb = (mb: number) => `${(mb / 1024).toFixed(mb < 10240 ? 1 : 0)} GB`;
    const used = g.vendor === "nvidia" ? "LocalFlow can use it" : "LocalFlow can use it for clean-up";
    return g.integrated
        ? `built in, shares up to ${gb(g.shared_mb)} of memory · ${used}`
        : `${gb(g.dedicated_mb)} of its own · ${used}`;
}

function Gauge({ label, value, fraction, tone }: { label: string; value: string; fraction: number; tone?: string }) {
    const pct = Math.max(2, Math.min(100, Math.round(fraction * 100)));
    return (
        <div className={`gauge${tone ? ` ${tone}` : ""}`}>
            <span className="label">{label}</span>
            <span className="value">{value}</span>
            <span className="track" aria-hidden>
                <span className="fill" style={{ width: `${pct}%` }} />
            </span>
        </div>
    );
}

function Where({ what, where, gpu }: { what: string; where: string; gpu: boolean }) {
    return (
        <div className="where">
            <span className="what">{what}</span>
            <span className={`device${gpu ? " gpu" : ""}`}>{where}</span>
        </div>
    );
}

function capitalise(text: string): string {
    return text ? text[0].toUpperCase() + text.slice(1) : "";
}

function clock(epochSeconds: number): string {
    return new Date(epochSeconds * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
