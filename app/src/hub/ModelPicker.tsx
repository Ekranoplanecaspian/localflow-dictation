import type { AutoPick, ModelChoice, ModelSwitch } from "./types";

type Props = {
    /** Accessible name for the group, e.g. "Speech model". */
    label: string;
    choices: ModelChoice[];
    change: ModelSwitch | null | undefined;
    /** False while the engine is away: nothing can be chosen. */
    enabled: boolean;
    onPick: (key: string) => void;
    /** Automatic, as the first option: on, and what it picked. Absent where there is no choice. */
    auto?: { on: boolean; pick: AutoPick | null };
    onAuto?: () => void;
};

/**
 * A list of models to move between. Each option says what it is good at, how big it is and
 * whether it is already on this machine, so choosing one that has to download gigabytes is
 * never a surprise. While a change is under way the other options wait.
 */
export function ModelPicker({ label, choices, change, enabled, onPick, auto, onAuto }: Props) {
    const busy = change != null && change.state !== "error";
    const autoOn = auto?.on ?? false;

    return (
        <div className="model-list" role="radiogroup" aria-label={label}>
            {auto && (
                <button
                    type="button"
                    role="radio"
                    aria-checked={autoOn}
                    className={`model-option${autoOn ? " current" : ""}`}
                    disabled={!enabled || autoOn || busy}
                    onClick={() => onAuto?.()}
                >
                    <span className="radio" aria-hidden />
                    <span className="body">
                        <span className="title">
                            Automatic
                            <em className="tag">Recommended</em>
                        </span>
                        <span className="blurb">
                            The most accurate model that is quick enough on this computer, and on
                            whichever of the graphics card or processor it is running on.
                        </span>
                        {autoOn && auto.pick && (
                            <span className="auto-pick">
                                Using <b>{auto.pick.label}</b>: {auto.pick.why}.
                            </span>
                        )}
                    </span>
                    <span className="action">{autoOn ? "On" : "Use"}</span>
                </button>
            )}
            {choices.map((m) => {
                const target = change?.to === m.key ? change : null;
                const inFlight = target != null && target.state !== "error";
                // Under Automatic the model in use can still be clicked: that pins it.
                const selected = m.current && !autoOn;
                const disabled = !enabled || selected || busy;
                return (
                    <button
                        key={m.key}
                        type="button"
                        role="radio"
                        aria-checked={selected}
                        className={`model-option${selected ? " current" : ""}${m.current && autoOn ? " auto-current" : ""}${inFlight ? " working" : ""}`}
                        disabled={disabled}
                        onClick={() => onPick(m.key)}
                    >
                        <span className="radio" aria-hidden />
                        <span className="body">
                            <span className="title">
                                {m.label}
                                {m.recommended && !auto && <em className="tag">Recommended</em>}
                            </span>
                            <span className="blurb">{m.blurb}</span>
                            <span className="facts">
                                {m.speed != null && <Rating name="Speed" value={m.speed} />}
                                {m.accuracy != null && <Rating name="Accuracy" value={m.accuracy} />}
                                {m.languages != null && (
                                    <span>{m.languages === 1 ? "English" : `${m.languages} languages`}</span>
                                )}
                                <span>{formatSize(m.size_gb)}</span>
                            </span>
                            {inFlight && <Progress change={target} />}
                            {target?.state === "error" && (
                                <span className="model-error">
                                    Could not switch: {target.error ?? "unknown error"}. Your previous
                                    model is still in use.
                                </span>
                            )}
                        </span>
                        <span className="action">{actionLabel(m, target, autoOn)}</span>
                    </button>
                );
            })}
        </div>
    );
}

function Rating({ name, value }: { name: string; value: number }) {
    return (
        <span className="rating" title={`${name}: ${value} of 5`}>
            {name}
            <span className="pips" aria-label={`${value} of 5`}>
                {[1, 2, 3, 4, 5].map((i) => (
                    <i key={i} className={i <= value ? "on" : ""} />
                ))}
            </span>
        </span>
    );
}

function Progress({ change }: { change: ModelSwitch }) {
    const loading = change.state === "loading";
    const pct = Math.round(change.progress * 100);
    return (
        <span className="model-progress">
            <span className={`track${loading ? " indeterminate" : ""}`}>
                <span className="fill" style={loading ? undefined : { width: `${pct}%` }} />
            </span>
            <span className="what">{loading ? "Loading the model…" : `Downloading… ${pct}%`}</span>
        </span>
    );
}

function actionLabel(m: ModelChoice, target: ModelSwitch | null, autoOn: boolean): string {
    if (m.current) return autoOn ? "In use · pin" : "In use";
    if (target?.state === "error") return "Try again";
    if (target) return target.state === "loading" ? "Loading" : "Downloading";
    return m.installed ? "Use" : "Download";
}

function formatSize(gb: number): string {
    if (gb < 1) return `${Math.round(gb * 1000)} MB`;
    return `${gb.toFixed(1)} GB`;
}
