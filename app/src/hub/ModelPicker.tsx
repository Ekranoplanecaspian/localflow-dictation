import { useEffect, useState } from "react";
import { downloadLine, FIT_LABEL, formatSize, sentence } from "./downloads";
import type { AutoPick, Download, ModelChoice, ModelSwitch } from "./types";

type Props = {
    /** Accessible name for the group, e.g. "Speech model". */
    label: string;
    choices: ModelChoice[];
    change: ModelSwitch | null | undefined;
    /** Every download the engine has, to show each model's own. */
    downloads: Download[];
    /** False while the engine is away: nothing can be chosen. */
    enabled: boolean;
    onPick: (key: string) => void;
    /** Download without switching; stop a download; take a model off the PC. */
    onDownload: (key: string) => void;
    onCancel: (id: string) => void;
    onRemove: (key: string) => void;
    /** Automatic, as the first option: on, and what it picked. Absent where there is no choice. */
    auto?: { on: boolean; pick: AutoPick | null };
    onAuto?: () => void;
};

/**
 * The model library for one kind of model. Each model says what it is good at, how it suits
 * this PC, whether it is here and what it takes on disk - and it can be downloaded to have it
 * ready, used, stopped while downloading, or removed. Choosing one that has to download
 * gigabytes is never a surprise, and while a change is under way the other options wait.
 */
export function ModelPicker(props: Props) {
    const { label, choices, change, downloads, enabled, onPick, auto, onAuto } = props;
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
            {choices.map((m) => (
                <Model
                    key={m.key}
                    m={m}
                    {...props}
                    busy={busy}
                    autoOn={autoOn}
                    job={downloads.find((d) => d.id === m.download) ?? null}
                    target={change?.to === m.key ? change : null}
                    enabled={enabled}
                    onPick={onPick}
                />
            ))}
        </div>
    );
}

function Model({
    m,
    busy,
    autoOn,
    job,
    target,
    enabled,
    onPick,
    onDownload,
    onCancel,
    onRemove,
}: Props & {
    m: ModelChoice;
    busy: boolean;
    autoOn: boolean;
    job: Download | null;
    target: ModelSwitch | null;
}) {
    const [confirm, setConfirm] = useState(false);
    useEffect(() => {
        if (!confirm) return;
        const t = window.setTimeout(() => setConfirm(false), 4000);
        return () => window.clearTimeout(t);
    }, [confirm]);

    const inFlight = target != null && target.state !== "error";
    // Under Automatic the model in use can still be clicked: that pins it.
    const selected = m.current && !autoOn;
    const tooBig = m.fit?.rating === "too-big";
    const disabled = !enabled || selected || busy;
    const downloading = job != null && (job.state === "downloading" || job.state === "queued");
    const classes = [
        "model-item",
        selected ? "current" : "",
        m.current && autoOn ? "auto-current" : "",
        inFlight || downloading ? "working" : "",
    ].join(" ");

    return (
        <div className={classes}>
            <button
                type="button"
                role="radio"
                aria-checked={selected}
                className="model-option"
                disabled={disabled}
                onClick={() => onPick(m.key)}
            >
                <span className="radio" aria-hidden />
                <span className="body">
                    <span className="title">
                        {m.label}
                        {m.current && <em className="tag">In use</em>}
                        {m.recommended && !m.current && <em className="tag">Recommended</em>}
                    </span>
                    <span className="blurb">{m.blurb}</span>
                    <span className="facts">
                        {m.speed != null && <Rating name="Speed" value={m.speed} />}
                        {m.accuracy != null && <Rating name="Accuracy" value={m.accuracy} />}
                        {m.languages != null && (
                            <span>{m.languages === 1 ? "English" : `${m.languages} languages`}</span>
                        )}
                    </span>
                    {m.fit && (
                        <span className={`fit ${m.fit.rating}`}>
                            <b>{FIT_LABEL[m.fit.rating]}</b>
                            <span>{sentence(m.fit.why)}.</span>
                        </span>
                    )}
                    <span className={`here ${m.installed ? "yes" : ""}`}>
                        {m.installed
                            ? `On this PC · ${formatSize(m.disk_gb || m.size_gb)}`
                            : `Not downloaded · ${formatSize(m.size_gb)} download`}
                    </span>
                    {downloading && job ? (
                        <Progress value={job.state === "queued" ? null : job.progress} text={downloadLine(job)} />
                    ) : (
                        inFlight &&
                        target.state === "loading" && <Progress value={null} text="Loading the model…" />
                    )}
                    {target?.state === "error" && (
                        <span className="model-error">
                            Could not switch: {target.error ?? "unknown error"}. Your previous model is
                            still in use.
                        </span>
                    )}
                    {job?.state === "error" && <span className="model-error">{downloadLine(job)}</span>}
                </span>
                <span className="action">{actionLabel(m, target, autoOn, tooBig, downloading)}</span>
            </button>
            {(downloading || !m.installed || m.removable) && (
                <div className="model-actions">
                    {downloading && job?.cancellable && (
                        <button type="button" className="ghost small" onClick={() => onCancel(job.id)}>
                            Cancel download
                        </button>
                    )}
                    {!m.installed && !downloading && !inFlight && (
                        <button
                            type="button"
                            className="ghost small"
                            disabled={!enabled}
                            onClick={() => onDownload(m.key)}
                            title="Download it now to have it ready; the model you use now stays in use"
                        >
                            Download only ({formatSize(m.size_gb)})
                        </button>
                    )}
                    {m.removable && !downloading && (
                        <button
                            type="button"
                            className={`ghost small${confirm ? " danger" : ""}`}
                            disabled={!enabled}
                            onClick={() => (confirm ? (setConfirm(false), onRemove(m.key)) : setConfirm(true))}
                        >
                            {confirm ? `Remove it and free ${formatSize(m.disk_gb ?? 0)}?` : "Remove"}
                        </button>
                    )}
                </div>
            )}
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

/** A progress bar with its words; `value` null runs the bar without a known end. */
export function Progress({ value, text }: { value: number | null; text: string }) {
    return (
        <span className="model-progress">
            <span className={`track${value == null ? " indeterminate" : ""}`}>
                <span className="fill" style={value == null ? undefined : { width: `${Math.round(value * 100)}%` }} />
            </span>
            <span className="what">{text}</span>
        </span>
    );
}

function actionLabel(m: ModelChoice, target: ModelSwitch | null, autoOn: boolean, tooBig: boolean,
                     downloading: boolean): string {
    if (m.current) return autoOn ? "In use · pin" : "In use";
    if (target?.state === "error") return "Try again";
    if (target) return target.state === "loading" ? "Loading" : "Downloading";
    if (m.installed) return "Use";
    if (downloading) return "Use when downloaded"; // choosing it joins the download
    return tooBig ? "Download and use anyway" : "Download and use";
}
