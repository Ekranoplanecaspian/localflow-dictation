import { StatusCard, type Go } from "./Status";
import type { Phase, SectionProps } from "./types";

/** Fourteen days of dictation counts, as a small bar chart. */
function Sparkline({ daily }: { daily: number[] }) {
    const peak = Math.max(1, ...daily);
    return (
        <div className="spark" aria-hidden>
            {daily.map((n, i) => (
                <span
                    key={i}
                    style={{ height: `${Math.max(3, (n / peak) * 100)}%` }}
                    className={n === 0 ? "empty" : ""}
                    title={`${n} dictations`}
                />
            ))}
        </div>
    );
}

function Stat({ value, label, hint }: { value: string; label: string; hint?: string }) {
    return (
        <div className="stat">
            <b>{value}</b>
            <span>{label}</span>
            {hint && <i>{hint}</i>}
        </div>
    );
}

export function Overview({
    data,
    phase,
    partial,
    onChange,
    say,
    go,
}: SectionProps & { phase: Phase; partial: string; go: Go }) {
    const { stats, link, settings } = data;
    const last = data.history[0];
    const hotkey = settings.hotkey.join(" + ").replace(/\bwin\b/i, "Win").replace(/\bctrl\b/i, "Ctrl");

    return (
        <>
            <header className="pane-head">
                <h1>Overview</h1>
                <p>
                    Hold <kbd>{hotkey}</kbd> and speak.{" "}
                    {settings.double_tap_hands_free && "Double-tap to keep it listening."}
                </p>
            </header>

            <StatusCard health={data.health} link={link} go={go} say={say} onChange={onChange} />

            <section className={`live ${phase}`}>
                {phase === "recording" && <span className="live-text">{partial || "Listening…"}</span>}
                {phase === "finishing" && <span className="live-text">{partial || "Transcribing…"}</span>}
                {phase === "idle" &&
                    (last ? (
                        <span className="live-text said">{last.text}</span>
                    ) : (
                        <span className="live-text muted">Nothing dictated yet.</span>
                    ))}
                {phase === "idle" && last && (
                    <span className="live-meta">
                        {last.app || "the caret"} · {Math.round(last.ms)} ms
                        {last.used_llm ? " · auto-edited" : ""}
                    </span>
                )}
            </section>

            <div className="stats">
                <Stat value={stats.dictations.toLocaleString()} label="dictations" />
                <Stat value={stats.words.toLocaleString()} label="words" />
                <Stat
                    value={stats.words_per_minute ? `${stats.words_per_minute}` : "—"}
                    label="words per minute"
                    hint="while speaking"
                />
                <Stat
                    value={stats.p50_ms ? `${stats.p50_ms} ms` : "—"}
                    label="release to text"
                    hint={stats.p95_ms ? `p95 ${stats.p95_ms} ms` : undefined}
                />
            </div>

            <div className="cards">
                <article className="card">
                    <h2>Last 14 days</h2>
                    <Sparkline daily={stats.daily} />
                    <p className="note">
                        {stats.with_auto_edits} of {stats.dictations} went through the clean-up model.
                    </p>
                </article>

                <article className="card">
                    <h2>Where you dictate</h2>
                    {stats.top_apps.length === 0 && <p className="note">No dictations yet.</p>}
                    <ul className="bars">
                        {stats.top_apps.map(([app, n]) => (
                            <li key={app}>
                                <span className="name">{app.replace(/\.exe$/i, "")}</span>
                                <span
                                    className="bar"
                                    style={{ width: `${(n / stats.top_apps[0][1]) * 100}%` }}
                                />
                                <span className="n">{n}</span>
                            </li>
                        ))}
                    </ul>
                </article>

            </div>
        </>
    );
}
