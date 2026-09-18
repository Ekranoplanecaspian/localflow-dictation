//! Microphone capture: always on, 16 kHz mono int16 out, with a pre-roll buffer.
//!
//! The stream runs from the moment the app starts, not from the moment the hotkey is pressed:
//! opening a WASAPI stream takes tens of milliseconds and the first words of a dictation live
//! in that gap. Instead the last half second is always kept in a ring buffer and flushed to the
//! engine the instant a take begins, so "hello" is not clipped to "ello".
//!
//! The device's mix format is whatever WASAPI shared mode says (usually 48 kHz stereo float),
//! so every block is downmixed and resampled here; the engine only ever sees 16 kHz mono.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde_json::json;
use tauri::{AppHandle, Emitter};

use crate::session::SessionManager;

pub const SAMPLE_RATE: u32 = 16_000;
/// Audio kept from before the hotkey press.
pub const PREROLL_MS: usize = 500;
const PREROLL_SAMPLES: usize = SAMPLE_RATE as usize * PREROLL_MS / 1000;
/// The flow bar's waveform wants a steady stream of levels, not one per audio block.
const METER_HZ: u64 = 50;
/// No audio for this long while a stream is supposed to be running means the device is gone
/// (unplugged, or the machine came back from sleep with a different default).
const SILENCE_REBUILD: Duration = Duration::from_secs(3);

struct Shared {
    recording: AtomicBool,
    /// Set with `recording`; the first callback afterwards flushes the pre-roll.
    arm: AtomicBool,
    /// Most recent peak, as f32 bits, for the level meter.
    level: AtomicU32,
    /// Counts 16 kHz samples produced, so the supervisor can tell a live stream from a dead one.
    produced: AtomicU64,
    preroll: Mutex<VecDeque<i16>>,
    /// Name of the device the current stream is on, to notice a default-device change.
    device: Mutex<String>,
    rebuild: AtomicBool,
    stop: AtomicBool,
}

#[derive(Clone)]
pub struct Capture {
    shared: Arc<Shared>,
}

impl Capture {
    pub fn start(app: AppHandle, sessions: Arc<SessionManager>) -> Capture {
        let shared = Arc::new(Shared {
            recording: AtomicBool::new(false),
            arm: AtomicBool::new(false),
            level: AtomicU32::new(0),
            produced: AtomicU64::new(0),
            preroll: Mutex::new(VecDeque::with_capacity(PREROLL_SAMPLES + 1024)),
            device: Mutex::new(String::new()),
            rebuild: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        });
        {
            let shared = shared.clone();
            let app = app.clone();
            std::thread::Builder::new()
                .name("audio-supervisor".into())
                .spawn(move || supervise(app, shared, sessions))
                .expect("audio thread");
        }
        {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name("level-meter".into())
                .spawn(move || meter(app, shared))
                .expect("meter thread");
        }
        Capture { shared }
    }

    /// Begin streaming to the engine, starting with the pre-roll.
    pub fn begin(&self) {
        self.shared.arm.store(true, Ordering::SeqCst);
        self.shared.recording.store(true, Ordering::SeqCst);
    }

    pub fn end(&self) {
        self.shared.recording.store(false, Ordering::SeqCst);
        self.shared.arm.store(false, Ordering::SeqCst);
    }

    pub fn level(&self) -> f32 {
        f32::from_bits(self.shared.level.load(Ordering::Relaxed))
    }

    pub fn device_name(&self) -> String {
        self.shared.device.lock().unwrap().clone()
    }

    /// True when a stream is open on a device.
    pub fn is_live(&self) -> bool {
        !self.shared.device.lock().unwrap().is_empty()
    }

    /// Reopen the stream, after the chosen microphone changed.
    pub fn rebuild(&self) {
        self.shared.rebuild.store(true, Ordering::SeqCst);
    }

    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------------------------

fn meter(app: AppHandle, shared: Arc<Shared>) {
    let period = Duration::from_millis(1000 / METER_HZ);
    let mut smoothed = 0f32;
    let mut settling = false;
    loop {
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
        std::thread::sleep(period);
        let recording = shared.recording.load(Ordering::Relaxed);
        if !recording && !settling {
            continue;
        }
        settling = recording || smoothed > 0.001;
        let level = f32::from_bits(shared.level.load(Ordering::Relaxed));
        // Fast attack, slow release: the bar should jump on a syllable and fall smoothly.
        smoothed = if level > smoothed { level } else { smoothed * 0.82 + level * 0.18 };
        let _ = app.emit("level", json!({"level": smoothed, "recording": recording}));
    }
}

fn supervise(app: AppHandle, shared: Arc<Shared>, sessions: Arc<SessionManager>) {
    let mut last_error: Option<String> = None;
    let mut last_device = String::new();
    // A stream that dies immediately must not be reopened in a tight loop: when several
    // instances were fighting over the microphone this rebuilt 115 times a minute.
    let mut backoff = Duration::ZERO;
    loop {
        if !backoff.is_zero() {
            std::thread::sleep(backoff);
        }
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
        match build_stream(&shared, &sessions) {
            Ok((stream, name)) => {
                if stream.play().is_err() {
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }
                *shared.device.lock().unwrap() = name.clone();
                last_error = None;
                if name != last_device {
                    crate::shell_log!("microphone: {name}");
                    last_device = name.clone();
                }
                let _ = app.emit("audio-device", json!({"device": name, "ok": true}));
                let opened = std::time::Instant::now();

                // Watch the stream: a dead device stops producing samples, and the default
                // device can change under us (a headset connecting, or sleep and resume).
                let mut last_count = shared.produced.load(Ordering::Relaxed);
                let mut quiet_for = Duration::ZERO;
                loop {
                    std::thread::sleep(Duration::from_millis(500));
                    if shared.stop.load(Ordering::Relaxed) {
                        return;
                    }
                    if shared.rebuild.swap(false, Ordering::SeqCst) {
                        break;
                    }
                    let count = shared.produced.load(Ordering::Relaxed);
                    if count == last_count {
                        quiet_for += Duration::from_millis(500);
                        if quiet_for >= SILENCE_REBUILD {
                            break;
                        }
                    } else {
                        quiet_for = Duration::ZERO;
                        last_count = count;
                    }
                    if current_device_name() != *shared.device.lock().unwrap() {
                        break;
                    }
                }
                drop(stream);
                shared.device.lock().unwrap().clear();
                // A stream that lasted a while was healthy; one that died at once was not.
                backoff = if opened.elapsed() > Duration::from_secs(10) {
                    Duration::ZERO
                } else {
                    (backoff * 2).clamp(Duration::from_millis(250), Duration::from_secs(5))
                };
                if !backoff.is_zero() {
                    crate::shell_log!("microphone stream lasted {:?}; retrying in {:?}", opened.elapsed(), backoff);
                }
            }
            Err(e) => {
                let msg = e.to_string();
                if last_error.as_deref() != Some(msg.as_str()) {
                    crate::shell_log!("microphone unavailable: {msg}");
                    let _ =
                        app.emit("audio-device", json!({"device": null, "ok": false, "error": msg}));
                    last_error = Some(msg);
                }
                std::thread::sleep(Duration::from_secs(2));
            }
        }
    }
}

fn device_name(device: &cpal::Device) -> String {
    device
        .description()
        .map(|d| d.name().to_owned())
        .unwrap_or_else(|_| "default".into())
}

/// Every microphone Windows will offer, for the Hub's picker.
pub fn input_devices() -> Vec<String> {
    let host = cpal::default_host();
    match host.input_devices() {
        Ok(devices) => devices.map(|d| device_name(&d)).collect(),
        Err(_) => Vec::new(),
    }
}

/// The device the user asked for, or the system default when they did not ask.
fn chosen_device() -> Option<cpal::Device> {
    let host = cpal::default_host();
    let wanted = crate::settings::load().microphone.trim().to_lowercase();
    if !wanted.is_empty() {
        if let Ok(devices) = host.input_devices() {
            for device in devices {
                if device_name(&device).to_lowercase().contains(&wanted) {
                    return Some(device);
                }
            }
        }
        // The chosen microphone is unplugged; the default is better than nothing at all.
    }
    host.default_input_device()
}

fn current_device_name() -> String {
    chosen_device().map(|d| device_name(&d)).unwrap_or_default()
}

fn build_stream(
    shared: &Arc<Shared>,
    sessions: &Arc<SessionManager>,
) -> Result<(cpal::Stream, String), anyhow::Error> {
    let device = chosen_device()
        .ok_or_else(|| anyhow::anyhow!("no microphone: is one plugged in and allowed?"))?;
    let name = device_name(&device);
    let supported = device.default_input_config()?;
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let channels = config.channels as usize;
    let rate = config.sample_rate;

    let mut pipe = Pipeline::new(rate, channels, shared.clone(), sessions.clone())?;
    let rebuild = shared.clone();
    let err_fn = move |err: cpal::Error| {
        // A device error is not recoverable in place; ask the supervisor for a new stream.
        // Through the log, not `eprintln!`: this runs on the audio callback thread, where a
        // panic from an unwritable stderr would take the microphone down with it - and a
        // windowed process launched from Explorer has no stderr. It belongs in the log
        // anyway, since a device that keeps failing is exactly what a report of "it stopped
        // hearing me" needs evidence for.
        crate::shell_log!("[audio] {err}");
        rebuild.rebuild.store(true, Ordering::SeqCst);
    };

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _: &_| pipe.push(data),
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _: &_| {
                let floats: Vec<f32> = data.iter().map(|s| *s as f32 / 32768.0).collect();
                pipe.push(&floats)
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            config,
            move |data: &[u16], _: &_| {
                let floats: Vec<f32> =
                    data.iter().map(|s| (*s as f32 - 32768.0) / 32768.0).collect();
                pipe.push(&floats)
            },
            err_fn,
            None,
        )?,
        other => return Err(anyhow::anyhow!("unsupported sample format {other:?}")),
    };
    Ok((stream, name))
}

// ---------------------------------------------------------------------------------------------
// downmix, resample, dispatch

const TAPS: usize = 32;
const HALF: usize = TAPS / 2;
const PHASES: usize = 256;

/// Windowed-sinc resampler, mono, arbitrary ratio.
///
/// The microphone runs at whatever WASAPI's shared mix format says (48 kHz here) and the speech
/// model wants 16 kHz. Dropping every third sample would fold everything above 8 kHz back into
/// the speech band, so each output sample is a 32-tap windowed sinc over the input, which
/// interpolates and low-passes in one step. The kernel is precomputed at 256 fractional phases,
/// so the hot loop is 32 multiply-adds per output sample.
pub struct Resampler {
    /// Input samples consumed per output sample.
    step: f64,
    /// Read position within `buf`, in input samples.
    pos: f64,
    buf: Vec<f32>,
    table: Vec<f32>,
}

impl Resampler {
    pub fn new(from_rate: u32, to_rate: u32) -> Resampler {
        // A little below Nyquist, so the transition band is inside the filter rather than at
        // the very edge where the window's ripple lives.
        let cutoff = (to_rate as f64 / from_rate as f64).min(1.0) * 0.92;
        let mut table = vec![0f32; PHASES * TAPS];
        for phase in 0..PHASES {
            let frac = phase as f64 / PHASES as f64;
            let mut sum = 0f64;
            for j in 0..TAPS {
                // Offset of tap j from the sample position: -HALF+1 ..= HALF input samples.
                let t = frac - (j as f64 - HALF as f64 + 1.0);
                let v = cutoff * sinc(cutoff * t) * blackman(t);
                table[phase * TAPS + j] = v as f32;
                sum += v;
            }
            // Normalise each phase to unit gain, or the output amplitude would ripple.
            if sum.abs() > 1e-9 {
                for j in 0..TAPS {
                    table[phase * TAPS + j] /= sum as f32;
                }
            }
        }
        Resampler {
            step: from_rate as f64 / to_rate as f64,
            pos: HALF as f64,
            // Leading zeros so the very first sample has history behind it.
            buf: vec![0.0; HALF],
            table,
        }
    }

    /// Append input and write however many output samples that produced.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<i16>) {
        self.buf.extend_from_slice(input);
        // The newest input a kernel centred at `pos` needs is floor(pos) + HALF.
        while (self.pos.floor() as usize) + HALF < self.buf.len() {
            let base = self.pos.floor() as usize;
            let frac = self.pos - base as f64;
            let phase = ((frac * PHASES as f64) as usize).min(PHASES - 1);
            let row = &self.table[phase * TAPS..phase * TAPS + TAPS];
            let start = base + 1 - HALF;
            let window = &self.buf[start..start + TAPS];
            let mut acc = 0f32;
            for (w, k) in window.iter().zip(row.iter()) {
                acc += w * k;
            }
            out.push(to_i16(acc));
            self.pos += self.step;
        }
        // Drop what no future kernel can reach.
        let keep_from = (self.pos.floor() as usize + 1).saturating_sub(HALF);
        if keep_from > 0 {
            self.buf.drain(..keep_from);
            self.pos -= keep_from as f64;
        }
    }
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

fn blackman(t: f64) -> f64 {
    let n = HALF as f64;
    if t.abs() > n {
        return 0.0;
    }
    let x = std::f64::consts::PI * t / n;
    0.42 + 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
}

struct Pipeline {
    channels: usize,
    /// None when the device already runs at 16 kHz.
    resampler: Option<Resampler>,
    mono: Vec<f32>,
    scratch: Vec<i16>,
    shared: Arc<Shared>,
    sessions: Arc<SessionManager>,
}

impl Pipeline {
    fn new(
        rate: u32,
        channels: usize,
        shared: Arc<Shared>,
        sessions: Arc<SessionManager>,
    ) -> Result<Self, anyhow::Error> {
        if channels == 0 {
            anyhow::bail!("the microphone reports no channels");
        }
        Ok(Pipeline {
            channels,
            resampler: (rate != SAMPLE_RATE).then(|| Resampler::new(rate, SAMPLE_RATE)),
            mono: Vec::with_capacity(2048),
            scratch: Vec::with_capacity(2048),
            shared,
            sessions,
        })
    }

    /// Called from the audio callback: no blocking, no waiting on the socket.
    fn push(&mut self, interleaved: &[f32]) {
        self.mono.clear();
        let mut peak = 0f32;
        for frame in interleaved.chunks(self.channels) {
            let m = frame.iter().sum::<f32>() / frame.len() as f32;
            peak = peak.max(m.abs());
            self.mono.push(m);
        }
        self.shared.level.store(peak.to_bits(), Ordering::Relaxed);

        self.scratch.clear();
        match self.resampler.as_mut() {
            Some(r) => {
                // The resampler borrows `mono` while writing `scratch`; swap it out so both
                // borrows are disjoint without copying.
                let mono = std::mem::take(&mut self.mono);
                r.process(&mono, &mut self.scratch);
                self.mono = mono;
            }
            None => {
                let mono = std::mem::take(&mut self.mono);
                self.scratch.extend(mono.iter().map(|s| to_i16(*s)));
                self.mono = mono;
            }
        }
        self.emit();
    }

    fn emit(&mut self) {
        if self.scratch.is_empty() {
            return;
        }
        self.shared.produced.fetch_add(self.scratch.len() as u64, Ordering::Relaxed);

        let mut ring = self.shared.preroll.lock().unwrap();
        ring.extend(self.scratch.iter().copied());
        while ring.len() > PREROLL_SAMPLES {
            ring.pop_front();
        }

        if self.shared.recording.load(Ordering::Relaxed) {
            if self.shared.arm.swap(false, Ordering::SeqCst) {
                // First block of a take: send everything we have, including what was said just
                // before the key went down.
                let pcm: Vec<i16> = ring.iter().copied().collect();
                drop(ring);
                self.sessions.feed(&pcm);
            } else {
                drop(ring);
                self.sessions.feed(&self.scratch);
            }
        }
    }
}

fn to_i16(s: f32) -> i16 {
    (s.clamp(-1.0, 1.0) * 32767.0) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1 kHz tone at 48 kHz must come out as a 1 kHz tone at 16 kHz with its amplitude
    /// intact: the right frequency, and no attenuation from a badly normalised kernel.
    #[test]
    fn resamples_a_tone_without_losing_it() {
        let mut r = Resampler::new(48_000, 16_000);
        let mut out = Vec::new();
        let input: Vec<f32> = (0..48_000)
            .map(|n| (2.0 * std::f64::consts::PI * 1000.0 * n as f64 / 48_000.0).sin() as f32 * 0.5)
            .collect();
        for block in input.chunks(480) {
            r.process(block, &mut out);
        }
        assert!(
            (out.len() as i64 - 16_000).abs() < 40,
            "expected about 16000 samples, got {}",
            out.len()
        );
        let tail = &out[1000..15_000];
        let peak = tail.iter().map(|s| s.abs() as f32 / 32768.0).fold(0.0, f32::max);
        assert!((peak - 0.5).abs() < 0.02, "amplitude drifted: {peak}");
        let crossings = tail.windows(2).filter(|w| (w[0] < 0) != (w[1] < 0)).count();
        let seconds = tail.len() as f64 / SAMPLE_RATE as f64;
        let freq = crossings as f64 / 2.0 / seconds;
        assert!((freq - 1000.0).abs() < 5.0, "frequency drifted: {freq}");
    }

    /// Anything above the new Nyquist must be filtered out rather than folded back into the
    /// speech band: that is the whole reason for a sinc kernel instead of dropping samples.
    #[test]
    fn rejects_frequencies_above_the_new_nyquist() {
        let mut r = Resampler::new(48_000, 16_000);
        let mut out = Vec::new();
        let input: Vec<f32> = (0..48_000)
            .map(|n| {
                (2.0 * std::f64::consts::PI * 12_000.0 * n as f64 / 48_000.0).sin() as f32 * 0.5
            })
            .collect();
        for block in input.chunks(480) {
            r.process(block, &mut out);
        }
        let tail = &out[2000..14_000];
        let rms = (tail.iter().map(|s| (*s as f64 / 32768.0).powi(2)).sum::<f64>()
            / tail.len() as f64)
            .sqrt();
        assert!(rms < 0.01, "12 kHz leaked through at rms {rms}");
    }
}
