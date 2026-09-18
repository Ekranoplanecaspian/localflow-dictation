"""Streaming voice activity detection and phrase segmentation.

Silero VAD (ONNX, CPU) scores 32 ms windows. The Segmenter turns those scores into
closed phrases as the audio arrives: a phrase closes after ~450 ms of silence, is padded
a little on both sides, and is never longer than the speech model's comfortable chunk.
Closed phrases are transcribed immediately while the user is still talking; only the
open tail is left for the moment the hotkey is released.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
HOP = 512  # samples per VAD window at 16 kHz (32 ms)
CONTEXT = 64  # samples of look-back Silero v5 expects in front of each window


class SileroVad:
    """Stateful, one-window-at-a-time Silero v5 wrapper."""

    def __init__(self):
        import onnx_asr

        vad = onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"])
        self._sess = vad._model  # reuse onnx-asr's downloaded session
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        """window: exactly HOP float32 samples. Returns P(speech)."""
        frame = np.concatenate([self._context, window])[None, :].astype(np.float32, copy=False)
        out, self._state = self._sess.run(["output", "stateN"],
                                          {"input": frame, "state": self._state, "sr": np.array([SAMPLE_RATE], dtype=np.int64)})
        self._context = window[-CONTEXT:]
        return float(out[0, 0])


@dataclass(frozen=True)
class Segment:
    start: int  # sample index into the session buffer, inclusive
    end: int  # exclusive

    @property
    def seconds(self) -> float:
        return (self.end - self.start) / SAMPLE_RATE


class Segmenter:
    """Feed 20 ms blocks in arrival order; get closed phrases back. `finish()` returns the tail."""

    def __init__(self, vad: SileroVad, *, threshold: float = 0.5, neg_threshold: float = 0.35,
                 min_silence_ms: int = 450, min_speech_ms: int = 250, pad_ms: int = 160,
                 max_segment_s: float = 20.0):
        self.vad = vad
        self.threshold, self.neg_threshold = threshold, neg_threshold
        self.min_silence = min_silence_ms * SAMPLE_RATE // 1000
        self.min_speech = min_speech_ms * SAMPLE_RATE // 1000
        self.pad = pad_ms * SAMPLE_RATE // 1000
        self.max_segment = int(max_segment_s * SAMPLE_RATE)
        self._pending = np.zeros(0, dtype=np.float32)  # samples not yet scored (less than one HOP)
        self.position = 0  # total samples scored so far
        self.last_end = 0  # end of the last closed segment
        self._in_speech = False
        self._speech_start = 0
        self._silence_run = 0
        self.last_speech_pos = -1  # end position of the last window that scored as speech
        self.vad.reset()

    def speech_within_last(self, ms: int) -> bool:
        """True if any window in the last `ms` of scored audio looked like speech (or if audio
        is still unscored, to stay on the safe side)."""
        if self.last_speech_pos < 0:
            return False
        return self.position - self.last_speech_pos < ms * SAMPLE_RATE // 1000

    def feed(self, block: np.ndarray) -> list[Segment]:
        closed: list[Segment] = []
        buf = np.concatenate([self._pending, block]) if self._pending.size else block
        n = (buf.size // HOP) * HOP
        for i in range(0, n, HOP):
            closed += self._score(buf[i:i + HOP])
        self._pending = buf[n:]
        return closed

    def _score(self, window: np.ndarray) -> list[Segment]:
        p = self.vad(window)
        self.position += HOP
        if p >= self.threshold:
            self.last_speech_pos = self.position
        closed: list[Segment] = []
        if not self._in_speech:
            if p >= self.threshold:
                self._in_speech = True
                self._speech_start = max(self.position - HOP - self.pad, self.last_end)
                self._silence_run = 0
            return closed
        # in speech
        if p >= self.threshold:
            self._silence_run = 0
        elif p < self.neg_threshold:
            self._silence_run += HOP
        if self._silence_run >= self.min_silence:
            end = min(self.position - self._silence_run + self.pad, self.position)
            closed += self._close(end)
        elif self.position - self._speech_start >= self.max_segment:
            closed += self._close(self.position)
        return closed

    def _close(self, end: int) -> list[Segment]:
        self._in_speech = False
        self._silence_run = 0
        start = self._speech_start
        if end - start >= self.min_speech:
            self.last_end = end
            return [Segment(start, end)]
        return []

    def finish(self, total_samples: int) -> Segment | None:
        """The open phrase (from the last closed segment to the end), or None when the key was
        released after the last phrase had already closed, so there is nothing left to hear."""
        if not self._in_speech:
            return None
        if total_samples - self.last_end < self.min_speech:
            return None
        return Segment(self.last_end, total_samples)
