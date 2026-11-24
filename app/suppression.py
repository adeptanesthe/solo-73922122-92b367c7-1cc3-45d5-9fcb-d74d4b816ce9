"""Automatic howling (acoustic-feedback) suppression for the monitor return.

Detection
---------
Every *original* real frame (the frame as received, before any concealment
fade or notch filtering) is analysed with an FFT.  A 20 ms frame at 16 kHz
gives exactly 50 Hz bin spacing, so bins 10..80 are the 500..4000 Hz band.
A bin qualifies on a frame when it is the strongest bin in that band and
holds at least 70 % of the band's total energy (the sum of those bins).  A
frame whose in-band energy is zero never qualifies.

The same bin qualifying on three consecutive real frames engages one notch
on the third frame.  The active bin must then qualify on every real frame;
three consecutive real frames without it release the notch on the third
frame.  A missing (concealed or silent) frame breaks both counts but leaves
the notch running.  Only one frequency is suppressed at a time.

Filtering
---------
A single RBJ notch biquad (Q = 20) is applied to the audio that is actually
played - true frames, recovery fade-ins and concealment alike.  Engaging and
releasing crossfade over 5 ms (80 samples); while the notch stays engaged its
filter state carries over from frame to frame, so there is no per-frame
transient.  The filtered samples are what the engine appends to its history,
so later concealment is derived from the notched signal as well.

All state is fixed-size; nothing grows with the length of the live stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .concealment import _to_pcm16
from .constants import FRAME_SAMPLES, SAMPLE_RATE

# ------------------------------------------------------------------ detection

BAND_LOW_HZ = 500
BAND_HIGH_HZ = 4_000
BIN_HZ = 50
# A 320-sample frame at 16 kHz has 50 Hz bins, so the band is an exact slice
# of the FFT.  Keep the indices derived so a format change fails loudly.
_BIN_STRIDE = SAMPLE_RATE // FRAME_SAMPLES
assert _BIN_STRIDE == BIN_HZ, "frame length must give 50 Hz bins"
_BAND_BINS = np.arange(BAND_LOW_HZ // BIN_HZ, BAND_HIGH_HZ // BIN_HZ + 1)
BAND_FREQS_HZ = _BAND_BINS * BIN_HZ

DOMINANCE_RATIO = 0.70
TRIGGER_FRAMES = 3
RELEASE_FRAMES = 3

# ------------------------------------------------------------------- filtering

NOTCH_Q = 20.0
NOTCH_RAMP_MS = 5
NOTCH_RAMP_SAMPLES = SAMPLE_RATE * NOTCH_RAMP_MS // 1000  # 80
_RAMP_IN = np.linspace(0.0, 1.0, NOTCH_RAMP_SAMPLES, dtype=np.float64)
_RAMP_OUT = _RAMP_IN[::-1].copy()


def notch_coefficients(
    notch_hz: float,
    q: float = NOTCH_Q,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[float, float, float, float, float, float]:
    """Return normalised ``b0, b1, b2, a0(=1), a1, a2`` for an RBJ notch."""

    if not 0.0 < notch_hz < sample_rate / 2.0:
        raise ValueError("notch frequency must be between 0 and Nyquist")
    if q <= 0.0:
        raise ValueError("Q must be positive")
    w0 = 2.0 * math.pi * notch_hz / sample_rate
    alpha = math.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha
    return (
        1.0 / a0,
        -2.0 * math.cos(w0) / a0,
        1.0 / a0,
        1.0,
        -2.0 * math.cos(w0) / a0,
        (1.0 - alpha) / a0,
    )


def notch_gain_db(
    notch_hz: float,
    freq_hz: float,
    q: float = NOTCH_Q,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    """Magnitude response of the notch at ``freq_hz`` in dB (for checks)."""

    b0, b1, b2, _a0, a1, a2 = notch_coefficients(notch_hz, q, sample_rate)
    z = np.exp(-2j * math.pi * freq_hz / sample_rate)
    numerator = b0 + b1 * z + b2 * z * z
    denominator = 1.0 + a1 * z + a2 * z * z
    magnitude = float(abs(numerator / denominator))
    if magnitude <= 0.0:
        return float("-inf")
    return float(20.0 * math.log10(magnitude))


# --------------------------------------------------------------- state objects


@dataclass(frozen=True)
class BandAnalysis:
    """Dominant in-band bin of one original frame."""

    peak_hz: int
    peak_ratio: float


@dataclass(frozen=True)
class SuppressionInfo:
    """What the suppressor did to one emitted frame, plus its new state."""

    enabled: bool
    applied_hz: int | None  # notch used while filtering this frame
    ramp: str | None  # "in", "out" or None
    notch_hz: int | None  # notch retained after this frame
    candidate_hz: int | None
    candidate_frames: int
    release_frames: int
    dominant_hz: int | None
    dominant_ratio: float | None


def analyse_band(frame: np.ndarray) -> BandAnalysis | None:
    """Return the strongest in-band bin and its share of in-band energy.

    ``None`` means the band is empty (zero total energy).  Only the bins on
    the 50 Hz grid between 500 and 4000 Hz are inspected.
    """

    x = np.asarray(frame, dtype=np.float64)
    if x.ndim != 1 or x.size != FRAME_SAMPLES:
        raise ValueError(f"frame must contain exactly {FRAME_SAMPLES} samples")
    power = np.abs(np.fft.rfft(x)) ** 2
    band_power = power[_BAND_BINS]
    total = float(band_power.sum())
    if not math.isfinite(total) or total <= 0.0:
        return None
    index = int(np.argmax(band_power))
    return BandAnalysis(
        peak_hz=int(BAND_FREQS_HZ[index]),
        peak_ratio=float(band_power[index] / total),
    )


class AutoSuppressor:
    """One-notch feedback suppressor with fixed-size state.

    The object is not internally locked: the engine calls it only while it
    holds its playout lock, so ``process`` calls are strictly sequential.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._notch_hz: int | None = None
        self._candidate_hz: int | None = None
        self._candidate_frames = 0
        self._release_frames = 0
        # Direct-form-I biquad history: x[n-1], x[n-2], y[n-1], y[n-2].
        self._filter_state = (0.0, 0.0, 0.0, 0.0)
        self._coefficients: tuple[float, float, float, float, float, float] | None = None
        self._coefficients_hz: int | None = None
        self.engaged_count = 0
        self.released_count = 0
        self.suppressed_frames = 0

    # ------------------------------------------------------------------ public

    def reset(self) -> None:
        """Forget detection and filter state (the enable flag is kept)."""

        self._notch_hz = None
        self._candidate_hz = None
        self._candidate_frames = 0
        self._release_frames = 0
        self._reset_filter()
        self.engaged_count = 0
        self.released_count = 0
        self.suppressed_frames = 0

    def process(
        self,
        emitted: np.ndarray,
        source: np.ndarray | None,
    ) -> tuple[np.ndarray, SuppressionInfo]:
        """Filter one emitted frame and advance the detector.

        ``emitted`` is the frame the engine would play (already concealment /
        recovery processed), ``source`` is the original received frame for
        this slot or ``None`` for a missing slot.  The detector only ever
        looks at ``source``, never at the notched output.
        """

        frame = np.asarray(emitted)
        if not self.enabled:
            return frame, SuppressionInfo(
                enabled=False,
                applied_hz=None,
                ramp=None,
                notch_hz=None,
                candidate_hz=None,
                candidate_frames=0,
                release_frames=0,
                dominant_hz=None,
                dominant_ratio=None,
            )

        analysis = analyse_band(source) if source is not None else None
        ramp = self._advance(analysis, present=source is not None)

        applied_hz = self._notch_hz
        if applied_hz is not None:
            filtered = self._filter_frame(frame, applied_hz, ramp)
            self.suppressed_frames += 1
            if ramp == "out":
                # The ramp reached dry at the end of this frame; retire the
                # notch and its state.
                self._notch_hz = None
                self._reset_filter()
            out = filtered
        else:
            out = frame

        return out, SuppressionInfo(
            enabled=True,
            applied_hz=applied_hz,
            ramp=ramp,
            notch_hz=self._notch_hz,
            candidate_hz=self._candidate_hz,
            candidate_frames=self._candidate_frames,
            release_frames=self._release_frames,
            dominant_hz=analysis.peak_hz if analysis else None,
            dominant_ratio=analysis.peak_ratio if analysis else None,
        )

    def snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "notch_hz": self._notch_hz,
            "candidate_hz": self._candidate_hz,
            "candidate_frames": self._candidate_frames,
            "release_frames": self._release_frames,
            "filter_state": self._notch_hz is not None,
            "engaged_count": self.engaged_count,
            "released_count": self.released_count,
            "suppressed_frames": self.suppressed_frames,
        }

    # ---------------------------------------------------------------- internal

    def _advance(self, analysis: BandAnalysis | None, present: bool) -> str | None:
        """Update streaks/notch.  Return the ramp to apply to this frame.

        ``present`` is False for a missing slot; ``analysis`` may still be
        None for a present frame whose in-band energy is zero.
        """

        if not present:
            # Missing slot: break both counts, keep any active notch running.
            self._candidate_hz = None
            self._candidate_frames = 0
            self._release_frames = 0
            return None

        # A present frame with an empty band is a real, qualifying-input frame
        # that simply failed the dominance test.
        qualifies = (
            analysis is not None and analysis.peak_ratio >= DOMINANCE_RATIO
        )

        if self._notch_hz is not None:
            if qualifies and analysis.peak_hz == self._notch_hz:
                self._release_frames = 0
                return None

            self._release_frames += 1
            if self._release_frames < RELEASE_FRAMES:
                return None
            # Third consecutive failure: ramp out over this frame.
            self._release_frames = 0
            self._candidate_hz = None
            self._candidate_frames = 0
            self.released_count += 1
            return "out"

        if qualifies:
            if analysis.peak_hz == self._candidate_hz:
                self._candidate_frames += 1
            else:
                self._candidate_hz = analysis.peak_hz
                self._candidate_frames = 1
            if self._candidate_frames >= TRIGGER_FRAMES:
                # Engage on this very frame, ramping in over its first 5 ms.
                self._notch_hz = analysis.peak_hz
                self._candidate_hz = None
                self._candidate_frames = 0
                self._release_frames = 0
                self._reset_filter()
                self.engaged_count += 1
                return "in"
        else:
            self._candidate_hz = None
            self._candidate_frames = 0
        return None

    def _coeffs(self, notch_hz: int):
        if self._coefficients is None or self._coefficients_hz != notch_hz:
            self._coefficients = notch_coefficients(notch_hz)
            self._coefficients_hz = notch_hz
        return self._coefficients

    def _reset_filter(self) -> None:
        self._filter_state = (0.0, 0.0, 0.0, 0.0)

    def _filter_frame(
        self,
        frame: np.ndarray,
        notch_hz: int,
        ramp: str | None,
    ) -> np.ndarray:
        """Run the biquad over the frame and crossfade dry/wet as requested."""

        b0, b1, b2, _a0, a1, a2 = self._coeffs(notch_hz)
        x = np.asarray(frame, dtype=np.float64)
        x1, x2, y1, y2 = self._filter_state
        wet = np.empty_like(x)
        for index in range(x.size):
            sample = float(x[index])
            value = b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, sample
            y2, y1 = y1, value
            wet[index] = value
        self._filter_state = (x1, x2, y1, y2)

        if ramp is None:
            alpha = 1.0
        elif ramp == "in":
            alpha = np.ones(x.size, dtype=np.float64)
            count = min(NOTCH_RAMP_SAMPLES, x.size)
            alpha[:count] = _RAMP_IN[:count]
        elif ramp == "out":
            alpha = np.zeros(x.size, dtype=np.float64)
            count = min(NOTCH_RAMP_SAMPLES, x.size)
            alpha[:count] = _RAMP_OUT[:count]
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown notch ramp {ramp!r}")

        mixed = x + alpha * (wet - x)
        return _to_pcm16(mixed)
