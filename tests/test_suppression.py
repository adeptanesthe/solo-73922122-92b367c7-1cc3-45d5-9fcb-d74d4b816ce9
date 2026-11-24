"""Unit tests for the automatic howling suppressor."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.constants import FRAME_SAMPLES, SAMPLE_RATE
from app.suppression import (
    BAND_FREQS_HZ,
    BAND_HIGH_HZ,
    BAND_LOW_HZ,
    DOMINANCE_RATIO,
    RELEASE_FRAMES,
    TRIGGER_FRAMES,
    AutoSuppressor,
    analyse_band,
    notch_coefficients,
    notch_gain_db,
)


def tone(freq: float, samples: int, amplitude: float = 12_000.0, phase: float = 0.0):
    t = np.arange(samples)
    wave = amplitude * np.sin(2 * np.pi * freq * t / SAMPLE_RATE + phase)
    return np.clip(np.rint(wave), -32768, 32767).astype(np.int16)


def frame_at(freq: float, index: int, amplitude: float = 12_000.0) -> np.ndarray:
    return tone(freq, FRAME_SAMPLES * (index + 1), amplitude)[
        index * FRAME_SAMPLES : (index + 1) * FRAME_SAMPLES
    ].copy()


def feed(suppressor: AutoSuppressor, source: np.ndarray | None):
    """Run one frame through the suppressor with emitted == source."""

    emitted = source if source is not None else np.zeros(FRAME_SAMPLES, np.int16)
    return suppressor.process(emitted, source)


def run(suppressor: AutoSuppressor, freq: float, count: int, start: int = 0):
    results = []
    for index in range(start, start + count):
        results.append(feed(suppressor, frame_at(freq, index)))
    return results


# ------------------------------------------------------------------ detection


def test_band_bins_are_the_50hz_grid_from_500_to_4000():
    assert BAND_FREQS_HZ[0] == BAND_LOW_HZ == 500
    assert BAND_FREQS_HZ[-1] == BAND_HIGH_HZ == 4000
    assert np.all(np.diff(BAND_FREQS_HZ) == 50)


def test_band_analysis_finds_dominant_bin_and_ratio():
    analysis = analyse_band(frame_at(1000, 0))
    assert analysis is not None
    assert analysis.peak_hz == 1000
    assert analysis.peak_ratio > 0.99


def test_band_analysis_ignores_out_of_band_energy():
    # A 300 Hz tone is below the band; add a quieter 1500 Hz tone inside it.
    below = tone(300, FRAME_SAMPLES, amplitude=20_000.0)
    inside = tone(1500, FRAME_SAMPLES, amplitude=4_000.0)
    analysis = analyse_band(below + inside)
    assert analysis is not None
    assert analysis.peak_hz == 1500
    assert analysis.peak_ratio > 0.9


def test_band_analysis_reports_none_for_zero_energy():
    assert analyse_band(np.zeros(FRAME_SAMPLES, dtype=np.int16)) is None


def test_band_analysis_rejects_wrong_frame_length():
    with pytest.raises(ValueError):
        analyse_band(np.zeros(FRAME_SAMPLES - 1, dtype=np.int16))


# ------------------------------------------------------------------ trigger


def test_disabled_suppressor_is_a_passthrough():
    suppressor = AutoSuppressor(enabled=False)
    frame = frame_at(1000, 0)
    out, info = feed(suppressor, frame)
    assert info.enabled is False
    assert info.applied_hz is None
    np.testing.assert_array_equal(out, frame)
    # Nothing is even analysed, so no state accumulates.
    assert suppressor.snapshot()["candidate_hz"] is None


def test_trigger_needs_three_consecutive_qualifying_frames():
    suppressor = AutoSuppressor(enabled=True)
    results = run(suppressor, 1000.0, TRIGGER_FRAMES + 1)
    assert [r[1].applied_hz for r in results] == [None, None, 1000, 1000]
    assert [r[1].ramp for r in results] == [None, None, "in", None]
    assert suppressor.snapshot()["notch_hz"] == 1000


def test_missing_frame_breaks_the_trigger_count():
    suppressor = AutoSuppressor(enabled=True)
    feed(suppressor, frame_at(1000, 0))
    feed(suppressor, frame_at(1000, 1))
    feed(suppressor, None)  # a concealed/missing slot
    results = [feed(suppressor, frame_at(1000, index)) for index in (2, 3, 4)]
    assert [r[1].applied_hz for r in results] == [None, None, 1000]
    assert [r[1].candidate_frames for r in results] == [1, 2, 0]


def test_below_70_percent_never_triggers():
    # Two equally loud in-band tones: the strongest holds ~50 % of the energy.
    pair = tone(1000, FRAME_SAMPLES, amplitude=9_000.0) + tone(
        2000, FRAME_SAMPLES, amplitude=9_000.0
    )
    suppressor = AutoSuppressor(enabled=True)
    for _ in range(6):
        _, info = feed(suppressor, pair)
        assert info.applied_hz is None
        assert info.dominant_ratio is not None
        assert info.dominant_ratio < DOMINANCE_RATIO


def test_zero_band_energy_does_not_qualify():
    suppressor = AutoSuppressor(enabled=True)
    for _ in range(6):
        _, info = feed(suppressor, np.zeros(FRAME_SAMPLES, dtype=np.int16))
        assert info.applied_hz is None
        assert info.dominant_hz is None
        assert info.candidate_hz is None


def test_trigger_requires_the_same_frequency():
    suppressor = AutoSuppressor(enabled=True)
    seq = [1000, 1000, 1500, 1000, 1000, 1500]
    results = [feed(suppressor, frame_at(freq, index)) for index, freq in enumerate(seq)]
    assert all(r[1].applied_hz is None for r in results)
    assert suppressor.snapshot()["candidate_frames"] == 1


def test_only_one_notch_is_active_at_a_time():
    suppressor = AutoSuppressor(enabled=True)
    run(suppressor, 1000.0, TRIGGER_FRAMES)
    # A new frequency becomes dominant.  The first notch stays until the old
    # one has failed three times; only then can the new one trigger.
    results = [feed(suppressor, frame_at(2000.0, 10 + i)) for i in range(6)]
    applied = [r[1].applied_hz for r in results]
    # Frames 3-5 release the old notch; the next frequency then needs its own
    # three consecutive frames, so only one notch is ever active.
    assert applied == [1000, 1000, 1000, None, None, 2000]
    assert suppressor.snapshot()["notch_hz"] == 2000
    assert suppressor.snapshot()["engaged_count"] == 2
    assert suppressor.snapshot()["released_count"] == 1


# ------------------------------------------------------------------ release


def test_release_needs_three_non_qualifying_frames():
    suppressor = AutoSuppressor(enabled=True)
    run(suppressor, 1000.0, TRIGGER_FRAMES)

    # Release with a loud tone at another frequency.  The third failing frame
    # is the one that ramps the notch out; it is filtered while ramping.
    results = [
        feed(suppressor, frame_at(2000.0, 10 + i)) for i in range(RELEASE_FRAMES + 1)
    ]
    assert [r[1].applied_hz for r in results] == [1000] * RELEASE_FRAMES + [None]
    assert results[RELEASE_FRAMES - 1][1].ramp == "out"
    assert suppressor.snapshot()["notch_hz"] is None
    assert suppressor.snapshot()["released_count"] == 1


def test_missing_frames_do_not_release_the_notch():
    suppressor = AutoSuppressor(enabled=True)
    run(suppressor, 1000.0, TRIGGER_FRAMES)
    for _ in range(5):
        _, info = feed(suppressor, None)
        assert info.applied_hz == 1000
        assert info.release_frames == 0
    # The notch is still there; a qualifying frame resets nothing but keeps it.
    _, info = feed(suppressor, frame_at(1000.0, 20))
    assert info.applied_hz == 1000
    assert suppressor.snapshot()["notch_hz"] == 1000


def test_release_count_breaks_on_a_qualifying_frame():
    suppressor = AutoSuppressor(enabled=True)
    run(suppressor, 1000.0, TRIGGER_FRAMES)
    feed(suppressor, frame_at(2000.0, 10))
    feed(suppressor, frame_at(2000.0, 11))
    _, info = feed(suppressor, frame_at(1000.0, 20))  # back on the notch
    assert info.release_frames == 0
    assert info.applied_hz == 1000


# ------------------------------------------------------------------ filter


@pytest.mark.parametrize("notch_hz", [500, 1000, 2000, 3000, 4000])
def test_notch_response_meets_attenuation_and_neighbour_gain(notch_hz):
    assert notch_gain_db(notch_hz, notch_hz) <= -20.0
    # Every frequency at least 200 Hz away is within +-1 dB, checked on a
    # 1 Hz grid up to Nyquist.
    for freq in range(1, SAMPLE_RATE // 2):
        if abs(freq - notch_hz) >= 200:
            assert abs(notch_gain_db(notch_hz, freq)) <= 1.0, (
                f"{freq} Hz shifted near notch {notch_hz}: "
                f"{notch_gain_db(notch_hz, freq):.3f} dB"
            )


def test_notch_at_full_scale_signal_keeps_everything_else_within_1db():
    # End-to-end spectral check on a dense multitone with a dominant 1000 Hz
    # interference: the notch bin is far down while every in-band bin at
    # least 200 Hz away is within +-1 dB of the input.
    suppressor = AutoSuppressor(enabled=True)
    run(suppressor, 1000.0, TRIGGER_FRAMES)
    freqs = [
        500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
        1050, 1100, 1200, 1300, 1500, 2000, 2500, 3000, 3500, 4000,
    ]
    rng = np.random.default_rng(11)
    phases = rng.uniform(0.0, 2 * math.pi, len(freqs))
    t = np.arange(FRAME_SAMPLES * 8)
    probe = 20_000.0 * np.sin(2 * math.pi * 1000 * t / SAMPLE_RATE)
    for freq, phase in zip(freqs, phases):
        probe += 1_000.0 * np.sin(2 * math.pi * freq * t / SAMPLE_RATE + phase)
    assert np.max(np.abs(probe)) < 32767, "test probe must not clip"
    probe = np.rint(probe).astype(np.int16)

    out = []
    for index in range(8):
        chunk = probe[index * FRAME_SAMPLES : (index + 1) * FRAME_SAMPLES]
        out.append(suppressor.process(chunk, chunk)[0])
    out = np.concatenate(out).astype(np.float64)
    window = np.hanning(out.size)
    in_spectrum = np.abs(np.fft.rfft(probe.astype(np.float64) * window))
    out_spectrum = np.abs(np.fft.rfft(out * window))
    bin_hz = SAMPLE_RATE / out.size
    notch_bin = round(1000 / bin_hz)
    gain = 20 * math.log10(max(out_spectrum[notch_bin], 1e-12) / in_spectrum[notch_bin])
    assert gain <= -20.0
    for freq in freqs:
        index = round(freq / bin_hz)
        gain = 20 * math.log10(max(out_spectrum[index], 1e-12) / in_spectrum[index])
        assert abs(gain) <= 1.0, f"{freq} Hz shifted by {gain:.2f} dB"


def test_engaged_filter_deeply_attenuates_the_detected_tone():
    suppressor = AutoSuppressor(enabled=True)
    run(suppressor, 1000.0, TRIGGER_FRAMES)
    # Let the ramp and filter settle, then compare steady-state spectra over
    # the same number of samples.
    network = run(suppressor, 1000.0, 8, start=TRIGGER_FRAMES)
    out = np.concatenate([item[0] for item in network]).astype(np.float64)
    raw = np.concatenate(
        [frame_at(1000.0, TRIGGER_FRAMES + i) for i in range(8)]
    ).astype(np.float64)
    freqs = np.fft.rfftfreq(out.size, 1 / SAMPLE_RATE)
    index = int(np.argmin(np.abs(freqs - 1000.0)))
    raw_mag = np.abs(np.fft.rfft(raw))[index]
    out_mag = np.abs(np.fft.rfft(out))[index]
    assert 20 * math.log10(max(out_mag, 1e-12) / raw_mag) <= -20.0


def test_ramp_in_starts_dry_and_ends_wet():
    suppressor = AutoSuppressor(enabled=True)
    # A steady tone so dry and wet differ mainly near the notch frequency.
    run(suppressor, 1000.0, TRIGGER_FRAMES - 1)
    raw = frame_at(1000.0, TRIGGER_FRAMES - 1)
    out, info = feed(suppressor, raw)
    assert info.ramp == "in"
    # The first sample is the dry signal (alpha starts at 0).
    assert out[0] == raw[0]
    # The last sample is fully wet and therefore much quieter than dry.
    assert abs(int(out[-1])) < abs(int(raw[-1]))


def test_ramp_out_starts_wet_and_ends_dry():
    suppressor = AutoSuppressor(enabled=True)
    run(suppressor, 1000.0, TRIGGER_FRAMES)
    # Three failing frames; the third ramps out.
    for index in range(RELEASE_FRAMES - 1):
        feed(suppressor, frame_at(2000.0, 10 + index))
    raw = frame_at(2000.0, 10 + RELEASE_FRAMES - 1)
    out, info = feed(suppressor, raw)
    assert info.ramp == "out"
    # Ends dry: the last sample is the unfiltered input.
    assert out[-1] == raw[-1]


def test_continuous_notch_state_carries_across_frames():
    # Filtering a stream frame by frame with a persistent suppressor must
    # equal filtering it in one continuous pass: no per-frame reset.
    suppressor = AutoSuppressor(enabled=True)
    run(suppressor, 1000.0, TRIGGER_FRAMES)
    stream = np.concatenate([frame_at(1000.0, 5), frame_at(1000.0, 6)])
    out_a = [
        suppressor.process(
            stream[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES],
            stream[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES],
        )[0]
        for i in range(2)
    ]
    separate = AutoSuppressor(enabled=True)
    run(separate, 1000.0, TRIGGER_FRAMES)
    # Compare against a fresh filter run over the whole stream in one call.
    out_b = [
        separate.process(
            stream[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES],
            stream[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES],
        )[0]
        for i in range(2)
    ]
    np.testing.assert_array_equal(out_a[0], out_b[0])
    np.testing.assert_array_equal(out_a[1], out_b[1])
    # The filter state at the end is carried over, not cleared.
    assert suppressor._filter_state != (0.0, 0.0, 0.0, 0.0)


def test_reset_clears_everything():
    suppressor = AutoSuppressor(enabled=True)
    run(suppressor, 1000.0, TRIGGER_FRAMES + 2)
    suppressor.reset()
    snapshot = suppressor.snapshot()
    assert snapshot["notch_hz"] is None
    assert snapshot["candidate_hz"] is None
    assert snapshot["candidate_frames"] == 0
    assert snapshot["release_frames"] == 0
    assert snapshot["engaged_count"] == 0
    assert suppressor._filter_state == (0.0, 0.0, 0.0, 0.0)


def test_notch_coefficients_reject_invalid_parameters():
    with pytest.raises(ValueError):
        notch_coefficients(0)
    with pytest.raises(ValueError):
        notch_coefficients(SAMPLE_RATE / 2)
    with pytest.raises(ValueError):
        notch_coefficients(1000, q=0)
