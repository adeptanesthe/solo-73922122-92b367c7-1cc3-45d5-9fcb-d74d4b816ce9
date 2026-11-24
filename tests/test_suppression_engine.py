"""Behavioural tests for auto-suppression inside the playout engine."""

from __future__ import annotations

import numpy as np
from helpers import tone_frames

from app.constants import FRAME_SAMPLES, HISTORY_SAMPLES
from app.engine import PlayoutEngine
from app.suppression import TRIGGER_FRAMES, AutoSuppressor


def tone_at(freq: float, count: int, amplitude: float = 15_000.0) -> list[np.ndarray]:
    full = np.clip(
        np.rint(
            amplitude
            * np.sin(2 * np.pi * freq * np.arange(FRAME_SAMPLES * count) / 16_000)
        ),
        -32768,
        32767,
    ).astype(np.int16)
    return [full[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES].copy() for i in range(count)]


def feed_all(engine: PlayoutEngine, frames: list[np.ndarray], start: int = 0) -> list:
    results = []
    for index, frame in enumerate(frames):
        engine.ingest(start + index, frame)
        results.append(engine.playout())
    return results


def test_disabled_by_default_and_audio_is_untouched():
    engine = PlayoutEngine()
    frames = tone_frames(6)
    for index, frame in enumerate(frames):
        engine.ingest(index, frame)
    results = [engine.playout() for _ in range(6)]
    for result, frame in zip(results, frames):
        np.testing.assert_array_equal(result.samples, frame)
        assert result.suppression.enabled is False
        assert result.suppression.applied_hz is None
    assert engine.snapshot()["suppression"]["engaged_count"] == 0


def test_enabled_engine_notches_the_third_repeated_frame():
    engine = PlayoutEngine(auto_suppression=True)
    results = feed_all(engine, tone_at(1000.0, TRIGGER_FRAMES + 2))
    applied = [r.suppression.applied_hz for r in results]
    assert applied == [None, None, 1000, 1000, 1000]
    assert all(r.kind == "true" for r in results)
    assert engine.snapshot()["suppression"]["notch_hz"] == 1000


def test_filtered_samples_are_used_for_concealment_history(monkeypatch):
    engine = PlayoutEngine(auto_suppression=True)
    frames = tone_at(1000.0, TRIGGER_FRAMES + 2)
    results = feed_all(engine, frames)
    assert results[-1].suppression.applied_hz == 1000

    played = np.concatenate([r.samples for r in results])
    history = engine.snapshot()["history"]
    # The played history is the notched signal, not the raw input.
    np.testing.assert_array_equal(history, played[-HISTORY_SAMPLES:])
    raw_tail = np.concatenate(frames)[-HISTORY_SAMPLES:]
    assert not np.array_equal(history, raw_tail)

    # The next missing frame builds its concealment from that notched history.
    seen: list[np.ndarray] = []
    from app import engine as engine_module

    original = engine_module.concealment_for

    def capture(value):
        seen.append(np.array(value))
        return original(value)

    monkeypatch.setattr(engine_module, "concealment_for", capture)
    missing = engine.playout()
    assert missing.kind == "concealed"
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], history)


def test_notch_also_filters_concealment_and_recovery_fade():
    engine = PlayoutEngine(auto_suppression=True)
    frames = tone_at(1000.0, TRIGGER_FRAMES + 2)
    results = feed_all(engine, frames)
    assert results[-1].suppression.applied_hz == 1000

    # Recovery: the fade-in frame is still passed through the notch.
    recovered_frame = tone_at(1000.0, 1)[0]
    engine.ingest(len(frames) + 1, recovered_frame)  # seq is now missing
    missing = engine.playout()
    assert missing.kind == "concealed"
    recovered = engine.playout()
    assert recovered.kind == "recovered"
    assert recovered.suppression.applied_hz == 1000
    # After the 5 ms fade the samples are not the raw ones (the notch is
    # still ringing), but they are much quieter than the input.
    assert np.max(np.abs(recovered.samples.astype(np.int32))) < np.max(
        np.abs(recovered_frame.astype(np.int32))
    )


def test_missing_frame_breaks_trigger_count_engine_level():
    engine = PlayoutEngine(auto_suppression=True)
    tone = tone_at(1000.0, 1, 9_000.0)[0]
    for index in range(2):
        engine.ingest(index, tone)
        engine.playout()
    # seq 2 missing: this pull is concealed, not a real frame.
    concealed = engine.playout()
    assert concealed.kind == "concealed"
    assert concealed.suppression.dominant_hz is None
    assert engine.snapshot()["suppression"]["candidate_frames"] == 0

    # Two more real frames only get the streak to two; a third is needed.
    for index in range(2):
        engine.ingest(3 + index, tone)
        result = engine.playout()
        assert result.suppression.applied_hz is None
    assert engine.snapshot()["suppression"]["candidate_frames"] == 2
    engine.ingest(5, tone)
    result = engine.playout()
    assert result.suppression.applied_hz == 1000
    assert engine.snapshot()["suppression"]["engaged_count"] == 1


def test_release_after_three_frames_at_another_frequency():
    engine = PlayoutEngine(auto_suppression=True)
    frames = tone_at(1000.0, TRIGGER_FRAMES)
    results = feed_all(engine, frames)
    assert results[-1].suppression.applied_hz == 1000

    others = tone_at(2000.0, 4)
    for index, frame in enumerate(others):
        engine.ingest(TRIGGER_FRAMES + index, frame)
    results = [engine.playout() for _ in range(4)]
    assert [r.suppression.applied_hz for r in results] == [1000, 1000, 1000, None]
    assert results[2].suppression.ramp == "out"
    assert engine.snapshot()["suppression"]["notch_hz"] is None
    assert engine.snapshot()["stats"]["notch_releases"] == 1


def test_reset_clears_detection_and_filter_state():
    engine = PlayoutEngine(auto_suppression=True)
    feed_all(engine, tone_at(1000.0, TRIGGER_FRAMES + 1))
    before = engine.snapshot()
    assert before["suppression"]["notch_hz"] == 1000

    engine.reset()
    after = engine.snapshot()
    assert after["suppression"]["notch_hz"] is None
    assert after["suppression"]["candidate_frames"] == 0
    assert after["suppression"]["engaged_count"] == 0
    assert after["suppression"]["suppressed_frames"] == 0
    assert after["suppression"]["filter_state"] is False

    # After the reset the same tone must take three frames again.
    results = feed_all(engine, tone_at(1000.0, 3))
    assert [r.suppression.applied_hz for r in results] == [None, None, 1000]


def test_state_size_is_constant_over_a_long_stream():
    engine = PlayoutEngine(auto_suppression=True)
    frames = tone_at(1000.0, 6)
    for index, frame in enumerate(frames):
        engine.ingest(index, frame)
    for _ in range(6):
        engine.playout()
    state_after_six = engine.snapshot()["suppression"]

    # Stream 600 more frames (6 s of audio) through the same engine.
    next_seq = engine.snapshot()["next_seq"]
    for index in range(600):
        engine.ingest(next_seq + index, frames[index % len(frames)])
        engine.playout()
    state_after_many = engine.snapshot()["suppression"]

    assert set(state_after_six) == set(state_after_many)
    # Only monotonic counters change; the structural fields are fixed-size.
    for key in ("notch_hz", "candidate_hz", "candidate_frames", "release_frames"):
        assert state_after_many[key] in (state_after_six[key], None, 0)
    assert state_after_many["suppressed_frames"] > state_after_six["suppressed_frames"]


def test_detector_and_filter_memory_does_not_grow_with_stream_length():
    suppressor = AutoSuppressor(enabled=True)
    frames = tone_at(1000.0, 8)

    def footprint():
        state = suppressor.__dict__
        total = 0
        for value in state.values():
            if isinstance(value, np.ndarray):
                total += value.nbytes
            elif isinstance(value, (tuple, list, dict, set)):
                total += len(value)
        return sorted(state), total

    run_shapes_after_one = None
    for index, frame in enumerate(frames):
        suppressor.process(frame, frame)
    run_shapes_after_one = footprint()

    # 4 000 more frames (~80 s) with the notch engaged.
    for index in range(4000):
        frame = frames[index % len(frames)]
        suppressor.process(frame, frame)
    assert footprint() == run_shapes_after_one


def test_suppression_works_under_concurrent_pulls():
    engine = PlayoutEngine(auto_suppression=True)
    frames = tone_at(1000.0, TRIGGER_FRAMES + 8)
    for index, frame in enumerate(frames):
        engine.ingest(index, frame)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: engine.playout(), range(len(frames))))

    assert sorted(r.seq for r in results) == list(range(len(frames)))
    assert all(r.kind == "true" for r in results)
    # Serialised pulls still saw a single deterministic engage at the 3rd frame.
    by_seq = {r.seq: r for r in results}
    assert by_seq[0].suppression.applied_hz is None
    assert by_seq[2].suppression.applied_hz == 1000
    assert by_seq[2].suppression.ramp == "in"
    assert engine.snapshot()["suppression"]["engaged_count"] == 1


def test_concealment_ignores_the_notched_output_for_detection():
    # Detection only ever sees original real frames.  A missing slot must not
    # produce a detection even though the concealed audio is notched.
    engine = PlayoutEngine(auto_suppression=True)
    feed_all(engine, tone_at(1000.0, TRIGGER_FRAMES + 1))
    before = engine.snapshot()["suppression"]["candidate_frames"]
    missing = engine.playout()
    assert missing.kind == "concealed"
    assert missing.suppression.dominant_hz is None
    after = engine.snapshot()["suppression"]["candidate_frames"]
    assert after == 0
    assert before == 0
