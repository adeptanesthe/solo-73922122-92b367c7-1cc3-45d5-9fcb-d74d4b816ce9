# Monitor Return

Single-channel monitor-return ("返听") service for a live mixing desk. The
engine plays numbered 20 ms frames strictly in order from sequence 0 and fills
missing frames from the pitch period of the most recent 40 ms of audio that was
actually played, so the engineer keeps hearing the dialogue instead of muting.

## Audio contract

| Item | Value |
| --- | --- |
| Sample rate | 16 000 Hz |
| Channels | 1 (mono) |
| Sample format | little-endian signed 16-bit PCM (`pcm_s16le`) |
| Frame | 20 ms = 320 samples = 640 bytes |
| Sequence | non-negative integer, starts at 0, may arrive in any order |
| History for concealment | last 640 samples (40 ms) that were actually emitted |
| Pitch search | lags 40..200 samples (80..400 Hz), inclusive |
| Correlation gate | peak normalised correlation must be >= 0.6 |
| Concealment | tile the last detected period over one frame at 0.5 gain |
| Recovery | first 5 ms (80 samples) of the first real frame after a gap is a linear blend from the hypothetical concealment; everything else is returned verbatim |

## HTTP API

| Method | Path | Meaning |
| --- | --- | --- |
| `POST` | `/frames?seq=N` | Raw 640-byte PCM16 frame. `202` with `accepted`, `duplicate` or `late_ignored`. Wrong size `400`, oversized `413`, backlog full `429`. |
| `GET`/`POST` | `/playout` | Pull exactly one frame. Binary body (640 bytes) plus `X-*` metadata headers. `?format=json` returns base64 samples and metadata. |
| `GET` | `/status` | Cursor, pending sequence numbers, counters, suppressor state, optional played history (`?include_history=true`). |
| `POST` | `/reset` | Clear all state, including detector and notch filter state (used by the demo). |
| `GET` | `/config` | The format/concealment/suppression parameters above. |
| `GET` | `/healthz` | Liveness for Docker. |

Playout metadata headers: `X-Sequence`, `X-Next-Sequence`, `X-Frame-Kind`
(`true` / `recovered` / `concealed` / `silence`), `X-Period-Samples`,
`X-Peak-Correlation`, `X-Conceal-Reason` (`periodic` / `insufficient_history` /
`autocorr_undefined` / `low_correlation`), `X-Crossfade-Samples`,
`X-History-Samples`, and for suppression `X-Notch-Enabled`, `X-Notch-Hz`
(the notch used on this frame, empty when none), `X-Notch-Ramp`
(`in` / `out` / empty), `X-Detector-Hz`, `X-Detector-Ratio`,
`X-Notch-Candidate-Hz`, `X-Notch-Candidate-Frames`,
`X-Notch-Release-Frames`.

## Semantics

* **Ordering.** The playout cursor only moves forward from 0. A missing
  sequence number is concealed once its turn is pulled; later numbers wait.
* **Concealment.** Mean-removed, per-lag normalised autocorrelation over the
  last 40 ms of *played* audio. The highest peak in lags 40..200 is the
  period. The last period is tiled over 320 samples at half amplitude.
  Shortest lag wins within 1e-9 of the maximum, so a periodic signal reports
  its fundamental rather than a multiple.
* **Consecutive loss.** Each concealed frame is appended to the history, so the
  next missing frame recursively repeats (and halves) the newest period.
* **Silence.** Fewer than 40 ms of history, an undefined autocorrelation (all
  samples equal or zero energy), or a peak below 0.6 produces a silent frame.
* **Recovery.** The first real frame after one or more missing frames is blended
  over its first 80 samples from what concealment *would* have produced for
  that slot, then returns the true samples untouched. Following frames are
  returned verbatim (and also fade in after a gap whose concealment was
  silence).
* **Duplicates.** The first copy of a sequence number is kept; later copies are
  acknowledged as `duplicate` and discarded.
* **Late packets.** Once a sequence number's turn has been pulled, any packet
  for it is `late_ignored`. It cannot change the emitted audio or the played
  history that later concealment is derived from.
* **Concurrency.** Ingestion and playout use separate locks for bookkeeping;
  NumPy work runs outside the state lock, so ``/frames`` keeps accepting while
  a frame is being concealed. Pulls are serialised, so concurrent pulls each
  return exactly one distinct, sequential frame.
* **Backlog bound.** At most 5 000 pending frames (100 s). Beyond that new
  future frames get `429`; nothing accepted is evicted and the cursor does not
  move.

## Automatic howling suppression

Off by default; the engineer enables it when starting the return service
(`MONITOR_AUTO_SUPPRESSION=true`). When enabled, every frame that is played is
also passed through one adaptive notch filter.

* **Detection.** The spectrum of each *original* real frame (the frame as
  received, before concealment, recovery fade or notching) is inspected on the
  50 Hz grid between 500 and 4 000 Hz. A bin qualifies on a frame when it is
  the strongest in that band and holds at least 70 % of the band's total
  energy. A frame with zero in-band energy never qualifies.
* **Trigger.** The same bin qualifying on three consecutive real frames
  engages the notch from the third frame onward; that third frame carries the
  5 ms fade-in, so it starts dry and ends fully notched. Only one frequency is
  suppressed at a time.
* **Release.** The active bin must keep qualifying; three consecutive real
  frames without it release the notch from the third frame onward. That third
  frame carries the 5 ms fade-out, so it starts notched and reaches the dry
  signal at the 5 ms point; later frames are unfiltered.
* **Missing frames.** A missing (concealed or silent) slot breaks both
  counters but leaves the current notch running.
* **Filtering.** An RBJ notch biquad (Q = 20) is applied to the audio that
  actually plays - true frames, the recovery fade-in and concealment alike.
  Engage and release each crossfade over 5 ms; while engaged, the filter state
  carries over between frames. The notched samples become the played history,
  so later concealment is derived from them too.
* **Steady state.** The suppressed bin is attenuated by well over 20 dB
  (measured > 70 dB for a full-scale tone); every frequency at least 200 Hz
  away stays within ±1 dB of unity.
* **Bounded cost.** Detector and filter state are fixed-size; nothing grows
  with the length of the live stream. Pulls do not wait for future packets —
  the detector only ever looks at the frame being played.
* **Reset.** `POST /reset` clears the streak counters, the notch and the
  filter history.

`GET /config` publishes the parameters under `auto_suppression`; `GET /status`
shows `suppression` (enabled, active notch, streak counters, counters) and the
playout headers include `X-Notch-*` / `X-Detector-*` per frame.

## Run with Docker Compose

```bash
docker compose up --build
# in another shell
python3 scripts/demo.py --url http://127.0.0.1:8000 --out demo_out

# with automatic howling suppression enabled on the return service
MONITOR_AUTO_SUPPRESSION=true docker compose up --build
python3 scripts/demo.py --url http://127.0.0.1:8000 --out demo_out --suppression

# or run the same demo as a one-shot compose service
docker compose --profile demo run --rm demo
```

The demo only needs the Python standard library. It plays a 100 Hz harmonic
test signal, delivers frames out of order with one duplicate and two late
retransmissions, loses sequences 2, 6, 9 and 10, pulls while frames are still
arriving, then pulls four frames concurrently. It checks the playout order,
frame kinds, the untouched samples after each 5 ms fade and the halving between
consecutive concealments. It writes `demo_out/playout.wav` (listenable) and
`demo_out/transcript.json`, and exits non-zero if a check fails.

`--suppression` additionally resets the service and exercises the suppressor:
three 1 000 Hz frames engage the notch on the third frame, a missing frame
keeps the notch while breaking the count, and three 2 000 Hz frames release it
on the third. The service must have been started with
`MONITOR_AUTO_SUPPRESSION=true`.

Raw curl equivalents of the key cases:

```bash
# frame N is the 640-byte body; generate with scripts/demo.py's make_frames
curl -sS -X POST --data-binary @f000.bin -H 'Content-Type: application/octet-stream' \
  'http://127.0.0.1:8000/frames?seq=0'
curl -sS -X POST --data-binary @f001.bin -H 'Content-Type: application/octet-stream' \
  'http://127.0.0.1:8000/frames?seq=1'
curl -sS -X POST --data-binary @f003.bin -H 'Content-Type: application/octet-stream' \
  'http://127.0.0.1:8000/frames?seq=3'        # 2 is lost, 3 arrives early
curl -sS -D - -o out0.pcm 'http://127.0.0.1:8000/playout'   # X-Frame-Kind: true
curl -sS -D - -o out1.pcm 'http://127.0.0.1:8000/playout'   # true
curl -sS -D - -o out2.pcm 'http://127.0.0.1:8000/playout'   # concealed
curl -sS -D - -o out3.pcm 'http://127.0.0.1:8000/playout'   # recovered
curl -sS -X POST --data-binary @f002.bin -H 'Content-Type: application/octet-stream' \
  'http://127.0.0.1:8000/frames?seq=2'        # late_ignored
```

Suppression over curl (service started with `MONITOR_AUTO_SUPPRESSION=true`;
`h000.bin`.. are 640-byte 1 000 Hz frames, `r000.bin`.. are 2 000 Hz):

```bash
# trigger: the third consecutive 1000 Hz frame engages the notch
for n in 000 001 002; do
  curl -sS -X POST --data-binary @h$n.bin \
    -H 'Content-Type: application/octet-stream' \
    "http://127.0.0.1:8000/frames?seq=$((10#$n))"
done
curl -sS -D - -o /dev/null 'http://127.0.0.1:8000/playout'   # X-Notch-Hz: (empty)
curl -sS -D - -o /dev/null 'http://127.0.0.1:8000/playout'   # X-Notch-Hz: (empty)
curl -sS -D - -o /dev/null 'http://127.0.0.1:8000/playout'   # X-Notch-Hz: 1000, X-Notch-Ramp: in
curl -sS -D - -o /dev/null 'http://127.0.0.1:8000/playout'   # seq 3 missing -> concealed, X-Notch-Hz: 1000
# release: three consecutive 2000 Hz frames, notch out on the third
for n in 000 001 002; do
  curl -sS -X POST --data-binary @r$n.bin \
    -H 'Content-Type: application/octet-stream' \
    "http://127.0.0.1:8000/frames?seq=$((4 + 10#$n))"
done
for _ in 1 2 3 4; do
  curl -sS -D - -o /dev/null 'http://127.0.0.1:8000/playout'
done   # notches 1000, 1000, 1000 (out on the 3rd), then empty
curl -sS 'http://127.0.0.1:8000/status'   # suppression.notch_hz is null
```

## Local development

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
pytest                                   # 86 tests
uvicorn app.main:app --port 8000         # then run scripts/demo.py
MONITOR_AUTO_SUPPRESSION=true uvicorn app.main:app --port 8000
```

Playback state lives in process memory, so run **one** uvicorn worker (the
Docker image does). Running several workers or replicas would give each its own
cursor; that needs sticky routing or an external shared state store.

Layout: `app/concealment.py` (NumPy period search, concealment, fade),
`app/suppression.py` (band detector, notch biquad, 5 ms ramps),
`app/engine.py` (ordered playout state machine), `app/main.py` (FastAPI),
`scripts/demo.py` (lossy/reordered demo + suppression scenario + self-check),
`tests/`.
