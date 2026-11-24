#!/usr/bin/env python3
"""Demonstrate lossy, reordered delivery against a running monitor-return API.

The script only uses the standard library so it can run on the host or inside
the same container image.  It:

1. generates a 16 kHz mono PCM16 test signal (a 100 Hz voice-like tone),
2. posts numbered frames in a deliberately reordered / duplicated / lossy
   order (with late retransmissions), interleaved with playout pulls,
3. exercises concurrent pulls after a prefetched burst,
4. optionally (``--suppression`` against a service started with
   ``MONITOR_AUTO_SUPPRESSION=true``) triggers the notch with three 1000 Hz
   frames, shows a missing frame keeping it, and releases it with three
   2000 Hz frames,
5. writes the concatenated playout to ``<out>/playout.wav`` and a transcript
   to ``<out>/transcript.json``.

Example:
    python3 scripts/demo.py --url http://127.0.0.1:8000 --out demo_out
    python3 scripts/demo.py --url http://127.0.0.1:8000 --out demo_out --suppression
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320
FRAME_BYTES = FRAME_SAMPLES * 2
FADE_SAMPLES = 80
PERIOD = 160  # 100 Hz fundamental -> detected pitch lag of 160 samples
HOWL_HZ = 1_000.0
HOWL_BAND_FRAMES = 3  # detection and release both need three frames


def _header(headers: dict, name: str) -> str:
    """HTTP header names are case-insensitive; urllib preserves wire casing."""

    value = headers.get(name)
    if value is None:
        value = headers.get(name.lower(), "")
    return value


def _request(url: str, method: str = "GET", body: bytes | None = None) -> tuple[int, bytes, dict]:
    headers = {"Content-Type": "application/octet-stream"} if body is not None else {}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def wait_for_health(base_url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            status, _, _ = _request(f"{base_url}/healthz")
            if status == 200:
                return
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"API did not become healthy at {base_url}")
        time.sleep(0.5)


def make_frames(count: int, amplitude: float = 11_000.0) -> list[bytes]:
    """Split one phase-continuous harmonic tone into 20 ms PCM16 frames."""

    total = count * FRAME_SAMPLES
    samples = []
    for n in range(total):
        t = n / SAMPLE_RATE
        # Slow envelope makes the signal read as "syllables", not a beep.
        envelope = 0.65 + 0.35 * math.sin(2 * math.pi * 2.0 * t)
        value = (
            0.62 * math.sin(2 * math.pi * n / PERIOD)
            + 0.27 * math.sin(2 * math.pi * 2 * n / PERIOD + 0.4)
            + 0.11 * math.sin(2 * math.pi * 3 * n / PERIOD + 1.1)
        )
        samples.append(round(amplitude * envelope * value))
    pcm = b"".join(struct.pack("<h", max(-32768, min(32767, s))) for s in samples)
    return [pcm[i * FRAME_BYTES : (i + 1) * FRAME_BYTES] for i in range(count)]


def howl_frames(count: int, freq: float = HOWL_HZ, amplitude: float = 15_000.0) -> list[bytes]:
    """Phase-continuous sine frames: one strong in-band peak per frame."""

    samples = [
        round(amplitude * math.sin(2 * math.pi * freq * n / SAMPLE_RATE))
        for n in range(count * FRAME_SAMPLES)
    ]
    pcm = b"".join(struct.pack("<h", max(-32768, min(32767, s))) for s in samples)
    return [pcm[i * FRAME_BYTES : (i + 1) * FRAME_BYTES] for i in range(count)]


class Demo:
    def __init__(self, base_url: str, out_dir: Path, enable_suppression: bool) -> None:
        self.base_url = base_url.rstrip("/")
        self.out_dir = out_dir
        self.enable_suppression = enable_suppression
        self.frames = make_frames(24)
        self.played: list[bytes] = []
        self.played_kinds: list[str] = []
        self.played_seqs: list[int] = []
        self.transcript: list[dict] = []

    def post(self, seq: int) -> str:
        return self._post_bytes(seq, self.frames[seq])

    def _post_bytes(self, seq: int, payload: bytes) -> str:
        status, body, _ = _request(
            f"{self.base_url}/frames?seq={seq}", "POST", payload
        )
        if status != 202:
            raise SystemExit(f"POST seq={seq} failed with {status}: {body!r}")
        result = json.loads(body)
        self._log("post", seq=seq, detail=result["status"])
        return result["status"]

    def pull(self, note: str = "") -> bytes:
        status, body, headers = _request(f"{self.base_url}/playout")
        if status != 200:
            raise SystemExit(f"GET /playout failed with {status}: {body!r}")
        if len(body) != FRAME_BYTES:
            raise SystemExit(f"unexpected playout size {len(body)}")
        self.played.append(body)
        seq = int(_header(headers, "x-sequence") or -1)
        kind = _header(headers, "x-frame-kind")
        self.played_seqs.append(seq)
        self.played_kinds.append(kind)
        self._log(
            "pull",
            seq=seq,
            detail=kind,
            period=_header(headers, "x-period-samples"),
            peak=_header(headers, "x-peak-correlation"),
            notch=_header(headers, "x-notch-hz"),
            detector=_header(headers, "x-detector-hz"),
            note=note,
        )
        return body

    def _log(self, action: str, seq: int, detail: str, **extra) -> None:
        entry = {"action": action, "seq": seq, "result": detail, **extra}
        self.transcript.append(entry)
        line = f"{action.upper():4s} seq={seq:<2d} {detail}"
        if extra.get("period"):
            line += (
                f"  period={extra['period']}samples"
                f" peak={float(extra['peak']):.3f}"
            )
        if extra.get("notch"):
            line += f"  notch={extra['notch']}Hz (detector {extra['detector']}Hz)"
        if extra.get("note"):
            line += f"  ({extra['note']})"
        print(line)

    def run(self) -> None:
        wait_for_health(self.base_url)
        # Start from a clean engine so repeated runs are deterministic.
        _request(f"{self.base_url}/reset", "POST")

        print("== loss + reordering + duplicate + late retransmission ==")
        self.post(0)
        self.post(1)
        self.post(4)  # arrives before 3
        self.post(3)
        self.post(3)  # same number resent: first copy wins

        self.pull("in order")
        self.pull("in order")
        self.pull("seq 2 has not arrived -> conceal")
        self.post(2)  # too late: its turn already passed
        self.pull("first real frame after gap -> 5 ms crossfade")
        self.pull("in order")

        self.post(5)
        self.pull("in order")

        self.post(7)
        self.post(8)
        self.pull("seq 6 missing -> conceal")
        self.post(6)  # too late
        self.pull("recovered from concealment")
        self.pull("in order")

        print("== two consecutive missing frames (9, 10) ==")
        self.post(11)  # future frame arrives early
        self.pull("seq 9 missing -> conceal")
        self.pull("seq 10 missing -> conceal again, half amplitude")
        self.pull("recovered from two-frame gap")

        print("== prefetched burst + concurrent pulls ==")
        for seq in range(12, 20):
            self.post(seq)
        with ThreadPoolExecutor(max_workers=4) as pool:
            payloads = list(pool.map(lambda _: self.pull("concurrent"), range(4)))
        if len(payloads) != 4:
            raise SystemExit("concurrent pulls returned the wrong frame count")

        status, body, _ = _request(f"{self.base_url}/status")
        if status != 200:
            raise SystemExit(f"GET /status failed with {status}")
        stats = json.loads(body)
        self._verify(stats)

        if self.enable_suppression:
            self._check_suppression_available(stats)
            self._run_suppression_scenario()
            status, body, _ = _request(f"{self.base_url}/status")
            stats = json.loads(body)
            self._verify_suppression(stats)

        self._write_outputs(stats)

    def _check_suppression_available(self, status: dict) -> None:
        if not status["suppression"]["enabled"]:
            raise SystemExit(
                "suppression demo requested but the service has auto suppression "
                "disabled; start it with MONITOR_AUTO_SUPPRESSION=true"
            )

    def _run_suppression_scenario(self) -> None:
        """Trigger, survive a gap, and release the notch over HTTP."""

        _request(f"{self.base_url}/reset", "POST")
        self.played.clear()
        self.played_kinds.clear()
        self.played_seqs.clear()

        print("== auto howling suppression: three 1000 Hz frames trigger ==")
        seq = 0
        for frame in howl_frames(HOWL_BAND_FRAMES):
            self._post_bytes(seq, frame)
            seq += 1
        notch = []
        for index in range(HOWL_BAND_FRAMES):
            headers = self._pull_headers(f"trigger {index + 1}/3")
            notch.append(_header(headers, "x-notch-hz"))
        if notch != ["", "", "1000"]:
            raise SystemExit(f"expected notch to engage on frame 3, got {notch}")

        print("== missing frame keeps the notch, breaks the count ==")
        headers = self._pull_headers("seq 3 missing -> concealed, notch holds")
        if _header(headers, "x-notch-hz") != "1000":
            raise SystemExit("notch was dropped on a missing frame")
        seq += 1  # seq 3 was consumed by the concealed pull

        print("== three 2000 Hz frames release the notch ==")
        release = howl_frames(HOWL_BAND_FRAMES, freq=2_000.0)
        for frame in release:
            self._post_bytes(seq, frame)
            seq += 1
        released: list[str] = []
        for _ in range(HOWL_BAND_FRAMES):
            headers = self._pull_headers("releasing")
            released.append(_header(headers, "x-notch-hz"))
        if released != ["1000", "1000", "1000"]:
            raise SystemExit(f"unexpected release sequence: {released}")
        headers = self._pull_headers("notch gone after three failing frames")
        if _header(headers, "x-notch-hz") != "":
            raise SystemExit("notch was not released after three failing frames")

    def _pull_headers(self, note: str) -> dict:
        status, body, headers = _request(f"{self.base_url}/playout")
        if status != 200 or len(body) != FRAME_BYTES:
            raise SystemExit(f"GET /playout failed with {status}")
        self.played.append(body)
        self.played_seqs.append(int(_header(headers, "x-sequence") or -1))
        self.played_kinds.append(_header(headers, "x-frame-kind"))
        self._log(
            "pull",
            seq=self.played_seqs[-1],
            detail=self.played_kinds[-1],
            notch=_header(headers, "x-notch-hz"),
            detector=_header(headers, "x-detector-hz"),
            note=note,
        )
        return headers

    def _verify_suppression(self, status: dict) -> None:
        suppression = status["suppression"]
        problems = []
        if suppression["engaged_count"] != 1:
            problems.append(
                f"expected 1 notch engagement, got {suppression['engaged_count']}"
            )
        if suppression["released_count"] != 1:
            problems.append(
                f"expected 1 notch release, got {suppression['released_count']}"
            )
        if suppression["notch_hz"] is not None:
            problems.append(f"notch still active: {suppression['notch_hz']}")
        if suppression["suppressed_frames"] != 2 * HOWL_BAND_FRAMES - 1:
            problems.append(
                "unexpected suppressed frame count "
                f"{suppression['suppressed_frames']}"
            )
        if problems:
            raise SystemExit(
                "suppression verification failed:\n  " + "\n  ".join(problems)
            )
        print("\nsuppression verification: OK")

    def _verify(self, status: dict) -> None:
        """Check the observable contract; the demo doubles as a smoke test."""

        expected = (
            ["true", "true", "concealed", "recovered", "true", "true"]
            + ["concealed", "recovered", "true"]
            + ["concealed", "concealed", "recovered"]
            + ["true"] * 4
        )
        problems: list[str] = []
        if self.played_seqs != list(range(len(expected))):
            problems.append(f"playout order {self.played_seqs} is not 0..N")
        if self.played_kinds != expected:
            problems.append(f"frame kinds {self.played_kinds} != {expected}")

        def pcm(index: int) -> list[int]:
            return list(struct.unpack(f"<{FRAME_SAMPLES}h", self.played[index]))

        # Recovered frame keeps the true audio after the 5 ms fade.
        for index, seq in ((3, 3), (7, 7), (11, 11)):
            true = struct.unpack(f"<{FRAME_SAMPLES}h", self.frames[seq])
            if pcm(index)[FADE_SAMPLES:] != list(true[FADE_SAMPLES:]):
                problems.append(f"seq {seq}: samples after fade were altered")

        # Two consecutive concealments: the second is half the first's level.
        first = max(abs(v) for v in pcm(9))
        second = max(abs(v) for v in pcm(10))
        if not first or abs(second - first / 2) > first * 0.1:
            problems.append(f"concealment levels did not halve ({first} -> {second})")

        if status["stats"]["late_frames"] != 2:
            problems.append(f"expected 2 late frames, got {status['stats']}")
        if status["stats"]["duplicate_frames"] != 1:
            problems.append(f"expected 1 duplicate, got {status['stats']}")

        if problems:
            raise SystemExit("demo verification failed:\n  " + "\n  ".join(problems))
        print("\nverification: OK")

    def _write_outputs(self, status: dict) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        wav_path = self.out_dir / "playout.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(b"".join(self.played))

        (self.out_dir / "transcript.json").write_text(
            json.dumps(
                {"frames": self.transcript, "status": status},
                indent=2,
            )
            + "\n"
        )

        counts: dict[str, int] = {}
        for kind in self.played_kinds:
            counts[kind] = counts.get(kind, 0) + 1
        print("\n== summary ==")
        print(f"pulled frames: {len(self.played)}  by kind: {counts}")
        print(f"engine stats:  {status['stats']}")
        print(f"wrote {wav_path} and transcript.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("MONITOR_URL", "http://127.0.0.1:8000"),
        help="base URL of the monitor-return API",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("MONITOR_DEMO_OUT", "demo_out"),
        help="directory for playout.wav and transcript.json",
    )
    parser.add_argument(
        "--suppression",
        action="store_true",
        help="also exercise auto howling suppression (server must have it enabled)",
    )
    args = parser.parse_args()
    Demo(args.url, Path(args.out), args.suppression).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
