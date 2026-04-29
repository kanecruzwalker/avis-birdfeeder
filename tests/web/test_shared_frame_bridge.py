"""Unit tests for src.web.shared_frame_bridge.

Covers:
    - Subscriber raises SegmentNotFound before any publisher exists
    - Round-trip: publisher.publish(jpeg) → subscriber.read() yields same bytes
    - Sequence number monotonicity across multiple publishes
    - Subscriber returns None when no new frame since last_seq
    - Publisher rejects bytes larger than MAX_FRAME_BYTES
    - Publisher rejects non-bytes input
    - Publisher reattach after a stale segment from a crashed prior run
      resets the header (subscriber sees no frame until next publish)
    - Pump thread forwards new frames into a duck-typed sink
    - Pump thread is idempotent (start twice = single thread)
    - close() is idempotent and safe to call from any thread

Notes
-----
Each test uses a unique segment name (uuid4 hex) so concurrent test
processes never collide. We always close + unlink in fixtures so a
flaky test doesn't leak /dev/shm/avis-test-* on Linux.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest

from src.web.shared_frame_bridge import (
    MAX_FRAME_BYTES,
    FrameTooLarge,
    SegmentNotFound,
    SharedFramePublisher,
    SharedFrameSubscriber,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _unique_name() -> str:
    # POSIX shm names start with '/'; SharedMemory adds it. The
    # uuid is plenty unique per test run.
    return f"avis-test-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def shm_name():
    name = _unique_name()
    yield name
    # Best-effort cleanup — segment may already be gone if the test
    # exercised unlink().
    try:
        from multiprocessing import shared_memory

        shared_memory.SharedMemory(name=name, create=False).unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


# ── Lifecycle ────────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_subscriber_before_publisher_raises(self, shm_name):
        with pytest.raises(SegmentNotFound):
            SharedFrameSubscriber(shm_name)

    def test_publisher_creates_segment(self, shm_name):
        pub = SharedFramePublisher(shm_name)
        try:
            sub = SharedFrameSubscriber(shm_name)
            sub.close()
        finally:
            pub.close()
            pub.unlink()

    def test_close_is_idempotent(self, shm_name):
        pub = SharedFramePublisher(shm_name)
        try:
            pub.close()
            pub.close()  # second call is a no-op
        finally:
            pub.unlink()


# ── Round-trip ───────────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_single_frame_round_trip(self, shm_name):
        pub = SharedFramePublisher(shm_name)
        try:
            payload = b"\xff\xd8\xff\xe0fake-jpeg" * 32
            pub.publish(payload)

            sub = SharedFrameSubscriber(shm_name)
            try:
                result = sub.read()
                assert result is not None
                seq, bytes_out = result
                assert seq == 1
                assert bytes_out == payload
            finally:
                sub.close()
        finally:
            pub.close()
            pub.unlink()

    def test_multiple_frames_monotonic_seq(self, shm_name):
        pub = SharedFramePublisher(shm_name)
        try:
            sub = SharedFrameSubscriber(shm_name)
            try:
                seqs = []
                for i in range(5):
                    pub.publish(f"frame-{i}".encode())
                    result = sub.read()
                    assert result is not None
                    seq, payload = result
                    seqs.append(seq)
                    assert payload == f"frame-{i}".encode()
                # Strictly increasing
                assert seqs == sorted(seqs)
                assert len(set(seqs)) == len(seqs)
            finally:
                sub.close()
        finally:
            pub.close()
            pub.unlink()

    def test_read_returns_none_when_no_new_frame(self, shm_name):
        pub = SharedFramePublisher(shm_name)
        try:
            sub = SharedFrameSubscriber(shm_name)
            try:
                # No publish yet — empty segment.
                assert sub.read() is None

                pub.publish(b"hello")
                assert sub.read() is not None
                # Same frame; no new seq.
                assert sub.read() is None
            finally:
                sub.close()
        finally:
            pub.close()
            pub.unlink()

    def test_subscriber_skips_to_latest_when_publisher_advances(self, shm_name):
        # Single-slot semantics: a slow subscriber sees only the
        # newest frame, not all the ones it missed.
        pub = SharedFramePublisher(shm_name)
        try:
            sub = SharedFrameSubscriber(shm_name)
            try:
                for i in range(10):
                    pub.publish(f"frame-{i}".encode())
                result = sub.read()
                assert result is not None
                seq, payload = result
                assert seq == 10
                assert payload == b"frame-9"
            finally:
                sub.close()
        finally:
            pub.close()
            pub.unlink()


# ── Validation ───────────────────────────────────────────────────────────────


class TestValidation:
    def test_rejects_oversized_frame(self, shm_name):
        pub = SharedFramePublisher(shm_name)
        try:
            with pytest.raises(FrameTooLarge):
                pub.publish(b"x" * (MAX_FRAME_BYTES + 1))
        finally:
            pub.close()
            pub.unlink()

    def test_rejects_non_bytes_input(self, shm_name):
        pub = SharedFramePublisher(shm_name)
        try:
            with pytest.raises(TypeError):
                pub.publish("not-bytes")  # type: ignore[arg-type]
        finally:
            pub.close()
            pub.unlink()

    def test_accepts_bytearray_and_memoryview(self, shm_name):
        pub = SharedFramePublisher(shm_name)
        try:
            sub = SharedFrameSubscriber(shm_name)
            try:
                pub.publish(bytearray(b"ba"))
                assert sub.read()[1] == b"ba"
                pub.publish(memoryview(b"mv-payload"))
                assert sub.read()[1] == b"mv-payload"
            finally:
                sub.close()
        finally:
            pub.close()
            pub.unlink()


# ── Reattach after crash ─────────────────────────────────────────────────────


class TestReattach:
    def test_publisher_reattaches_existing_segment_and_resets_header(self, shm_name):
        # Simulate a previous publisher that crashed leaving a frame
        # in the segment. A fresh publisher reattaches and zeroes
        # the header so subscribers don't latch onto the stale frame.
        pub1 = SharedFramePublisher(shm_name)
        pub1.publish(b"stale-frame")
        # Detach without unlinking — segment persists.
        pub1.close()

        pub2 = SharedFramePublisher(shm_name)
        try:
            sub = SharedFrameSubscriber(shm_name)
            try:
                # No frame published since the new publisher started —
                # subscriber sees an empty slot, not the stale frame.
                assert sub.read() is None

                pub2.publish(b"fresh-frame")
                result = sub.read()
                assert result is not None
                seq, payload = result
                assert seq == 1  # fresh publisher's count starts at 1
                assert payload == b"fresh-frame"
            finally:
                sub.close()
        finally:
            pub2.close()
            pub2.unlink()


# ── Pump thread ──────────────────────────────────────────────────────────────


class _RecordingSink:
    """Duck-typed sink standing in for StreamBuffer.publish."""

    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self._lock = threading.Lock()

    def publish(self, jpeg: bytes) -> None:
        with self._lock:
            self.frames.append(bytes(jpeg))

    def snapshot(self) -> list[bytes]:
        with self._lock:
            return list(self.frames)


class TestPump:
    def test_pump_forwards_new_frames_into_sink(self, shm_name):
        pub = SharedFramePublisher(shm_name)
        try:
            sub = SharedFrameSubscriber(shm_name)
            sink = _RecordingSink()
            try:
                sub.start_pump(sink, poll_interval=0.01)

                for i in range(3):
                    pub.publish(f"f{i}".encode())
                    time.sleep(0.05)

                # Wait briefly for the pump to drain the last publish.
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    if len(sink.snapshot()) >= 3:
                        break
                    time.sleep(0.02)

                forwarded = sink.snapshot()
                assert len(forwarded) >= 1
                # We can't guarantee all three landed (single-slot;
                # pump may skip), but the last one must be present.
                assert b"f2" in forwarded
            finally:
                sub.close()
        finally:
            pub.close()
            pub.unlink()

    def test_pump_start_is_idempotent(self, shm_name):
        pub = SharedFramePublisher(shm_name)
        try:
            sub = SharedFrameSubscriber(shm_name)
            sink = _RecordingSink()
            try:
                sub.start_pump(sink, poll_interval=0.05)
                first_thread = sub._pump_thread
                sub.start_pump(sink, poll_interval=0.05)
                # Same thread instance — second call is a no-op.
                assert sub._pump_thread is first_thread
            finally:
                sub.close()
        finally:
            pub.close()
            pub.unlink()

    def test_pump_stops_on_close(self, shm_name):
        pub = SharedFramePublisher(shm_name)
        try:
            sub = SharedFrameSubscriber(shm_name)
            sink = _RecordingSink()
            sub.start_pump(sink, poll_interval=0.01)
            pump_thread = sub._pump_thread
            sub.close()
            # close() joins with a 1.0s timeout; a healthy stop
            # should complete in well under that.
            assert pump_thread is not None
            pump_thread.join(timeout=1.0)
            assert not pump_thread.is_alive()
        finally:
            pub.close()
            pub.unlink()

    def test_pump_survives_sink_exception(self, shm_name):
        # A sink that raises on publish shouldn't kill the pump
        # thread — bugs in the dashboard's stream buffer must not
        # prevent the bridge from continuing to consume frames.
        class _FlakySink:
            def __init__(self) -> None:
                self.calls = 0

            def publish(self, _: bytes) -> None:
                self.calls += 1
                raise RuntimeError("boom")

        pub = SharedFramePublisher(shm_name)
        try:
            sub = SharedFrameSubscriber(shm_name)
            flaky = _FlakySink()
            try:
                sub.start_pump(flaky, poll_interval=0.01)
                for i in range(3):
                    pub.publish(f"f{i}".encode())
                    time.sleep(0.05)
                # Pump kept calling despite exceptions.
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    if flaky.calls >= 1:
                        break
                    time.sleep(0.02)
                assert flaky.calls >= 1
                # And the thread is still alive.
                assert sub._pump_thread is not None
                assert sub._pump_thread.is_alive()
            finally:
                sub.close()
        finally:
            pub.close()
            pub.unlink()
