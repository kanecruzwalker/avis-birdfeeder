"""Unit tests for src.web.stream_buffer.

Covers:
    - construction and validation
    - publish + latest round-trip
    - capacity-bounded eviction
    - sequence numbers monotonic, never reused
    - subscriber receives newly-published frames in order
    - subscriber that falls behind sees only the latest frame
    - subscriber limit enforced via SubscriberLimitExceeded
    - closing the buffer wakes subscribers cleanly
    - publish after close is a no-op (no exception)
    - thread-safety under concurrent publish + subscribe
"""

from __future__ import annotations

import threading
import time

import pytest

from src.web.stream_buffer import (
    StreamBuffer,
    SubscriberLimitExceeded,
    Subscription,
)

# ── Construction ─────────────────────────────────────────────────────────────


class TestConstruction:
    def test_default_capacity(self):
        buf = StreamBuffer()
        assert buf.capacity == 30

    def test_default_max_subscribers(self):
        buf = StreamBuffer()
        assert buf.max_subscribers == 5

    def test_custom_capacity(self):
        buf = StreamBuffer(capacity=10)
        assert buf.capacity == 10

    def test_custom_max_subscribers(self):
        buf = StreamBuffer(max_subscribers=2)
        assert buf.max_subscribers == 2

    def test_zero_capacity_rejected(self):
        with pytest.raises(ValueError, match="capacity"):
            StreamBuffer(capacity=0)

    def test_zero_max_subscribers_rejected(self):
        with pytest.raises(ValueError, match="max_subscribers"):
            StreamBuffer(max_subscribers=0)

    def test_starts_empty(self):
        buf = StreamBuffer()
        assert len(buf) == 0
        assert buf.latest() is None
        assert buf.latest_seq() == 0
        assert buf.subscriber_count == 0
        assert buf.closed is False


# ── Publisher path ───────────────────────────────────────────────────────────


class TestPublish:
    def test_publish_then_latest(self):
        buf = StreamBuffer()
        buf.publish(b"frame-A")
        assert buf.latest() == b"frame-A"

    def test_seq_increments_per_publish(self):
        buf = StreamBuffer()
        assert buf.latest_seq() == 0
        buf.publish(b"a")
        assert buf.latest_seq() == 1
        buf.publish(b"b")
        assert buf.latest_seq() == 2

    def test_publish_accepts_bytearray(self):
        buf = StreamBuffer()
        buf.publish(bytearray(b"frame"))
        assert buf.latest() == b"frame"

    def test_publish_accepts_memoryview(self):
        buf = StreamBuffer()
        buf.publish(memoryview(b"frame"))
        assert buf.latest() == b"frame"

    def test_publish_rejects_str(self):
        buf = StreamBuffer()
        with pytest.raises(TypeError):
            buf.publish("not-bytes")  # type: ignore[arg-type]

    def test_publish_rejects_int(self):
        buf = StreamBuffer()
        with pytest.raises(TypeError):
            buf.publish(42)  # type: ignore[arg-type]


class TestCapacity:
    def test_overflow_evicts_oldest(self):
        buf = StreamBuffer(capacity=3)
        for n in range(5):
            buf.publish(f"frame-{n}".encode())
        # Only the last 3 frames should remain
        assert len(buf) == 3
        assert buf.latest() == b"frame-4"
        assert buf.latest_seq() == 5

    def test_capacity_one(self):
        buf = StreamBuffer(capacity=1)
        buf.publish(b"a")
        buf.publish(b"b")
        assert len(buf) == 1
        assert buf.latest() == b"b"


class TestClose:
    def test_publish_after_close_is_noop(self):
        buf = StreamBuffer()
        buf.close()
        assert buf.closed is True
        # Should not raise
        buf.publish(b"frame")
        assert buf.latest() is None
        assert buf.latest_seq() == 0


# ── Subscriber path ──────────────────────────────────────────────────────────


class TestSubscribe:
    def test_subscribe_returns_subscription(self):
        buf = StreamBuffer()
        sub = buf.subscribe()
        assert isinstance(sub, Subscription)
        sub.close()

    def test_subscriber_count_tracks_active(self):
        buf = StreamBuffer()
        assert buf.subscriber_count == 0
        sub = buf.subscribe()
        assert buf.subscriber_count == 1
        sub.close()
        assert buf.subscriber_count == 0

    def test_context_manager_decrements_on_exit(self):
        buf = StreamBuffer()
        with buf.subscribe():
            assert buf.subscriber_count == 1
        assert buf.subscriber_count == 0

    def test_double_close_safe(self):
        buf = StreamBuffer()
        sub = buf.subscribe()
        sub.close()
        sub.close()  # should not double-decrement
        assert buf.subscriber_count == 0

    def test_subscriber_limit_enforced(self):
        buf = StreamBuffer(max_subscribers=2)
        a = buf.subscribe()
        b = buf.subscribe()
        with pytest.raises(SubscriberLimitExceeded):
            buf.subscribe()
        a.close()
        # Slot freed
        c = buf.subscribe()
        c.close()
        b.close()

    def test_subscriber_skips_pre_existing_frames(self):
        """A late-joining subscriber should not replay frames that
        were published before subscribe() was called."""
        buf = StreamBuffer()
        buf.publish(b"old-1")
        buf.publish(b"old-2")
        with buf.subscribe(timeout=0.05) as sub:
            # No new frames, the iterator should time out without
            # yielding any of the pre-existing frames.
            frames = list(sub)
        assert frames == []

    def test_subscriber_receives_new_frames(self):
        buf = StreamBuffer()
        with buf.subscribe(timeout=0.5) as sub:
            buf.publish(b"new-1")
            it = iter(sub)
            assert next(it) == b"new-1"

    def test_subscriber_jumps_to_latest_skipping_backlog(self):
        """When the publisher gets ahead of the subscriber, the
        subscriber should jump to the newest frame on its next
        iteration (live-preview semantics, no queue)."""
        buf = StreamBuffer()
        with buf.subscribe(timeout=0.5) as sub:
            buf.publish(b"a")
            buf.publish(b"b")
            buf.publish(b"c")
            it = iter(sub)
            assert next(it) == b"c"

    def test_subscriber_timeout_ends_iterator(self):
        buf = StreamBuffer()
        with buf.subscribe(timeout=0.05) as sub:
            t0 = time.monotonic()
            frames = list(sub)
            elapsed = time.monotonic() - t0
        assert frames == []
        assert elapsed < 1.0  # didn't hang

    def test_close_buffer_wakes_subscriber(self):
        """When the buffer closes, any sleeping subscriber should
        wake up and StopIteration cleanly -- not hang on the
        condition."""
        buf = StreamBuffer()
        results: list = []
        sub = buf.subscribe(timeout=None)  # would otherwise wait forever

        def consume():
            try:
                results.extend(list(sub))
            finally:
                sub.close()

        thread = threading.Thread(target=consume, daemon=True)
        thread.start()
        # Give the consumer a moment to enter wait()
        time.sleep(0.05)
        buf.close()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "subscriber thread did not wake on close"
        assert results == []


# ── Concurrency ──────────────────────────────────────────────────────────────


class TestConcurrency:
    def test_concurrent_publishers_dont_lose_seq(self):
        """Sequence numbers must remain unique across concurrent
        publishers (no torn writes, no dropped increments)."""
        buf = StreamBuffer(capacity=1000)
        n_threads = 4
        per_thread = 50

        def producer(tag: int):
            for i in range(per_thread):
                buf.publish(f"t{tag}-i{i}".encode())

        threads = [
            threading.Thread(target=producer, args=(t,), daemon=True) for t in range(n_threads)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=5.0)

        assert buf.latest_seq() == n_threads * per_thread

    def test_concurrent_publish_and_subscribe(self):
        """A subscriber should see at least one frame when a
        producer publishes during its iteration window."""
        buf = StreamBuffer()
        seen: list[bytes] = []
        stop = threading.Event()

        def subscriber():
            with buf.subscribe(timeout=0.1) as sub:
                for frame in sub:
                    seen.append(frame)
                    if stop.is_set():
                        break

        consumer = threading.Thread(target=subscriber, daemon=True)
        consumer.start()
        # Give the subscriber a moment to register.
        time.sleep(0.05)

        for n in range(20):
            buf.publish(f"frame-{n}".encode())
            time.sleep(0.005)

        stop.set()
        # Force an unblock if the consumer is still in wait()
        buf.publish(b"final")
        consumer.join(timeout=2.0)

        assert not consumer.is_alive()
        assert len(seen) > 0
        # Each yielded frame must have been published at some point
        all_payloads = {f"frame-{n}".encode() for n in range(20)} | {b"final"}
        for frame in seen:
            assert frame in all_payloads


# ── Subscription init helper ─────────────────────────────────────────────────


class TestSubscriptionDirectConstruction:
    """The Subscription constructor is private API -- StreamBuffer.subscribe
    is the public path -- but a couple of invariants are worth pinning."""

    def test_starts_at_provided_seq(self):
        buf = StreamBuffer()
        buf.publish(b"a")
        buf.publish(b"b")
        # Manually start at seq=1, simulating "I've already seen frame 1"
        sub = Subscription(buf, initial_seq=1, timeout=0.05)
        try:
            it = iter(sub)
            # Should jump to the newest unseen frame (b)
            assert next(it) == b"b"
        finally:
            sub.close()
