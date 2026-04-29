"""Thread-safe ring buffer for the live MJPEG preview stream.

The vision capture loop calls :meth:`StreamBuffer.publish` every
~200ms with a downsized JPEG; the dashboard's ``/api/stream``
route subscribes to the buffer and forwards each new frame to
viewers as ``multipart/x-mixed-replace``.

Semantics
---------
- Fixed-size ring (default 30 frames). Old frames evict on
  overflow; the publisher never blocks.
- A condition variable wakes subscribers on each publish.
- Subscribers always jump to the most-recent frame on their next
  iteration — no replay, no backlog. Slow consumers don't build
  latency, they skip frames. This is the right policy for a live
  preview; queueing would grow latency unboundedly.
- A configurable subscriber cap (default 5) protects Pi outbound
  bandwidth. Past the cap, ``subscribe()`` raises
  :class:`SubscriberLimitExceeded`; the route maps it to 503.

Threading
---------
``publish()`` runs on the capture-loop thread; subscribers iterate
on a request thread (FastAPI runs sync generators in its
threadpool). One :class:`threading.Condition` guards everything;
critical sections are tiny.

Dependency direction
--------------------
``src.vision`` imports nothing from ``src.web``. ``VisionCapture``
takes a duck-typed sink with ``publish(bytes)``; this module is
that sink in production.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Iterator

logger = logging.getLogger(__name__)


_DEFAULT_CAPACITY = 30
_DEFAULT_MAX_SUBSCRIBERS = 5


class SubscriberLimitExceeded(RuntimeError):
    """Raised when ``subscribe()`` is called past the configured cap."""


class StreamBuffer:
    """Thread-safe ring buffer for JPEG preview frames.

    Args:
        capacity: max frames retained. Default 30 (~1MB at 30KB/frame).
        max_subscribers: cap on concurrent ``subscribe()`` sessions.
    """

    def __init__(
        self,
        capacity: int = _DEFAULT_CAPACITY,
        max_subscribers: int = _DEFAULT_MAX_SUBSCRIBERS,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if max_subscribers < 1:
            raise ValueError("max_subscribers must be at least 1")
        self._capacity = capacity
        self._max_subscribers = max_subscribers
        self._frames: deque[tuple[int, bytes]] = deque(maxlen=capacity)
        self._seq = 0
        self._closed = False
        self._cond = threading.Condition()
        self._subscriber_count = 0

    # ── Properties (diagnostics) ──────────────────────────────────────────────

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def max_subscribers(self) -> int:
        return self._max_subscribers

    @property
    def subscriber_count(self) -> int:
        with self._cond:
            return self._subscriber_count

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def __len__(self) -> int:
        with self._cond:
            return len(self._frames)

    # ── Publisher API ─────────────────────────────────────────────────────────

    def publish(self, jpeg_bytes: bytes) -> None:
        """Push a new JPEG. Non-blocking; evicts oldest on overflow.

        Closed buffers swallow further publishes silently — the
        capture thread shouldn't crash on dashboard shutdown.
        """
        if not isinstance(jpeg_bytes, bytes | bytearray | memoryview):
            raise TypeError(f"publish() expects bytes-like, got {type(jpeg_bytes).__name__}")
        payload = bytes(jpeg_bytes)
        with self._cond:
            if self._closed:
                return
            self._seq += 1
            self._frames.append((self._seq, payload))
            self._cond.notify_all()

    def latest(self) -> bytes | None:
        with self._cond:
            return self._frames[-1][1] if self._frames else None

    def latest_seq(self) -> int:
        """Sequence number of the most-recent frame; 0 if empty."""
        with self._cond:
            return self._seq

    def close(self) -> None:
        """Mark closed and wake all subscribers."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # ── Subscriber API ────────────────────────────────────────────────────────

    def subscribe(self, *, timeout: float | None = None) -> Subscription:
        """Start a subscription. Raises ``SubscriberLimitExceeded`` past the cap.

        Args:
            timeout: per-frame wait timeout in seconds. ``None``
                waits indefinitely. Routes typically pass a value
                so an idle stream doesn't pin a worker thread when
                the publisher stalls.
        """
        with self._cond:
            if self._subscriber_count >= self._max_subscribers:
                raise SubscriberLimitExceeded(
                    f"max {self._max_subscribers} concurrent subscribers reached"
                )
            self._subscriber_count += 1
            initial_seq = self._seq
        return Subscription(self, initial_seq=initial_seq, timeout=timeout)

    def _release_subscriber_slot(self) -> None:
        with self._cond:
            self._subscriber_count = max(0, self._subscriber_count - 1)

    def _pop_next_for_subscriber(
        self,
        last_seq: int,
        timeout: float | None,
    ) -> tuple[int, bytes] | None:
        """Return the most-recent frame newer than ``last_seq``.

        Blocks on the condition variable until either a new frame
        is published, the buffer closes, or the timeout elapses.
        Returns ``None`` to signal end-of-iteration in any of those
        terminal cases.
        """
        with self._cond:
            while True:
                if self._closed:
                    return None
                if self._seq > last_seq:
                    return self._frames[-1]
                if not self._cond.wait(timeout=timeout):
                    return None  # timeout


class Subscription:
    """Iterator over new frames from a :class:`StreamBuffer`.

    Use as::

        with stream_buffer.subscribe() as sub:
            for frame in sub:
                yield mjpeg_chunk(frame)

    Late-joining subscribers see only frames published after
    ``subscribe()`` returned. A subscriber that falls behind the
    publisher jumps to the newest frame on its next iteration —
    "newest only" semantics, no queue.
    """

    def __init__(
        self,
        buffer: StreamBuffer,
        *,
        initial_seq: int,
        timeout: float | None = None,
    ) -> None:
        self._buffer = buffer
        self._timeout = timeout
        self._last_seq = initial_seq
        self._closed = False

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._buffer._release_subscriber_slot()

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        if self._closed:
            raise StopIteration
        result = self._buffer._pop_next_for_subscriber(self._last_seq, self._timeout)
        if result is None:
            raise StopIteration
        seq, payload = result
        self._last_seq = seq
        return payload


__all__ = [
    "StreamBuffer",
    "SubscriberLimitExceeded",
    "Subscription",
]
