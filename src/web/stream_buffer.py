"""Thread-safe ring buffer for the live MJPEG preview stream.

The vision capture loop publishes a downsized JPEG every ~200ms via
:meth:`StreamBuffer.publish`. The web dashboard's ``/api/stream``
endpoint subscribes to the buffer and forwards each new frame to
connected viewers as ``multipart/x-mixed-replace``.

Design
------
- Fixed-size ring (default 30 frames). Old frames are evicted on
  overflow; the buffer never blocks the publisher.
- A condition variable wakes subscribers when a new frame lands.
- Each subscriber tracks its own "last seen" sequence number, so
  subscribers that join late only get the frames published after
  they joined (no replay), and slow subscribers don't hold up the
  publisher (each subscriber jumps to the most-recent frame on its
  next iteration -- "newest only" semantics, which is exactly what
  a live preview wants).
- A configurable subscriber cap (default 5) limits how many
  concurrent viewers we accept. New connections beyond the cap raise
  :class:`SubscriberLimitExceeded` and the route turns that into a
  503 response. The Pi's outbound bandwidth is the practical limit
  on viewer count; refusing cleanly is preferable to degrading an
  existing stream.

Threading model
---------------
``publish()`` is called from the capture-loop thread. Subscribers
iterate from a request-handler thread (FastAPI runs sync generators
in its threadpool). Both paths take the same ``threading.Condition``
lock; contention is minimal because the critical sections are tiny
(append + seq increment + notify on the publisher side; pointer read
+ optional wait on the subscriber side).

Dependency direction
--------------------
This module has no FastAPI imports and no agent imports -- it sits
in ``src.web`` only as a convenient location, but ``src.vision``
imports nothing from ``src.web``. ``VisionCapture`` accepts a
duck-typed ``publish(bytes)`` sink, and tests pass a ``StreamBuffer``
or any other compatible object.
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
    """Raised when ``subscribe()`` is called past the configured cap.

    The route layer maps this to HTTP 503. Pi outbound bandwidth and
    JPEG encoding CPU both scale with viewer count; refusing cleanly
    is preferable to degrading an existing viewer's stream.
    """


class StreamBuffer:
    """Thread-safe ring buffer for JPEG preview frames.

    Args:
        capacity: max frames retained in the ring. Older frames are
            evicted on overflow. Default 30 (~1MB at 30KB/frame, the
            measured size for 640x360 q=75 JPEGs from the Pi camera).
        max_subscribers: hard cap on concurrent ``subscribe()``
            sessions. Past the cap, ``subscribe()`` raises
            :class:`SubscriberLimitExceeded`. Default 5.
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
        # Monotonic sequence number for the most recent frame. Starts
        # at 0; the first publish becomes seq=1. Subscribers compare
        # against this to skip frames they've already seen.
        self._seq = 0
        self._closed = False

        # Single condition variable guards everything: the deque, the
        # sequence number, the closed flag, and the subscriber count.
        # Critical sections are tiny so a single lock is fine.
        self._cond = threading.Condition()
        self._subscriber_count = 0

    # ── Properties ────────────────────────────────────────────────────────────

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
        """Push a new JPEG frame onto the ring.

        Non-blocking: evicts the oldest frame on overflow so the
        capture loop never stalls. Wakes all current subscribers.

        Closed buffers swallow further publishes silently. The
        capture thread shouldn't crash on dashboard shutdown, so we
        return cleanly instead of raising.
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
        """Return the most recent JPEG, or ``None`` when empty."""
        with self._cond:
            if not self._frames:
                return None
            return self._frames[-1][1]

    def latest_seq(self) -> int:
        """Sequence number of the most-recent published frame.

        ``0`` when nothing has been published yet. Useful for
        diagnostics ("is the publisher alive?") and as the starting
        seq for a subscriber that wants to skip replay.
        """
        with self._cond:
            return self._seq

    def close(self) -> None:
        """Mark the buffer closed and wake all subscribers.

        Subscribers iterating after ``close()`` see their iterator
        end (``StopIteration``). Subsequent ``publish()`` calls are
        no-ops.
        """
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # ── Subscriber API ────────────────────────────────────────────────────────

    def subscribe(self, *, timeout: float | None = None) -> Subscription:
        """Start a new subscription.

        Args:
            timeout: per-frame wait timeout in seconds. ``None`` waits
                indefinitely. Routes typically pass a value (e.g. 30s)
                so an idle stream doesn't hold a worker thread forever
                if the agent stalls.

        Returns:
            A :class:`Subscription` -- iterate it for new frames, or
            use it as a context manager to ensure ``close()`` runs.

        Raises:
            SubscriberLimitExceeded: when the per-buffer cap is full.
        """
        with self._cond:
            if self._subscriber_count >= self._max_subscribers:
                raise SubscriberLimitExceeded(
                    f"max {self._max_subscribers} concurrent subscribers reached"
                )
            self._subscriber_count += 1
            initial_seq = self._seq
        return Subscription(self, initial_seq=initial_seq, timeout=timeout)

    def _decrement_subscriber(self) -> None:
        with self._cond:
            self._subscriber_count = max(0, self._subscriber_count - 1)


class Subscription:
    """Context-managed iterator over new frames.

    Use as::

        with stream_buffer.subscribe() as sub:
            for frame in sub:
                yield mjpeg_chunk(frame)

    The iterator yields each newly-published JPEG. A subscriber that
    falls behind the publisher only ever sees the most-recent frame
    on its next ``__next__`` -- intermediate frames are skipped, not
    queued. For a live preview, "newest only" is the correct
    semantic; queueing would build latency unboundedly.
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
        # Start AT the current seq -- only frames published after
        # subscribe() returns are yielded. The route layer can also
        # call buffer.latest() once before iterating to send an
        # immediate initial frame so the <img> isn't blank.
        self._last_seq = initial_seq
        self._closed = False

    def __enter__(self) -> Subscription:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._buffer._decrement_subscriber()

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        buf = self._buffer
        with buf._cond:
            while True:
                if self._closed:
                    raise StopIteration
                if buf._closed:
                    raise StopIteration
                if buf._seq > self._last_seq:
                    # Always jump to the most-recent frame -- skip
                    # any backlog. This is the live-preview policy.
                    seq, payload = buf._frames[-1]
                    self._last_seq = seq
                    return payload
                # Wait for publish or close. wait() returns False on
                # timeout (Python 3.2+).
                got = buf._cond.wait(timeout=self._timeout)
                if not got:
                    # Idle timeout -- end the iterator cleanly so the
                    # streaming response can return. The browser will
                    # typically reconnect when the user refocuses the
                    # tab (per the design doc's bandwidth mitigation).
                    raise StopIteration


__all__ = [
    "StreamBuffer",
    "SubscriberLimitExceeded",
    "Subscription",
]
