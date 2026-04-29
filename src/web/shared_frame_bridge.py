"""Cross-process JPEG frame bridge backed by ``multiprocessing.shared_memory``.

Production runs the agent (``avis.service``) and the dashboard
(``avis-web.service``) as separate systemd units, so the in-process
:class:`StreamBuffer` is invisible to the dashboard. This module hands
those two processes a shared single-slot frame buffer: the agent
publishes annotated JPEG bytes; the dashboard polls and forwards each
new frame into its own in-process ``StreamBuffer`` so the existing
``/api/stream`` route serves them unchanged.

Why single-slot instead of a ring
---------------------------------
The capture loop publishes at ~5 fps; a subscriber that falls behind
already drops to "newest only" by design (see ``StreamBuffer``'s
"slow consumer skips frames" semantics). Carrying that semantic into
shared memory means we only ever need the latest frame — no offsets,
no head/tail, no overflow. Saves ~80 LOC of ring math.

Why polling instead of a cross-process condvar
----------------------------------------------
Cross-process condvars exist (``multiprocessing.Condition``) but
require a managed ``Manager`` server, which adds a third process and
its own failure modes. A 50 ms poll on a uint64 sequence counter
costs single-digit microseconds per tick and is robust to either
side crashing. The dashboard's stream route is the only consumer of
this signal; tightening the poll past 50 ms doesn't help the user
experience at 5 fps capture.

Layout
------
The shared segment is a fixed-size byte array with this header:

    bytes 0-7    sequence number (u64, little-endian)
                 only the publisher writes; monotonic; 0 means
                 "no frame published yet"
    bytes 8-11   payload size (u32) — bytes used by the current frame
    bytes 12-15  reserved (zeroed)
    bytes 16+    payload (JPEG bytes, up to ``MAX_FRAME_BYTES``)

Torn-read protection: the subscriber reads ``seq`` before and after
copying the payload. A mismatch means the publisher rolled the slot
mid-read; the subscriber retries. Since we only need the latest
frame, a "skip and try the next one" fallback is also fine after a
few retries.

Single-writer, multi-reader. Concurrent writers would race on the
slot — there's only ever one agent publishing.
"""

from __future__ import annotations

import logging
import struct
import threading
from multiprocessing import shared_memory

logger = logging.getLogger(__name__)


# Header fields packed in little-endian to match every platform we
# realistically run on (Pi 5 = aarch64 LE; laptops = x86_64 LE).
_SEQ_FORMAT = "<Q"  # u64 sequence number
_SIZE_FORMAT = "<I"  # u32 payload size
_HEADER_BYTES = 16  # seq(8) + size(4) + reserved(4)

# 256 KB total payload room. Production preview JPEGs are ~30 KB at
# 640x360 q=75; this is ~8x headroom for higher-quality demo frames
# without reallocating. Both sides must agree on the size — bump
# both processes together if you change it.
MAX_FRAME_BYTES = 256_000
_SEGMENT_BYTES = _HEADER_BYTES + MAX_FRAME_BYTES


# Subscriber poll cadence. 50 ms = 20 Hz, comfortably faster than the
# 5 fps publish rate so latency added by the bridge stays under a
# frame interval.
_DEFAULT_POLL_INTERVAL = 0.05

# Subscriber retry budget when seq changes mid-read. Three retries
# tolerate a publisher that's just slightly faster than the
# subscriber; beyond that we accept a torn read and retry on the
# next poll tick.
_MAX_TORN_READ_RETRIES = 3


# ── Errors ──────────────────────────────────────────────────────────────────


class FrameTooLarge(ValueError):
    """Raised when ``publish()`` is given a JPEG larger than ``MAX_FRAME_BYTES``."""


class SegmentNotFound(RuntimeError):
    """Raised when the subscriber tries to attach but the segment doesn't exist yet."""


# ── Publisher ──────────────────────────────────────────────────────────────


class SharedFramePublisher:
    """Single-writer JPEG slot in shared memory.

    ``publish(jpeg_bytes)`` matches the ``StreamBuffer.publish`` shape
    so :class:`VisionCapture` accepts this as a drop-in preview sink.

    Lifecycle:
        - First-toucher creates the segment.
        - If the segment already exists from a previous run that
          crashed, we attach + reinitialise the header (zeroes seq,
          which makes subscribers see "no frame" until the next
          publish — the desired behaviour after a publisher restart).
        - ``close()`` detaches; ``unlink()`` removes the segment from
          the OS so the next run starts clean.

    Thread safety: a single capture loop publishes; ``publish()`` is
    not safe to call from multiple threads concurrently.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        try:
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=_SEGMENT_BYTES)
            logger.info(
                "SharedFramePublisher created segment %r (requested=%d, allocated=%d).",
                name,
                _SEGMENT_BYTES,
                self._shm.size,
            )
        except FileExistsError:
            # Stale segment from a crash — reattach and reset header.
            self._shm = shared_memory.SharedMemory(name=name, create=False)
            # Note: OS may round size up to a page boundary (Windows
            # NamedFileMapping does this). Require enough room for our
            # layout, not exact equality.
            if self._shm.size < _SEGMENT_BYTES:
                self._shm.close()
                raise RuntimeError(
                    f"Stale shared-memory segment {name!r} has size {self._shm.size}, "
                    f"need at least {_SEGMENT_BYTES}. Unlink it manually and retry "
                    f"(/dev/shm/{name} on Linux)."
                ) from None
            logger.info("SharedFramePublisher reattached existing segment %r.", name)

        self._seq = 0
        # Reset header so subscribers don't latch onto a stale frame
        # that was in the segment when this publisher started.
        self._shm.buf[:_HEADER_BYTES] = b"\x00" * _HEADER_BYTES

    @property
    def name(self) -> str:
        return self._name

    @property
    def latest_seq(self) -> int:
        """Most recent sequence number this publisher has written."""
        return self._seq

    def publish(self, jpeg_bytes: bytes) -> None:
        """Write ``jpeg_bytes`` to the slot and bump the sequence.

        Closed segments are silently dropped — the agent capture loop
        shouldn't crash on dashboard shutdown. This mirrors
        :class:`StreamBuffer.publish`.

        Order of writes is load-bearing for torn-read protection:
        size first, then payload, then sequence last. A subscriber
        that reads sequence before this method finishes will see the
        old sequence and skip. A subscriber that reads sequence
        twice (before + after copy) catches a publisher that rolled
        mid-read.
        """
        if not isinstance(jpeg_bytes, bytes | bytearray | memoryview):
            raise TypeError(f"publish() expects bytes-like, got {type(jpeg_bytes).__name__}")
        payload = bytes(jpeg_bytes)
        if len(payload) > MAX_FRAME_BYTES:
            raise FrameTooLarge(
                f"frame size {len(payload)} exceeds MAX_FRAME_BYTES={MAX_FRAME_BYTES}"
            )
        if self._shm is None:
            return  # closed

        size = len(payload)
        # Header: write size, copy payload, then bump seq.
        # Reading the buffer back from a closed shm raises ValueError;
        # treat that as "publish dropped" the same way StreamBuffer does.
        try:
            self._shm.buf[8:12] = struct.pack(_SIZE_FORMAT, size)
            self._shm.buf[_HEADER_BYTES : _HEADER_BYTES + size] = payload
            self._seq += 1
            self._shm.buf[0:8] = struct.pack(_SEQ_FORMAT, self._seq)
        except ValueError:
            # buffer detached mid-write (shouldn't happen in practice
            # because we only close from the same thread that publishes)
            return

    def close(self) -> None:
        """Detach this process from the segment. Idempotent."""
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                logger.exception("SharedFramePublisher close failed (ignored)")
            self._shm = None

    def unlink(self) -> None:
        """Remove the segment from the OS. Call after ``close()`` from
        the owning process when shutting down for good."""
        try:
            shared_memory.SharedMemory(name=self._name, create=False).unlink()
            logger.info("SharedFramePublisher unlinked segment %r.", self._name)
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("SharedFramePublisher unlink failed (ignored)")


# ── Subscriber ─────────────────────────────────────────────────────────────


class SharedFrameSubscriber:
    """Reads the latest JPEG slot from a publisher's shared segment.

    Typical use is the dashboard's ``__main__``: attach to the
    publisher's segment, then ``start_pump(stream_buffer)`` to forward
    each new frame into a local in-process :class:`StreamBuffer`. The
    existing ``/api/stream`` route then sees frames the same way it
    sees in-process publishes.

    ``read()`` is a low-level "give me the latest frame if it's newer
    than what I last saw" call; the pump thread is built on top of it.
    """

    def __init__(self, name: str) -> None:
        try:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
        except FileNotFoundError as exc:
            raise SegmentNotFound(
                f"shared-memory segment {name!r} does not exist. "
                f"Start the publisher (the agent) before the subscriber, "
                f"or recheck AVIS_STREAM_SHM matches on both sides."
            ) from exc
        # OS may round size up to a page boundary (Windows
        # NamedFileMapping does this); require enough room, not
        # exact equality. A smaller segment means publisher and
        # subscriber were built against different MAX_FRAME_BYTES.
        if self._shm.size < _SEGMENT_BYTES:
            self._shm.close()
            raise RuntimeError(
                f"Shared-memory segment {name!r} has size {self._shm.size}, "
                f"need at least {_SEGMENT_BYTES}. Publisher and subscriber are "
                f"out of sync — bump both processes together."
            )
        self._name = name
        self._last_seq = 0
        self._stop_event = threading.Event()
        self._pump_thread: threading.Thread | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def last_seq(self) -> int:
        return self._last_seq

    def read(self) -> tuple[int, bytes] | None:
        """Return ``(seq, jpeg_bytes)`` if a newer frame is available.

        Returns ``None`` when the publisher hasn't advanced past
        ``last_seq`` yet (or the segment is still empty). Updates
        ``last_seq`` on success.

        Retries up to ``_MAX_TORN_READ_RETRIES`` times if the
        publisher rolled the slot mid-read.
        """
        if self._shm is None:
            return None

        for _ in range(_MAX_TORN_READ_RETRIES):
            seq_before = struct.unpack(_SEQ_FORMAT, bytes(self._shm.buf[0:8]))[0]
            if seq_before == 0 or seq_before == self._last_seq:
                return None
            size = struct.unpack(_SIZE_FORMAT, bytes(self._shm.buf[8:12]))[0]
            if size == 0 or size > MAX_FRAME_BYTES:
                # Header inconsistent — likely a torn read; retry.
                continue
            payload = bytes(self._shm.buf[_HEADER_BYTES : _HEADER_BYTES + size])
            seq_after = struct.unpack(_SEQ_FORMAT, bytes(self._shm.buf[0:8]))[0]
            if seq_after != seq_before:
                continue  # torn read; retry
            self._last_seq = seq_after
            return seq_after, payload

        return None

    def start_pump(
        self,
        stream_buffer,  # noqa: ANN001 — duck-typed (StreamBuffer or any obj with publish)
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ) -> None:
        """Spawn a daemon thread that forwards new frames into ``stream_buffer``.

        Idempotent — calling twice is a no-op. The thread runs until
        :meth:`close` is called or the process exits. Exceptions
        inside the loop are caught + logged so a transient publisher
        glitch doesn't kill the dashboard.
        """
        if self._pump_thread is not None and self._pump_thread.is_alive():
            return

        def _pump() -> None:
            logger.info(
                "SharedFrameSubscriber pump started for %r (poll=%.3fs).",
                self._name,
                poll_interval,
            )
            while not self._stop_event.wait(poll_interval):
                try:
                    result = self.read()
                except Exception:
                    logger.exception("SharedFrameSubscriber pump read failed")
                    continue
                if result is None:
                    continue
                _, payload = result
                try:
                    stream_buffer.publish(payload)
                except Exception:
                    logger.exception("SharedFrameSubscriber pump forward failed")
            logger.info("SharedFrameSubscriber pump stopped for %r.", self._name)

        self._stop_event.clear()
        self._pump_thread = threading.Thread(
            target=_pump,
            name=f"shm-pump-{self._name}",
            daemon=True,
        )
        self._pump_thread.start()

    def close(self) -> None:
        """Stop the pump thread (if running) and detach. Idempotent."""
        self._stop_event.set()
        if self._pump_thread is not None:
            self._pump_thread.join(timeout=1.0)
            self._pump_thread = None
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                logger.exception("SharedFrameSubscriber close failed (ignored)")
            self._shm = None


__all__ = [
    "MAX_FRAME_BYTES",
    "FrameTooLarge",
    "SegmentNotFound",
    "SharedFramePublisher",
    "SharedFrameSubscriber",
]
