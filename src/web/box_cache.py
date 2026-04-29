"""Last-known YOLO bounding box with TTL + fade.

The agent calls :meth:`BoxCache.update` after the gate detector
returns a box and the classifier produces a species; the live
preview reads from :meth:`peek_fresh` to overlay the box on
streamed frames.

Lifecycle:
    age <= ttl_seconds - fade_seconds   →  alpha = 1.0
    ttl - fade < age <= ttl             →  alpha decays linearly to 0
    age > ttl                           →  peek_fresh returns None

Defaults match the investigation doc: 3 s total TTL with the last
1 s fading. The agent updates ~5 fps when YOLO is active, so the
cache stays warm during a detection burst and decays cleanly when
the bird leaves frame.

Thread-safety: a single lock around the read and write paths is
plenty — one writer (agent loop) and a small number of readers
(capture-loop annotator).
"""

from __future__ import annotations

import threading
import time

# (x1, y1, x2, y2) in the camera's native pixel coordinates — same
# shape ``BirdDetection`` and ``CaptureResult.detection_box`` use.
Box = tuple[int, int, int, int]

# (box, species, confidence, alpha) — alpha already collapses
# elapsed-time into the [0, 1] range so the annotator stays simple.
FreshBox = tuple[Box, str, float, float]


_DEFAULT_TTL_SECONDS = 3.0
_DEFAULT_FADE_SECONDS = 1.0


class BoxCache:
    """Single-slot box cache with monotonic-time TTL and linear fade.

    Args:
        ttl_seconds: total lifetime of a cached box. Past this age,
            ``peek_fresh`` returns ``None``.
        fade_seconds: trailing window inside the TTL during which
            alpha decays from 1.0 → 0.0 linearly. Must be < ttl.
    """

    def __init__(
        self,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        fade_seconds: float = _DEFAULT_FADE_SECONDS,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if fade_seconds < 0:
            raise ValueError("fade_seconds must be non-negative")
        if fade_seconds >= ttl_seconds:
            raise ValueError("fade_seconds must be less than ttl_seconds")
        self._ttl = ttl_seconds
        self._fade = fade_seconds
        self._lock = threading.Lock()
        self._box: Box | None = None
        self._species: str = ""
        self._confidence: float = 0.0
        self._updated_at: float = 0.0

    def update(self, box: Box, species: str, confidence: float) -> None:
        """Store the latest box. Overwrites any prior entry."""
        now = time.monotonic()
        with self._lock:
            self._box = box
            self._species = species
            self._confidence = confidence
            self._updated_at = now

    def clear(self) -> None:
        with self._lock:
            self._box = None
            self._species = ""
            self._confidence = 0.0
            self._updated_at = 0.0

    def peek_fresh(self, *, now: float | None = None) -> FreshBox | None:
        """Return the cached box if still within the TTL window.

        Args:
            now: monotonic time to evaluate the age against. ``None``
                uses ``time.monotonic()`` — the test hook lets
                deterministic tests advance time without sleeping.

        Returns:
            ``(box, species, confidence, alpha)`` or ``None`` when no
            entry is set or it has aged out.
        """
        t = time.monotonic() if now is None else now
        with self._lock:
            if self._box is None:
                return None
            age = t - self._updated_at
            if age > self._ttl:
                return None
            alpha = self._alpha_for_age(age)
            return self._box, self._species, self._confidence, alpha

    def _alpha_for_age(self, age: float) -> float:
        # Solid for the first (ttl - fade) seconds; linear decay over
        # the trailing fade window. Clamped to avoid float wobble at
        # the boundaries.
        solid_window = self._ttl - self._fade
        if age <= solid_window or self._fade == 0:
            return 1.0
        decay = (age - solid_window) / self._fade
        return max(0.0, min(1.0, 1.0 - decay))


__all__ = ["Box", "BoxCache", "FreshBox"]
