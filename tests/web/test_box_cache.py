"""Unit tests for src.web.box_cache."""

from __future__ import annotations

import threading
import time

import pytest

from src.web.box_cache import BoxCache

# ── Construction ─────────────────────────────────────────────────────────────


class TestConstruction:
    def test_defaults(self):
        cache = BoxCache()
        # Defaults aren't part of the public surface but a fresh
        # cache should start empty.
        assert cache.peek_fresh() is None

    def test_custom_ttl_and_fade(self):
        cache = BoxCache(ttl_seconds=2.0, fade_seconds=0.5)
        assert cache.peek_fresh() is None

    def test_zero_ttl_rejected(self):
        with pytest.raises(ValueError, match="ttl"):
            BoxCache(ttl_seconds=0)

    def test_negative_ttl_rejected(self):
        with pytest.raises(ValueError, match="ttl"):
            BoxCache(ttl_seconds=-1)

    def test_negative_fade_rejected(self):
        with pytest.raises(ValueError, match="fade"):
            BoxCache(fade_seconds=-0.1)

    def test_fade_must_be_less_than_ttl(self):
        with pytest.raises(ValueError, match="fade"):
            BoxCache(ttl_seconds=1.0, fade_seconds=1.0)


# ── Update + peek ────────────────────────────────────────────────────────────


class TestUpdateAndPeek:
    def test_update_then_peek_returns_solid(self):
        cache = BoxCache(ttl_seconds=3.0, fade_seconds=1.0)
        cache.update((10, 20, 30, 40), "HOFI", 0.95)
        result = cache.peek_fresh()
        assert result is not None
        box, species, conf, alpha = result
        assert box == (10, 20, 30, 40)
        assert species == "HOFI"
        assert conf == 0.95
        assert alpha == pytest.approx(1.0)

    def test_peek_none_when_empty(self):
        cache = BoxCache()
        assert cache.peek_fresh() is None

    def test_clear_drops_entry(self):
        cache = BoxCache()
        cache.update((1, 2, 3, 4), "AMRO", 0.8)
        assert cache.peek_fresh() is not None
        cache.clear()
        assert cache.peek_fresh() is None

    def test_update_overwrites(self):
        cache = BoxCache()
        cache.update((1, 2, 3, 4), "HOFI", 0.5)
        cache.update((10, 20, 30, 40), "AMRO", 0.9)
        result = cache.peek_fresh()
        assert result is not None
        box, species, conf, _ = result
        assert box == (10, 20, 30, 40)
        assert species == "AMRO"
        assert conf == 0.9


# ── TTL + fade ───────────────────────────────────────────────────────────────


class TestTtlAndFade:
    """Drive time deterministically via the ``now=`` hook so the
    tests don't have to sleep. The cache stores ``time.monotonic()``
    on update; passing ``now=`` to ``peek_fresh`` controls the read
    side. Reading the stored update time isn't part of the public
    surface, so we capture it via a snapshot of monotonic time
    around the update call."""

    def test_solid_alpha_inside_solid_window(self):
        cache = BoxCache(ttl_seconds=3.0, fade_seconds=1.0)
        cache.update((1, 2, 3, 4), "HOFI", 0.9)
        t0 = time.monotonic()
        # 0.5 s after update is well inside the 2 s solid window.
        result = cache.peek_fresh(now=t0 + 0.5)
        assert result is not None
        assert result[3] == pytest.approx(1.0)

    def test_alpha_fades_linearly_in_fade_window(self):
        cache = BoxCache(ttl_seconds=3.0, fade_seconds=1.0)
        cache.update((1, 2, 3, 4), "HOFI", 0.9)
        t0 = time.monotonic()
        # Halfway through the fade (age = 2.5 s of a 3 s ttl with
        # 1 s fade) → alpha ~= 0.5.
        result = cache.peek_fresh(now=t0 + 2.5)
        assert result is not None
        assert result[3] == pytest.approx(0.5, abs=0.05)

    def test_alpha_near_zero_at_ttl_boundary(self):
        cache = BoxCache(ttl_seconds=3.0, fade_seconds=1.0)
        cache.update((1, 2, 3, 4), "HOFI", 0.9)
        t0 = time.monotonic()
        # Just inside the ttl boundary — alpha should be near 0.
        # (We can't pin age exactly at ttl because update() captured
        # its own monotonic time and we sample t0 after that, so
        # ``t0 + 3.0`` is technically just past ttl.)
        result = cache.peek_fresh(now=t0 + 2.99)
        assert result is not None
        assert result[3] == pytest.approx(0.0, abs=0.02)

    def test_none_past_ttl(self):
        cache = BoxCache(ttl_seconds=3.0, fade_seconds=1.0)
        cache.update((1, 2, 3, 4), "HOFI", 0.9)
        t0 = time.monotonic()
        assert cache.peek_fresh(now=t0 + 5.0) is None

    def test_zero_fade_means_solid_then_drop(self):
        """fade_seconds=0 → no fade window: alpha is 1.0 for the
        entire ttl, then None. (Edge of the validation: 0 is allowed
        because it's a perfectly valid 'no fade' policy.)"""
        cache = BoxCache(ttl_seconds=2.0, fade_seconds=0.0)
        cache.update((1, 2, 3, 4), "HOFI", 0.9)
        t0 = time.monotonic()
        assert cache.peek_fresh(now=t0 + 1.0)[3] == pytest.approx(1.0)
        assert cache.peek_fresh(now=t0 + 2.5) is None


# ── Concurrency ──────────────────────────────────────────────────────────────


class TestConcurrency:
    def test_concurrent_update_and_peek(self):
        """Hammer the cache from multiple threads. peek_fresh should
        always return either None or a self-consistent tuple — never
        a torn read."""
        cache = BoxCache(ttl_seconds=10.0, fade_seconds=1.0)
        stop = threading.Event()
        bad_reads: list[object] = []

        def writer():
            i = 0
            while not stop.is_set():
                cache.update((i, i, i + 10, i + 10), "HOFI", 0.5 + (i % 50) / 100)
                i += 1

        def reader():
            while not stop.is_set():
                r = cache.peek_fresh()
                if r is None:
                    continue
                box, species, conf, alpha = r
                if not (
                    isinstance(box, tuple)
                    and len(box) == 4
                    and species == "HOFI"
                    and 0.0 <= alpha <= 1.0
                    and 0.0 <= conf <= 1.0
                ):
                    bad_reads.append(r)

        threads = [
            threading.Thread(target=writer, daemon=True),
            threading.Thread(target=reader, daemon=True),
            threading.Thread(target=reader, daemon=True),
        ]
        for t in threads:
            t.start()
        time.sleep(0.2)
        stop.set()
        for t in threads:
            t.join(timeout=2.0)
        assert bad_reads == []
