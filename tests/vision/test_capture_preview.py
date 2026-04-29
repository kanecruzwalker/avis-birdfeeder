"""Tests for VisionCapture's preview-publish path.

Exercises ``_maybe_publish_preview`` directly with synthetic raw
frames and a fake stream_buffer / box_source. Avoids picamera2 —
we never call ``capture_frames``.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.vision.capture import VisionCapture

# ── Test doubles ─────────────────────────────────────────────────────────────


class FakeStreamBuffer:
    def __init__(self) -> None:
        self.published: list[bytes] = []

    def publish(self, jpeg_bytes: bytes) -> None:
        self.published.append(jpeg_bytes)


class FakeBoxSource:
    def __init__(self, *, fresh=None) -> None:
        self._fresh = fresh

    def peek_fresh(self):
        return self._fresh


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_capture(tmp_path, *, stream_buffer=None, box_source=None):
    """Construct a VisionCapture with the smallest-possible config.

    The test never calls ``capture_frames`` (which needs picamera2);
    it drives ``_maybe_publish_preview`` directly. Crop/motion
    parameters are placeholders — they don't matter on this path.
    """
    return VisionCapture(
        primary_index=0,
        secondary_index=1,
        capture_width=320,
        capture_height=240,
        capture_fps=30,
        classification_width=224,
        classification_height=224,
        crop_x=0,
        crop_y=0,
        crop_width=320,
        crop_height=240,
        motion_threshold=0.005,
        background_history=5,
        output_dir=tmp_path / "out",
        stream_buffer=stream_buffer,
        stream_preview_size=(160, 120),
        stream_preview_quality=75,
        box_source=box_source,
    )


def _solid_raw_frame(h=240, w=320, color=(40, 40, 40)) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = color
    return arr


def _decode(jpeg: bytes) -> Image.Image:
    return Image.open(BytesIO(jpeg)).convert("RGB")


# ── No buffer ────────────────────────────────────────────────────────────────


class TestNoBuffer:
    def test_no_buffer_is_silent(self, tmp_path):
        """Without a sink, the call is a no-op — no exception, no
        encode work. Mirrors the no-dashboard rollback path."""
        capture = _make_capture(tmp_path, stream_buffer=None)
        capture._maybe_publish_preview(_solid_raw_frame())
        # No assertion needed beyond "did not raise".


# ── Buffer, no box source ────────────────────────────────────────────────────


class TestBufferOnly:
    def test_publishes_unannotated_jpeg(self, tmp_path):
        sink = FakeStreamBuffer()
        capture = _make_capture(tmp_path, stream_buffer=sink, box_source=None)
        capture._maybe_publish_preview(_solid_raw_frame())
        assert len(sink.published) == 1
        # Output is a valid JPEG.
        jpeg = sink.published[0]
        assert jpeg[:3] == b"\xff\xd8\xff"
        # Output is the configured preview size.
        img = _decode(jpeg)
        assert img.size == (160, 120)


# ── Buffer + fresh box ───────────────────────────────────────────────────────


class TestBoxAnnotation:
    def test_publishes_annotated_when_box_fresh(self, tmp_path):
        """The box source returns a fresh entry; the published JPEG
        should differ from the unannotated baseline (something was
        drawn) and the box should be drawn at the SCALED coordinates
        in the preview's pixel space."""
        sink_baseline = FakeStreamBuffer()
        baseline_capture = _make_capture(tmp_path, stream_buffer=sink_baseline)
        baseline_capture._maybe_publish_preview(_solid_raw_frame())
        baseline = sink_baseline.published[0]

        # Camera-native box (320x240); preview is 160x120, so the
        # scaled box is at (40, 30) → (120, 90).
        box_source = FakeBoxSource(
            fresh=((80, 60, 240, 180), "HOFI", 0.95, 1.0),
        )
        sink_annotated = FakeStreamBuffer()
        capture = _make_capture(
            tmp_path,
            stream_buffer=sink_annotated,
            box_source=box_source,
        )
        capture._maybe_publish_preview(_solid_raw_frame())

        annotated = sink_annotated.published[0]
        assert annotated != baseline, "annotated frame should differ from baseline"

        # Sample the top edge of the scaled box (around y=30) — should
        # be greenish from the rectangle outline.
        img = _decode(annotated)
        r, g, b = img.getpixel((80, 31))
        assert g > r + 30, f"green not dominant at scaled box top edge: rgb=({r},{g},{b})"
        assert g > b + 30, f"green not dominant at scaled box top edge: rgb=({r},{g},{b})"

    def test_no_box_when_source_returns_none(self, tmp_path):
        """An empty cache leaves the JPEG bit-identical to the
        no-box-source case (fast path)."""
        # Baseline: no box source.
        sink_baseline = FakeStreamBuffer()
        capture_baseline = _make_capture(tmp_path, stream_buffer=sink_baseline)
        capture_baseline._maybe_publish_preview(_solid_raw_frame())

        # Box source returns None (nothing fresh).
        sink_with_source = FakeStreamBuffer()
        capture = _make_capture(
            tmp_path,
            stream_buffer=sink_with_source,
            box_source=FakeBoxSource(fresh=None),
        )
        capture._maybe_publish_preview(_solid_raw_frame())

        assert sink_with_source.published[0] == sink_baseline.published[0]

    def test_alpha_zero_skips_annotation(self, tmp_path):
        """An aged-out box at alpha=0 returns the unannotated bytes
        from the annotator, so the publish must match the baseline."""
        sink_baseline = FakeStreamBuffer()
        capture_baseline = _make_capture(tmp_path, stream_buffer=sink_baseline)
        capture_baseline._maybe_publish_preview(_solid_raw_frame())

        sink = FakeStreamBuffer()
        capture = _make_capture(
            tmp_path,
            stream_buffer=sink,
            box_source=FakeBoxSource(
                fresh=((80, 60, 240, 180), "HOFI", 0.95, 0.0),
            ),
        )
        capture._maybe_publish_preview(_solid_raw_frame())
        assert sink.published[0] == sink_baseline.published[0]


# ── Failure isolation ────────────────────────────────────────────────────────


class TestFailureIsolation:
    def test_box_source_exception_doesnt_break_capture(self, tmp_path):
        """A misbehaving box source must not crash capture — the
        whole publish is best-effort."""

        class BrokenSource:
            def peek_fresh(self):
                raise RuntimeError("simulated cache failure")

        sink = FakeStreamBuffer()
        capture = _make_capture(
            tmp_path,
            stream_buffer=sink,
            box_source=BrokenSource(),
        )
        # Should not raise. The outer try/except in
        # _maybe_publish_preview swallows the failure; nothing is
        # published since we bail early on the exception.
        capture._maybe_publish_preview(_solid_raw_frame())


@pytest.fixture
def _no_basetemp_warning():
    """No-op: kept for parity with other test modules' fixtures."""
    yield
