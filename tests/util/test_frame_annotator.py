"""Unit tests for src.util.frame_annotator."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from src.util.frame_annotator import annotate_jpeg


def _solid_jpeg(size=(160, 90), color=(20, 20, 20), quality=85) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _decode(jpeg: bytes) -> Image.Image:
    return Image.open(BytesIO(jpeg)).convert("RGB")


# ── Round-trip ───────────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_returns_valid_jpeg(self):
        src = _solid_jpeg()
        out = annotate_jpeg(src, box=(10, 10, 50, 50), label="HOFI 0.95", alpha=1.0)
        # Decodable, same dimensions.
        img = _decode(out)
        assert img.size == (160, 90)
        assert img.format is None  # _decode strips the source format reference
        # Output is valid JPEG (header magic).
        assert out[:3] == b"\xff\xd8\xff"

    def test_output_differs_from_input_when_drawing(self):
        src = _solid_jpeg()
        out = annotate_jpeg(src, box=(20, 20, 80, 60), label="X", alpha=1.0)
        assert out != src


# ── Alpha fast-path ──────────────────────────────────────────────────────────


class TestAlpha:
    def test_alpha_zero_returns_input_unchanged(self):
        """alpha=0 must skip the decode/encode entirely so the
        publish path doesn't burn CPU on a no-op overlay."""
        src = _solid_jpeg()
        out = annotate_jpeg(src, box=(10, 10, 50, 50), label="X", alpha=0.0)
        assert out is src or out == src

    def test_alpha_negative_returns_input_unchanged(self):
        src = _solid_jpeg()
        out = annotate_jpeg(src, box=(10, 10, 50, 50), label="X", alpha=-0.5)
        assert out == src

    def test_alpha_above_one_clamps(self):
        """Caller sloppiness shouldn't crash — clamp to 1.0."""
        src = _solid_jpeg()
        out = annotate_jpeg(src, box=(10, 10, 50, 50), label="", alpha=2.0)
        # No exception. Output differs from source (something drew).
        assert out != src

    def test_partial_alpha_modifies_pixels(self):
        src = _solid_jpeg()
        full = annotate_jpeg(src, box=(20, 20, 80, 60), label="", alpha=1.0)
        half = annotate_jpeg(src, box=(20, 20, 80, 60), label="", alpha=0.5)
        # Both differ from source and from each other.
        assert full != src
        assert half != src
        assert full != half


# ── Visual sanity (pixel checks) ─────────────────────────────────────────────


class TestPixelEffect:
    def test_box_pixels_become_greenish(self):
        """A bright-green outline should be visibly green at the box
        edge. JPEG is lossy so we tolerate compression noise — green
        channel just has to dominate red and blue."""
        src = _solid_jpeg(color=(20, 20, 20))
        out = annotate_jpeg(src, box=(40, 40, 100, 70), label="", alpha=1.0)
        img = _decode(out)
        # Sample on the top edge of the rectangle.
        r, g, b = img.getpixel((70, 41))
        assert g > r + 30, f"green not dominant at top edge: rgb=({r},{g},{b})"
        assert g > b + 30, f"green not dominant at top edge: rgb=({r},{g},{b})"

    def test_pixels_inside_box_unchanged(self):
        """Outline-only — interior pixels should still be (close to)
        the source colour."""
        src_color = (40, 40, 40)
        src = _solid_jpeg(color=src_color)
        out = annotate_jpeg(src, box=(40, 30, 120, 80), label="", alpha=1.0)
        img = _decode(out)
        # Centre of the box, well away from the outline.
        r, g, b = img.getpixel((80, 55))
        for ch_actual, ch_expected in zip((r, g, b), src_color, strict=True):
            # JPEG quantization can wander a few units; allow slack.
            assert abs(ch_actual - ch_expected) < 25


# ── Robustness ───────────────────────────────────────────────────────────────


class TestRobustness:
    def test_invalid_jpeg_returns_input(self):
        """Annotation is best-effort; if PIL can't decode, the
        unannotated bytes still go through."""
        garbage = b"definitely-not-a-jpeg"
        out = annotate_jpeg(garbage, box=(0, 0, 10, 10), label="", alpha=1.0)
        assert out == garbage

    def test_empty_label_omits_text(self):
        src = _solid_jpeg()
        # No raise. Output is a valid JPEG with the rectangle drawn.
        out = annotate_jpeg(src, box=(20, 20, 80, 60), label="", alpha=1.0)
        assert out[:3] == b"\xff\xd8\xff"
