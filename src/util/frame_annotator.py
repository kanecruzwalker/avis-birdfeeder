"""Draw a YOLO bounding box + label onto a JPEG frame.

Pure PIL helper used by ``VisionCapture`` (publish-time
annotation) and any future caller that needs to overlay a box on
a JPEG. Lives in ``src.util`` rather than ``src.web`` so
``src.vision`` can import it without violating the
``src.web → src.vision`` dependency rule.

All coordinates are in the JPEG's pixel space — callers scale
camera-native boxes down to preview size before invoking. The
function decodes the JPEG, alpha-composites a green overlay,
and re-encodes (~5–8 ms on a Pi 5 for a 640×360 q=75 JPEG).
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

# (x1, y1, x2, y2) in target-image pixel coordinates.
Box = tuple[int, int, int, int]

_DEFAULT_BOX_COLOR = (0, 255, 0)  # bright green; visible against most foliage
_DEFAULT_BOX_WIDTH = 4
_LABEL_PADDING_Y = 14  # pixels above the box


def annotate_jpeg(
    jpeg_bytes: bytes,
    *,
    box: Box,
    label: str,
    alpha: float,
    color: tuple[int, int, int] = _DEFAULT_BOX_COLOR,
    width: int = _DEFAULT_BOX_WIDTH,
    quality: int = 75,
) -> bytes:
    """Return a new JPEG with ``box`` + ``label`` overlaid at ``alpha``.

    Args:
        jpeg_bytes: source JPEG.
        box: ``(x1, y1, x2, y2)`` in source-image pixel coordinates.
        label: text drawn just above the box. Empty string skips text.
        alpha: opacity in ``[0, 1]``. ``0`` is a fast no-op (returns
            ``jpeg_bytes`` unchanged); the annotator pays the
            decode/encode cost only when there's something to draw.
        color: RGB triple for the rectangle outline + label.
        width: rectangle outline thickness in pixels.
        quality: JPEG re-encode quality.

    Returns:
        Annotated JPEG bytes. Falls back to the input bytes if PIL
        raises during overlay — annotation is best-effort, the
        unannotated frame is still useful.
    """
    if alpha <= 0.0:
        return jpeg_bytes
    alpha = min(1.0, alpha)

    try:
        with Image.open(io.BytesIO(jpeg_bytes)) as src:
            base = src.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        rgba = (*color, int(round(255 * alpha)))
        draw.rectangle(box, outline=rgba, width=width)
        if label:
            x1, y1, _, _ = box
            text_y = max(y1 - _LABEL_PADDING_Y, 0)
            draw.text((x1, text_y), label, fill=rgba)
        composed = Image.alpha_composite(base, overlay).convert("RGB")
        out = io.BytesIO()
        composed.save(out, format="JPEG", quality=quality)
        return out.getvalue()
    except Exception:  # noqa: BLE001 — annotation must not crash capture
        return jpeg_bytes


__all__ = ["Box", "annotate_jpeg"]
