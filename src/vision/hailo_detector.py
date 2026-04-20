"""
src/vision/hailo_detector.py

YOLOv8s bird detector running on the Hailo HAILO8L NPU.

Replaces the fixed feeder_crop zone with per-bird bounding box detection.
The detector finds birds in the full camera frame and returns the tightest
crop around the highest-confidence detection, which is then passed to
VisualClassifier for species identification.

Architecture:
    Full frame (1536×864) → resize to 640×640 → YOLOv8s HEF inference
    → NMS postprocessed detections → filter class 14 (bird, COCO)
    → highest-confidence box → scale back to original resolution
    → return crop coordinates for VisualClassifier

Why YOLO instead of fixed crop?
    Fixed crop assumes the bird is always in a specific pixel region.
    YOLO finds wherever the bird actually is, handles multiple birds,
    adapts to camera mounting variations, and enables per-bird crops
    that represent exactly what the classifier should see.
    This is the detect-then-classify paradigm vs classify-then-hope.

Model: yolov8s_h8l.hef (pre-installed at /usr/share/hailo-models/)
    - Input:  640×640×3 uint8
    - Output: flat uint8 buffer, NMS postprocessed, built into HEF
    - Classes: 80 COCO classes, class 14 = bird
    - Buffer size: NUM_CLASSES × MAX_PROPOSALS × 20 bytes
      Each detection: 5 × float32 = [x1, y1, x2, y2, score] (20 bytes)

Output format (per detection, 20 bytes = 5 × float32):
    [x1, y1, x2, y2, score]
    Coordinates are normalized [0, 1] relative to 640×640 input.

COCO class 14 = bird. This covers all bird species detectable by a
YOLOv8 model trained on COCO — not species-specific, just bird vs not-bird.
Species identification is still handled by VisualClassifier.

Usage:
    detector = HailoDetector.from_config("configs/hardware.yaml")
    detector.open()
    crop = detector.detect(full_frame_uint8)   # (H, W, 3) uint8
    if crop is not None:
        result = classifier.predict(preprocess_frame(crop))
    detector.close()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# COCO dataset class index for "bird"
COCO_BIRD_CLASS_ID = 14

# YOLOv8s HEF input resolution (fixed by compilation)
YOLO_INPUT_SIZE = 640

# Number of COCO classes in the YOLOv8s model
YOLO_NUM_CLASSES = 80

# Bytes per detection in the NMS output buffer
# Each detection = 5 × float32 = [x1, y1, x2, y2, score]
BYTES_PER_DETECTION = 20

try:
    from hailo_platform import VDevice  # type: ignore[import]

    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    VDevice = None  # type: ignore[assignment,misc]


def _resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Resize frame to size×size. Uses cv2 if available, falls back to PIL."""
    try:
        import cv2  # noqa: PLC0415

        return cv2.resize(frame, (size, size))
    except ImportError:
        from PIL import Image  # noqa: PLC0415

        return np.array(Image.fromarray(frame).resize((size, size)))


@dataclass
class Detection:
    """
    A single object detection from YOLOv8s.

    Coordinates are in original frame pixel space (before any resizing).
    """

    x1: int  # left edge (pixels)
    y1: int  # top edge (pixels)
    x2: int  # right edge (pixels)
    y2: int  # bottom edge (pixels)
    confidence: float  # detection confidence [0, 1]
    class_id: int  # COCO class index (14 = bird)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_crop(self, frame: np.ndarray, padding: int = 20) -> np.ndarray:
        """
        Extract this detection as a cropped sub-image from the frame.

        Adds optional padding around the bounding box, clamped to frame bounds.
        The crop is what gets passed to VisualClassifier for species ID.

        Args:
            frame:   Full camera frame (H, W, 3) uint8.
            padding: Pixels to add around the bounding box on each side.

        Returns:
            Cropped region as uint8 array.
        """
        h, w = frame.shape[:2]
        x1 = max(0, self.x1 - padding)
        y1 = max(0, self.y1 - padding)
        x2 = min(w, self.x2 + padding)
        y2 = min(h, self.y2 + padding)
        return frame[y1:y2, x1:x2]


class HailoDetector:
    """
    YOLOv8s bird detector running on the Hailo HAILO8L NPU.

    Accepts a full camera frame, runs YOLO detection, and returns the
    crop around the highest-confidence bird detection. Falls back
    gracefully when hailo_platform is unavailable (laptop/CI).

    The detector is stateful — call open() before detect() and close()
    when done. Supports use as a context manager.

    Usage:
        with HailoDetector(hef_path) as detector:
            crop = detector.detect(frame)
    """

    # Default NMS thresholds — tunable via hardware.yaml
    DEFAULT_SCORE_THRESHOLD = 0.25
    DEFAULT_MAX_PROPOSALS = 10
    DEFAULT_IOU_THRESHOLD = 0.45

    def __init__(
        self,
        hef_path: str,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        max_proposals_per_class: int = DEFAULT_MAX_PROPOSALS,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        min_bird_confidence: float = 0.25,
        min_crop_size: int = 40,
        shared_vdevice: object | None = None,
    ) -> None:
        """
        Args:
            hef_path:               Path to yolov8s_h8l.hef.
            score_threshold:        NMS score threshold. Detections below
                                    this are discarded by the chip.
            max_proposals_per_class: Maximum detections per class per frame.
                                    Controls output buffer size.
            iou_threshold:          NMS IoU overlap threshold.
            min_bird_confidence:    Minimum confidence to accept a bird detection.
                                    Applied after chip NMS as a secondary filter.
            min_crop_size:          Minimum width or height of a valid crop in pixels.
                                    Crops smaller than this are discarded.
        """
        self.hef_path = Path(hef_path)
        self.score_threshold = score_threshold
        self.max_proposals_per_class = max_proposals_per_class
        self.iou_threshold = iou_threshold
        self.min_bird_confidence = min_bird_confidence
        self.min_crop_size = min_crop_size

        self._vdevice = None
        self._configured = None
        self._out_buf_size: int = 0
        self.is_open: bool = False

        self._shared_vdevice = shared_vdevice
        self._owns_vdevice = shared_vdevice is None

        logger.info(
            "HailoDetector | hef=%s score_threshold=%.2f max_proposals=%d",
            self.hef_path.name,
            self.score_threshold,
            self.max_proposals_per_class,
        )

    @classmethod
    def from_config(cls, hardware_config_path: str) -> HailoDetector:
        """
        Construct a HailoDetector from configs/hardware.yaml.

        Reads:
            hailo.models.yolo_hef         → hef_path
            hailo.yolo.score_threshold    → score_threshold (optional)
            hailo.yolo.max_proposals      → max_proposals_per_class (optional)
            hailo.yolo.min_bird_confidence → min_bird_confidence (optional)

        Falls back to defaults if yolo sub-section is absent.

        Args:
            hardware_config_path: Path to configs/hardware.yaml.

        Returns:
            Configured HailoDetector instance (not yet opened).
        """
        path = Path(hardware_config_path)
        with path.open() as f:
            hw = yaml.safe_load(f)

        hailo_cfg = hw.get("hailo", {})
        hef_path = hailo_cfg.get("models", {}).get(
            "yolo_hef", "/usr/share/hailo-models/yolov8s_h8l.hef"
        )
        yolo_cfg = hailo_cfg.get("yolo", {})

        return cls(
            hef_path=hef_path,
            score_threshold=yolo_cfg.get("score_threshold", cls.DEFAULT_SCORE_THRESHOLD),
            max_proposals_per_class=yolo_cfg.get("max_proposals", cls.DEFAULT_MAX_PROPOSALS),
            min_bird_confidence=yolo_cfg.get("min_bird_confidence", cls.DEFAULT_SCORE_THRESHOLD),
        )

    def open(self) -> None:
        """
        Initialize the Hailo VDevice and configure the YOLO model.

        Must be called before detect(). Safe to call multiple times —
        subsequent calls are no-ops if already open.

        Raises:
            RuntimeError: If hailo_platform is not available or HEF is missing.
        """
        if self.is_open:
            return
        if not HAILO_AVAILABLE:
            raise RuntimeError("hailo_platform not available — HailoDetector requires Pi hardware.")
        if not self.hef_path.exists():
            raise RuntimeError(
                f"YOLO HEF not found at {self.hef_path}. "
                "Expected at /usr/share/hailo-models/yolov8s_h8l.hef on Pi."
            )

        if self._shared_vdevice is not None:
            self._vdevice = self._shared_vdevice
        else:
            self._vdevice = VDevice()

        model = self._vdevice.create_infer_model(str(self.hef_path))
        model.set_batch_size(1)

        # Configure NMS thresholds before compilation
        out_stream = model.output("yolov8s/yolov8_nms_postprocess")
        out_stream.set_nms_score_threshold(self.score_threshold)
        out_stream.set_nms_max_proposals_per_class(self.max_proposals_per_class)
        out_stream.set_nms_iou_threshold(self.iou_threshold)

        # Each class slot = 4-byte count header + max_proposals × 20 bytes per detection
        self._out_buf_size = YOLO_NUM_CLASSES * (
            4 + self.max_proposals_per_class * BYTES_PER_DETECTION
        )
        # NOTE: Do NOT call .activate() on the configured network group.
        # The shared VDevice uses HailoSchedulingAlgorithm.ROUND_ROBIN
        # (see src.vision.capture._create_shared_vdevice), which means the
        # chip's core-op scheduler handles activation automatically at
        # inference time. Calling .activate() manually raises
        # HAILO_INVALID_OPERATION(6) "Manually activate a core-op is not
        # allowed when the core-op scheduler is active!".
        # This matches the pattern in src/vision/hailo_extractor.py.
        self._configured = model.configure()
        self.is_open = True

        logger.info(
            "HailoDetector opened — YOLO ready. out_buf_size=%d bytes",
            self._out_buf_size,
        )

    def close(self) -> None:
        """
        Release Hailo resources. Safe to call multiple times.

        The scheduler handles deactivation automatically when the configured
        network group is garbage-collected, so we do not call .deactivate().
        Calling it manually under a ROUND_ROBIN scheduler raises
        HAILO_INVALID_OPERATION(6).
        """
        if not self.is_open:
            return
        if self._configured is not None:
            self._configured = None

        if self._owns_vdevice:
            self._vdevice = None

        self.is_open = False
        logger.info("HailoDetector closed.")

    def __enter__(self) -> HailoDetector:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def detect(self, frame: np.ndarray) -> Detection | None:
        """
        Run YOLO detection on a full camera frame and return the
        highest-confidence bird detection.

        Args:
            frame: Full camera frame, uint8, any resolution.
                   Will be resized to 640×640 for inference.
                   Must be HWC format, RGB or BGR (both work for detection).

        Returns:
            Detection with bounding box in original frame coordinates,
            or None if no bird detected above min_bird_confidence.

        Raises:
            RuntimeError: If open() has not been called.
        """
        if not self.is_open:
            raise RuntimeError("HailoDetector.open() must be called before detect().")

        orig_h, orig_w = frame.shape[:2]

        # ── Preprocess: resize to 640×640 uint8 ──────────────────────────────
        frame_640 = _resize_frame(frame, YOLO_INPUT_SIZE)
        if frame_640.dtype != np.uint8:
            frame_640 = (frame_640 * 255).clip(0, 255).astype(np.uint8)

        # ── Inference ─────────────────────────────────────────────────────────
        bindings = self._configured.create_bindings()
        bindings.input().set_buffer(frame_640)
        out_buf = np.zeros(self._out_buf_size, dtype=np.uint8)
        bindings.output().set_buffer(out_buf)
        self._configured.wait_for_async_ready(timeout_ms=1000)
        job = self._configured.run_async([bindings])
        job.wait(1000)

        # ── Decode NMS output buffer ──────────────────────────────────────────
        detections = self._decode_nms_output(out_buf, orig_w, orig_h)

        if not detections:
            logger.debug("HailoDetector: no birds detected.")
            return None

        # Return highest-confidence detection
        best = max(detections, key=lambda d: d.confidence)
        logger.info(
            "HailoDetector: bird detected conf=%.3f box=(%d,%d,%d,%d)",
            best.confidence,
            best.x1,
            best.y1,
            best.x2,
            best.y2,
        )
        return best

    def detect_all(self, frame: np.ndarray) -> list[Detection]:
        """
        Run YOLO detection and return all bird detections above threshold.

        Useful for multi-bird scenes or when you want to process all detections
        rather than just the highest-confidence one.

        Args:
            frame: Full camera frame, uint8, any resolution.

        Returns:
            List of Detection objects sorted by confidence (highest first).
            Empty list if no birds detected.
        """
        if not self.is_open:
            raise RuntimeError("HailoDetector.open() must be called before detect_all().")

        orig_h, orig_w = frame.shape[:2]
        frame_640 = _resize_frame(frame, YOLO_INPUT_SIZE)
        if frame_640.dtype != np.uint8:
            frame_640 = (frame_640 * 255).clip(0, 255).astype(np.uint8)

        bindings = self._configured.create_bindings()
        bindings.input().set_buffer(frame_640)
        out_buf = np.zeros(self._out_buf_size, dtype=np.uint8)
        bindings.output().set_buffer(out_buf)
        self._configured.wait_for_async_ready(timeout_ms=1000)
        job = self._configured.run_async([bindings])
        job.wait(1000)

        detections = self._decode_nms_output(out_buf, orig_w, orig_h)
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def _decode_nms_output(
        self,
        out_buf: np.ndarray,
        orig_w: int,
        orig_h: int,
    ) -> list[Detection]:
        """
        Decode the flat NMS output buffer into Detection objects.

        Hailo YOLOv8 NMS output format (uint8 buffer):
            Layout: [class_0_detections | class_1_detections | ... | class_79_detections]
            Per class: [count (4 bytes) | det_0 (20 bytes) | det_1 (20 bytes) | ...]
            Per detection (20 bytes = 5 × float32): [x1, y1, x2, y2, score]
            Coordinates: normalized [0, 1] relative to 640×640 input

        We only decode class 14 (bird) to keep this fast.

        Args:
            out_buf:  Raw uint8 output buffer from Hailo inference.
            orig_w:   Original frame width for coordinate scaling.
            orig_h:   Original frame height for coordinate scaling.

        Returns:
            List of Detection objects with pixel coordinates in original frame space.
        """
        # Bytes per class slot: 4 (count) + max_proposals × 20 (detections)
        bytes_per_class = 4 + self.max_proposals_per_class * BYTES_PER_DETECTION

        # Offset to bird class (14) slot
        bird_offset = COCO_BIRD_CLASS_ID * bytes_per_class

        # Read detection count for bird class (first 4 bytes of slot = uint32)
        count = int(np.frombuffer(out_buf[bird_offset : bird_offset + 4], dtype=np.float32)[0])
        if count == 0:
            return []

        detections = []
        det_start = bird_offset + 4  # skip the count bytes

        for i in range(count):
            offset = det_start + i * BYTES_PER_DETECTION
            values = np.frombuffer(out_buf[offset : offset + BYTES_PER_DETECTION], dtype=np.float32)
            if len(values) < 5:
                continue

            x1_norm, y1_norm, x2_norm, y2_norm, score = values

            if score < self.min_bird_confidence:
                continue

            # Scale normalized coordinates back to original frame resolution
            x1 = int(x1_norm * orig_w)
            y1 = int(y1_norm * orig_h)
            x2 = int(x2_norm * orig_w)
            y2 = int(y2_norm * orig_h)

            # Clamp to frame bounds
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            # Discard degenerate boxes
            if (x2 - x1) < self.min_crop_size or (y2 - y1) < self.min_crop_size:
                logger.debug(
                    "HailoDetector: discarding small box (%d×%d)",
                    x2 - x1,
                    y2 - y1,
                )
                continue

            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(score),
                    class_id=COCO_BIRD_CLASS_ID,
                )
            )

        return detections
