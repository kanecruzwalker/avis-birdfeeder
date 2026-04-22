"""
src/vision/detector.py

Protocol and implementations for bird-presence detection.

The BirdDetector runs as a gate before species classification:
motion triggers → detector asks "is there a bird in this frame?" →
if yes, classifier runs on the configured crop (fixed ROI or detector
bounding box depending on detection_mode); if no, the observation is
logged with gate_reason='no_bird_detected' and classification is skipped.

Two backends:
    CPUYOLODetector:   ultralytics YOLOv8s on CPU. Default for Branch 2.
                       ~1000ms per inference on Pi 5 CPU. Acceptable given
                       the ~8s motion-gated cycle.
    HailoYOLODetector: Hailo NPU inference (Branch 5 — not yet available).
                       ~13ms per inference when compiled with correct
                       preprocessing contract.

Both implement the BirdDetector protocol and are interchangeable via the
hardware.yaml: detector.backend config toggle.

See docs/investigations/hailo-2026-04-22.md for the full investigation that
motivated CPU-first default: the stock `yolov8s_h8l.hef` does not produce
usable detections via the VDevice Python API. A custom-compiled HEF is
deferred to Branch 5 (post-report).

Adding a new backend:
    1. Implement the BirdDetector protocol (see class docstring).
    2. Wire it into the load_detector() factory below.
    3. Add a config block under hardware.yaml: detector.<name>.
    4. Add unit tests in tests/vision/test_detector.py following the
       CPUYOLODetector pattern (mock the underlying library, test the
       protocol contract).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ── BirdDetection dataclass ───────────────────────────────────────────────────
# A backend-agnostic detection type returned by BirdDetector implementations.
# This is separate from src.vision.hailo_detector.Detection so the protocol
# stays free of Hailo-specific fields and conventions. A HailoYOLODetector
# wrapper (Branch 5) would convert its internal Detection → BirdDetection
# at the protocol boundary.


@dataclass
class BirdDetection:
    """
    A single bird detected in a frame by any BirdDetector backend.

    Coordinates are in the ORIGINAL frame's pixel space, not the resized
    or letterboxed input the detector internally used. Backends are
    responsible for translating their internal coordinates back to the
    input frame's space before constructing a BirdDetection.

    Attributes:
        x1, y1: Top-left corner of the bounding box (pixels, original frame).
        x2, y2: Bottom-right corner of the bounding box (pixels, original frame).
                x2 > x1 and y2 > y1 are guaranteed; backends must validate.
        confidence: Detector's confidence in [0.0, 1.0].
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_crop(self, frame: np.ndarray, padding: int = 0) -> np.ndarray:
        """
        Return the region of `frame` enclosed by this detection, with optional
        padding, clamped to the frame's bounds.

        The returned array is a view into `frame` (numpy slicing), not a copy.
        If the caller needs to mutate independently of the source frame, call
        `.copy()` on the result.

        Args:
            frame:   Source frame the detection was computed on. Shape (H, W, C).
            padding: Pixels to expand the box on all sides before cropping.
                     Expansion is clamped to stay within frame bounds.

        Returns:
            Cropped region of the frame. Shape (h, w, C) with h, w > 0
            unless the detection was already degenerate.
        """
        h, w = frame.shape[:2]
        x1 = max(0, self.x1 - padding)
        y1 = max(0, self.y1 - padding)
        x2 = min(w, self.x2 + padding)
        y2 = min(h, self.y2 + padding)
        return frame[y1:y2, x1:x2]


# ── BirdDetector protocol ─────────────────────────────────────────────────────


@runtime_checkable
class BirdDetector(Protocol):
    """
    Interface for bird-presence detection backends.

    Contract:
        is_open (bool attribute):
            True after open() completes, False after close() or before open().
            Readable without requiring the detector to be open.

        open() -> None:
            Prepares the detector for inference — may load weights, open
            hardware, warm caches. Idempotent: calling open() on an already-
            open detector must be a no-op (not an error).

        detect(frame) -> BirdDetection | None:
            Runs inference and returns the highest-confidence bird detection
            in the frame, or None if no bird was detected above the backend's
            configured confidence threshold.
            Must not raise on frames without detections.

        detect_all(frame) -> list[BirdDetection]:
            Returns all bird detections above threshold, sorted by confidence
            descending. Empty list if no detections (never None).

        close() -> None:
            Releases resources. Idempotent. Safe to call multiple times or
            on a detector that was never opened.

    Implementations may require open() before detect() / detect_all().
    Calling detect() on a not-open detector SHOULD raise RuntimeError with
    a clear message ("call open() first") rather than silently failing.

    Backends are responsible for any preprocessing the underlying model
    requires (resize, letterbox, normalization, channel order, etc.).
    Callers pass raw uint8 RGB frames of arbitrary size.
    """

    is_open: bool

    def open(self) -> None: ...

    def detect(self, frame: np.ndarray) -> BirdDetection | None: ...

    def detect_all(self, frame: np.ndarray) -> list[BirdDetection]: ...

    def close(self) -> None: ...


# ── CPUYOLODetector — ultralytics-based CPU implementation ───────────────────


class CPUYOLODetector:
    """
    Bird-presence detector using ultralytics YOLOv8s on CPU.

    This is the default backend for Branch 2, selected when
    hardware.yaml: detector.backend == "cpu".

    Latency on Raspberry Pi 5 ARM64: ~1000ms per inference at imgsz=640.
    This is acceptable given the ~8-second motion-gated cycle interval —
    the detector only runs when motion fires, and an additional second
    of latency on a motion-triggered cycle is a ~12% worst-case increase.

    Preprocessing: ultralytics handles all of it internally. We pass a
    raw RGB uint8 numpy array of arbitrary size; ultralytics resizes,
    letterboxes, normalizes, and returns results in the original frame's
    coordinate space. No manual preprocessing is required here.

    Model weights: yolov8s.pt is downloaded by ultralytics to CWD on
    first use (or resolved to a path configured via model_path). At the
    time of writing the download is ~22MB.

    Filtering: this class is specifically a bird-presence detector —
    detect() and detect_all() only consider COCO class 14 (bird) and
    discard all other detected objects.
    """

    COCO_BIRD_CLASS_ID = 14

    def __init__(
        self,
        model_path: str | Path = "yolov8s.pt",
        confidence_threshold: float = 0.25,
        imgsz: int = 640,
    ) -> None:
        """
        Args:
            model_path:          Path to the YOLOv8s weights file. Passed
                                 directly to ultralytics.YOLO(); can be a
                                 local path or a name like "yolov8s.pt"
                                 which ultralytics resolves via its own
                                 download logic.
            confidence_threshold: Minimum YOLO confidence to accept a bird
                                 detection. 0.25 is a reasonable default
                                 for backyard birds; tune based on false
                                 positive / false negative tradeoff.
            imgsz:               Input image size for YOLO inference.
                                 640 is the YOLOv8s canonical training size.
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        self._model = None  # lazy loaded in open()
        self.is_open = False

        logger.info(
            "CPUYOLODetector initialized | model_path=%s conf=%.2f imgsz=%d",
            self.model_path,
            self.confidence_threshold,
            self.imgsz,
        )

    @classmethod
    def from_config(cls, config_dir: str | Path) -> CPUYOLODetector:
        """
        Construct a CPUYOLODetector from configs/hardware.yaml.

        Reads the `detector.cpu.*` block:
            model_path: path or name passed to ultralytics.YOLO()
            confidence_threshold: minimum YOLO confidence to accept
            imgsz: YOLO input size (usually 640)

        Args:
            config_dir: Path to the configs/ directory.

        Returns:
            Configured CPUYOLODetector instance (weights not yet loaded —
            call open() when ready to inference).
        """
        config_dir = Path(config_dir)
        with (config_dir / "hardware.yaml").open() as f:
            hw = yaml.safe_load(f)

        cpu_cfg = hw.get("detector", {}).get("cpu", {})
        return cls(
            model_path=cpu_cfg.get("model_path", "yolov8s.pt"),
            confidence_threshold=float(cpu_cfg.get("confidence_threshold", 0.25)),
            imgsz=int(cpu_cfg.get("imgsz", 640)),
        )

    def open(self) -> None:
        """
        Load YOLOv8s weights and warm up the model with a dummy inference.

        The warm-up is important because ultralytics' first real inference
        can be ~30% slower than subsequent ones due to lazy imports and
        kernel compilation. Doing a warm-up here means the agent's first
        real motion-triggered YOLO call has predictable latency.

        Idempotent — safe to call on an already-open detector.
        """
        if self.is_open:
            return

        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. On Pi: "
                "pip install ultralytics --break-system-packages --no-deps. "
                "In dev venv: pip install ultralytics."
            ) from exc

        self._model = YOLO(str(self.model_path))

        # Warm up with a dummy inference so the first real call isn't slow.
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        _ = self._model(dummy, conf=self.confidence_threshold, verbose=False)

        self.is_open = True
        logger.info(
            "CPUYOLODetector opened | model=%s warmed up at imgsz=%d",
            self.model_path,
            self.imgsz,
        )

    def detect(self, frame: np.ndarray) -> BirdDetection | None:
        """
        Run inference and return the highest-confidence bird detection.

        Args:
            frame: Raw RGB uint8 numpy array of arbitrary (H, W, 3) shape.

        Returns:
            Highest-confidence BirdDetection, or None if no bird detected.

        Raises:
            RuntimeError: If the detector has not been opened via open().
        """
        all_dets = self.detect_all(frame)
        return all_dets[0] if all_dets else None

    def detect_all(self, frame: np.ndarray) -> list[BirdDetection]:
        """
        Run inference and return all bird detections above threshold.

        ultralytics handles resize/letterbox internally and returns
        bounding boxes in the original frame's pixel coordinate space.

        Args:
            frame: Raw RGB uint8 numpy array of arbitrary (H, W, 3) shape.

        Returns:
            List of BirdDetection objects sorted by confidence descending.
            Empty list if no birds detected.

        Raises:
            RuntimeError: If the detector has not been opened via open().
        """
        if not self.is_open or self._model is None:
            raise RuntimeError("CPUYOLODetector not open. Call open() before detect/detect_all.")

        results = self._model(frame, conf=self.confidence_threshold, verbose=False)
        r = results[0]

        birds: list[BirdDetection] = []
        if r.boxes is None:
            return birds

        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id != self.COCO_BIRD_CLASS_ID:
                continue

            # xyxy returns a single-row tensor; convert to int pixel coords
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
            conf = float(box.conf[0])

            birds.append(
                BirdDetection(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    confidence=conf,
                )
            )

        birds.sort(key=lambda d: d.confidence, reverse=True)
        return birds

    def close(self) -> None:
        """
        Release the loaded model and mark the detector closed.

        Idempotent — safe to call on a closed or never-opened detector.
        """
        self._model = None
        self.is_open = False
        logger.info("CPUYOLODetector closed.")

    def __enter__(self) -> CPUYOLODetector:
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ── Factory ──────────────────────────────────────────────────────────────────


def load_detector(config_dir: str | Path) -> BirdDetector:
    """
    Construct a BirdDetector instance based on hardware.yaml: detector.backend.

    Supported backends:
        "cpu":   CPUYOLODetector (ultralytics, default)
        "hailo": HailoYOLODetector — deferred to Branch 5, raises
                 NotImplementedError until a working custom HEF is compiled.

    This factory is the single entry point callers use to obtain a detector;
    backend-specific construction details are encapsulated behind the
    BirdDetector protocol.

    Args:
        config_dir: Path to the configs/ directory.

    Returns:
        A BirdDetector implementation, ready to open() and detect() with.

    Raises:
        NotImplementedError: If hailo backend is requested (deferred to Branch 5).
        ValueError:          If an unknown backend name is configured.
        FileNotFoundError:   If hardware.yaml does not exist.
    """
    config_dir = Path(config_dir)
    hw_path = config_dir / "hardware.yaml"

    if not hw_path.exists():
        raise FileNotFoundError(f"hardware.yaml not found at {hw_path}")

    with hw_path.open() as f:
        hw = yaml.safe_load(f)

    backend = hw.get("detector", {}).get("backend", "cpu")

    if backend == "cpu":
        return CPUYOLODetector.from_config(config_dir)

    if backend == "hailo":
        raise NotImplementedError(
            "The 'hailo' detector backend is deferred to Branch 5 "
            "(see docs/investigations/hailo-2026-04-22.md). The stock "
            "yolov8s_h8l.hef does not produce usable detections via the "
            "VDevice API, and a custom-compiled replacement is planned as "
            "post-report future work. Use backend='cpu' until then."
        )

    raise ValueError(
        f"Unknown detector backend: {backend!r}. "
        f"Valid values are 'cpu' (available now) and 'hailo' "
        f"(deferred to Branch 5). Check hardware.yaml: detector.backend."
    )
