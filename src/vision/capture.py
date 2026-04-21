"""
src/vision/capture.py

Captures frames from both Pi Camera Module 3 sensors and prepares them
for EfficientNet classification.

Hardware:
    Camera 0 (primary):   i2c@88000, IMX708 sensor — left stereo position
    Camera 1 (secondary): i2c@80000, IMX708 sensor — right stereo position
    Mode:                 1536x864 @ 120fps (highest framerate available)
    Simultaneous capture: confirmed via picamera2 on deployed Pi

Pipeline per cycle (fixed_crop mode):
    picamera2 capture → both cameras simultaneously
        → apply feeder crop (configurable ROI from hardware.yaml)
        → motion gate (background model comparison — skip empty frames)
        → save cropped frame to data/captures/images/ if above threshold
        → preprocess_frame() → (224, 224, 3) float32 normalized arrays
        → return (CaptureResult, CaptureResult | None)

Pipeline per cycle (yolo mode):
    picamera2 capture → both cameras simultaneously
        → motion gate on fixed crop (background model — skip empty frames)
        → if motion: run YOLOv8s on full frame to find bird bounding box
        → if bird detected: crop to bounding box (+ padding)
        → if no bird: fall back to fixed crop
        → save crop, preprocess, return CaptureResult

Why two detection modes?
    fixed_crop assumes the bird is always in a specific pixel region.
    yolo finds wherever the bird actually is, handles mounting variation,
    and enables per-bird crops that represent exactly what the classifier
    should see. The ExperimentOrchestrator can switch between modes on a
    timer to A/B compare classification accuracy across modes.

Why crop before classification?
    The full 1536x864 frame includes background sky, branches, and surroundings.
    Cropping before downsampling to 224x224 focuses EfficientNet on the bird
    rather than background, reducing false classifications from background clutter.

Why save the cropped frame rather than the full-resolution frame?
    The cropped frame is what the classifier actually saw — it is the correct
    visual record of the detection event and what gets attached to Pushover
    notifications (Phase 6). Full-resolution PNG can exceed Pushover's 2.5MB
    attachment limit and is not needed for notifications or the observation record.

Config keys consumed:
    hardware.yaml: cameras.*, hailo.detection_mode, hailo.models.yolo_hef,
                   hailo.yolo.*
    paths.yaml:    captures.images
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

try:
    from hailo_platform import (  # type: ignore[import]
        HailoSchedulingAlgorithm as _HailoSchedulingAlgorithm,
    )
    from hailo_platform import (
        VDevice as _HailoVDevice,
    )

    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    _HailoVDevice = None  # type: ignore[assignment]
    _HailoSchedulingAlgorithm = None  # type: ignore[assignment]


from src.vision.preprocess import preprocess_frame

logger = logging.getLogger(__name__)


def _create_shared_vdevice() -> object:
    """
    Create a Hailo VDevice configured with the ROUND_ROBIN scheduler.

    The ROUND_ROBIN scheduling algorithm is REQUIRED for the modern HailoRT
    InferModel API (HailoRT 4.23.0). Without it, inference calls return
    HAILO_STREAM_NOT_ACTIVATED(72) and the output buffer is filled with zeros.

    This requirement is documented in Hailo's HailoRT python user guide and
    in our own src/vision/hailo_extractor.py module notes.

    When multiple models share one VDevice (YOLO detector + EfficientNet
    extractor), ROUND_ROBIN lets the chip's scheduler switch between them
    automatically rather than requiring manual activate/deactivate around
    each inference call.

    Returns:
        Configured VDevice instance.

    Raises:
        RuntimeError: If hailo_platform is not available.
    """
    if not HAILO_AVAILABLE:
        raise RuntimeError("hailo_platform not available — cannot create shared VDevice.")
    params = _HailoVDevice.create_params()
    params.scheduling_algorithm = _HailoSchedulingAlgorithm.ROUND_ROBIN
    return _HailoVDevice(params)


# Detection mode constants
DETECTION_MODE_FIXED_CROP = "fixed_crop"
DETECTION_MODE_YOLO = "yolo"


@dataclass
class CaptureResult:
    """
    Output of a single camera capture, post-crop and post-preprocess.

    Attributes:
        frame:          Preprocessed float32 array (224, 224, 3), ImageNet-normalized.
                        Ready for VisualClassifier.predict().
        raw_frame:      Raw uint8 array at full capture resolution (H, W, 3).
                        Preserved in memory for stereo depth estimation in Phase 6.
        camera_index:   Which camera produced this frame (0=primary, 1=secondary).
        image_path:     Absolute path where the cropped frame was saved to disk,
                        or None if the save failed.
        motion_score:   Mean absolute pixel difference from background model.
        detection_mode: Which detection mode produced this crop ("fixed_crop"|"yolo").
        detection_box:  (x1, y1, x2, y2) bounding box in original frame coordinates
                        if YOLO detected a bird, else None.
    """

    frame: np.ndarray  # (224, 224, 3) float32 preprocessed
    raw_frame: np.ndarray  # (H, W, 3) uint8 full resolution
    camera_index: int
    image_path: Path | None
    motion_score: float
    detection_mode: str = DETECTION_MODE_FIXED_CROP
    detection_box: tuple[int, int, int, int] | None = None


class VisionCapture:
    """
    Captures frames from both Pi Camera Module 3 sensors simultaneously.

    Supports two detection modes controlled by hardware.yaml:
        fixed_crop: Apply a fixed ROI crop before classification (Phase 5 default).
        yolo:       Run YOLOv8s on the full frame to find bird bounding boxes,
                    then crop to the detected region (Phase 6 detect-then-classify).

    Usage:
        capture = VisionCapture.from_config("configs/")
        primary, secondary = capture.capture_frames()
        if primary is not None:
            result_0 = classifier.predict(primary.frame, camera_index=0)
        capture.stop()
    """

    def __init__(
        self,
        primary_index: int,
        secondary_index: int,
        capture_width: int,
        capture_height: int,
        capture_fps: int,
        classification_width: int,
        classification_height: int,
        crop_x: int,
        crop_y: int,
        crop_width: int,
        crop_height: int,
        crop_x_cam1: int | None = None,
        crop_y_cam1: int | None = None,
        crop_width_cam1: int | None = None,
        crop_height_cam1: int | None = None,
        motion_threshold: float = 0.005,
        background_history: int = 30,
        output_dir: str | Path = "data/captures/images",
        detection_mode: str = DETECTION_MODE_FIXED_CROP,
        hailo_yolo_hef: str | None = None,
        yolo_score_threshold: float = 0.25,
        yolo_max_proposals: int = 10,
        yolo_min_bird_confidence: float = 0.25,
        yolo_crop_padding: int = 20,
        shared_vdevice: object | None = None,
    ) -> None:
        """
        Args:
            primary_index:          picamera2 index for primary camera (left stereo).
            secondary_index:        picamera2 index for secondary camera (right stereo).
            capture_width:          Full capture frame width in pixels (1536).
            capture_height:         Full capture frame height in pixels (864).
            capture_fps:            Capture framerate (120).
            classification_width:   EfficientNet input width (224).
            classification_height:  EfficientNet input height (224).
            crop_x:                 Left edge of feeder crop ROI in pixels.
            crop_y:                 Top edge of feeder crop ROI in pixels.
            crop_width:             Width of feeder crop ROI in pixels.
            crop_height:            Height of feeder crop ROI in pixels.
            motion_threshold:       Min mean abs pixel diff to accept a frame.
            background_history:     Number of frames to average for background model.
            output_dir:             Directory where cropped frames are saved.
            detection_mode:         "fixed_crop" or "yolo". Controlled by
                                    hardware.yaml hailo.detection_mode.
            hailo_yolo_hef:         Path to yolov8s_h8l.hef. Only used when
                                    detection_mode is "yolo".
            yolo_score_threshold:   YOLO NMS score threshold.
            yolo_max_proposals:     Max YOLO detections per class per frame.
            yolo_min_bird_confidence: Min confidence to accept a bird detection.
            yolo_crop_padding:      Pixels to add around YOLO bounding box.
        """
        self.primary_index = primary_index
        self.secondary_index = secondary_index
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.capture_fps = capture_fps
        self.classification_width = classification_width
        self.classification_height = classification_height
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_width = crop_width
        self.crop_height = crop_height
        # Per-camera crop overrides — fall back to shared crop if not set
        self.crop_x_cam1 = crop_x_cam1 if crop_x_cam1 is not None else crop_x
        self.crop_y_cam1 = crop_y_cam1 if crop_y_cam1 is not None else crop_y
        self.crop_width_cam1 = crop_width_cam1 if crop_width_cam1 is not None else crop_width
        self.crop_height_cam1 = crop_height_cam1 if crop_height_cam1 is not None else crop_height
        self.motion_threshold = motion_threshold
        self.background_history = background_history
        self.output_dir = Path(output_dir).resolve()
        self.detection_mode = detection_mode
        self.hailo_yolo_hef = hailo_yolo_hef
        self.yolo_score_threshold = yolo_score_threshold
        self.yolo_max_proposals = yolo_max_proposals
        self.yolo_min_bird_confidence = yolo_min_bird_confidence
        self.yolo_crop_padding = yolo_crop_padding

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Rolling background models — one per camera
        self._bg_primary: np.ndarray | None = None
        self._bg_secondary: np.ndarray | None = None
        self._bg_count: int = 0

        # picamera2 instances — lazy loaded on first capture_frames()
        self._picam0 = None
        self._picam1 = None

        # HailoDetector — lazy loaded on first YOLO detection
        self._detector = None
        self._detector_open = False

        # Shared Hailo VDevice — created eagerly if YOLO mode, reused by VisualClassifier.
        # ROUND_ROBIN scheduling is required for the InferModel API (HailoRT 4.23.0)
        # when multiple models share a VDevice — see _create_shared_vdevice() docstring.
        self._shared_vdevice = None
        if self.detection_mode == DETECTION_MODE_YOLO and self.hailo_yolo_hef:
            try:
                self._shared_vdevice = _create_shared_vdevice()
                logger.info("Shared Hailo VDevice created (YOLO mode, ROUND_ROBIN scheduler).")
            except Exception as exc:
                logger.warning("Could not create shared VDevice: %s", exc)

        logger.info(
            "VisionCapture initialized | cameras=[%d, %d] resolution=%dx%d "
            "crop=(%d,%d,%dx%d) motion_threshold=%.3f output=%s mode=%s",
            primary_index,
            secondary_index,
            capture_width,
            capture_height,
            crop_x,
            crop_y,
            crop_width,
            crop_height,
            motion_threshold,
            self.output_dir,
            detection_mode,
        )

    @classmethod
    def from_config(cls, config_dir: str | Path) -> VisionCapture:
        """
        Construct a VisionCapture from the configs/ directory.

        Reads hardware.yaml and paths.yaml.

        Hailo YOLO config is read from hardware.yaml:
            hailo.detection_mode    → detection_mode ("fixed_crop" or "yolo")
            hailo.models.yolo_hef   → hailo_yolo_hef
            hailo.yolo.*            → yolo thresholds

        Args:
            config_dir: Path to the configs/ directory.

        Returns:
            Configured VisionCapture instance (cameras not yet opened).
        """
        config_dir = Path(config_dir)

        with (config_dir / "hardware.yaml").open() as f:
            hw = yaml.safe_load(f)
        with (config_dir / "paths.yaml").open() as f:
            paths = yaml.safe_load(f)

        cam = hw["cameras"]
        crop = cam["feeder_crop"]
        crop0 = cam.get("feeder_crop_cam0", crop)
        crop1 = cam.get("feeder_crop_cam1", crop)

        # Read Hailo YOLO config
        hailo_cfg = hw.get("hailo", {})
        detection_mode = hailo_cfg.get("detection_mode", DETECTION_MODE_FIXED_CROP)
        hailo_yolo_hef = hailo_cfg.get("models", {}).get("yolo_hef")
        yolo_cfg = hailo_cfg.get("yolo", {})

        return cls(
            primary_index=cam["primary_index"],
            secondary_index=cam["secondary_index"],
            capture_width=cam["capture_width"],
            capture_height=cam["capture_height"],
            capture_fps=cam["capture_fps"],
            classification_width=cam["classification_width"],
            classification_height=cam["classification_height"],
            crop_x=crop0["x"],
            crop_y=crop0["y"],
            crop_width=crop0["width"],
            crop_height=crop0["height"],
            crop_x_cam1=crop1["x"],
            crop_y_cam1=crop1["y"],
            crop_width_cam1=crop1["width"],
            crop_height_cam1=crop1["height"],
            motion_threshold=cam["motion_threshold"],
            background_history=cam["background_history"],
            output_dir=paths["captures"]["images"],
            detection_mode=detection_mode,
            hailo_yolo_hef=hailo_yolo_hef,
            yolo_score_threshold=yolo_cfg.get("score_threshold", 0.25),
            yolo_max_proposals=yolo_cfg.get("max_proposals", 10),
            yolo_min_bird_confidence=yolo_cfg.get("min_bird_confidence", 0.25),
        )

    def _open_cameras(self) -> None:
        """
        Open both cameras via picamera2 for simultaneous capture.

        Called lazily on first capture_frames(). Configures format,
        resolution, and continuous autofocus before starting the sensors.

        Format choice (picamera2 gotcha, addressed in PR #50):
            picamera2's "RGB888" format actually returns bytes in B-G-R
            order in the numpy array, a long-standing libcamera convention
            where the format name describes the source pixel format rather
            than the resulting memory layout. To get true R-G-B order in
            the array (matching NABirds training data and PIL's default
            interpretation), we request "BGR888", which libcamera then
            delivers as R-G-B in memory. Without this, every frame fed
            to EfficientNet had red and blue channels swapped, which
            silently degraded classification on warm-plumage species
            (house finches, sparrows, orioles) during Phase 5 deployment.

        Autofocus (added in PR #50):
            Pi Camera Module 3 has hardware autofocus, but libcamera's
            default is manual focus at lens position 0.0 (infinity).
            Birds at the feeder sit ~30 cm from the lens, well within
            macro range, so fixed infinity focus produces the soft
            captures observed in Phase 5. AfModeEnum.Continuous lets
            each sensor track focus as birds land and move on the tray.
            If a camera does not support autofocus (e.g. Module 2), we
            log and continue so the system still captures at fixed focus.
        """
        try:
            from picamera2 import Picamera2  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 is not installed or not available. "
                "VisionCapture requires a Raspberry Pi with picamera2."
            ) from exc

        # libcamera provides the AfModeEnum. Imported separately from
        # picamera2 because older hardware without autofocus may still
        # have picamera2 installed. Fall back gracefully if absent.
        try:
            from libcamera import controls  # type: ignore[import]
        except ImportError:
            controls = None

        logger.info(
            "Opening cameras %d and %d via picamera2...",
            self.primary_index,
            self.secondary_index,
        )

        self._picam0 = Picamera2(self.primary_index)
        self._picam1 = Picamera2(self.secondary_index)

        # "BGR888" is picamera2's counterintuitive name for true-RGB
        # memory layout. See _open_cameras() docstring for full detail.
        config0 = self._picam0.create_still_configuration(
            main={"size": (self.capture_width, self.capture_height), "format": "BGR888"},
        )
        config1 = self._picam1.create_still_configuration(
            main={"size": (self.capture_width, self.capture_height), "format": "BGR888"},
        )

        self._picam0.configure(config0)
        self._picam1.configure(config1)

        # Continuous autofocus for birds at macro range (~30cm). Set
        # after configure() so the control applies to the active config.
        # Wrapped in try/except so a single camera without AF does not
        # prevent the other camera (or fixed-focus operation) from
        # functioning.
        if controls is not None:
            for name, cam in [("cam0", self._picam0), ("cam1", self._picam1)]:
                try:
                    cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                    logger.info("Continuous autofocus enabled on %s.", name)
                except Exception as exc:
                    logger.warning(
                        "Continuous autofocus unavailable on %s: %s. "
                        "Falling back to fixed focus for this sensor.",
                        name,
                        exc,
                    )
        else:
            logger.warning(
                "libcamera controls module unavailable — "
                "autofocus cannot be configured. Using fixed focus."
            )

        self._picam0.start()
        self._picam1.start()

        logger.info("Both cameras opened and started.")

    def _load_detector(self) -> bool:
        """
        Lazily load HailoDetector for YOLO mode.

        Returns True if detector is ready, False if unavailable.
        Falls back silently — caller continues with fixed_crop.
        """
        if self._detector_open:
            return True
        if not self.hailo_yolo_hef:
            logger.warning("YOLO mode requested but no yolo_hef path configured.")
            return False
        try:
            from src.vision.hailo_detector import HailoDetector

            if self._shared_vdevice is None and HAILO_AVAILABLE:
                self._shared_vdevice = _create_shared_vdevice()
                logger.info("Shared Hailo VDevice created (lazy, ROUND_ROBIN scheduler).")

            self._detector = HailoDetector(
                hef_path=self.hailo_yolo_hef,
                score_threshold=self.yolo_score_threshold,
                max_proposals_per_class=self.yolo_max_proposals,
                min_bird_confidence=self.yolo_min_bird_confidence,
                shared_vdevice=self._shared_vdevice,
            )
            self._detector.open()

            self._detector_open = True
            logger.info(
                "HailoDetector opened for YOLO detection | hef=%s",
                self.hailo_yolo_hef,
            )
            return True
        except Exception as exc:
            logger.warning("HailoDetector failed to load (%s) — falling back to fixed_crop.", exc)
            self._detector = None
            return False

    def capture_frames(self) -> tuple[CaptureResult | None, CaptureResult | None]:
        """
        Capture one frame from each camera simultaneously and apply motion gate.

        Returns:
            Tuple of (primary_result, secondary_result).
            Either element is None if the camera's frame was below the
            motion threshold. Both None means the feeder is empty this cycle.
        """
        if self._picam0 is None:
            self._open_cameras()

        try:
            raw0 = self._picam0.capture_array("main")
            raw1 = self._picam1.capture_array("main")
        except Exception as exc:
            logger.exception("Camera capture failed: %s", exc)
            return None, None

        result0 = self._process_frame(raw0, camera_index=self.primary_index)
        result1 = self._process_frame(raw1, camera_index=self.secondary_index)

        return result0, result1

    def _process_frame(
        self,
        raw_frame: np.ndarray,
        camera_index: int,
    ) -> CaptureResult | None:
        """
        Apply motion gate, crop (fixed or YOLO), save, and preprocess a frame.

        In fixed_crop mode:
            Apply feeder_crop ROI → motion gate → save → preprocess.

        In yolo mode:
            Apply feeder_crop for motion gate only → if motion detected,
            run YOLO on full frame → use detected bird box as crop region
            (falls back to fixed_crop if no bird detected by YOLO).

        Args:
            raw_frame:    Full resolution (H, W, 3) uint8 RGB from picamera2.
            camera_index: Which camera this frame came from.

        Returns:
            CaptureResult if frame passes motion gate, else None.
        """
        cx = self.crop_x_cam1 if camera_index == self.secondary_index else self.crop_x
        cy = self.crop_y_cam1 if camera_index == self.secondary_index else self.crop_y
        cw = self.crop_width_cam1 if camera_index == self.secondary_index else self.crop_width
        ch = self.crop_height_cam1 if camera_index == self.secondary_index else self.crop_height

        fixed_crop = raw_frame[
            cy : cy + ch,
            cx : cx + cw,
        ]

        # ── Motion gate ───────────────────────────────────────────────────────
        motion_score, bg = self._compute_motion(fixed_crop, camera_index)
        self._update_background(fixed_crop, camera_index, bg)

        if motion_score < self.motion_threshold:
            logger.debug(
                "Camera %d: below motion threshold (%.4f < %.4f)",
                camera_index,
                motion_score,
                self.motion_threshold,
            )
            return None

        logger.debug(
            "Camera %d: motion detected (score=%.4f >= %.4f)",
            camera_index,
            motion_score,
            self.motion_threshold,
        )

        # ── Crop selection — fixed_crop or YOLO ───────────────────────────────
        crop_to_classify = fixed_crop
        detection_mode = DETECTION_MODE_FIXED_CROP
        detection_box = None

        if self.detection_mode == DETECTION_MODE_YOLO:
            yolo_ready = self._load_detector()
            if yolo_ready and self._detector is not None:
                try:
                    detection = self._detector.detect(raw_frame)
                    if detection is not None:
                        crop_to_classify = detection.as_crop(
                            raw_frame, padding=self.yolo_crop_padding
                        )
                        detection_mode = DETECTION_MODE_YOLO
                        detection_box = (
                            detection.x1,
                            detection.y1,
                            detection.x2,
                            detection.y2,
                        )
                        logger.debug(
                            "Camera %d: YOLO bird detected conf=%.3f " "box=(%d,%d,%d,%d)",
                            camera_index,
                            detection.confidence,
                            detection.x1,
                            detection.y1,
                            detection.x2,
                            detection.y2,
                        )
                    else:
                        logger.debug(
                            "Camera %d: YOLO no bird — falling back to fixed_crop.",
                            camera_index,
                        )
                except Exception as exc:
                    logger.warning(
                        "Camera %d: YOLO detection failed (%s) — " "falling back to fixed_crop.",
                        camera_index,
                        exc,
                    )

        # ── Save crop ─────────────────────────────────────────────────────────
        image_path = self._save_frame(crop_to_classify, camera_index)

        # ── Preprocess for classification ─────────────────────────────────────
        preprocessed = preprocess_frame(
            crop_to_classify,
            width=self.classification_width,
            height=self.classification_height,
        )

        return CaptureResult(
            frame=preprocessed,
            raw_frame=raw_frame,
            camera_index=camera_index,
            image_path=image_path,
            motion_score=motion_score,
            detection_mode=detection_mode,
            detection_box=detection_box,
        )

    def _compute_motion(
        self,
        frame: np.ndarray,
        camera_index: int,
    ) -> tuple[float, np.ndarray | None]:
        """
        Compute mean absolute pixel difference from the background model.

        On the first background_history frames we return motion_threshold + 1
        so initial frames always pass through to build the background model.
        """
        bg = self._bg_primary if camera_index == self.primary_index else self._bg_secondary

        if bg is None or self._bg_count < self.background_history:
            return self.motion_threshold + 1.0, bg

        frame_float = frame.astype(np.float32) / 255.0
        diff = np.abs(frame_float - bg)
        score = float(diff.mean())
        return score, bg

    def _update_background(
        self,
        frame: np.ndarray,
        camera_index: int,
        current_bg: np.ndarray | None,
    ) -> None:
        """
        Update the rolling background model with the current frame.

        Uses exponential moving average: bg = alpha * frame + (1 - alpha) * bg
        where alpha = 1 / background_history.
        """
        frame_float = frame.astype(np.float32) / 255.0
        alpha = 1.0 / self.background_history

        if current_bg is None:
            new_bg = frame_float.copy()
        else:
            new_bg = alpha * frame_float + (1.0 - alpha) * current_bg

        if camera_index == self.primary_index:
            self._bg_primary = new_bg
        else:
            self._bg_secondary = new_bg

        self._bg_count += 1

    def _save_frame(self, cropped_frame: np.ndarray, camera_index: int) -> Path | None:
        """
        Save a cropped frame to disk as a PNG.

        Uses PIL for PNG encoding — no OpenCV dependency required.
        output_dir is absolute (resolved in __init__) so the returned path
        is always absolute regardless of working directory.
        """
        try:
            from PIL import Image  # type: ignore[import]

            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{ts}_cam{camera_index}.png"
            path = self.output_dir / filename

            Image.fromarray(cropped_frame).save(path)
            logger.debug("Saved cropped frame → %s", filename)
            return path

        except Exception as exc:
            logger.warning("Failed to save frame from camera %d: %s", camera_index, exc)
            return None

    def stop(self) -> None:
        """
        Stop both cameras and release all resources including HailoDetector.

        Safe to call even if cameras were never opened.
        """
        for cam_attr in ("_picam0", "_picam1"):
            cam = getattr(self, cam_attr, None)
            if cam is not None:
                try:
                    cam.stop()
                    cam.close()
                    logger.info("Camera %s stopped.", cam_attr)
                except Exception as exc:
                    logger.warning("Error stopping %s: %s", cam_attr, exc)

        if self._detector is not None and self._detector_open:
            try:
                self._detector.close()
                logger.info("HailoDetector closed.")
            except Exception as exc:
                logger.warning("Error closing HailoDetector: %s", exc)
            self._detector = None
            self._detector_open = False

        if self._shared_vdevice is not None:
            try:
                self._shared_vdevice = None
                logger.info("Shared Hailo VDevice released.")
            except Exception as exc:
                logger.warning("Error releasing shared VDevice: %s", exc)

    def get_shared_vdevice(self) -> object | None:
        """Return the shared Hailo VDevice for use by VisualClassifier."""
        return self._shared_vdevice
