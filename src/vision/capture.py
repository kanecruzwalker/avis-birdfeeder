"""
src/vision/capture.py

Captures frames from both Pi Camera Module 3 sensors and prepares them
for EfficientNet classification.

Hardware:
    Camera 0 (primary):   i2c@88000, IMX708 sensor — left stereo position
    Camera 1 (secondary): i2c@80000, IMX708 sensor — right stereo position
    Mode:                 1536x864 @ 120fps (highest framerate available)
    Simultaneous capture: confirmed via picamera2 on deployed Pi

Pipeline per cycle:
    picamera2 capture → both cameras simultaneously
        → apply feeder crop (configurable ROI from hardware.yaml)
        → motion gate (background model comparison — skip empty frames)
        → save cropped frame to data/captures/images/ if above threshold
        → preprocess_frame() → (224, 224, 3) float32 normalized arrays
        → return (CaptureResult, CaptureResult | None)

Why crop before classification?
    The full 1536x864 frame includes background sky, branches, and surroundings.
    Cropping to the feeder perch zone (400x400px configured in hardware.yaml)
    before downsampling to 224x224 focuses EfficientNet on the bird rather
    than background, reducing false classifications from background clutter.
    The crop coordinates are tunable post-mounting via hardware.yaml —
    no code changes needed when repointing the cameras.

Why save the cropped frame rather than the full-resolution frame?
    The cropped frame (400x400px) is what the classifier actually saw — it is
    the correct visual record of the detection event and is what gets attached
    to Pushover notifications (Phase 6). Full-resolution PNG at 1536x864 can
    exceed the Pushover 2.5MB attachment limit and is not needed for
    notifications or the observation record.

    The full-resolution raw_frame is preserved in memory on CaptureResult for
    use by StereoEstimator in Phase 6. When stereo depth estimation is active,
    StereoEstimator will consume raw_frame directly from the CaptureResult —
    no full-resolution disk save is required until scripts/calibrate_stereo.py
    needs to persist calibration frames, which is a Phase 6 task.

Why motion gate?
    At 120fps both cameras produce ~240 frames/second. Most frames contain
    an empty feeder. The motion gate compares each frame to a rolling background
    model (mean of last N frames) and only passes frames where the mean absolute
    pixel difference exceeds motion_threshold. This reduces classifier compute
    to only the moments when something is actually at the feeder.

Config keys consumed:
    hardware.yaml: cameras.primary_index, cameras.secondary_index,
                   cameras.capture_width, cameras.capture_height,
                   cameras.capture_fps, cameras.classification_width,
                   cameras.classification_height, cameras.feeder_crop,
                   cameras.motion_threshold, cameras.background_history
    paths.yaml:    captures.images

Dependencies:
    picamera2   — Pi Camera Module 3 capture (Pi only)
    numpy       — array operations and background model
    src.vision.preprocess — frame normalization for EfficientNet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from src.vision.preprocess import preprocess_frame

logger = logging.getLogger(__name__)


@dataclass
class CaptureResult:
    """
    Output of a single camera capture, post-crop and post-preprocess.

    Attributes:
        frame:        Preprocessed float32 array (224, 224, 3), ImageNet-normalized.
                      Ready for VisualClassifier.predict().
        raw_frame:    Raw uint8 array at full capture resolution (H, W, 3).
                      Preserved in memory for stereo depth estimation in Phase 6.
                      StereoEstimator consumes this directly — not read from disk.
        camera_index: Which camera produced this frame (0=primary, 1=secondary).
        image_path:   Absolute path where the cropped frame was saved to disk,
                      or None if the save failed. Used for Pushover notifications
                      and the observation log. Points to the cropped frame
                      (crop_width × crop_height) not the full-resolution frame.
        motion_score: Mean absolute pixel difference from background model.
                      Used to decide whether to classify this frame.
    """

    frame: np.ndarray  # (224, 224, 3) float32 preprocessed
    raw_frame: np.ndarray  # (H, W, 3) uint8 full resolution — for Phase 6 stereo
    camera_index: int
    image_path: Path | None  # absolute path to saved cropped frame
    motion_score: float


class VisionCapture:
    """
    Captures frames from both Pi Camera Module 3 sensors simultaneously.

    Each call to capture_frames() returns preprocessed results from both
    cameras, filtering out frames below the motion threshold.

    Usage:
        capture = VisionCapture.from_config("configs/")
        primary, secondary = capture.capture_frames()
        if primary is not None:
            result_0 = classifier.predict(primary.frame, camera_index=0)
        if secondary is not None:
            result_1 = classifier.predict(secondary.frame, camera_index=1)
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
        motion_threshold: float,
        background_history: int,
        output_dir: str | Path,
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
            output_dir:             Directory where cropped frames are saved on
                                    detection. Resolved to absolute path at
                                    construction time so image_path on CaptureResult
                                    is always absolute regardless of working directory.
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
        self.motion_threshold = motion_threshold
        self.background_history = background_history
        # resolve() makes the path absolute relative to cwd at construction time.
        # The agent is always launched from the project root, so this guarantees
        # image_path on CaptureResult is absolute and portable across modules.
        self.output_dir = Path(output_dir).resolve()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Rolling background models — one per camera, shape (H, W, 3) float32
        # Initialized on first capture, updated each cycle
        self._bg_primary: np.ndarray | None = None
        self._bg_secondary: np.ndarray | None = None
        self._bg_count: int = 0

        # picamera2 instance — lazy loaded on first capture_frames() call
        self._picam = None

        logger.info(
            "VisionCapture initialized | cameras=[%d, %d] resolution=%dx%d "
            "crop=(%d,%d,%dx%d) motion_threshold=%.3f output=%s",
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
        )

    @classmethod
    def from_config(cls, config_dir: str | Path) -> VisionCapture:
        """
        Construct a VisionCapture from the configs/ directory.

        Reads hardware.yaml and paths.yaml.

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

        return cls(
            primary_index=cam["primary_index"],
            secondary_index=cam["secondary_index"],
            capture_width=cam["capture_width"],
            capture_height=cam["capture_height"],
            capture_fps=cam["capture_fps"],
            classification_width=cam["classification_width"],
            classification_height=cam["classification_height"],
            crop_x=crop["x"],
            crop_y=crop["y"],
            crop_width=crop["width"],
            crop_height=crop["height"],
            motion_threshold=cam["motion_threshold"],
            background_history=cam["background_history"],
            output_dir=paths["captures"]["images"],
        )

    def _open_cameras(self) -> None:
        """
        Open both cameras via picamera2 for simultaneous capture.

        Called lazily on first capture_frames(). Separated from __init__ so
        that constructing VisionCapture in tests does not require Pi hardware.

        Raises:
            RuntimeError: If picamera2 is not installed or cameras are not detected.
        """
        try:
            from picamera2 import Picamera2  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 is not installed or not available. "
                "VisionCapture requires a Raspberry Pi with picamera2. "
                "On laptop, mock VisionCapture.capture_frames() in tests."
            ) from exc

        logger.info(
            "Opening cameras %d and %d via picamera2...",
            self.primary_index,
            self.secondary_index,
        )

        self._picam0 = Picamera2(self.primary_index)
        self._picam1 = Picamera2(self.secondary_index)

        config0 = self._picam0.create_still_configuration(
            main={"size": (self.capture_width, self.capture_height), "format": "RGB888"},
        )
        config1 = self._picam1.create_still_configuration(
            main={"size": (self.capture_width, self.capture_height), "format": "RGB888"},
        )

        self._picam0.configure(config0)
        self._picam1.configure(config1)
        self._picam0.start()
        self._picam1.start()

        logger.info("Both cameras opened and started.")

    def capture_frames(self) -> tuple[CaptureResult | None, CaptureResult | None]:
        """
        Capture one frame from each camera simultaneously and apply motion gate.

        Both cameras are captured in sequence (picamera2 does not support
        true hardware-synchronized simultaneous capture on Pi 5 via CSI —
        frames are within ~8ms of each other at 120fps, sufficient for
        classification and stereo estimation at feeder distances).

        Returns:
            Tuple of (primary_result, secondary_result).
            Either element is None if the camera's frame was below the
            motion threshold — caller should skip classification for that camera.
            Both None means the feeder is empty this cycle.
        """
        if self._picam is None and not hasattr(self, "_picam0"):
            self._open_cameras()

        try:
            raw0 = self._picam0.capture_array("main")  # (H, W, 3) uint8 RGB
            raw1 = self._picam1.capture_array("main")  # (H, W, 3) uint8 RGB
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
        Apply crop, motion gate, save, and preprocess a single raw frame.

        The cropped frame is saved to disk for notifications and the observation
        log. The full-resolution raw_frame is preserved on CaptureResult in
        memory for Phase 6 stereo estimation.

        Args:
            raw_frame:    Full resolution (H, W, 3) uint8 RGB from picamera2.
            camera_index: Which camera this frame came from.

        Returns:
            CaptureResult if frame passes motion gate, else None.
        """
        # ── Apply feeder crop ─────────────────────────────────────────────────
        cropped = raw_frame[
            self.crop_y : self.crop_y + self.crop_height,
            self.crop_x : self.crop_x + self.crop_width,
        ]

        # ── Motion gate ───────────────────────────────────────────────────────
        motion_score, bg = self._compute_motion(cropped, camera_index)
        self._update_background(cropped, camera_index, bg)

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

        # ── Save cropped frame ────────────────────────────────────────────────
        # Save the cropped frame — this is what the classifier saw and what
        # gets attached to Pushover notifications. Full-resolution raw_frame
        # stays in memory on CaptureResult for Phase 6 stereo estimation.
        image_path = self._save_frame(cropped, camera_index)

        # ── Preprocess for classification ─────────────────────────────────────
        preprocessed = preprocess_frame(
            cropped,
            width=self.classification_width,
            height=self.classification_height,
        )

        return CaptureResult(
            frame=preprocessed,
            raw_frame=raw_frame,  # full resolution preserved in memory for Phase 6
            camera_index=camera_index,
            image_path=image_path,
            motion_score=motion_score,
        )

    def _compute_motion(
        self,
        frame: np.ndarray,
        camera_index: int,
    ) -> tuple[float, np.ndarray | None]:
        """
        Compute mean absolute pixel difference from the background model.

        On the first background_history frames we return motion_threshold + 1
        so that initial frames are always passed through to build the background
        model before we start gating.

        Args:
            frame:        Cropped uint8 RGB frame.
            camera_index: Determines which background model to use.

        Returns:
            Tuple of (motion_score, current_background).
            motion_score is mean abs diff normalized to [0, 1].
            current_background is the current model array, or None if not built.
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

        Args:
            frame:        Current cropped frame (uint8).
            camera_index: Which camera's model to update.
            current_bg:   Current background model, or None if not yet initialized.
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

        Saves the post-crop frame (crop_width × crop_height) rather than the
        full-resolution capture. This keeps file sizes small (typically 50-200KB
        vs 1-3MB for full resolution), stays well within Pushover's 2.5MB
        attachment limit, and represents exactly what the classifier saw.

        Uses PIL for PNG encoding — no OpenCV dependency required.
        output_dir is absolute (resolved in __init__) so the returned path
        is always absolute regardless of working directory.

        Args:
            cropped_frame: Cropped (crop_height × crop_width, 3) uint8 RGB frame.
            camera_index:  Used in filename to distinguish cameras.

        Returns:
            Absolute Path to the saved file, or None if save fails (non-fatal).
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
        Stop both cameras and release picamera2 resources.

        Call when the agent loop terminates to cleanly shut down the hardware.
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
