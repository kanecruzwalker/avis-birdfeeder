"""
tests/vision/test_capture.py

Unit tests for src.vision.capture — VisionCapture and CaptureResult.

Design principles:
    - Zero hardware dependencies — picamera2 is never imported.
    - All tests construct VisionCapture directly via __init__ (not from_config)
      so no real config files are required beyond what tmp_path provides.
    - _process_frame(), _compute_motion(), _update_background(), and
      _save_frame() are all tested directly — they are pure numpy/PIL logic
      with no hardware dependency.
    - capture_frames() is not tested here — it requires picamera2 and is
      exercised by integration tests on the Pi.
    - Synthetic frames use np.random.randint (same strategy as test_preprocess.py).

Markers:
    No special markers — all tests run on any machine with the venv active.
    Pi-hardware tests (capture_frames with real cameras) are marked
    @pytest.mark.hardware and excluded from laptop CI.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.vision.capture import CaptureResult, VisionCapture

# ── Helpers ───────────────────────────────────────────────────────────────────

# Matches hardware.yaml defaults
_CAPTURE_W, _CAPTURE_H = 1536, 864
_CROP_X, _CROP_Y, _CROP_W, _CROP_H = 568, 232, 400, 400
_CLASS_W, _CLASS_H = 224, 224


def _make_capture(tmp_path: Path, **kwargs) -> VisionCapture:
    """
    Build a VisionCapture with sensible defaults and a tmp output dir.

    All picamera2 calls are lazy — constructing VisionCapture never touches
    hardware. Tests that need _process_frame() call it directly with synthetic
    frames rather than going through capture_frames().
    """
    defaults = dict(
        primary_index=0,
        secondary_index=1,
        capture_width=_CAPTURE_W,
        capture_height=_CAPTURE_H,
        capture_fps=120,
        classification_width=_CLASS_W,
        classification_height=_CLASS_H,
        crop_x=_CROP_X,
        crop_y=_CROP_Y,
        crop_width=_CROP_W,
        crop_height=_CROP_H,
        motion_threshold=0.05,
        background_history=30,
        output_dir=str(tmp_path / "captures"),
    )
    defaults.update(kwargs)
    return VisionCapture(**defaults)


def _raw_frame(h: int = _CAPTURE_H, w: int = _CAPTURE_W) -> np.ndarray:
    """Synthetic full-resolution uint8 RGB frame."""
    return np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def _solid_frame(h: int = _CAPTURE_H, w: int = _CAPTURE_W, value: int = 128) -> np.ndarray:
    """Solid-colour frame — useful for predictable motion score tests."""
    return np.full((h, w, 3), value, dtype=np.uint8)


# ── TestVisionCaptureInit ─────────────────────────────────────────────────────


class TestVisionCaptureInit:
    def test_stores_primary_index(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path, primary_index=0)
        assert vc.primary_index == 0

    def test_stores_secondary_index(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path, secondary_index=1)
        assert vc.secondary_index == 1

    def test_stores_crop_params(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path, crop_x=10, crop_y=20, crop_width=300, crop_height=300)
        assert vc.crop_x == 10
        assert vc.crop_y == 20
        assert vc.crop_width == 300
        assert vc.crop_height == 300

    def test_stores_motion_threshold(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path, motion_threshold=0.08)
        assert vc.motion_threshold == pytest.approx(0.08)

    def test_output_dir_is_absolute(self, tmp_path: Path) -> None:
        """output_dir must be absolute so image_path on CaptureResult is portable."""
        vc = _make_capture(tmp_path, output_dir="data/captures/images")
        assert vc.output_dir.is_absolute()

    def test_output_dir_created(self, tmp_path: Path) -> None:
        out = tmp_path / "new" / "nested" / "dir"
        _make_capture(tmp_path, output_dir=str(out))
        assert out.exists()

    def test_background_models_none_on_init(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        assert vc._bg_primary is None
        assert vc._bg_secondary is None

    def test_bg_count_zero_on_init(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        assert vc._bg_count == 0

    def test_from_config_constructs(self) -> None:
        """from_config reads hardware.yaml and paths.yaml without error."""
        vc = VisionCapture.from_config("configs/")
        assert vc.primary_index == 0
        assert vc.secondary_index == 1

    def test_from_config_output_dir_is_absolute(self) -> None:
        vc = VisionCapture.from_config("configs/")
        assert vc.output_dir.is_absolute()


# ── TestHailoVDeviceCreation ──────────────────────────────────────────────────


class TestHailoVDeviceCreation:
    """
    Verifies that shared VDevice creation is gated on hailo_enabled, not on
    detection_mode. Before Branch 1, the VDevice was created only when
    detection_mode == "yolo", which meant fixed_crop runs couldn't share a
    VDevice with the classifier. After Branch 1, the VDevice follows the
    Hailo availability flag directly.

    These tests pass regardless of whether Hailo hardware is present because
    the conditional in __init__ checks hailo_enabled *before* attempting
    VDevice creation. If Hailo is not installed, the create call inside
    the try/except fails gracefully and _shared_vdevice stays None.
    """

    def test_no_vdevice_when_hailo_disabled(self, tmp_path: Path) -> None:
        """When hailo_enabled=False, _shared_vdevice must be None regardless of detection_mode."""
        vc = _make_capture(tmp_path, hailo_enabled=False, detection_mode="fixed_crop")
        assert vc._shared_vdevice is None

    def test_no_vdevice_when_hailo_disabled_even_in_yolo_mode(self, tmp_path: Path) -> None:
        """Old code created VDevice in yolo mode regardless of enable flag — verify fix."""
        vc = _make_capture(tmp_path, hailo_enabled=False, detection_mode="yolo")
        assert vc._shared_vdevice is None

    def test_hailo_enabled_stored(self, tmp_path: Path) -> None:
        """hailo_enabled parameter is stored on the instance for later inspection."""
        vc = _make_capture(tmp_path, hailo_enabled=True)
        assert vc.hailo_enabled is True

    def test_hailo_disabled_default(self, tmp_path: Path) -> None:
        """Default for hailo_enabled is False — safe default for dev environments."""
        vc = _make_capture(tmp_path)
        assert vc.hailo_enabled is False

    def test_from_config_reads_hailo_enabled(self) -> None:
        """from_config reads hardware.yaml:hailo.enabled and propagates to __init__."""
        vc = VisionCapture.from_config("configs/")
        # Current Pi config has hailo.enabled: true, but on dev laptops it's typically false.
        # Either is valid — just verify the attribute exists and is a bool.
        assert isinstance(vc.hailo_enabled, bool)


# ── TestSaveFrame ─────────────────────────────────────────────────────────────


class TestSaveFrame:
    def test_returns_path_on_success(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        result = vc._save_frame(frame, camera_index=0)
        assert result is not None
        assert isinstance(result, Path)

    def test_saved_file_exists(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        path = vc._save_frame(frame, camera_index=0)
        assert path is not None
        assert path.exists()

    def test_saved_path_is_absolute(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        path = vc._save_frame(frame, camera_index=0)
        assert path is not None
        assert path.is_absolute()

    def test_filename_contains_camera_index(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        path = vc._save_frame(frame, camera_index=1)
        assert path is not None
        assert "cam1" in path.name

    def test_saved_file_is_png(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        path = vc._save_frame(frame, camera_index=0)
        assert path is not None
        assert path.suffix == ".png"

    def test_saved_image_dimensions_match_cropped_frame(self, tmp_path: Path) -> None:
        """Saved PNG dimensions must match the cropped frame, not full resolution."""
        vc = _make_capture(tmp_path)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        path = vc._save_frame(frame, camera_index=0)
        assert path is not None
        img = Image.open(path)
        assert img.size == (_CROP_W, _CROP_H)  # PIL size is (width, height)

    def test_returns_none_on_pil_failure(self, tmp_path: Path) -> None:
        """If PIL save fails, _save_frame returns None rather than raising."""
        vc = _make_capture(tmp_path)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        with patch("PIL.Image.Image.save", side_effect=OSError("disk full")):
            result = vc._save_frame(frame, camera_index=0)
        assert result is None

    def test_two_saves_produce_distinct_paths(self, tmp_path: Path) -> None:
        """Each save generates a unique timestamped filename."""
        vc = _make_capture(tmp_path)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        path_a = vc._save_frame(frame, camera_index=0)
        path_b = vc._save_frame(frame, camera_index=0)
        assert path_a != path_b


# ── TestComputeMotion ─────────────────────────────────────────────────────────


class TestComputeMotion:
    def test_returns_above_threshold_before_bg_built(self, tmp_path: Path) -> None:
        """Before background_history frames, score always exceeds threshold."""
        vc = _make_capture(tmp_path, background_history=30, motion_threshold=0.05)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        score, _ = vc._compute_motion(frame, camera_index=0)
        assert score > vc.motion_threshold

    def test_returns_low_score_for_identical_frame(self, tmp_path: Path) -> None:
        """A frame identical to the background model should score near zero."""
        vc = _make_capture(tmp_path, background_history=1)
        frame = _solid_frame(_CROP_H, _CROP_W, value=100)
        # Build background from the same frame
        vc._update_background(frame, camera_index=0, current_bg=None)
        score, _ = vc._compute_motion(frame, camera_index=0)
        assert score < 0.01

    def test_returns_high_score_for_different_frame(self, tmp_path: Path) -> None:
        """A frame very different from background should score high."""
        vc = _make_capture(tmp_path, background_history=1)
        bg_frame = _solid_frame(_CROP_H, _CROP_W, value=0)
        diff_frame = _solid_frame(_CROP_H, _CROP_W, value=255)
        vc._update_background(bg_frame, camera_index=0, current_bg=None)
        score, _ = vc._compute_motion(diff_frame, camera_index=0)
        assert score > 0.9

    def test_primary_and_secondary_use_independent_models(self, tmp_path: Path) -> None:
        """Primary and secondary cameras maintain separate background models."""
        vc = _make_capture(tmp_path, background_history=1)
        bg0 = _solid_frame(_CROP_H, _CROP_W, value=50)
        bg1 = _solid_frame(_CROP_H, _CROP_W, value=200)
        vc._update_background(bg0, camera_index=0, current_bg=None)
        vc._update_background(bg1, camera_index=1, current_bg=None)

        # Primary's score against its own background — should be low
        score0, _ = vc._compute_motion(bg0, camera_index=0)
        # Secondary's score against its own background — should be low
        score1, _ = vc._compute_motion(bg1, camera_index=1)
        assert score0 < 0.01
        assert score1 < 0.01


# ── TestUpdateBackground ──────────────────────────────────────────────────────


class TestUpdateBackground:
    def test_initialises_bg_from_none(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        vc._update_background(frame, camera_index=0, current_bg=None)
        assert vc._bg_primary is not None

    def test_bg_shape_matches_frame(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        vc._update_background(frame, camera_index=0, current_bg=None)
        assert vc._bg_primary is not None
        assert vc._bg_primary.shape == (_CROP_H, _CROP_W, 3)

    def test_bg_dtype_is_float32(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        vc._update_background(frame, camera_index=0, current_bg=None)
        assert vc._bg_primary is not None
        assert vc._bg_primary.dtype == np.float32

    def test_bg_count_increments(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        vc._update_background(frame, camera_index=0, current_bg=None)
        vc._update_background(frame, camera_index=0, current_bg=vc._bg_primary)
        assert vc._bg_count == 2

    def test_secondary_bg_updated_independently(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        frame = np.random.randint(0, 256, (_CROP_H, _CROP_W, 3), dtype=np.uint8)
        vc._update_background(frame, camera_index=1, current_bg=None)
        assert vc._bg_secondary is not None
        assert vc._bg_primary is None


# ── TestProcessFrame ──────────────────────────────────────────────────────────


class TestProcessFrame:
    def test_returns_none_after_background_built_for_static_scene(self, tmp_path: Path) -> None:
        """Static scene — after background is built, same frame should be suppressed."""
        vc = _make_capture(tmp_path, background_history=1, motion_threshold=0.05)
        frame = _solid_frame(value=128)
        # Build background from this frame
        vc._update_background(
            frame[_CROP_Y : _CROP_Y + _CROP_H, _CROP_X : _CROP_X + _CROP_W],
            camera_index=0,
            current_bg=None,
        )
        vc._bg_count = 1  # mark as built

        result = vc._process_frame(frame, camera_index=0)
        assert result is None

    def test_returns_capture_result_before_bg_built(self, tmp_path: Path) -> None:
        """Before background is built, every frame passes the motion gate."""
        vc = _make_capture(tmp_path, background_history=30)
        frame = _raw_frame()
        result = vc._process_frame(frame, camera_index=0)
        assert result is not None
        assert isinstance(result, CaptureResult)

    def test_result_frame_shape_is_224(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        result = vc._process_frame(_raw_frame(), camera_index=0)
        assert result is not None
        assert result.frame.shape == (_CLASS_H, _CLASS_W, 3)

    def test_result_frame_dtype_is_float32(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        result = vc._process_frame(_raw_frame(), camera_index=0)
        assert result is not None
        assert result.frame.dtype == np.float32

    def test_result_raw_frame_is_full_resolution(self, tmp_path: Path) -> None:
        """raw_frame on CaptureResult must be the original full-resolution frame."""
        vc = _make_capture(tmp_path)
        frame = _raw_frame()
        result = vc._process_frame(frame, camera_index=0)
        assert result is not None
        assert result.raw_frame.shape == (_CAPTURE_H, _CAPTURE_W, 3)

    def test_result_camera_index_preserved(self, tmp_path: Path) -> None:
        vc = _make_capture(tmp_path)
        result = vc._process_frame(_raw_frame(), camera_index=1)
        assert result is not None
        assert result.camera_index == 1

    def test_result_image_path_is_absolute(self, tmp_path: Path) -> None:
        """image_path must be absolute so notifier can find it from any cwd."""
        vc = _make_capture(tmp_path)
        result = vc._process_frame(_raw_frame(), camera_index=0)
        assert result is not None
        assert result.image_path is not None
        assert result.image_path.is_absolute()

    def test_saved_image_is_cropped_size(self, tmp_path: Path) -> None:
        """The file on disk must be crop-sized, not full-resolution."""
        vc = _make_capture(tmp_path)
        result = vc._process_frame(_raw_frame(), camera_index=0)
        assert result is not None
        assert result.image_path is not None
        img = Image.open(result.image_path)
        assert img.size == (_CROP_W, _CROP_H)

    def test_image_path_none_when_save_fails(self, tmp_path: Path) -> None:
        """If _save_frame fails, image_path is None but result is still returned."""
        vc = _make_capture(tmp_path)
        with patch.object(vc, "_save_frame", return_value=None):
            result = vc._process_frame(_raw_frame(), camera_index=0)
        assert result is not None
        assert result.image_path is None

    def test_background_updated_even_when_below_threshold(self, tmp_path: Path) -> None:
        """Background model must update on every frame, not just passing ones."""
        vc = _make_capture(tmp_path, background_history=1, motion_threshold=0.05)
        frame = _solid_frame(value=128)
        cropped = frame[_CROP_Y : _CROP_Y + _CROP_H, _CROP_X : _CROP_X + _CROP_W]
        # Pre-build background so motion gate suppresses this frame
        vc._update_background(cropped, camera_index=0, current_bg=None)
        vc._bg_count = 1

        count_before = vc._bg_count
        vc._process_frame(frame, camera_index=0)
        assert vc._bg_count > count_before


# ── TestCaptureResult ─────────────────────────────────────────────────────────


class TestCaptureResult:
    def test_can_construct_directly(self, tmp_path: Path) -> None:
        result = CaptureResult(
            frame=np.zeros((_CLASS_H, _CLASS_W, 3), dtype=np.float32),
            raw_frame=np.zeros((_CAPTURE_H, _CAPTURE_W, 3), dtype=np.uint8),
            camera_index=0,
            image_path=tmp_path / "frame.png",
            motion_score=0.12,
        )
        assert result.camera_index == 0
        assert result.motion_score == pytest.approx(0.12)

    def test_image_path_can_be_none(self) -> None:
        result = CaptureResult(
            frame=np.zeros((_CLASS_H, _CLASS_W, 3), dtype=np.float32),
            raw_frame=np.zeros((_CAPTURE_H, _CAPTURE_W, 3), dtype=np.uint8),
            camera_index=1,
            image_path=None,
            motion_score=0.07,
        )
        assert result.image_path is None
