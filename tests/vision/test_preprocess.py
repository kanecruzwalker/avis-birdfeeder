"""
tests/vision/test_preprocess.py

Unit tests for src.vision.preprocess.

Design principles:
    - Zero hardware dependencies — all images are generated synthetically with NumPy.
    - No real image files required — file-based tests write to a temporary directory
      via pytest's tmp_path fixture so nothing is left on disk after the run.
    - Tests validate behavior, output shape, dtype, and value range — not exact
      pixel values — so they remain valid as PIL versions evolve.
    - Each test covers one clearly-named contract, making CI failure messages
      immediately actionable.

Synthetic image strategy:
    We generate random uint8 RGB arrays using np.random.randint. This gives us
    full control over shape, dtype, and value range without needing any real
    bird images. The preprocessing math (resize, normalize) works identically
    on random noise as on real images for the purposes of shape/range testing.

    For file-based tests, we save synthetic images as PNG via PIL and load them
    back — this exercises the full disk I/O path without any dataset dependency.

Markers:
    No special markers needed — all tests run on any machine with the venv active.
    Hardware-dependent tests (real camera frames) are deferred to Phase 5 and
    will be marked @pytest.mark.hardware.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.vision.preprocess import (
    _DEFAULT_HEIGHT,
    _DEFAULT_WIDTH,
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    augment,
    load_image,
    normalize,
    preprocess_file,
    preprocess_frame,
    resize,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic image generation
# ---------------------------------------------------------------------------

_H, _W = 480, 640  # typical camera resolution to start from


def _random_frame(height: int = _H, width: int = _W) -> np.ndarray:
    """
    Generate a random uint8 RGB frame using NumPy.

    Args:
        height: Frame height in pixels.
        width:  Frame width in pixels.

    Returns:
        NumPy array of shape (height, width, 3), dtype uint8, values in [0, 255].
    """
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


def _solid_frame(height: int = _H, width: int = _W, value: int = 128) -> np.ndarray:
    """
    Generate a solid-color uint8 RGB frame (all pixels the same value).

    Useful for testing normalization math precisely, since the expected
    output value is predictable for a constant input.

    Args:
        height: Frame height in pixels.
        width:  Frame width in pixels.
        value:  Pixel value in [0, 255] applied to all channels.

    Returns:
        NumPy array of shape (height, width, 3), dtype uint8.
    """
    return np.full((height, width, 3), value, dtype=np.uint8)


def _write_png(path: Path, frame: np.ndarray) -> None:
    """Save a uint8 RGB NumPy array as a PNG file using PIL."""
    Image.fromarray(frame).save(str(path))


# ---------------------------------------------------------------------------
# Tests: resize
# ---------------------------------------------------------------------------


class TestResize:
    """Tests for resize() — spatial downsampling to model input size."""

    def test_output_shape_matches_target(self) -> None:
        """resize returns an array with exactly (height, width, 3) shape."""
        frame = _random_frame()
        result = resize(frame, width=224, height=224)
        assert result.shape == (224, 224, 3)

    def test_output_dtype_is_uint8(self) -> None:
        """resize preserves uint8 dtype — normalization happens separately."""
        frame = _random_frame()
        result = resize(frame)
        assert result.dtype == np.uint8

    def test_custom_target_size(self) -> None:
        """resize respects arbitrary target dimensions, not just 224×224."""
        frame = _random_frame()
        result = resize(frame, width=128, height=96)
        assert result.shape == (96, 128, 3)

    def test_channels_preserved(self) -> None:
        """resize always produces exactly 3 channels."""
        frame = _random_frame()
        result = resize(frame)
        assert result.shape[2] == 3

    def test_already_correct_size_is_unchanged(self) -> None:
        """A frame already at target size passes through with the same shape."""
        frame = _random_frame(height=224, width=224)
        result = resize(frame, width=224, height=224)
        assert result.shape == (224, 224, 3)

    def test_raises_on_non_3channel_input(self) -> None:
        """resize raises ValueError if the input does not have 3 channels."""
        grayscale = np.zeros((224, 224, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="3 channels"):
            resize(grayscale)

    def test_raises_on_2d_input(self) -> None:
        """resize raises ValueError if the input is 2-D (no channel dimension)."""
        flat = np.zeros((224, 224), dtype=np.uint8)
        with pytest.raises(ValueError, match="3 channels"):
            resize(flat)

    def test_upsample_works(self) -> None:
        """resize handles upsampling (small → large) without error."""
        small = _random_frame(height=32, width=32)
        result = resize(small, width=224, height=224)
        assert result.shape == (224, 224, 3)


# ---------------------------------------------------------------------------
# Tests: normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    """Tests for normalize() — ImageNet standardization to float32."""

    def test_output_dtype_is_float32(self) -> None:
        """normalize always returns float32."""
        frame = _random_frame(height=224, width=224)
        result = normalize(frame)
        assert result.dtype == np.float32

    def test_output_shape_unchanged(self) -> None:
        """normalize does not change spatial dimensions or channel count."""
        frame = _random_frame(height=224, width=224)
        result = normalize(frame)
        assert result.shape == frame.shape

    def test_zero_pixel_maps_to_negative(self) -> None:
        """
        A fully black frame (all zeros) should normalize to negative values.

        After scaling: 0 / 255 = 0.0
        After normalization: (0.0 - mean) / std — all negative since mean > 0.
        """
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        result = normalize(frame)
        assert np.all(result < 0)

    def test_255_pixel_maps_to_positive(self) -> None:
        """
        A fully white frame (all 255s) should normalize to positive values.

        After scaling: 255 / 255 = 1.0
        After normalization: (1.0 - mean) / std — all positive since mean < 1.
        """
        frame = np.full((224, 224, 3), 255, dtype=np.uint8)
        result = normalize(frame)
        assert np.all(result > 0)

    def test_solid_128_frame_close_to_zero(self) -> None:
        """
        A mid-gray frame (all 128s) should normalize close to zero per channel.

        128 / 255 ≈ 0.502, which is close to the ImageNet mean (~0.45 avg).
        Result won't be exactly zero but should be in a small range around it.
        """
        frame = _solid_frame(height=224, width=224, value=128)
        result = normalize(frame)
        assert np.all(np.abs(result) < 2.0)

    def test_no_nan_or_inf(self) -> None:
        """normalize output must be finite — NaN/Inf would crash model inference."""
        frame = _random_frame(height=224, width=224)
        result = normalize(frame)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_custom_mean_and_std(self) -> None:
        """normalize respects custom mean/std — not hardcoded to ImageNet values."""
        frame = _solid_frame(height=4, width=4, value=255)
        # With mean=1.0 and std=1.0: (1.0 - 1.0) / 1.0 = 0.0
        result = normalize(frame, mean=(1.0, 1.0, 1.0), std=(1.0, 1.0, 1.0))
        assert np.allclose(result, 0.0, atol=1e-5)

    def test_raises_on_non_3channel_input(self) -> None:
        """normalize raises ValueError for non-3-channel input."""
        bad = np.zeros((224, 224, 1), dtype=np.uint8)
        with pytest.raises(ValueError):
            normalize(bad)

    def test_math_correctness_single_pixel(self) -> None:
        """
        Verify the exact normalization formula on a known single-pixel input.

        For pixel value 128 in channel R:
            x = 128 / 255 ≈ 0.50196
            y = (0.50196 - 0.485) / 0.229 ≈ 0.0742
        """
        frame = np.full((1, 1, 3), 128, dtype=np.uint8)
        result = normalize(frame)
        expected_r = (128.0 / 255.0 - _IMAGENET_MEAN[0]) / _IMAGENET_STD[0]
        assert np.isclose(result[0, 0, 0], expected_r, atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: augment
# ---------------------------------------------------------------------------


class TestAugment:
    """Tests for augment() — training-time data augmentation."""

    def test_output_shape_unchanged(self) -> None:
        """augment always returns the same shape as the input."""
        frame = _random_frame(height=224, width=224)
        result = augment(frame)
        assert result.shape == frame.shape

    def test_output_dtype_is_uint8(self) -> None:
        """augment preserves uint8 dtype — normalization happens after."""
        frame = _random_frame(height=224, width=224)
        result = augment(frame)
        assert result.dtype == np.uint8

    def test_output_values_in_valid_range(self) -> None:
        """augment never produces pixel values outside [0, 255]."""
        frame = _random_frame(height=224, width=224)
        result = augment(frame)
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_does_not_mutate_input(self) -> None:
        """augment works on a copy — the original frame is not modified."""
        frame = _random_frame(height=224, width=224)
        original = frame.copy()
        augment(frame)
        np.testing.assert_array_equal(frame, original)

    def test_no_nan_or_inf(self) -> None:
        """augment output must be finite."""
        frame = _random_frame(height=224, width=224)
        result = augment(frame)
        assert not np.any(np.isnan(result.astype(np.float32)))
        assert not np.any(np.isinf(result.astype(np.float32)))

    def test_augment_disabled_flags(self) -> None:
        """With both augmentations disabled, output equals input exactly."""
        frame = _random_frame(height=224, width=224)
        result = augment(frame, horizontal_flip=False, color_jitter=False)
        np.testing.assert_array_equal(result, frame)


# ---------------------------------------------------------------------------
# Tests: preprocess_frame (integration — exercises resize → normalize chain)
# ---------------------------------------------------------------------------


class TestPreprocessFrame:
    """Integration tests for preprocess_frame() — full in-memory pipeline."""

    def test_returns_float32_array(self) -> None:
        """preprocess_frame returns float32."""
        frame = _random_frame()
        result = preprocess_frame(frame)
        assert result.dtype == np.float32

    def test_output_shape_is_hwc_224(self) -> None:
        """preprocess_frame output is (224, 224, 3) by default."""
        frame = _random_frame()
        result = preprocess_frame(frame)
        assert result.shape == (_DEFAULT_HEIGHT, _DEFAULT_WIDTH, 3)

    def test_custom_target_size(self) -> None:
        """preprocess_frame respects custom width and height."""
        frame = _random_frame()
        result = preprocess_frame(frame, width=128, height=128)
        assert result.shape == (128, 128, 3)

    def test_no_nan_or_inf(self) -> None:
        """Full pipeline output must be finite."""
        frame = _random_frame()
        result = preprocess_frame(frame)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_augment_flag_does_not_change_shape(self) -> None:
        """preprocess_frame with augment_=True still returns (224, 224, 3)."""
        frame = _random_frame()
        result = preprocess_frame(frame, augment_=True)
        assert result.shape == (_DEFAULT_HEIGHT, _DEFAULT_WIDTH, 3)
        assert result.dtype == np.float32

    def test_values_in_expected_range(self) -> None:
        """
        Normalized output should be in roughly [-3, 3].

        ImageNet normalization maps [0, 255] → roughly [-2.1, 2.6].
        We use a generous ±3.5 bound to account for floating point.
        """
        frame = _random_frame()
        result = preprocess_frame(frame)
        assert result.min() > -3.5
        assert result.max() < 3.5


# ---------------------------------------------------------------------------
# Tests: load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    """Tests for load_image() — disk-based frame loading."""

    def test_returns_uint8_array(self, tmp_path: Path) -> None:
        """load_image returns uint8."""
        img_path = tmp_path / "bird.png"
        _write_png(img_path, _random_frame())
        result = load_image(img_path)
        assert result.dtype == np.uint8

    def test_returns_3channel_rgb(self, tmp_path: Path) -> None:
        """load_image always returns 3-channel RGB regardless of source format."""
        img_path = tmp_path / "bird.png"
        _write_png(img_path, _random_frame())
        result = load_image(img_path)
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_shape_matches_saved_image(self, tmp_path: Path) -> None:
        """load_image returns the same spatial dimensions as the saved image."""
        frame = _random_frame(height=100, width=150)
        img_path = tmp_path / "bird.png"
        _write_png(img_path, frame)
        result = load_image(img_path)
        assert result.shape == (100, 150, 3)

    def test_accepts_path_object_and_string(self, tmp_path: Path) -> None:
        """load_image accepts both Path objects and plain strings."""
        frame = _random_frame()
        img_path = tmp_path / "bird.png"
        _write_png(img_path, frame)
        result_path = load_image(img_path)
        result_str = load_image(str(img_path))
        np.testing.assert_array_equal(result_path, result_str)

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        """load_image raises FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError, match="Image file not found"):
            load_image(tmp_path / "ghost.png")

    def test_grayscale_converted_to_rgb(self, tmp_path: Path) -> None:
        """load_image converts grayscale images to 3-channel RGB."""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        img_path = tmp_path / "gray.png"
        Image.fromarray(gray, mode="L").save(str(img_path))
        result = load_image(img_path)
        assert result.shape == (100, 100, 3)


# ---------------------------------------------------------------------------
# Tests: preprocess_file (integration — disk → full pipeline)
# ---------------------------------------------------------------------------


class TestPreprocessFile:
    """Integration tests for preprocess_file() — full disk-based pipeline."""

    def test_returns_float32_array(self, tmp_path: Path) -> None:
        """preprocess_file returns float32."""
        img_path = tmp_path / "bird.png"
        _write_png(img_path, _random_frame())
        result = preprocess_file(img_path)
        assert result.dtype == np.float32

    def test_output_shape_is_default_hwc(self, tmp_path: Path) -> None:
        """preprocess_file output is (224, 224, 3) by default."""
        img_path = tmp_path / "bird.png"
        _write_png(img_path, _random_frame())
        result = preprocess_file(img_path)
        assert result.shape == (_DEFAULT_HEIGHT, _DEFAULT_WIDTH, 3)

    def test_no_nan_or_inf(self, tmp_path: Path) -> None:
        """Full disk pipeline output must be finite."""
        img_path = tmp_path / "bird.png"
        _write_png(img_path, _random_frame())
        result = preprocess_file(img_path)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        """preprocess_file propagates FileNotFoundError for missing input."""
        with pytest.raises(FileNotFoundError):
            preprocess_file(tmp_path / "ghost.png")

    def test_accepts_path_object_and_string(self, tmp_path: Path) -> None:
        """preprocess_file accepts both Path objects and plain strings."""
        img_path = tmp_path / "bird.png"
        _write_png(img_path, _random_frame())
        result_path = preprocess_file(img_path)
        result_str = preprocess_file(str(img_path))
        assert result_path.shape == result_str.shape
