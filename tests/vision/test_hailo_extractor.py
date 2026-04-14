"""
tests/vision/test_hailo_extractor.py

Unit tests for src.vision.hailo_extractor.HailoVisualExtractor.

Design principles (matching existing test style):
    - Zero hardware dependencies in CI — hailo_platform is mocked throughout.
    - Tests marked @pytest.mark.hardware require a Pi with HailoRT and the
      compiled HEF. They are skipped in CI unless HAILO_AVAILABLE=1 is set.
    - All mocks are minimal and transparent — they test the class interface
      and error handling, not the Hailo SDK internals.

Hardware tests run on Pi:
    pytest tests/vision/test_hailo_extractor.py -m hardware -v

CI tests (no hardware):
    pytest tests/vision/test_hailo_extractor.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_dummy_frame(val: int = 128) -> np.ndarray:
    """Synthetic (224, 224, 3) uint8 RGB frame."""
    return np.full((224, 224, 3), val, dtype=np.uint8)


def _mock_hef_info(input_shape=(224, 224, 3), output_shape=(1280,)):
    """Mock HEF with correct shapes."""
    mock_input = MagicMock()
    mock_input.name = "efficientnet_b0_avis/input_layer1"
    mock_input.shape = input_shape

    mock_output = MagicMock()
    mock_output.name = "efficientnet_b0_avis/avgpool17"
    mock_output.shape = output_shape

    mock_hef = MagicMock()
    mock_hef.get_network_group_names.return_value = ["efficientnet_b0_avis"]
    mock_hef.get_input_vstream_infos.return_value = [mock_input]
    mock_hef.get_output_vstream_infos.return_value = [mock_output]
    return mock_hef


def _make_mock_vdevice_context():
    """Full mock of the VDevice + InferModel context chain."""
    mock_configured_model = MagicMock()
    mock_configured_model.__enter__ = MagicMock(return_value=mock_configured_model)
    mock_configured_model.__exit__ = MagicMock(return_value=False)

    mock_bindings = MagicMock()
    mock_configured_model.create_bindings.return_value = mock_bindings

    def mock_run(bindings_list, timeout):
        # Simulate real inference: fill output buffer with non-zero values
        for b in bindings_list:
            buf = b.output().get_buffer()
            if buf is not None and hasattr(buf, "__setitem__"):
                buf[:] = 42

    mock_configured_model.run.side_effect = mock_run

    mock_infer_model = MagicMock()
    mock_infer_model.configure.return_value = mock_configured_model

    mock_vdevice = MagicMock()
    mock_vdevice.__enter__ = MagicMock(return_value=mock_vdevice)
    mock_vdevice.__exit__ = MagicMock(return_value=False)
    mock_vdevice.create_infer_model.return_value = mock_infer_model

    return mock_vdevice, mock_configured_model, mock_bindings


# ── Construction tests ────────────────────────────────────────────────────────


class TestHailoVisualExtractorInit:
    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_extractor.HEF")
    def test_init_valid_hef(self, mock_hef_cls, tmp_path):
        """Constructor reads HEF metadata and stores shapes."""
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        mock_hef_cls.return_value = _mock_hef_info()

        from src.vision.hailo_extractor import HailoVisualExtractor

        extractor = HailoVisualExtractor(str(hef_path))

        assert extractor.hef_path == hef_path
        assert extractor._input_shape == (224, 224, 3)
        assert extractor._output_shape == (1280,)
        assert not extractor.is_open

    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    def test_init_missing_hef_raises(self, tmp_path):
        """FileNotFoundError if HEF file does not exist."""
        from src.vision.hailo_extractor import HailoVisualExtractor

        with pytest.raises(FileNotFoundError, match="HEF file not found"):
            HailoVisualExtractor(str(tmp_path / "nonexistent.hef"))

    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", False)
    def test_init_no_hailo_raises(self, tmp_path):
        """RuntimeError when hailo_platform is not installed."""
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")

        from src.vision.hailo_extractor import HailoVisualExtractor

        with pytest.raises(RuntimeError, match="hailo_platform is not installed"):
            HailoVisualExtractor(str(hef_path))

    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_extractor.HEF")
    def test_not_open_on_construction(self, mock_hef_cls, tmp_path):
        """VDevice is not opened at construction time."""
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        mock_hef_cls.return_value = _mock_hef_info()

        from src.vision.hailo_extractor import HailoVisualExtractor

        extractor = HailoVisualExtractor(str(hef_path))
        assert not extractor.is_open


# ── open() / close() / is_open ───────────────────────────────────────────────


class TestHailoVisualExtractorOpen:
    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_extractor.HailoSchedulingAlgorithm")
    @patch("src.vision.hailo_extractor.VDevice")
    @patch("src.vision.hailo_extractor.HEF")
    def test_open_sets_is_open(self, mock_hef_cls, mock_vdevice_cls, mock_algo, tmp_path):
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        mock_hef_cls.return_value = _mock_hef_info()
        mock_vdevice, mock_cm, _ = _make_mock_vdevice_context()
        mock_vdevice_cls.return_value = mock_vdevice

        from src.vision.hailo_extractor import HailoVisualExtractor

        extractor = HailoVisualExtractor(str(hef_path))
        extractor.open()

        assert extractor.is_open

    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_extractor.HailoSchedulingAlgorithm")
    @patch("src.vision.hailo_extractor.VDevice")
    @patch("src.vision.hailo_extractor.HEF")
    def test_close_clears_is_open(self, mock_hef_cls, mock_vdevice_cls, mock_algo, tmp_path):
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        mock_hef_cls.return_value = _mock_hef_info()
        mock_vdevice, _, _ = _make_mock_vdevice_context()
        mock_vdevice_cls.return_value = mock_vdevice

        from src.vision.hailo_extractor import HailoVisualExtractor

        extractor = HailoVisualExtractor(str(hef_path))
        extractor.open()
        extractor.close()

        assert not extractor.is_open

    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_extractor.HailoSchedulingAlgorithm")
    @patch("src.vision.hailo_extractor.VDevice")
    @patch("src.vision.hailo_extractor.HEF")
    def test_double_open_is_safe(self, mock_hef_cls, mock_vdevice_cls, mock_algo, tmp_path):
        """Calling open() twice should not raise."""
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        mock_hef_cls.return_value = _mock_hef_info()
        mock_vdevice, _, _ = _make_mock_vdevice_context()
        mock_vdevice_cls.return_value = mock_vdevice

        from src.vision.hailo_extractor import HailoVisualExtractor

        extractor = HailoVisualExtractor(str(hef_path))
        extractor.open()
        extractor.open()  # should not raise

    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_extractor.HEF")
    def test_double_close_is_safe(self, mock_hef_cls, tmp_path):
        """Calling close() without open() should not raise."""
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        mock_hef_cls.return_value = _mock_hef_info()

        from src.vision.hailo_extractor import HailoVisualExtractor

        extractor = HailoVisualExtractor(str(hef_path))
        extractor.close()  # never opened — should not raise
        extractor.close()  # again — safe


# ── Context manager ───────────────────────────────────────────────────────────


class TestHailoVisualExtractorContextManager:
    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_extractor.HailoSchedulingAlgorithm")
    @patch("src.vision.hailo_extractor.VDevice")
    @patch("src.vision.hailo_extractor.HEF")
    def test_context_manager_opens_and_closes(
        self, mock_hef_cls, mock_vdevice_cls, mock_algo, tmp_path
    ):
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        mock_hef_cls.return_value = _mock_hef_info()
        mock_vdevice, _, _ = _make_mock_vdevice_context()
        mock_vdevice_cls.return_value = mock_vdevice

        from src.vision.hailo_extractor import HailoVisualExtractor

        with HailoVisualExtractor(str(hef_path)) as ext:
            assert ext.is_open
        assert not ext.is_open


# ── extract() input validation ────────────────────────────────────────────────


class TestHailoVisualExtractorExtract:
    def _make_extractor(self, tmp_path):
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        with (
            patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True),
            patch("src.vision.hailo_extractor.HEF") as mock_hef_cls,
        ):
            mock_hef_cls.return_value = _mock_hef_info()
            from src.vision.hailo_extractor import HailoVisualExtractor

            return HailoVisualExtractor(str(hef_path))

    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_extractor.HEF")
    def test_extract_without_open_raises(self, mock_hef_cls, tmp_path):
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        mock_hef_cls.return_value = _mock_hef_info()

        from src.vision.hailo_extractor import HailoVisualExtractor

        extractor = HailoVisualExtractor(str(hef_path))

        with pytest.raises(RuntimeError, match="open\\(\\) must be called"):
            extractor.extract(_make_dummy_frame())

    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_extractor.HailoSchedulingAlgorithm")
    @patch("src.vision.hailo_extractor.VDevice")
    @patch("src.vision.hailo_extractor.HEF")
    def test_extract_wrong_shape_raises(self, mock_hef_cls, mock_vdevice_cls, mock_algo, tmp_path):
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        mock_hef_cls.return_value = _mock_hef_info()
        mock_vdevice, _, _ = _make_mock_vdevice_context()
        mock_vdevice_cls.return_value = mock_vdevice

        from src.vision.hailo_extractor import HailoVisualExtractor

        extractor = HailoVisualExtractor(str(hef_path))
        extractor.open()

        bad_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame shape mismatch"):
            extractor.extract(bad_frame)

    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_extractor.HailoSchedulingAlgorithm")
    @patch("src.vision.hailo_extractor.VDevice")
    @patch("src.vision.hailo_extractor.HEF")
    def test_extract_returns_float32(self, mock_hef_cls, mock_vdevice_cls, mock_algo, tmp_path):
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        mock_hef_cls.return_value = _mock_hef_info()
        mock_vdevice, _, _ = _make_mock_vdevice_context()
        mock_vdevice_cls.return_value = mock_vdevice

        from src.vision.hailo_extractor import HailoVisualExtractor

        extractor = HailoVisualExtractor(str(hef_path))
        extractor.open()

        result = extractor.extract(_make_dummy_frame())
        assert result.dtype == np.float32

    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_extractor.HailoSchedulingAlgorithm")
    @patch("src.vision.hailo_extractor.VDevice")
    @patch("src.vision.hailo_extractor.HEF")
    def test_extract_output_shape(self, mock_hef_cls, mock_vdevice_cls, mock_algo, tmp_path):
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        mock_hef_cls.return_value = _mock_hef_info()
        mock_vdevice, _, _ = _make_mock_vdevice_context()
        mock_vdevice_cls.return_value = mock_vdevice

        from src.vision.hailo_extractor import HailoVisualExtractor

        extractor = HailoVisualExtractor(str(hef_path))
        extractor.open()

        result = extractor.extract(_make_dummy_frame())
        assert result.shape == (1, 1280)

    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_extractor.HailoSchedulingAlgorithm")
    @patch("src.vision.hailo_extractor.VDevice")
    @patch("src.vision.hailo_extractor.HEF")
    def test_extract_output_range(self, mock_hef_cls, mock_vdevice_cls, mock_algo, tmp_path):
        """Output values should be in [0.0, 1.0] after dequantization."""
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        mock_hef_cls.return_value = _mock_hef_info()
        mock_vdevice, _, _ = _make_mock_vdevice_context()
        mock_vdevice_cls.return_value = mock_vdevice

        from src.vision.hailo_extractor import HailoVisualExtractor

        extractor = HailoVisualExtractor(str(hef_path))
        extractor.open()

        result = extractor.extract(_make_dummy_frame())
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @patch("src.vision.hailo_extractor.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_extractor.HailoSchedulingAlgorithm")
    @patch("src.vision.hailo_extractor.VDevice")
    @patch("src.vision.hailo_extractor.HEF")
    def test_extract_float32_input_converted_to_uint8(
        self, mock_hef_cls, mock_vdevice_cls, mock_algo, tmp_path
    ):
        """float32 input is accepted and converted to uint8 internally."""
        hef_path = tmp_path / "test.hef"
        hef_path.write_bytes(b"fakehef")
        mock_hef_cls.return_value = _mock_hef_info()
        mock_vdevice, _, _ = _make_mock_vdevice_context()
        mock_vdevice_cls.return_value = mock_vdevice

        from src.vision.hailo_extractor import HailoVisualExtractor

        extractor = HailoVisualExtractor(str(hef_path))
        extractor.open()

        float_frame = np.full((224, 224, 3), 0.5, dtype=np.float32)
        result = extractor.extract(float_frame)  # should not raise
        assert result.shape == (1, 1280)


# ── Hardware integration tests (Pi only) ─────────────────────────────────────


@pytest.mark.hardware
class TestHailoVisualExtractorHardware:
    """
    Integration tests requiring Hailo hardware and the compiled HEF.
    Run with: pytest tests/vision/test_hailo_extractor.py -m hardware -v

    These tests are skipped in CI unless HAILO_AVAILABLE=1 is set.
    """

    HEF_PATH = Path("models/visual/efficientnet_b0_avis_v2.hef")

    @pytest.fixture(autouse=True)
    def require_hardware(self):
        if os.environ.get("HAILO_AVAILABLE") != "1":
            pytest.skip("Hailo hardware not available (set HAILO_AVAILABLE=1 on Pi)")
        if not self.HEF_PATH.exists():
            pytest.skip(f"HEF not found: {self.HEF_PATH}")

    def test_hardware_extract_returns_correct_shape(self):
        from src.vision.hailo_extractor import HailoVisualExtractor

        with HailoVisualExtractor(str(self.HEF_PATH)) as ext:
            frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            result = ext.extract(frame)
            assert result.shape == (1, 1280)
            assert result.dtype == np.float32

    def test_hardware_extract_nonzero_output(self):
        """Verify hardware produces non-zero activations (not all-zeros corruption)."""
        from src.vision.hailo_extractor import HailoVisualExtractor

        with HailoVisualExtractor(str(self.HEF_PATH)) as ext:
            frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            result = ext.extract(frame)
            assert np.count_nonzero(result) > 0, (
                "All-zero output detected — check ROUND_ROBIN scheduler is active "
                "and HEF was not corrupted during transfer."
            )

    def test_hardware_consistent_across_calls(self):
        """Same input produces same output (deterministic inference)."""
        from src.vision.hailo_extractor import HailoVisualExtractor

        frame = np.full((224, 224, 3), 100, dtype=np.uint8)
        with HailoVisualExtractor(str(self.HEF_PATH)) as ext:
            result1 = ext.extract(frame)
            result2 = ext.extract(frame)
            np.testing.assert_array_equal(result1, result2)
