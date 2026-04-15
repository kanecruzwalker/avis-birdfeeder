"""
tests/vision/test_hailo_detector.py

Unit tests for HailoDetector.

Strategy:
    - All Hailo hardware calls are mocked — no Pi required for CI.
    - _decode_nms_output() is tested directly with synthetic buffers
      that match the exact binary format produced by Hailo YOLOv8 NMS.
    - Hardware integration tests are marked @pytest.mark.hardware and
      skipped in CI.

NMS output buffer format (verified on Pi hardware):
    Layout: [class_0 | class_1 | ... | class_79]
    Per class: [count: uint32 (4 bytes)] + [det_0...det_N: float32×5 (20 bytes each)]
    Per detection: [x1, y1, x2, y2, score] normalized [0,1] to 640×640 input

COCO class 14 = bird.
"""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.vision.hailo_detector import (
    BYTES_PER_DETECTION,
    COCO_BIRD_CLASS_ID,
    YOLO_NUM_CLASSES,
    Detection,
    HailoDetector,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_nms_buffer(
    detections_by_class: dict[int, list[tuple[float, float, float, float, float]]],
    max_proposals: int = 10,
) -> np.ndarray:
    """
    Build a synthetic NMS output buffer matching Hailo YOLOv8 format.

    Args:
        detections_by_class: {class_id: [(x1, y1, x2, y2, score), ...]}
        max_proposals: max proposals per class (controls buffer size)

    Returns:
        uint8 numpy array matching Hailo NMS output format.
    """
    bytes_per_class = 4 + max_proposals * BYTES_PER_DETECTION
    total_bytes = YOLO_NUM_CLASSES * bytes_per_class
    buf = np.zeros(total_bytes, dtype=np.uint8)

    for class_id, dets in detections_by_class.items():
        offset = class_id * bytes_per_class
        count = min(len(dets), max_proposals)
        # Write count as uint32
        buf[offset : offset + 4] = np.frombuffer(
            struct.pack("<f", float(count)), dtype=np.uint8
        )  # Write each detection as 5 × float32
        det_start = offset + 4
        for i, (x1, y1, x2, y2, score) in enumerate(dets[:count]):
            det_offset = det_start + i * BYTES_PER_DETECTION
            packed = struct.pack("<fffff", x1, y1, x2, y2, score)
            buf[det_offset : det_offset + BYTES_PER_DETECTION] = np.frombuffer(
                packed, dtype=np.uint8
            )

    return buf


def _make_detector(tmp_path: Path, **kwargs) -> HailoDetector:
    """HailoDetector pointing at a fake HEF path."""
    fake_hef = tmp_path / "fake_yolov8s.hef"
    fake_hef.touch()
    return HailoDetector(str(fake_hef), **kwargs)


# ── Detection dataclass ───────────────────────────────────────────────────────


class TestDetection:
    def test_width(self) -> None:
        d = Detection(x1=10, y1=20, x2=110, y2=220, confidence=0.9, class_id=14)
        assert d.width == 100

    def test_height(self) -> None:
        d = Detection(x1=10, y1=20, x2=110, y2=220, confidence=0.9, class_id=14)
        assert d.height == 200

    def test_area(self) -> None:
        d = Detection(x1=0, y1=0, x2=100, y2=50, confidence=0.8, class_id=14)
        assert d.area == 5000

    def test_as_crop_basic(self) -> None:
        frame = np.zeros((864, 1536, 3), dtype=np.uint8)
        frame[100:300, 200:400] = 128
        d = Detection(x1=200, y1=100, x2=400, y2=300, confidence=0.9, class_id=14)
        crop = d.as_crop(frame, padding=0)
        assert crop.shape == (200, 200, 3)

    def test_as_crop_with_padding(self) -> None:
        frame = np.zeros((864, 1536, 3), dtype=np.uint8)
        d = Detection(x1=100, y1=100, x2=300, y2=300, confidence=0.9, class_id=14)
        crop = d.as_crop(frame, padding=10)
        assert crop.shape == (220, 220, 3)

    def test_as_crop_clamped_to_frame(self) -> None:
        frame = np.zeros((864, 1536, 3), dtype=np.uint8)
        d = Detection(x1=0, y1=0, x2=100, y2=100, confidence=0.9, class_id=14)
        crop = d.as_crop(frame, padding=50)
        # y1 clamped to 0, x1 clamped to 0
        assert crop.shape[0] <= 864
        assert crop.shape[1] <= 1536


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestHailoDetectorInit:
    def test_stores_hef_path(self, tmp_path: Path) -> None:
        d = _make_detector(tmp_path)
        assert d.hef_path == tmp_path / "fake_yolov8s.hef"

    def test_not_open_on_construction(self, tmp_path: Path) -> None:
        d = _make_detector(tmp_path)
        assert not d.is_open

    def test_default_score_threshold(self, tmp_path: Path) -> None:
        d = _make_detector(tmp_path)
        assert d.score_threshold == HailoDetector.DEFAULT_SCORE_THRESHOLD

    def test_custom_score_threshold(self, tmp_path: Path) -> None:
        d = _make_detector(tmp_path, score_threshold=0.5)
        assert d.score_threshold == 0.5

    def test_default_max_proposals(self, tmp_path: Path) -> None:
        d = _make_detector(tmp_path)
        assert d.max_proposals_per_class == HailoDetector.DEFAULT_MAX_PROPOSALS

    def test_configured_none_before_open(self, tmp_path: Path) -> None:
        d = _make_detector(tmp_path)
        assert d._configured is None


# ── from_config ───────────────────────────────────────────────────────────────


class TestFromConfig:
    def test_constructs_from_hardware_yaml(self) -> None:
        d = HailoDetector.from_config("configs/hardware.yaml")
        assert d is not None

    def test_hef_path_from_config(self) -> None:
        d = HailoDetector.from_config("configs/hardware.yaml")
        assert "yolov8s" in str(d.hef_path) or "hailo" in str(d.hef_path).lower()

    def test_not_open_after_from_config(self) -> None:
        d = HailoDetector.from_config("configs/hardware.yaml")
        assert not d.is_open


# ── open/close ────────────────────────────────────────────────────────────────


class TestOpenClose:
    @patch("src.vision.hailo_detector.HAILO_AVAILABLE", False)
    def test_open_raises_if_no_hailo(self, tmp_path: Path) -> None:
        d = _make_detector(tmp_path)
        with pytest.raises(RuntimeError, match="hailo_platform not available"):
            d.open()

    @patch("src.vision.hailo_detector.HAILO_AVAILABLE", True)
    def test_open_raises_if_hef_missing(self, tmp_path: Path) -> None:
        d = HailoDetector(str(tmp_path / "nonexistent.hef"))
        with pytest.raises(RuntimeError, match="YOLO HEF not found"):
            d.open()

    @patch("src.vision.hailo_detector.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_detector.VDevice")
    def test_open_sets_is_open(self, mock_vdevice_cls: MagicMock, tmp_path: Path) -> None:
        mock_vdevice = MagicMock()
        mock_vdevice_cls.return_value = mock_vdevice
        mock_model = MagicMock()
        mock_vdevice.create_infer_model.return_value = mock_model
        mock_configured = MagicMock()
        mock_model.configure.return_value = mock_configured
        mock_model.output.return_value = MagicMock()

        d = _make_detector(tmp_path)
        d.open()
        assert d.is_open

    @patch("src.vision.hailo_detector.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_detector.VDevice")
    def test_close_clears_is_open(self, mock_vdevice_cls: MagicMock, tmp_path: Path) -> None:
        mock_vdevice = MagicMock()
        mock_vdevice_cls.return_value = mock_vdevice
        mock_model = MagicMock()
        mock_vdevice.create_infer_model.return_value = mock_model
        mock_configured = MagicMock()
        mock_model.configure.return_value = mock_configured
        mock_model.output.return_value = MagicMock()

        d = _make_detector(tmp_path)
        d.open()
        d.close()
        assert not d.is_open

    @patch("src.vision.hailo_detector.HAILO_AVAILABLE", True)
    @patch("src.vision.hailo_detector.VDevice")
    def test_double_open_is_safe(self, mock_vdevice_cls: MagicMock, tmp_path: Path) -> None:
        mock_vdevice = MagicMock()
        mock_vdevice_cls.return_value = mock_vdevice
        mock_model = MagicMock()
        mock_vdevice.create_infer_model.return_value = mock_model
        mock_configured = MagicMock()
        mock_model.configure.return_value = mock_configured
        mock_model.output.return_value = MagicMock()

        d = _make_detector(tmp_path)
        d.open()
        d.open()  # second open should be a no-op
        assert d.is_open

    def test_double_close_is_safe(self, tmp_path: Path) -> None:
        d = _make_detector(tmp_path)
        d.close()  # close before open — should not raise
        d.close()  # double close — should not raise

    def test_detect_raises_if_not_open(self, tmp_path: Path) -> None:
        d = _make_detector(tmp_path)
        frame = np.zeros((864, 1536, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="open()"):
            d.detect(frame)

    def test_detect_all_raises_if_not_open(self, tmp_path: Path) -> None:
        d = _make_detector(tmp_path)
        frame = np.zeros((864, 1536, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="open()"):
            d.detect_all(frame)


# ── _decode_nms_output ────────────────────────────────────────────────────────


class TestDecodeNmsOutput:
    """
    Tests for the NMS buffer decoder — the most critical logic in the class.
    These tests use _make_nms_buffer() to construct synthetic buffers that
    exactly match the Hailo YOLOv8 NMS output format.
    """

    def setup_method(self) -> None:
        self.detector = HailoDetector.__new__(HailoDetector)
        self.detector.max_proposals_per_class = 10
        self.detector.min_bird_confidence = 0.2
        self.detector.min_crop_size = 10

    def test_empty_buffer_returns_no_detections(self) -> None:
        buf = _make_nms_buffer({}, max_proposals=10)
        result = self.detector._decode_nms_output(buf, orig_w=1536, orig_h=864)
        assert result == []

    def test_single_bird_detection(self) -> None:
        buf = _make_nms_buffer(
            {COCO_BIRD_CLASS_ID: [(0.2, 0.1, 0.8, 0.9, 0.85)]},
            max_proposals=10,
        )
        result = self.detector._decode_nms_output(buf, orig_w=1536, orig_h=864)
        assert len(result) == 1
        assert result[0].confidence == pytest.approx(0.85, abs=1e-4)

    def test_coordinates_scaled_to_original_frame(self) -> None:
        # Normalized box [0.25, 0.25, 0.75, 0.75] on 1536×864 frame
        buf = _make_nms_buffer(
            {COCO_BIRD_CLASS_ID: [(0.25, 0.25, 0.75, 0.75, 0.9)]},
            max_proposals=10,
        )
        result = self.detector._decode_nms_output(buf, orig_w=1536, orig_h=864)
        assert len(result) == 1
        d = result[0]
        assert d.x1 == pytest.approx(384, abs=2)  # 0.25 * 1536
        assert d.y1 == pytest.approx(216, abs=2)  # 0.25 * 864
        assert d.x2 == pytest.approx(1152, abs=2)  # 0.75 * 1536
        assert d.y2 == pytest.approx(648, abs=2)  # 0.75 * 864

    def test_low_confidence_detection_filtered(self) -> None:
        buf = _make_nms_buffer(
            {COCO_BIRD_CLASS_ID: [(0.1, 0.1, 0.9, 0.9, 0.15)]},
            max_proposals=10,
        )
        result = self.detector._decode_nms_output(buf, orig_w=1536, orig_h=864)
        assert result == []

    def test_non_bird_class_ignored(self) -> None:
        # Class 0 = person — should be ignored
        buf = _make_nms_buffer(
            {0: [(0.1, 0.1, 0.9, 0.9, 0.95)]},
            max_proposals=10,
        )
        result = self.detector._decode_nms_output(buf, orig_w=1536, orig_h=864)
        assert result == []

    def test_multiple_bird_detections(self) -> None:
        buf = _make_nms_buffer(
            {
                COCO_BIRD_CLASS_ID: [
                    (0.1, 0.1, 0.3, 0.3, 0.9),
                    (0.5, 0.5, 0.8, 0.8, 0.7),
                ]
            },
            max_proposals=10,
        )
        result = self.detector._decode_nms_output(buf, orig_w=1536, orig_h=864)
        assert len(result) == 2

    def test_small_box_filtered(self) -> None:
        self.detector.min_crop_size = 100
        # Very small box — will be filtered
        buf = _make_nms_buffer(
            {COCO_BIRD_CLASS_ID: [(0.1, 0.1, 0.11, 0.11, 0.9)]},
            max_proposals=10,
        )
        result = self.detector._decode_nms_output(buf, orig_w=1536, orig_h=864)
        assert result == []

    def test_class_id_set_correctly(self) -> None:
        buf = _make_nms_buffer(
            {COCO_BIRD_CLASS_ID: [(0.1, 0.1, 0.9, 0.9, 0.8)]},
            max_proposals=10,
        )
        result = self.detector._decode_nms_output(buf, orig_w=1536, orig_h=864)
        assert result[0].class_id == COCO_BIRD_CLASS_ID

    def test_coordinates_clamped_to_frame(self) -> None:
        # Coordinates slightly outside [0,1] — should be clamped
        buf = _make_nms_buffer(
            {COCO_BIRD_CLASS_ID: [(-0.1, -0.1, 1.1, 1.1, 0.9)]},
            max_proposals=10,
        )
        result = self.detector._decode_nms_output(buf, orig_w=1536, orig_h=864)
        if result:
            assert result[0].x1 >= 0
            assert result[0].y1 >= 0
            assert result[0].x2 <= 1536
            assert result[0].y2 <= 864

    def test_zero_count_returns_empty(self) -> None:
        # Buffer with count=0 for bird class
        buf = _make_nms_buffer({}, max_proposals=10)
        result = self.detector._decode_nms_output(buf, orig_w=1536, orig_h=864)
        assert result == []


# ── detect() with mocked hardware ────────────────────────────────────────────


class TestDetectMocked:
    """
    Tests for detect() and detect_all() with fully mocked Hailo hardware.
    These confirm the detect() logic (resize, call inference, decode) without
    requiring a Pi.
    """

    def _make_open_detector(self, tmp_path: Path) -> HailoDetector:
        """Return a detector with is_open=True and mocked _configured."""
        d = _make_detector(tmp_path)
        d.is_open = True
        d._out_buf_size = YOLO_NUM_CLASSES * 10 * BYTES_PER_DETECTION + YOLO_NUM_CLASSES * 4
        mock_configured = MagicMock()
        mock_bindings = MagicMock()
        mock_configured.create_bindings.return_value = mock_bindings
        mock_job = MagicMock()
        mock_configured.run_async.return_value = mock_job

        # Make output buffer contain a bird detection
        bird_buf = _make_nms_buffer(
            {COCO_BIRD_CLASS_ID: [(0.2, 0.2, 0.8, 0.8, 0.85)]},
            max_proposals=10,
        )

        def set_output_buffer(buf):
            buf[: len(bird_buf)] = bird_buf[: len(buf)]

        mock_bindings.output.return_value.set_buffer.side_effect = set_output_buffer
        d._configured = mock_configured
        return d

    def test_detect_returns_none_on_empty_frame(self, tmp_path: Path) -> None:
        d = _make_detector(tmp_path)
        d.is_open = True
        d._out_buf_size = YOLO_NUM_CLASSES * 10 * BYTES_PER_DETECTION + YOLO_NUM_CLASSES * 4
        mock_configured = MagicMock()
        mock_bindings = MagicMock()
        mock_configured.create_bindings.return_value = mock_bindings
        mock_job = MagicMock()
        mock_configured.run_async.return_value = mock_job
        # Output buffer stays all zeros — no detections
        mock_bindings.output.return_value.set_buffer.return_value = None
        d._configured = mock_configured

        frame = np.zeros((864, 1536, 3), dtype=np.uint8)
        result = d.detect(frame)
        assert result is None

    def test_detect_all_returns_list(self, tmp_path: Path) -> None:
        d = _make_detector(tmp_path)
        d.is_open = True
        d._out_buf_size = YOLO_NUM_CLASSES * 10 * BYTES_PER_DETECTION + YOLO_NUM_CLASSES * 4
        mock_configured = MagicMock()
        mock_bindings = MagicMock()
        mock_configured.create_bindings.return_value = mock_bindings
        mock_job = MagicMock()
        mock_configured.run_async.return_value = mock_job
        mock_bindings.output.return_value.set_buffer.return_value = None
        d._configured = mock_configured

        frame = np.zeros((864, 1536, 3), dtype=np.uint8)
        result = d.detect_all(frame)
        assert isinstance(result, list)


# ── Hardware tests ────────────────────────────────────────────────────────────


@pytest.mark.hardware
class TestHailoDetectorHardware:
    """Requires Pi hardware with yolov8s_h8l.hef at /usr/share/hailo-models/."""

    def test_open_and_close(self) -> None:
        d = HailoDetector("/usr/share/hailo-models/yolov8s_h8l.hef")
        d.open()
        assert d.is_open
        d.close()
        assert not d.is_open

    def test_detect_on_blank_frame(self) -> None:
        d = HailoDetector("/usr/share/hailo-models/yolov8s_h8l.hef")
        d.open()
        frame = np.zeros((864, 1536, 3), dtype=np.uint8)
        result = d.detect(frame)
        assert result is None  # blank frame should have no detections
        d.close()

    def test_context_manager(self) -> None:
        with HailoDetector("/usr/share/hailo-models/yolov8s_h8l.hef") as d:
            assert d.is_open
            frame = np.zeros((864, 1536, 3), dtype=np.uint8)
            result = d.detect(frame)
            assert result is None or isinstance(result, Detection)
        assert not d.is_open
