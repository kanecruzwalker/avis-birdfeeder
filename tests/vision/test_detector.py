"""
tests/vision/test_detector.py

Unit tests for src.vision.detector — BirdDetection, BirdDetector protocol,
CPUYOLODetector implementation, and load_detector factory.

Strategy:
    - ultralytics is mocked in all unit tests to avoid requiring the model
      download or any real inference.
    - Real-model integration tests are marked @pytest.mark.requires_model
      and excluded from default CI (see pyproject.toml addopts).
    - CPUYOLODetector construction, open, detect, close lifecycle is tested
      against a mocked YOLO model.
    - load_detector factory tested with real hardware.yaml (backend="cpu")
      and with synthetic hardware.yaml variants in tmp_path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.vision.detector import BirdDetection, CPUYOLODetector, load_detector

# ── BirdDetection dataclass ───────────────────────────────────────────────────


class TestBirdDetection:
    def test_width_height(self) -> None:
        d = BirdDetection(x1=10, y1=20, x2=110, y2=220, confidence=0.9)
        assert d.width == 100
        assert d.height == 200

    def test_as_crop_basic(self) -> None:
        frame = np.random.randint(0, 255, (500, 800, 3), dtype=np.uint8)
        d = BirdDetection(x1=100, y1=50, x2=300, y2=250, confidence=0.8)
        crop = d.as_crop(frame, padding=0)
        assert crop.shape == (200, 200, 3)

    def test_as_crop_with_padding(self) -> None:
        frame = np.random.randint(0, 255, (500, 800, 3), dtype=np.uint8)
        d = BirdDetection(x1=100, y1=100, x2=300, y2=300, confidence=0.8)
        crop = d.as_crop(frame, padding=10)
        assert crop.shape == (220, 220, 3)

    def test_as_crop_clamps_to_frame(self) -> None:
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        d = BirdDetection(x1=0, y1=0, x2=50, y2=50, confidence=0.8)
        crop = d.as_crop(frame, padding=100)
        # Padding would push coords negative and beyond frame; should clamp.
        assert crop.shape[0] <= 100
        assert crop.shape[1] <= 100


# ── CPUYOLODetector ───────────────────────────────────────────────────────────


class TestCPUYOLODetectorInit:
    def test_default_values(self) -> None:
        d = CPUYOLODetector()
        assert d.model_path == Path("yolov8s.pt")
        assert d.confidence_threshold == 0.25
        assert d.imgsz == 640
        assert d.is_open is False

    def test_custom_values(self) -> None:
        d = CPUYOLODetector(
            model_path="/custom/path.pt",
            confidence_threshold=0.5,
            imgsz=320,
        )
        assert d.model_path == Path("/custom/path.pt")
        assert d.confidence_threshold == 0.5
        assert d.imgsz == 320


# ── Helper: stub ultralytics module for tests ────────────────────────────────
# Our CPUYOLODetector does a late import of ultralytics inside open() so the
# module is only required at runtime, not at import time. Tests on machines
# without ultralytics installed can't patch "ultralytics.YOLO" directly
# (the patch decorator tries to resolve the path at setup, which fails
# with ModuleNotFoundError). Instead we inject a fake ultralytics module
# into sys.modules before calling open(), so the late import resolves to
# our stub and we can control its behavior.


class _FakeYoloResults:
    """Behaves like the single element of ultralytics' results list."""

    def __init__(self, boxes=None):
        self.boxes = boxes


class _FakeYoloBox:
    """
    Behaves like a single ultralytics box from Results.boxes iteration.

    Uses real ints, floats, and a numpy array so the detector's
    `int(box.cls[0])`, `float(box.conf[0])`, and `box.xyxy[0].cpu().numpy()
    .astype(int).tolist()` calls all work without any MagicMock dunder
    method shenanigans.
    """

    def __init__(self, cls_id: int, confidence: float, xyxy=(10, 20, 110, 220)):
        # cls[0] and conf[0] must coerce to int and float respectively.
        # Using single-element lists of real numbers is the cleanest way.
        self.cls = [cls_id]
        self.conf = [confidence]
        # xyxy[0] must support .cpu().numpy().astype(int).tolist()
        # The real ultralytics returns a torch tensor here; we use a numpy
        # array that implements the chain via a simple wrapper class.
        self.xyxy = [_FakeXyxyTensor(xyxy)]


class _FakeXyxyTensor:
    """Mimics the .cpu().numpy().astype(int).tolist() chain."""

    def __init__(self, xyxy: tuple[int, int, int, int]):
        self._xyxy = xyxy

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self._xyxy, dtype=np.float32)


def _install_fake_ultralytics(mock_model: MagicMock) -> MagicMock:
    """
    Install a fake `ultralytics` module in sys.modules whose YOLO class
    returns the given mock_model. Returns the mock_model for further
    configuration. Safe to call multiple times in a test; each call
    overwrites the previous stub.
    """
    import sys
    import types

    fake_module = types.ModuleType("ultralytics")
    fake_yolo_cls = MagicMock(return_value=mock_model)
    fake_module.YOLO = fake_yolo_cls
    sys.modules["ultralytics"] = fake_module
    return fake_yolo_cls


def _uninstall_fake_ultralytics() -> None:
    """Remove the fake ultralytics stub so subsequent tests get a clean slate."""
    import sys

    sys.modules.pop("ultralytics", None)


class TestCPUYOLODetectorOpen:
    def teardown_method(self) -> None:
        _uninstall_fake_ultralytics()

    def test_open_loads_model(self) -> None:
        mock_model = MagicMock()
        mock_model.return_value = [_FakeYoloResults(boxes=None)]
        fake_yolo_cls = _install_fake_ultralytics(mock_model)

        d = CPUYOLODetector(model_path="yolov8s.pt")
        d.open()

        fake_yolo_cls.assert_called_once_with("yolov8s.pt")
        assert d.is_open is True

    def test_open_is_idempotent(self) -> None:
        mock_model = MagicMock()
        mock_model.return_value = [_FakeYoloResults(boxes=None)]
        fake_yolo_cls = _install_fake_ultralytics(mock_model)

        d = CPUYOLODetector()
        d.open()
        d.open()  # second call should be a no-op

        assert fake_yolo_cls.call_count == 1

    def test_open_raises_if_ultralytics_missing(self) -> None:
        # Force the import to fail by installing a sentinel that raises on access
        import sys

        sys.modules["ultralytics"] = None  # type: ignore[assignment]
        try:
            d = CPUYOLODetector()
            with pytest.raises(RuntimeError, match="ultralytics is not installed"):
                d.open()
        finally:
            sys.modules.pop("ultralytics", None)


class TestCPUYOLODetectorDetect:
    def _make_open_detector(self) -> tuple[CPUYOLODetector, MagicMock]:
        """Return (detector, mock_model) with detector in is_open=True state."""
        d = CPUYOLODetector()
        mock_model = MagicMock()
        d._model = mock_model
        d.is_open = True
        return d, mock_model

    def test_detect_raises_if_not_open(self) -> None:
        d = CPUYOLODetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="not open"):
            d.detect(frame)

    def test_detect_all_returns_empty_when_no_birds(self) -> None:
        d, mock_model = self._make_open_detector()
        mock_model.return_value = [_FakeYoloResults(boxes=None)]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        birds = d.detect_all(frame)
        assert birds == []

    def test_detect_all_filters_non_bird_classes(self) -> None:
        d, mock_model = self._make_open_detector()
        # class 0 = person, class 14 = bird, class 17 = horse
        boxes = [
            _FakeYoloBox(cls_id=0, confidence=0.9),
            _FakeYoloBox(cls_id=14, confidence=0.7),
            _FakeYoloBox(cls_id=17, confidence=0.8),
        ]
        mock_model.return_value = [_FakeYoloResults(boxes=boxes)]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        birds = d.detect_all(frame)
        assert len(birds) == 1
        assert birds[0].confidence == pytest.approx(0.7)

    def test_detect_returns_none_when_no_birds(self) -> None:
        d, mock_model = self._make_open_detector()
        mock_model.return_value = [_FakeYoloResults(boxes=None)]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert d.detect(frame) is None

    def test_detect_returns_highest_confidence(self) -> None:
        d, mock_model = self._make_open_detector()
        boxes = [
            _FakeYoloBox(cls_id=14, confidence=0.6),
            _FakeYoloBox(cls_id=14, confidence=0.9),
            _FakeYoloBox(cls_id=14, confidence=0.3),
        ]
        mock_model.return_value = [_FakeYoloResults(boxes=boxes)]

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        top = d.detect(frame)
        assert top is not None
        assert top.confidence == pytest.approx(0.9)


class TestCPUYOLODetectorClose:
    def test_close_marks_not_open(self) -> None:
        d = CPUYOLODetector()
        d._model = MagicMock()
        d.is_open = True
        d.close()
        assert d.is_open is False
        assert d._model is None

    def test_close_is_idempotent(self) -> None:
        d = CPUYOLODetector()
        d.close()
        d.close()  # should not raise


class TestCPUYOLODetectorContext:
    def teardown_method(self) -> None:
        _uninstall_fake_ultralytics()

    def test_context_manager(self) -> None:
        mock_model = MagicMock()
        mock_model.return_value = [_FakeYoloResults(boxes=None)]
        _install_fake_ultralytics(mock_model)

        with CPUYOLODetector() as d:
            assert d.is_open is True

        assert d.is_open is False


# ── load_detector factory ─────────────────────────────────────────────────────


class TestLoadDetector:
    def _write_hw(self, tmp_path: Path, content: str) -> Path:
        configs = tmp_path / "configs"
        configs.mkdir()
        (configs / "hardware.yaml").write_text(content)
        return configs

    def test_raises_if_hardware_yaml_missing(self, tmp_path: Path) -> None:
        configs = tmp_path / "configs"
        configs.mkdir()
        with pytest.raises(FileNotFoundError, match="hardware.yaml"):
            load_detector(configs)

    def test_cpu_backend_returns_cpu_detector(self, tmp_path: Path) -> None:
        configs = self._write_hw(
            tmp_path,
            "detector:\n  backend: cpu\n  cpu:\n    model_path: yolov8s.pt\n    confidence_threshold: 0.3\n    imgsz: 640\n",
        )
        detector = load_detector(configs)
        assert isinstance(detector, CPUYOLODetector)
        assert detector.confidence_threshold == 0.3

    def test_hailo_backend_raises_not_implemented(self, tmp_path: Path) -> None:
        configs = self._write_hw(tmp_path, "detector:\n  backend: hailo\n")
        with pytest.raises(NotImplementedError, match="Branch 5"):
            load_detector(configs)

    def test_unknown_backend_raises(self, tmp_path: Path) -> None:
        configs = self._write_hw(tmp_path, "detector:\n  backend: cuda\n")
        with pytest.raises(ValueError, match="Unknown detector backend"):
            load_detector(configs)

    def test_missing_detector_section_defaults_to_cpu(self, tmp_path: Path) -> None:
        """No detector block → defaults to cpu (matches factory default)."""
        configs = self._write_hw(tmp_path, "hailo:\n  enabled: false\n")
        detector = load_detector(configs)
        assert isinstance(detector, CPUYOLODetector)


# ── Real-model integration test (skipped in CI) ──────────────────────────────


@pytest.mark.requires_model
class TestCPUYOLOIntegration:
    """
    Integration tests that actually load YOLOv8s.pt and run inference.
    Skipped in CI. Run locally with:
        pytest tests/vision/test_detector.py -m requires_model
    """

    def test_detect_real_model(self) -> None:
        detector = CPUYOLODetector()
        detector.open()
        try:
            # Random frame — may or may not produce detections, but should not error
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = detector.detect_all(frame)
            assert isinstance(result, list)
        finally:
            detector.close()
