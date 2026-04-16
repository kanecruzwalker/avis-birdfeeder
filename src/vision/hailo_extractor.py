"""
src/vision/hailo_extractor.py

Hailo HAILO8L inference path for EfficientNet-B0 feature extraction.

Replaces the CPU PyTorch forward pass in VisualClassifier when a compiled
.hef file is available and hardware.yaml has hailo.enabled: true.

Architecture decision:
    VisualClassifier calls either _HailoCPUExtractor or HailoVisualExtractor
    depending on config. Both return float32 (1, 1280) feature vectors for the
    shared sklearn LogReg head. The swap is transparent to BirdAgent.

Hailo API notes (HailoRT 4.23.0):
    - VDevice must be created with HailoSchedulingAlgorithm.ROUND_ROBIN.
      Without it, InferModel API returns HAILO_STREAM_NOT_ACTIVATED(72)
      and writes zeros to the output buffer on every call.
    - Input buffer dtype: uint8  (chip uses INT8 internally after quantization)
    - Output buffer dtype: uint8 (dequantized by dividing by 255.0 before LogReg)
    - Output shape: (1, 1280) — the GlobalAveragePool output of EfficientNet-B0

Preprocessing difference vs CPU path:
    CPU path:  resize → float32 → ImageNet mean/std normalize → CHW tensor
    Hailo path: resize → uint8 (raw pixel values, no normalization)
    The Hailo quantizer was calibrated on normalized [0,1] float images, so
    HailoRT handles the uint8→float conversion internally at the chip boundary.

Phase 6 only: This module is NOT imported by CI tests unless
HAILO_AVAILABLE=1 is set. All tests requiring hardware are marked
@pytest.mark.hardware and skipped in CI.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import — hailo_platform is Pi-only, not available in CI
try:
    from hailo_platform import HEF, HailoSchedulingAlgorithm, VDevice  # type: ignore[import]

    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    HEF = None  # type: ignore[assignment]
    VDevice = None  # type: ignore[assignment]
    HailoSchedulingAlgorithm = None  # type: ignore[assignment]


class HailoVisualExtractor:
    """
    EfficientNet-B0 feature extractor running on the Hailo HAILO8L NPU.

    Accepts a preprocessed uint8 RGB frame (224, 224, 3) and returns a
    float32 feature vector (1280,) ready for the sklearn LogReg head.

    Usage:
        extractor = HailoVisualExtractor(hef_path="models/visual/efficientnet_b0_avis_v2.hef")
        features  = extractor.extract(frame_uint8)   # frame: (224, 224, 3) uint8
        # features: (1, 1280) float32 — pass to sklearn_pipeline for classification

    Note on lifecycle:
        The VDevice context must stay open for the duration of inference.
        Call open() before any extract() calls, close() when done.
        The agent loop calls open() once at startup and close() on shutdown.
        Using the extractor as a context manager is preferred for short-lived use.
    """

    def __init__(self, hef_path: str | Path, shared_vdevice: object | None = None) -> None:
        """
        Args:
            hef_path: Path to the compiled .hef file on the Pi.
                      Typically models/visual/efficientnet_b0_avis_v2.hef
        """
        if not HAILO_AVAILABLE:
            raise RuntimeError(
                "hailo_platform is not installed. HailoVisualExtractor requires "
                "a Raspberry Pi with HailoRT installed. "
                "Set hardware.yaml hailo.enabled: false to use the CPU path."
            )

        self.hef_path = Path(hef_path)
        if not self.hef_path.exists():
            raise FileNotFoundError(
                f"HEF file not found: {self.hef_path}. "
                "Run scripts/compile_hailo_hef.py to generate it, or set "
                "hardware.yaml hailo.enabled: false to use the CPU path."
            )

        self._vdevice = None
        self._configured_model = None
        self._bindings = None
        self._output_buffer: np.ndarray | None = None

        # Validate HEF metadata on construction (fast — no hardware opened yet)
        hef = HEF(str(self.hef_path))
        groups = hef.get_network_group_names()
        if not groups:
            raise ValueError(f"HEF contains no network groups: {self.hef_path}")
        self._network_group_name = groups[0]
        inputs = hef.get_input_vstream_infos(self._network_group_name)
        outputs = hef.get_output_vstream_infos(self._network_group_name)
        self._input_shape = inputs[0].shape  # expected (224, 224, 3)
        self._output_shape = outputs[0].shape  # expected (1280,)

        self._shared_vdevice = shared_vdevice
        self._owns_vdevice = shared_vdevice is None

        logger.info(
            "HailoVisualExtractor | hef=%s network=%s input=%s output=%s",
            self.hef_path.name,
            self._network_group_name,
            self._input_shape,
            self._output_shape,
        )

    def open(self) -> None:
        """
        Open the VDevice and configure the inference model.

        Must be called before any extract() calls.
        Creates the ROUND_ROBIN VDevice, loads the HEF, and pre-allocates
        the output buffer for zero-copy inference.
        """
        if self._vdevice is not None:
            logger.warning("HailoVisualExtractor.open() called when already open — ignoring.")
            return

        if self._shared_vdevice is not None:
            self._vdevice = self._shared_vdevice
        else:
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            self._vdevice = VDevice(params).__enter__()

        infer_model = self._vdevice.create_infer_model(str(self.hef_path))
        infer_model.set_batch_size(1)
        self._configured_model = infer_model.configure().__enter__()

        # Pre-allocate output buffer once — reused across all extract() calls
        self._output_buffer = np.empty((1, *self._output_shape), dtype=np.uint8)
        self._bindings = self._configured_model.create_bindings()
        self._bindings.output().set_buffer(self._output_buffer)

        logger.info("HailoVisualExtractor opened — VDevice ready.")

    def close(self) -> None:
        """
        Release the VDevice and all Hailo resources.
        Safe to call multiple times.
        """
        if self._configured_model is not None:
            try:
                self._configured_model.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("Error closing configured model: %s", exc)
            self._configured_model = None

        if self._owns_vdevice and self._vdevice is not None:
            try:
                self._vdevice.__exit__(None, None, None)
            except Exception as exc:
                logger.warning("Error closing VDevice: %s", exc)
            self._vdevice = None

        self._bindings = None
        self._output_buffer = None
        logger.info("HailoVisualExtractor closed.")

    def __enter__(self) -> HailoVisualExtractor:
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Run one forward pass through the EfficientNet-B0 HEF.

        Args:
            frame: (224, 224, 3) uint8 RGB image.
                   Caller is responsible for resizing to 224x224 before calling.
                   No normalization needed — Hailo handles this internally.

        Returns:
            (1, 1280) float32 feature vector.
            Values are dequantized uint8 / 255.0 — suitable for the sklearn
            StandardScaler + LogReg pipeline trained on CPU float32 features.

        Raises:
            RuntimeError: If open() has not been called.
            ValueError:   If frame shape is wrong.
        """
        if self._configured_model is None or self._bindings is None:
            raise RuntimeError("HailoVisualExtractor.open() must be called before extract().")

        expected = (self._input_shape[0], self._input_shape[1], self._input_shape[2])
        if frame.shape != expected:
            raise ValueError(
                f"Frame shape mismatch: expected {expected}, got {frame.shape}. "
                "Resize frame to 224×224 before calling extract()."
            )

        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        # Set input buffer (zero-copy — sets pointer to frame's memory)
        self._bindings.input().set_buffer(frame.reshape(1, *frame.shape))

        # Run inference (blocking — returns when output buffer is written)
        self._configured_model.run([self._bindings], timeout=1000)

        # Dequantize: uint8 [0, 255] → float32 [0, 1]
        # This normalization is consistent with the float32 features the
        # sklearn pipeline was trained on (after StandardScaler).
        features = self._output_buffer.astype(np.float32) / 255.0
        return features  # (1, 1280) float32

    @property
    def is_open(self) -> bool:
        """True if the VDevice is open and ready for inference."""
        return self._vdevice is not None
