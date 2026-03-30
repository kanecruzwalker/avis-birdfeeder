"""
src/vision/classify.py

Wraps EfficientNet (or MobileNet) inference to produce a ClassificationResult.

EfficientNet-B0 is our primary visual backbone. It was chosen because:
    - Strong accuracy/efficiency tradeoff (outperforms ResNet-50 at fewer params)
    - Well-documented deployment path to Hailo via the TAPPAS toolkit
    - Pretrained ImageNet weights available via torchvision / timm

MobileNetV2 is our lightweight fallback if EfficientNet proves too slow on
the Hailo HAT+ for real-time inference.

Input:  preprocessed frame (from src.vision.preprocess)
Output: ClassificationResult (from src.data.schema)

Phase 4 will integrate the actual model weights and Hailo inference pipeline.
"""

from __future__ import annotations

import numpy as np

from src.data.schema import ClassificationResult


class VisualClassifier:
    """
    Wrapper around EfficientNet/MobileNet for bird species classification.

    Usage:
        classifier = VisualClassifier.from_config("configs/paths.yaml")
        frame = preprocess_frame(raw_frame)
        result = classifier.predict(frame)
    """

    def __init__(self, model_path: str, species_list_path: str) -> None:
        """
        Args:
            model_path: Path to compiled model weights (.hef for Hailo, .pt otherwise).
            species_list_path: Path to configs/species.yaml.
        """
        self.model_path = model_path
        self.species_list_path = species_list_path
        self._model = None  # loaded lazily in Phase 4

    @classmethod
    def from_config(cls, config_path: str) -> VisualClassifier:
        """
        Construct a VisualClassifier from a paths config YAML.

        Args:
            config_path: Path to configs/paths.yaml.

        Returns:
            Configured VisualClassifier instance.
        """
        raise NotImplementedError("Implement in Phase 4.")

    def predict(self, frame: np.ndarray) -> ClassificationResult:
        """
        Run inference on a preprocessed frame and return a classification result.

        Args:
            frame: Float32 array of shape (224, 224, 3), ImageNet-normalized.

        Returns:
            ClassificationResult with the top species prediction and confidence.
        """
        raise NotImplementedError("Implement in Phase 4.")
