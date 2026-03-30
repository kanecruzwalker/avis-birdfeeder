"""
src/audio/classify.py

Wraps BirdNET inference to produce a ClassificationResult.

BirdNET is a pretrained audio classifier trained on ~6,000 bird species from
Xeno-canto and Macaulay Library recordings. We use it as our audio backbone
and fine-tune on San Diego region species.

Input:  mel spectrogram (from src.audio.preprocess)
Output: ClassificationResult (from src.data.schema)

Phase 4 will integrate the actual BirdNET model weights and Hailo inference.
"""

from __future__ import annotations

import numpy as np

from src.data.schema import ClassificationResult


class AudioClassifier:
    """
    Wrapper around BirdNET (or our fine-tuned variant) for species classification.

    Usage:
        classifier = AudioClassifier.from_config("configs/paths.yaml")
        spectrogram = preprocess_file("clip.wav")
        result = classifier.predict(spectrogram)
    """

    def __init__(self, model_path: str, species_list_path: str) -> None:
        """
        Args:
            model_path: Path to the model weights (.hef for Hailo, .tflite otherwise).
            species_list_path: Path to the species list YAML (configs/species.yaml).
        """
        # Phase 4: load model weights and species list
        self.model_path = model_path
        self.species_list_path = species_list_path
        self._model = None  # loaded lazily in Phase 4

    @classmethod
    def from_config(cls, config_path: str) -> AudioClassifier:
        """
        Construct an AudioClassifier from a paths config YAML.

        Args:
            config_path: Path to configs/paths.yaml.

        Returns:
            Configured AudioClassifier instance.
        """
        raise NotImplementedError("Implement in Phase 4.")

    def predict(self, spectrogram: np.ndarray) -> ClassificationResult:
        """
        Run inference on a mel spectrogram and return a classification result.

        Args:
            spectrogram: 2-D float32 array of shape (n_mels, time_frames).

        Returns:
            ClassificationResult with the top species prediction and confidence.
        """
        raise NotImplementedError("Implement in Phase 4.")
