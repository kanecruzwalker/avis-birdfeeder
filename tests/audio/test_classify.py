"""
tests/audio/test_classify.py

Unit tests for AudioClassifier.

Strategy:
    - All tests run without model weights on disk (lazy loading pattern).
    - Model loading path is tested via RuntimeError on missing weights.
    - predict() shape and dtype contracts are tested with a mock _load().
    - from_config() is tested with a real paths.yaml to verify key resolution.
    - No real audio files or GPU required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.audio.classify import AudioClassifier, _build_audio_cnn
from src.data.schema import ClassificationResult, Modality

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def classifier_no_weights(tmp_path: Path) -> AudioClassifier:
    """AudioClassifier pointing at a nonexistent model path — weights not loaded."""
    return AudioClassifier(
        model_path=str(tmp_path / "nonexistent.pt"),
        label_map_path=str(tmp_path / "label_map.json"),
        species_list_path="configs/species.yaml",
    )


@pytest.fixture()
def label_map(tmp_path: Path) -> Path:
    """Write a minimal label_map.json to tmp_path."""
    lm = {"0": "HOFI", "1": "MODO", "2": "ANHU"}
    p = tmp_path / "label_map.json"
    p.write_text(json.dumps(lm))
    return p


@pytest.fixture()
def loaded_classifier(tmp_path: Path, label_map: Path) -> AudioClassifier:
    """
    AudioClassifier with _load() bypassed — _model, _label_map, _species_meta
    populated directly so predict() can run without real weights.
    """
    clf = AudioClassifier(
        model_path=str(tmp_path / "fake.pt"),
        label_map_path=str(label_map),
        species_list_path="configs/species.yaml",
    )
    # Populate internals directly
    clf._label_map = {0: "HOFI", 1: "MODO", 2: "ANHU"}
    clf._species_meta = {
        "HOFI": {"common_name": "House Finch", "scientific_name": "Haemorhous mexicanus"},
        "MODO": {"common_name": "Mourning Dove", "scientific_name": "Zenaida macroura"},
        "ANHU": {"common_name": "Anna's Hummingbird", "scientific_name": "Calypte anna"},
    }
    # Stub model: always predicts class 0 (HOFI) with high confidence
    n_classes = 3
    fake_model = _build_audio_cnn(n_classes=n_classes)
    with torch.no_grad():
        # Zero all weights so softmax output is uniform — just need it to run
        for p in fake_model.parameters():
            p.zero_()
    fake_model.eval()
    clf._model = fake_model
    # Point model_path to a fake existing file so predict() doesn't raise
    fake_pt = tmp_path / "fake.pt"
    fake_pt.write_bytes(b"")
    clf.model_path = fake_pt
    return clf


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestAudioClassifierInit:
    def test_stores_paths(self, tmp_path: Path) -> None:
        clf = AudioClassifier(
            model_path="models/audio/finetuned_sdbirds.pt",
            label_map_path="models/label_map.json",
            species_list_path="configs/species.yaml",
        )
        assert clf.model_path == Path("models/audio/finetuned_sdbirds.pt")
        assert clf._model is None  # lazy — not loaded yet

    def test_device_auto_detected(self, tmp_path: Path) -> None:
        clf = AudioClassifier(
            model_path="x.pt",
            label_map_path="x.json",
            species_list_path="configs/species.yaml",
        )
        assert clf.device.type in ("cpu", "cuda")

    def test_device_explicit(self) -> None:
        clf = AudioClassifier(
            model_path="x.pt",
            label_map_path="x.json",
            species_list_path="configs/species.yaml",
            device="cpu",
        )
        assert clf.device == torch.device("cpu")


# ── from_config ───────────────────────────────────────────────────────────────


class TestFromConfig:
    def test_resolves_paths_from_yaml(self) -> None:
        clf = AudioClassifier.from_config("configs/paths.yaml")
        assert "finetuned_sdbirds" in str(clf.model_path)
        assert "label_map" in str(clf.label_map_path)
        assert "species.yaml" in str(clf.species_list_path)

    def test_model_not_loaded_after_from_config(self) -> None:
        clf = AudioClassifier.from_config("configs/paths.yaml")
        assert clf._model is None


# ── predict — error paths ─────────────────────────────────────────────────────


class TestPredictErrors:
    def test_raises_if_weights_missing(self, classifier_no_weights: AudioClassifier) -> None:
        spectrogram = np.zeros((128, 282), dtype=np.float32)
        with pytest.raises(RuntimeError, match="Model weights not found"):
            classifier_no_weights.predict(spectrogram)

    def test_raises_on_wrong_ndim(self, loaded_classifier: AudioClassifier) -> None:
        bad = np.zeros((128,), dtype=np.float32)  # 1-D — should be 2-D
        with pytest.raises(ValueError, match="Expected 2-D spectrogram"):
            loaded_classifier.predict(bad)

    def test_raises_on_3d_input(self, loaded_classifier: AudioClassifier) -> None:
        bad = np.zeros((128, 282, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 2-D spectrogram"):
            loaded_classifier.predict(bad)


# ── predict — happy path ──────────────────────────────────────────────────────


class TestPredictOutput:
    def test_returns_classification_result(self, loaded_classifier: AudioClassifier) -> None:
        spectrogram = np.zeros((128, 282), dtype=np.float32)
        result = loaded_classifier.predict(spectrogram)
        assert isinstance(result, ClassificationResult)

    def test_modality_is_audio(self, loaded_classifier: AudioClassifier) -> None:
        spectrogram = np.zeros((128, 282), dtype=np.float32)
        result = loaded_classifier.predict(spectrogram)
        assert result.modality == Modality.AUDIO

    def test_confidence_in_range(self, loaded_classifier: AudioClassifier) -> None:
        spectrogram = np.zeros((128, 282), dtype=np.float32)
        result = loaded_classifier.predict(spectrogram)
        assert 0.0 <= result.confidence <= 1.0

    def test_species_code_in_label_map(self, loaded_classifier: AudioClassifier) -> None:
        spectrogram = np.zeros((128, 282), dtype=np.float32)
        result = loaded_classifier.predict(spectrogram)
        assert result.species_code in {"HOFI", "MODO", "ANHU"}

    def test_species_code_uppercase(self, loaded_classifier: AudioClassifier) -> None:
        spectrogram = np.zeros((128, 282), dtype=np.float32)
        result = loaded_classifier.predict(spectrogram)
        assert result.species_code == result.species_code.upper()

    def test_model_version_set(self, loaded_classifier: AudioClassifier) -> None:
        spectrogram = np.zeros((128, 282), dtype=np.float32)
        result = loaded_classifier.predict(spectrogram)
        assert result.model_version == AudioClassifier.MODEL_VERSION

    def test_different_spectrogram_shapes_accepted(
        self, loaded_classifier: AudioClassifier
    ) -> None:
        """Classifier should handle varying time dimensions gracefully."""
        for time_frames in [128, 282, 400]:
            spec = np.random.rand(128, time_frames).astype(np.float32)
            result = loaded_classifier.predict(spec)
            assert isinstance(result, ClassificationResult)


# ── _build_audio_cnn ──────────────────────────────────────────────────────────


class TestBuildAudioCnn:
    def test_output_shape(self) -> None:
        model = _build_audio_cnn(n_classes=18)
        x = torch.zeros(1, 1, 128, 282)
        out = model(x)
        assert out.shape == (1, 18)

    def test_different_class_counts(self) -> None:
        for n in [3, 10, 18, 19]:
            model = _build_audio_cnn(n_classes=n)
            x = torch.zeros(1, 1, 128, 282)
            assert model(x).shape == (1, n)

    def test_variable_time_dim(self) -> None:
        """AdaptiveAvgPool2d should handle any time dimension."""
        model = _build_audio_cnn(n_classes=18)
        for t in [128, 200, 400]:
            x = torch.zeros(1, 1, 128, t)
            assert model(x).shape == (1, 18)
