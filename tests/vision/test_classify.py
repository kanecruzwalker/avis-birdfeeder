"""
tests/vision/test_classify.py

Unit tests for VisualClassifier.

Strategy mirrors tests/audio/test_classify.py:
    - No real weights or GPU required.
    - _load() is bypassed via direct internal population in loaded_classifier.
    - predict() input contract (shape, dtype) is fully tested.
    - from_config() verified against real paths.yaml.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.schema import ClassificationResult, Modality
from src.vision.classify import VisualClassifier, _build_efficientnet

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def label_map(tmp_path: Path) -> Path:
    lm = {"0": "HOFI", "1": "MODO", "2": "DOWO"}
    p = tmp_path / "label_map.json"
    p.write_text(json.dumps(lm))
    return p


@pytest.fixture()
def loaded_classifier(tmp_path: Path, label_map: Path) -> VisualClassifier:
    """
    VisualClassifier with _load() bypassed — model populated directly
    so predict() runs without real weights or internet access.
    """
    clf = VisualClassifier(
        model_path=str(tmp_path / "fake.pt"),
        label_map_path=str(label_map),
        species_list_path="configs/species.yaml",
        device="cpu",
    )
    clf._label_map = {0: "HOFI", 1: "MODO", 2: "DOWO"}
    clf._species_meta = {
        "HOFI": {"common_name": "House Finch", "scientific_name": "Haemorhous mexicanus"},
        "MODO": {"common_name": "Mourning Dove", "scientific_name": "Zenaida macroura"},
        "DOWO": {"common_name": "Downy Woodpecker", "scientific_name": "Dryobates pubescens"},
    }
    # Use real EfficientNet-B0 structure but random weights — no checkpoint needed
    clf._model = _build_efficientnet(n_classes=3)
    clf._model.eval()
    fake_pt = tmp_path / "fake.pt"
    fake_pt.write_bytes(b"")
    clf.model_path = fake_pt
    return clf


@pytest.fixture()
def classifier_no_weights(tmp_path: Path) -> VisualClassifier:
    return VisualClassifier(
        model_path=str(tmp_path / "nonexistent.pt"),
        label_map_path=str(tmp_path / "label_map.json"),
        species_list_path="configs/species.yaml",
    )


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestVisualClassifierInit:
    def test_stores_paths(self) -> None:
        clf = VisualClassifier(
            model_path="models/visual/finetuned_sdbirds.pt",
            label_map_path="models/label_map.json",
            species_list_path="configs/species.yaml",
        )
        assert "finetuned_sdbirds" in str(clf.model_path)
        assert clf._model is None

    def test_device_auto_detected(self) -> None:
        clf = VisualClassifier(
            model_path="x.pt",
            label_map_path="x.json",
            species_list_path="configs/species.yaml",
        )
        assert clf.device.type in ("cpu", "cuda")

    def test_device_explicit_cpu(self) -> None:
        clf = VisualClassifier(
            model_path="x.pt",
            label_map_path="x.json",
            species_list_path="configs/species.yaml",
            device="cpu",
        )
        assert clf.device == torch.device("cpu")


# ── from_config ───────────────────────────────────────────────────────────────


class TestFromConfig:
    def test_resolves_visual_finetuned_path(self) -> None:
        clf = VisualClassifier.from_config("configs/paths.yaml")
        assert "finetuned_sdbirds" in str(clf.model_path)

    def test_resolves_label_map_path(self) -> None:
        clf = VisualClassifier.from_config("configs/paths.yaml")
        assert "label_map" in str(clf.label_map_path)

    def test_model_not_loaded_after_construction(self) -> None:
        clf = VisualClassifier.from_config("configs/paths.yaml")
        assert clf._model is None


# ── predict — error paths ─────────────────────────────────────────────────────


class TestPredictErrors:
    def test_raises_if_weights_missing(self, classifier_no_weights: VisualClassifier) -> None:
        frame = np.zeros((224, 224, 3), dtype=np.float32)
        with pytest.raises(RuntimeError, match="Model weights not found"):
            classifier_no_weights.predict(frame)

    def test_raises_on_wrong_shape(self, loaded_classifier: VisualClassifier) -> None:
        bad = np.zeros((128, 128, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected frame of shape"):
            loaded_classifier.predict(bad)

    def test_raises_on_2d_input(self, loaded_classifier: VisualClassifier) -> None:
        bad = np.zeros((224, 224), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected frame of shape"):
            loaded_classifier.predict(bad)

    def test_raises_on_wrong_channels(self, loaded_classifier: VisualClassifier) -> None:
        bad = np.zeros((224, 224, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected frame of shape"):
            loaded_classifier.predict(bad)


# ── predict — happy path ──────────────────────────────────────────────────────


class TestPredictOutput:
    def test_returns_classification_result(self, loaded_classifier: VisualClassifier) -> None:
        frame = np.random.rand(224, 224, 3).astype(np.float32)
        result = loaded_classifier.predict(frame)
        assert isinstance(result, ClassificationResult)

    def test_modality_is_visual(self, loaded_classifier: VisualClassifier) -> None:
        frame = np.random.rand(224, 224, 3).astype(np.float32)
        result = loaded_classifier.predict(frame)
        assert result.modality == Modality.VISUAL

    def test_confidence_in_range(self, loaded_classifier: VisualClassifier) -> None:
        frame = np.random.rand(224, 224, 3).astype(np.float32)
        result = loaded_classifier.predict(frame)
        assert 0.0 <= result.confidence <= 1.0

    def test_species_code_in_label_map(self, loaded_classifier: VisualClassifier) -> None:
        frame = np.random.rand(224, 224, 3).astype(np.float32)
        result = loaded_classifier.predict(frame)
        assert result.species_code in {"HOFI", "MODO", "DOWO"}

    def test_species_code_uppercase(self, loaded_classifier: VisualClassifier) -> None:
        frame = np.random.rand(224, 224, 3).astype(np.float32)
        result = loaded_classifier.predict(frame)
        assert result.species_code == result.species_code.upper()

    def test_model_version_set(self, loaded_classifier: VisualClassifier) -> None:
        frame = np.random.rand(224, 224, 3).astype(np.float32)
        result = loaded_classifier.predict(frame)
        assert result.model_version == VisualClassifier.MODEL_VERSION

    def test_common_name_populated(self, loaded_classifier: VisualClassifier) -> None:
        frame = np.random.rand(224, 224, 3).astype(np.float32)
        result = loaded_classifier.predict(frame)
        assert result.common_name != ""


# ── _build_efficientnet ───────────────────────────────────────────────────────


class TestBuildEfficientnet:
    def test_output_shape_19_classes(self) -> None:
        model = _build_efficientnet(n_classes=19)
        x = torch.zeros(1, 3, 224, 224)
        assert model(x).shape == (1, 19)

    def test_output_shape_various(self) -> None:
        for n in [3, 10, 18, 19]:
            model = _build_efficientnet(n_classes=n)
            x = torch.zeros(1, 3, 224, 224)
            assert model(x).shape == (1, n)
