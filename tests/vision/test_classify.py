"""
tests/vision/test_classify.py

Unit tests for VisualClassifier.

Phase 5 changes from Phase 4:
    - VisualClassifier.__init__() now takes extractor_path and sklearn_path
      instead of model_path and label_map_path.
    - _load() loads two artifacts: frozen_extractor.pt + sklearn_pipeline.pkl.
    - predict() runs EfficientNet feature extraction then sklearn predict_proba().
    - predict() accepts an optional camera_index parameter, passed through to
      ClassificationResult.camera_index.
    - _build_efficientnet() no longer takes n_classes — it's a feature extractor
      (num_classes=0, global_pool="avg") outputting 1280-dim vectors.
    - _extractor replaces _model as the lazy-loaded internal attribute.

Strategy:
    - No real weights required — loaded_classifier fixture bypasses _load()
      by directly populating _extractor, _scaler, _clf, _label_map,
      _species_meta with synthetic equivalents.
    - predict() is tested end-to-end through the synthetic pipeline.
    - from_config() tested with real configs/paths.yaml.
    - _build_efficientnet() tested for correct output dimension (1280).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.schema import ClassificationResult, Modality
from src.vision.classify import VisualClassifier, _build_efficientnet

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def loaded_classifier(tmp_path: Path) -> VisualClassifier:
    """
    VisualClassifier with _load() bypassed — all internals populated directly
    so predict() runs without real artifacts on disk.

    Uses:
        - Real _build_efficientnet() extractor (random ImageNet weights via timm)
        - Synthetic sklearn scaler and LogReg that always predict HOFI
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    clf = VisualClassifier(
        extractor_path=str(tmp_path / "fake_extractor.pt"),
        sklearn_path=str(tmp_path / "fake_sklearn.pkl"),
        species_list_path="configs/species.yaml",
        device="cpu",
    )

    # Real EfficientNet-B0 feature extractor — random weights, outputs 1280-dim
    extractor = _build_efficientnet()
    extractor.eval()
    clf._extractor = extractor

    # Synthetic label map and species metadata
    clf._label_map = {0: "HOFI", 1: "MODO", 2: "DOWO"}
    clf._species_meta = {
        "HOFI": {"common_name": "House Finch", "scientific_name": "Haemorhous mexicanus"},
        "MODO": {"common_name": "Mourning Dove", "scientific_name": "Zenaida macroura"},
        "DOWO": {"common_name": "Downy Woodpecker", "scientific_name": "Dryobates pubescens"},
    }

    # Synthetic sklearn pipeline — fit on random data, 3 classes
    X_fake = np.random.randn(30, 1280).astype(np.float32)
    y_fake = np.array([0, 1, 2] * 10)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fake)
    logreg = LogisticRegression(C=0.1, max_iter=200, random_state=42)
    logreg.fit(X_scaled, y_fake)

    clf._scaler = scaler
    clf._clf = logreg

    return clf


@pytest.fixture()
def classifier_no_artifacts(tmp_path: Path) -> VisualClassifier:
    """VisualClassifier pointing at nonexistent artifact files."""
    return VisualClassifier(
        extractor_path=str(tmp_path / "nonexistent_extractor.pt"),
        sklearn_path=str(tmp_path / "nonexistent_sklearn.pkl"),
        species_list_path="configs/species.yaml",
    )


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestVisualClassifierInit:
    def test_stores_extractor_path(self) -> None:
        clf = VisualClassifier(
            extractor_path="models/visual/frozen_extractor.pt",
            sklearn_path="models/visual/sklearn_pipeline.pkl",
            species_list_path="configs/species.yaml",
        )
        assert clf.extractor_path == Path("models/visual/frozen_extractor.pt")

    def test_stores_sklearn_path(self) -> None:
        clf = VisualClassifier(
            extractor_path="models/visual/frozen_extractor.pt",
            sklearn_path="models/visual/sklearn_pipeline.pkl",
            species_list_path="configs/species.yaml",
        )
        assert clf.sklearn_path == Path("models/visual/sklearn_pipeline.pkl")

    def test_extractor_none_before_load(self) -> None:
        """Lazy loading — extractor not loaded until first predict()."""
        clf = VisualClassifier(
            extractor_path="x.pt",
            sklearn_path="x.pkl",
            species_list_path="configs/species.yaml",
        )
        assert clf._extractor is None

    def test_scaler_none_before_load(self) -> None:
        clf = VisualClassifier(
            extractor_path="x.pt",
            sklearn_path="x.pkl",
            species_list_path="configs/species.yaml",
        )
        assert clf._scaler is None

    def test_device_auto_detected(self) -> None:
        clf = VisualClassifier(
            extractor_path="x.pt",
            sklearn_path="x.pkl",
            species_list_path="configs/species.yaml",
        )
        assert clf.device.type in ("cpu", "cuda")

    def test_device_explicit_cpu(self) -> None:
        clf = VisualClassifier(
            extractor_path="x.pt",
            sklearn_path="x.pkl",
            species_list_path="configs/species.yaml",
            device="cpu",
        )
        assert clf.device == torch.device("cpu")


# ── from_config ───────────────────────────────────────────────────────────────


class TestFromConfig:
    def test_constructs_without_error(self) -> None:
        clf = VisualClassifier.from_config("configs/paths.yaml")
        assert clf is not None

    def test_resolves_extractor_path(self) -> None:
        clf = VisualClassifier.from_config("configs/paths.yaml")
        assert "frozen_extractor" in str(clf.extractor_path)

    def test_resolves_sklearn_path(self) -> None:
        clf = VisualClassifier.from_config("configs/paths.yaml")
        assert "sklearn_pipeline" in str(clf.sklearn_path)

    def test_extractor_not_loaded_after_from_config(self) -> None:
        clf = VisualClassifier.from_config("configs/paths.yaml")
        assert clf._extractor is None

    def test_species_list_path_set(self) -> None:
        clf = VisualClassifier.from_config("configs/paths.yaml")
        assert "species.yaml" in str(clf.species_list_path)


# ── predict — error paths ─────────────────────────────────────────────────────


class TestPredictErrors:
    def test_raises_if_extractor_missing(self, classifier_no_artifacts: VisualClassifier) -> None:
        frame = np.zeros((224, 224, 3), dtype=np.float32)
        with pytest.raises(RuntimeError, match="Frozen extractor not found"):
            classifier_no_artifacts.predict(frame)

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

    def test_camera_index_none_by_default(self, loaded_classifier: VisualClassifier) -> None:
        frame = np.random.rand(224, 224, 3).astype(np.float32)
        result = loaded_classifier.predict(frame)
        assert result.camera_index is None

    def test_camera_index_passed_through(self, loaded_classifier: VisualClassifier) -> None:
        """camera_index is stored on ClassificationResult for fusion tracking."""
        frame = np.random.rand(224, 224, 3).astype(np.float32)
        result = loaded_classifier.predict(frame, camera_index=1)
        assert result.camera_index == 1

    def test_deterministic_on_same_input(self, loaded_classifier: VisualClassifier) -> None:
        """Same frame → same prediction (no randomness in inference)."""
        frame = np.random.rand(224, 224, 3).astype(np.float32)
        result1 = loaded_classifier.predict(frame)
        result2 = loaded_classifier.predict(frame)
        assert result1.species_code == result2.species_code
        assert result1.confidence == pytest.approx(result2.confidence)


# ── _build_efficientnet ───────────────────────────────────────────────────────


class TestBuildEfficientnet:
    def test_output_is_1280_dim_feature_vector(self) -> None:
        """Feature extractor (num_classes=0) outputs 1280-dim vectors, not class logits."""
        model = _build_efficientnet()
        model.eval()
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1280)

    def test_batch_size_preserved(self) -> None:
        model = _build_efficientnet()
        model.eval()
        for batch in [1, 2, 4]:
            x = torch.zeros(batch, 3, 224, 224)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (batch, 1280)

    def test_output_is_finite(self) -> None:
        """No NaN or Inf in feature extractor output."""
        model = _build_efficientnet()
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()
