"""
tests/audio/test_classify.py

Unit tests for AudioClassifier.

Phase 5 changes from Phase 4:
    - AudioClassifier.__init__() now takes only species_list_path and min_conf.
      model_path and label_map_path are gone — BirdNET loads its own weights
      from the birdnetlib package installation.
    - predict() now takes a WAV file path (str | Path), not a spectrogram array.
    - NoBirdDetectedError is raised when BirdNET finds no SD species above min_conf.
    - _analyzer replaces _model as the lazy-loaded internal attribute.
    - _build_audio_cnn() is preserved for training reproducibility but is not
      used for inference. Its tests remain unchanged.

Strategy:
    - No real audio files, no BirdNET weights, no network access required.
    - predict() is tested by mocking _analyzer and controlling birdnetlib
      Recording behavior via monkeypatching.
    - from_config() verified with real configs/paths.yaml.
    - NoBirdDetectedError tested for all cases: no detections, no SD match,
      inference failure.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.audio.classify import AudioClassifier, NoBirdDetectedError, _build_audio_cnn
from src.data.schema import ClassificationResult, Modality

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def classifier(tmp_path: Path) -> AudioClassifier:
    """AudioClassifier with species_list_path pointing at real configs/species.yaml."""
    return AudioClassifier(species_list_path="configs/species.yaml")


@pytest.fixture()
def loaded_classifier(tmp_path: Path) -> AudioClassifier:
    """
    AudioClassifier with _load() bypassed — _analyzer and lookup tables
    populated directly so predict() can run without BirdNET weights.

    predict() is tested by mocking the birdnetlib Recording class.
    """
    clf = AudioClassifier(species_list_path="configs/species.yaml")

    # Populate lookup tables directly — mirrors what _load() would do
    clf._sci_to_code = {
        "Haemorhous mexicanus": "HOFI",
        "Zenaida macroura": "MODO",
        "Calypte anna": "ANHU",
    }
    clf._species_meta = {
        "HOFI": {"common_name": "House Finch", "scientific_name": "Haemorhous mexicanus"},
        "MODO": {"common_name": "Mourning Dove", "scientific_name": "Zenaida macroura"},
        "ANHU": {"common_name": "Anna's Hummingbird", "scientific_name": "Calypte anna"},
    }
    # Stub analyzer — the object just needs to exist; Recording is mocked per-test
    clf._analyzer = MagicMock()
    return clf


@pytest.fixture()
def fake_wav(tmp_path: Path) -> Path:
    """Write a minimal valid WAV file to tmp_path."""
    import struct
    import wave

    path = tmp_path / "test.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        # 3 seconds of silence = 144000 samples
        wf.writeframes(struct.pack("<144000h", *([0] * 144000)))
    return path


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestAudioClassifierInit:
    def test_stores_species_list_path(self) -> None:
        clf = AudioClassifier(species_list_path="configs/species.yaml")
        assert clf.species_list_path == Path("configs/species.yaml")

    def test_stores_min_conf(self) -> None:
        clf = AudioClassifier(species_list_path="configs/species.yaml", min_conf=0.25)
        assert clf.min_conf == 0.25

    def test_default_min_conf(self) -> None:
        clf = AudioClassifier(species_list_path="configs/species.yaml")
        assert clf.min_conf == 0.1

    def test_analyzer_none_before_load(self) -> None:
        """Lazy loading — BirdNET Analyzer not loaded until first predict()."""
        clf = AudioClassifier(species_list_path="configs/species.yaml")
        assert clf._analyzer is None

    def test_model_none_before_load(self) -> None:
        """CNN _model kept for backward compat — still None before load."""
        clf = AudioClassifier(species_list_path="configs/species.yaml")
        assert clf._model is None


# ── from_config ───────────────────────────────────────────────────────────────


class TestFromConfig:
    def test_constructs_without_error(self) -> None:
        clf = AudioClassifier.from_config("configs/paths.yaml")
        assert clf is not None

    def test_species_list_path_points_to_species_yaml(self) -> None:
        clf = AudioClassifier.from_config("configs/paths.yaml")
        assert "species.yaml" in str(clf.species_list_path)

    def test_analyzer_not_loaded_after_from_config(self) -> None:
        clf = AudioClassifier.from_config("configs/paths.yaml")
        assert clf._analyzer is None

    def test_model_version_set(self) -> None:
        assert "birdnet" in AudioClassifier.MODEL_VERSION.lower()


# ── predict — file not found ──────────────────────────────────────────────────


class TestPredictFileNotFound:
    def test_raises_file_not_found(
        self, loaded_classifier: AudioClassifier, tmp_path: Path
    ) -> None:
        missing = tmp_path / "nonexistent.wav"
        with pytest.raises(FileNotFoundError):
            loaded_classifier.predict(missing)

    def test_accepts_path_object(self, loaded_classifier: AudioClassifier, fake_wav: Path) -> None:
        """predict() accepts Path objects — test with mocked Recording."""
        mock_recording = MagicMock()
        mock_recording.detections = [
            {"scientific_name": "Haemorhous mexicanus", "confidence": 0.85}
        ]
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            result = loaded_classifier.predict(fake_wav)
        assert isinstance(result, ClassificationResult)

    def test_accepts_string_path(self, loaded_classifier: AudioClassifier, fake_wav: Path) -> None:
        mock_recording = MagicMock()
        mock_recording.detections = [
            {"scientific_name": "Haemorhous mexicanus", "confidence": 0.85}
        ]
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            result = loaded_classifier.predict(str(fake_wav))
        assert isinstance(result, ClassificationResult)


# ── predict — NoBirdDetectedError cases ──────────────────────────────────────


class TestNoBirdDetectedError:
    def test_raises_when_no_detections(
        self, loaded_classifier: AudioClassifier, fake_wav: Path
    ) -> None:
        """BirdNET returns empty detections list."""
        mock_recording = MagicMock()
        mock_recording.detections = []
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            with pytest.raises(NoBirdDetectedError):
                loaded_classifier.predict(fake_wav)

    def test_raises_when_no_sd_species_match(
        self, loaded_classifier: AudioClassifier, fake_wav: Path
    ) -> None:
        """BirdNET detects a species not in our SD list."""
        mock_recording = MagicMock()
        mock_recording.detections = [
            {"scientific_name": "Corvus corax", "confidence": 0.9}  # Common Raven — not in SD list
        ]
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            with pytest.raises(NoBirdDetectedError):
                loaded_classifier.predict(fake_wav)

    def test_raises_when_birdnet_inference_fails(
        self, loaded_classifier: AudioClassifier, fake_wav: Path
    ) -> None:
        """BirdNET Recording.analyze() raises an exception."""
        mock_recording = MagicMock()
        mock_recording.analyze.side_effect = RuntimeError("tflite error")
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            with pytest.raises(NoBirdDetectedError):
                loaded_classifier.predict(fake_wav)


# ── predict — happy path ──────────────────────────────────────────────────────


class TestPredictOutput:
    def test_returns_classification_result(
        self, loaded_classifier: AudioClassifier, fake_wav: Path
    ) -> None:
        mock_recording = MagicMock()
        mock_recording.detections = [
            {"scientific_name": "Haemorhous mexicanus", "confidence": 0.85}
        ]
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            result = loaded_classifier.predict(fake_wav)
        assert isinstance(result, ClassificationResult)

    def test_modality_is_audio(self, loaded_classifier: AudioClassifier, fake_wav: Path) -> None:
        mock_recording = MagicMock()
        mock_recording.detections = [
            {"scientific_name": "Haemorhous mexicanus", "confidence": 0.85}
        ]
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            result = loaded_classifier.predict(fake_wav)
        assert result.modality == Modality.AUDIO

    def test_confidence_in_range(self, loaded_classifier: AudioClassifier, fake_wav: Path) -> None:
        mock_recording = MagicMock()
        mock_recording.detections = [
            {"scientific_name": "Haemorhous mexicanus", "confidence": 0.85}
        ]
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            result = loaded_classifier.predict(fake_wav)
        assert 0.0 <= result.confidence <= 1.0

    def test_species_code_matches_detection(
        self, loaded_classifier: AudioClassifier, fake_wav: Path
    ) -> None:
        mock_recording = MagicMock()
        mock_recording.detections = [
            {"scientific_name": "Haemorhous mexicanus", "confidence": 0.85}
        ]
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            result = loaded_classifier.predict(fake_wav)
        assert result.species_code == "HOFI"

    def test_species_code_uppercase(
        self, loaded_classifier: AudioClassifier, fake_wav: Path
    ) -> None:
        mock_recording = MagicMock()
        mock_recording.detections = [{"scientific_name": "Zenaida macroura", "confidence": 0.75}]
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            result = loaded_classifier.predict(fake_wav)
        assert result.species_code == result.species_code.upper()

    def test_model_version_set(self, loaded_classifier: AudioClassifier, fake_wav: Path) -> None:
        mock_recording = MagicMock()
        mock_recording.detections = [
            {"scientific_name": "Haemorhous mexicanus", "confidence": 0.85}
        ]
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            result = loaded_classifier.predict(fake_wav)
        assert result.model_version == AudioClassifier.MODEL_VERSION

    def test_picks_highest_confidence_sd_detection(
        self, loaded_classifier: AudioClassifier, fake_wav: Path
    ) -> None:
        """When multiple SD species detected, highest confidence wins."""
        mock_recording = MagicMock()
        mock_recording.detections = [
            {"scientific_name": "Zenaida macroura", "confidence": 0.6},
            {"scientific_name": "Haemorhous mexicanus", "confidence": 0.85},
            {"scientific_name": "Calypte anna", "confidence": 0.4},
        ]
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            result = loaded_classifier.predict(fake_wav)
        assert result.species_code == "HOFI"
        assert result.confidence == pytest.approx(0.85)

    def test_common_name_populated(
        self, loaded_classifier: AudioClassifier, fake_wav: Path
    ) -> None:
        mock_recording = MagicMock()
        mock_recording.detections = [
            {"scientific_name": "Haemorhous mexicanus", "confidence": 0.85}
        ]
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            result = loaded_classifier.predict(fake_wav)
        assert result.common_name == "House Finch"

    def test_non_sd_detections_filtered_out(
        self, loaded_classifier: AudioClassifier, fake_wav: Path
    ) -> None:
        """Non-SD species in detections are ignored; SD species still found."""
        mock_recording = MagicMock()
        mock_recording.detections = [
            {"scientific_name": "Corvus corax", "confidence": 0.95},  # not in SD list
            {"scientific_name": "Haemorhous mexicanus", "confidence": 0.72},  # SD species
        ]
        with patch("src.audio.classify.Recording", return_value=mock_recording):
            result = loaded_classifier.predict(fake_wav)
        assert result.species_code == "HOFI"


# ── _build_audio_cnn — unchanged from Phase 4 ────────────────────────────────


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
