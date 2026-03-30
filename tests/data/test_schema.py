"""
tests/data/test_schema.py

Unit tests for src/data/schema.py

These tests have zero hardware or model dependencies and run in CI on every push.
They verify that our shared data contracts (ClassificationResult, BirdObservation)
behave correctly — including validation, field coercion, and edge cases.

If these tests break, something fundamental to the whole pipeline has changed.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.data.schema import BirdObservation, ClassificationResult, Modality

# ── ClassificationResult ──────────────────────────────────────────────────────


class TestClassificationResult:
    def test_valid_construction(self):
        """A fully specified ClassificationResult should construct without error."""
        result = ClassificationResult(
            species_code="amro",  # lowercase — should be coerced to uppercase
            common_name="American Robin",
            scientific_name="Turdus migratorius",
            confidence=0.92,
            modality=Modality.AUDIO,
        )
        assert result.species_code == "AMRO"
        assert result.confidence == 0.92
        assert result.modality == Modality.AUDIO

    def test_species_code_coerced_to_uppercase(self):
        """Species codes must always be uppercase regardless of input case."""
        result = ClassificationResult(
            species_code="hofi",
            common_name="House Finch",
            scientific_name="Haemorhous mexicanus",
            confidence=0.75,
            modality=Modality.VISUAL,
        )
        assert result.species_code == "HOFI"

    def test_confidence_boundary_zero(self):
        """Confidence of exactly 0.0 should be valid."""
        result = ClassificationResult(
            species_code="AMRO",
            common_name="American Robin",
            scientific_name="Turdus migratorius",
            confidence=0.0,
            modality=Modality.AUDIO,
        )
        assert result.confidence == 0.0

    def test_confidence_boundary_one(self):
        """Confidence of exactly 1.0 should be valid."""
        result = ClassificationResult(
            species_code="AMRO",
            common_name="American Robin",
            scientific_name="Turdus migratorius",
            confidence=1.0,
            modality=Modality.VISUAL,
        )
        assert result.confidence == 1.0

    def test_confidence_below_zero_raises(self):
        """Confidence below 0.0 is invalid and should raise ValidationError."""
        with pytest.raises(ValidationError):
            ClassificationResult(
                species_code="AMRO",
                common_name="American Robin",
                scientific_name="Turdus migratorius",
                confidence=-0.1,
                modality=Modality.AUDIO,
            )

    def test_confidence_above_one_raises(self):
        """Confidence above 1.0 is invalid and should raise ValidationError."""
        with pytest.raises(ValidationError):
            ClassificationResult(
                species_code="AMRO",
                common_name="American Robin",
                scientific_name="Turdus migratorius",
                confidence=1.01,
                modality=Modality.VISUAL,
            )

    def test_default_timestamp_is_set(self):
        """timestamp should default to a recent UTC datetime if not provided."""
        before = datetime.now(UTC)
        result = ClassificationResult(
            species_code="AMRO",
            common_name="American Robin",
            scientific_name="Turdus migratorius",
            confidence=0.8,
            modality=Modality.AUDIO,
        )
        after = datetime.now(UTC)
        assert before <= result.timestamp <= after

    def test_default_model_version(self):
        """model_version should default to 'unknown' when not specified."""
        result = ClassificationResult(
            species_code="AMRO",
            common_name="American Robin",
            scientific_name="Turdus migratorius",
            confidence=0.8,
            modality=Modality.AUDIO,
        )
        assert result.model_version == "unknown"

    def test_modality_enum_values(self):
        """All three Modality variants should be constructable."""
        for modality in (Modality.AUDIO, Modality.VISUAL, Modality.FUSED):
            result = ClassificationResult(
                species_code="AMRO",
                common_name="American Robin",
                scientific_name="Turdus migratorius",
                confidence=0.5,
                modality=modality,
            )
            assert result.modality == modality


# ── BirdObservation ───────────────────────────────────────────────────────────


def _make_result(modality: Modality, confidence: float = 0.85) -> ClassificationResult:
    """Helper: build a valid ClassificationResult for use in BirdObservation tests."""
    return ClassificationResult(
        species_code="AMRO",
        common_name="American Robin",
        scientific_name="Turdus migratorius",
        confidence=confidence,
        modality=modality,
    )


class TestBirdObservation:
    def test_valid_construction_with_both_modalities(self):
        """BirdObservation with both audio and visual results should construct."""
        obs = BirdObservation(
            species_code="AMRO",
            common_name="American Robin",
            scientific_name="Turdus migratorius",
            fused_confidence=0.88,
            audio_result=_make_result(Modality.AUDIO),
            visual_result=_make_result(Modality.VISUAL),
        )
        assert obs.species_code == "AMRO"
        assert obs.fused_confidence == 0.88
        assert obs.has_both_modalities is True

    def test_valid_construction_audio_only(self):
        """BirdObservation with only audio result should construct (visual is optional)."""
        obs = BirdObservation(
            species_code="AMRO",
            common_name="American Robin",
            scientific_name="Turdus migratorius",
            fused_confidence=0.80,
            audio_result=_make_result(Modality.AUDIO),
        )
        assert obs.visual_result is None
        assert obs.has_both_modalities is False

    def test_valid_construction_visual_only(self):
        """BirdObservation with only visual result should construct (audio is optional)."""
        obs = BirdObservation(
            species_code="AMRO",
            common_name="American Robin",
            scientific_name="Turdus migratorius",
            fused_confidence=0.78,
            visual_result=_make_result(Modality.VISUAL),
        )
        assert obs.audio_result is None
        assert obs.has_both_modalities is False

    def test_species_code_coerced_to_uppercase(self):
        """Species codes on BirdObservation should also be uppercased."""
        obs = BirdObservation(
            species_code="hofi",
            common_name="House Finch",
            scientific_name="Haemorhous mexicanus",
            fused_confidence=0.72,
        )
        assert obs.species_code == "HOFI"

    def test_optional_media_paths_default_none(self):
        """image_path and audio_path should default to None."""
        obs = BirdObservation(
            species_code="AMRO",
            common_name="American Robin",
            scientific_name="Turdus migratorius",
            fused_confidence=0.85,
        )
        assert obs.image_path is None
        assert obs.audio_path is None

    def test_fused_confidence_below_zero_raises(self):
        """fused_confidence below 0.0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            BirdObservation(
                species_code="AMRO",
                common_name="American Robin",
                scientific_name="Turdus migratorius",
                fused_confidence=-0.1,
            )

    def test_json_serialization_roundtrip(self):
        """BirdObservation should serialize to JSON and deserialize back cleanly."""
        obs = BirdObservation(
            species_code="AMRO",
            common_name="American Robin",
            scientific_name="Turdus migratorius",
            fused_confidence=0.91,
            audio_result=_make_result(Modality.AUDIO),
        )
        json_str = obs.model_dump_json()
        obs2 = BirdObservation.model_validate_json(json_str)
        assert obs2.species_code == obs.species_code
        assert obs2.fused_confidence == obs.fused_confidence
        assert obs2.audio_result is not None
