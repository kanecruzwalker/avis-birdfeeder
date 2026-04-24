"""Unit tests for tools.labeler.schema.

Tests Pydantic validation logic, species code normalization, and the
shape of the persisted records. No Gemini calls here.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tools.labeler.schema import (
    ALLOWED_CODES,
    KNOWN_SPECIES_CODES,
    PreLabel,
    PreLabelResponse,
    VerifiedLabel,
)


class TestPreLabelResponse:
    """LLM-response schema that with_structured_output constrains Gemini to."""

    def test_valid_bird_response(self) -> None:
        r = PreLabelResponse(
            bird_visible=True,
            species_code="HOFI",
            confidence=0.92,
            reasoning="Small streaky brown bird with red head, consistent with male HOFI.",
        )
        assert r.species_code == "HOFI"
        assert r.bird_visible is True

    def test_valid_none_response(self) -> None:
        r = PreLabelResponse(
            bird_visible=False,
            species_code="NONE",
            confidence=1.0,
            reasoning="Empty feeder with seed visible, no bird present.",
        )
        assert r.species_code == "NONE"

    def test_species_code_normalised_to_uppercase(self) -> None:
        r = PreLabelResponse(
            bird_visible=True,
            species_code="hofi",
            confidence=0.9,
            reasoning="test",
        )
        assert r.species_code == "HOFI"

    def test_species_code_stripped_of_whitespace(self) -> None:
        r = PreLabelResponse(
            bird_visible=True,
            species_code="  HOFI  ",
            confidence=0.9,
            reasoning="test",
        )
        assert r.species_code == "HOFI"

    def test_invalid_species_code_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not in allowed codes"):
            PreLabelResponse(
                bird_visible=True,
                species_code="NOTABIRD",
                confidence=0.9,
                reasoning="test",
            )

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PreLabelResponse(
                bird_visible=True,
                species_code="HOFI",
                confidence=-0.1,
                reasoning="test",
            )

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PreLabelResponse(
                bird_visible=True,
                species_code="HOFI",
                confidence=1.5,
                reasoning="test",
            )

    def test_uncertain_between_normalised(self) -> None:
        r = PreLabelResponse(
            bird_visible=True,
            species_code="HOFI",
            confidence=0.5,
            reasoning="Could be HOFI or HOSP from this angle.",
            uncertain_between=["hosp", "  LEGO  "],
        )
        assert r.uncertain_between == ["HOSP", "LEGO"]

    def test_uncertain_between_rejects_invalid_codes(self) -> None:
        with pytest.raises(ValidationError, match="invalid codes"):
            PreLabelResponse(
                bird_visible=True,
                species_code="HOFI",
                confidence=0.5,
                reasoning="test",
                uncertain_between=["HOSP", "NOTABIRD"],
            )

    def test_uncertain_between_defaults_to_none(self) -> None:
        r = PreLabelResponse(
            bird_visible=True,
            species_code="HOFI",
            confidence=0.9,
            reasoning="test",
        )
        assert r.uncertain_between is None


class TestPreLabel:
    """Persisted record written to pre_labels.jsonl."""

    def _response(self) -> PreLabelResponse:
        return PreLabelResponse(
            bird_visible=True,
            species_code="HOFI",
            confidence=0.9,
            reasoning="test reasoning",
        )

    def test_minimal_valid_prelabel(self) -> None:
        p = PreLabel(
            image_path="/fake/path/img.png",
            image_filename="img.png",
            llm_response=self._response(),
            model_name="gemini-2.5-flash",
            prompt_version="v1.0",
            elapsed_seconds=2.5,
        )
        assert p.image_filename == "img.png"
        assert p.audio_hint is None
        assert p.capture_timestamp is None

    def test_prelabel_with_audio_hint(self) -> None:
        p = PreLabel(
            image_path="/fake/path/img.png",
            image_filename="img.png",
            audio_hint="HOFI",
            audio_confidence=0.85,
            llm_response=self._response(),
            model_name="gemini-2.5-flash",
            prompt_version="v1.0",
            elapsed_seconds=2.5,
        )
        assert p.audio_hint == "HOFI"
        assert p.audio_confidence == 0.85

    def test_labeled_at_defaults_to_now(self) -> None:
        before = datetime.now(UTC)
        p = PreLabel(
            image_path="/fake/path/img.png",
            image_filename="img.png",
            llm_response=self._response(),
            model_name="gemini-2.5-flash",
            prompt_version="v1.0",
            elapsed_seconds=2.5,
        )
        after = datetime.now(UTC)
        assert before <= p.labeled_at <= after

    def test_serialises_to_json_round_trip(self) -> None:
        p = PreLabel(
            image_path="/fake/path/img.png",
            image_filename="img.png",
            audio_hint="HOFI",
            audio_confidence=0.85,
            llm_response=self._response(),
            model_name="gemini-2.5-flash",
            prompt_version="v1.0",
            elapsed_seconds=2.5,
        )
        json_str = p.model_dump_json()
        reconstructed = PreLabel.model_validate_json(json_str)
        assert reconstructed.image_filename == p.image_filename
        assert reconstructed.llm_response.species_code == "HOFI"
        assert reconstructed.audio_confidence == 0.85


class TestVerifiedLabel:
    """Human-review output schema (consumed by future Layer 2 PR)."""

    def test_verified_label_valid(self) -> None:
        v = VerifiedLabel(
            image_path="/fake/path/img.png",
            image_filename="img.png",
            species_code="HOFI",
        )
        assert v.species_code == "HOFI"

    def test_verified_label_rejects_invalid_code(self) -> None:
        with pytest.raises(ValidationError):
            VerifiedLabel(
                image_path="/fake/path/img.png",
                image_filename="img.png",
                species_code="NOTABIRD",
            )

    def test_verified_label_accepts_none_sentinel(self) -> None:
        v = VerifiedLabel(
            image_path="/fake/path/img.png",
            image_filename="img.png",
            species_code="NONE",
        )
        assert v.species_code == "NONE"


class TestAllowedCodes:
    """Verify the allowed-codes constants are correctly built."""

    def test_twenty_known_species(self) -> None:
        assert len(KNOWN_SPECIES_CODES) == 20

    def test_allowed_codes_includes_sentinels(self) -> None:
        assert "NONE" in ALLOWED_CODES
        assert "UNKNOWN" in ALLOWED_CODES

    def test_no_duplicates(self) -> None:
        assert len(ALLOWED_CODES) == len(set(ALLOWED_CODES))
