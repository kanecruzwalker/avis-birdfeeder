"""Unit tests for tools.labeler.schema.

Tests Pydantic validation logic, species code normalization, and the
shape of the persisted records. No Gemini calls here.

Test classes (in order):
- TestPreLabelResponse — LLM-response schema
- TestPreLabel — persisted pre-label record
- TestVerifiedLabel — human-review output schema (legacy subset)
- TestAllowedCodes — constants sanity checks
- TestOtherSentinel — NEW: OTHER sentinel and other_species_code rules
- TestVerifiedLabelOther — NEW: VerifiedLabel-specific OTHER handling
- TestBackwardCompat — NEW: real pre_labels.jsonl records parse unchanged
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tools.labeler.schema import (
    ALLOWED_CODES,
    KNOWN_SPECIES_CODES,
    SENTINEL_NO_BIRD,
    SENTINEL_OTHER,
    SENTINEL_UNKNOWN,
    SENTINELS,
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

    def test_other_species_code_defaults_to_none(self) -> None:
        """New field added in Layer 2 — must default to None so existing
        records parse without it."""
        r = PreLabelResponse(
            bird_visible=True,
            species_code="HOFI",
            confidence=0.9,
            reasoning="test",
        )
        assert r.other_species_code is None


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
    """Human-review output schema (legacy subset — see TestVerifiedLabelOther
    for the full Layer 2 review UI test set)."""

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

    def test_allowed_codes_includes_all_sentinels(self) -> None:
        assert SENTINEL_NO_BIRD in ALLOWED_CODES
        assert SENTINEL_UNKNOWN in ALLOWED_CODES
        assert SENTINEL_OTHER in ALLOWED_CODES

    def test_sentinels_tuple_matches_constants(self) -> None:
        assert SENTINELS == (SENTINEL_NO_BIRD, SENTINEL_UNKNOWN, SENTINEL_OTHER)

    def test_no_duplicates(self) -> None:
        assert len(ALLOWED_CODES) == len(set(ALLOWED_CODES))

    def test_allowed_codes_length(self) -> None:
        # 20 species + 3 sentinels
        assert len(ALLOWED_CODES) == 23


class TestOtherSentinel:
    """OTHER sentinel + other_species_code invariants, introduced in Layer 2."""

    def test_other_with_valid_candidate_code(self) -> None:
        """A bird that's genuinely out-of-vocab: OTHER + CALT for California
        Towhee. This is the core new capability for Layer 2."""
        r = PreLabelResponse(
            bird_visible=True,
            species_code="OTHER",
            confidence=0.8,
            reasoning="Brown towhee-like bird with long tail, not in the 20-species list.",
            other_species_code="CALT",
        )
        assert r.species_code == "OTHER"
        assert r.other_species_code == "CALT"

    def test_other_without_other_species_code_rejected(self) -> None:
        """species_code=OTHER without an other_species_code is meaningless data."""
        with pytest.raises(ValidationError, match="requires other_species_code"):
            PreLabelResponse(
                bird_visible=True,
                species_code="OTHER",
                confidence=0.8,
                reasoning="test",
            )

    def test_other_species_code_without_other_sentinel_rejected(self) -> None:
        """A record with species_code=HOFI and other_species_code=CALT is
        incoherent — reviewer must pick one mode, not both."""
        with pytest.raises(ValidationError, match="only valid when species_code='OTHER'"):
            PreLabelResponse(
                bird_visible=True,
                species_code="HOFI",
                confidence=0.9,
                reasoning="test",
                other_species_code="CALT",
            )

    def test_other_species_code_collision_with_known_rejected(self) -> None:
        """If the reviewer types a code that matches a known species,
        make them use species_code= directly rather than OTHER + known-code."""
        with pytest.raises(ValidationError, match="already a known species"):
            PreLabelResponse(
                bird_visible=True,
                species_code="OTHER",
                confidence=0.8,
                reasoning="test",
                other_species_code="HOFI",
            )

    def test_other_species_code_collision_with_sentinel_rejected(self) -> None:
        """NONE / UNKNOWN / OTHER cannot be used as other_species_code."""
        with pytest.raises(ValidationError, match="already a known species"):
            PreLabelResponse(
                bird_visible=True,
                species_code="OTHER",
                confidence=0.8,
                reasoning="test",
                other_species_code="NONE",
            )

    def test_other_species_code_format_three_letters_rejected(self) -> None:
        """Candidate codes must be exactly 4 letters to match AOU convention."""
        with pytest.raises(ValidationError, match="4 uppercase letters"):
            PreLabelResponse(
                bird_visible=True,
                species_code="OTHER",
                confidence=0.8,
                reasoning="test",
                other_species_code="CAT",
            )

    def test_other_species_code_format_with_digits_rejected(self) -> None:
        """Only letters — AOU codes don't use digits."""
        with pytest.raises(ValidationError, match="4 uppercase letters"):
            PreLabelResponse(
                bird_visible=True,
                species_code="OTHER",
                confidence=0.8,
                reasoning="test",
                other_species_code="CA1T",
            )

    def test_other_species_code_lowercase_normalised(self) -> None:
        """Reviewer can type lowercase; schema normalises."""
        r = PreLabelResponse(
            bird_visible=True,
            species_code="OTHER",
            confidence=0.8,
            reasoning="test",
            other_species_code="calt",
        )
        assert r.other_species_code == "CALT"

    def test_other_species_code_whitespace_stripped(self) -> None:
        r = PreLabelResponse(
            bird_visible=True,
            species_code="OTHER",
            confidence=0.8,
            reasoning="test",
            other_species_code="  CALT  ",
        )
        assert r.other_species_code == "CALT"

    def test_other_round_trips_through_json(self) -> None:
        """Verify the persistence path doesn't drop other_species_code."""
        r = PreLabelResponse(
            bird_visible=True,
            species_code="OTHER",
            confidence=0.8,
            reasoning="Out-of-vocab bird.",
            other_species_code="CALT",
        )
        p = PreLabel(
            image_path="/fake/path/img.png",
            image_filename="img.png",
            llm_response=r,
            model_name="gemini-2.5-flash",
            prompt_version="v1.1",
            elapsed_seconds=2.5,
        )
        reconstructed = PreLabel.model_validate_json(p.model_dump_json())
        assert reconstructed.llm_response.species_code == "OTHER"
        assert reconstructed.llm_response.other_species_code == "CALT"


class TestVerifiedLabelOther:
    """VerifiedLabel parallels the PreLabelResponse OTHER rules."""

    def test_verified_label_with_other(self) -> None:
        v = VerifiedLabel(
            image_path="/fake/path/img.png",
            image_filename="img.png",
            species_code="OTHER",
            other_species_code="CALT",
        )
        assert v.species_code == "OTHER"
        assert v.other_species_code == "CALT"

    def test_verified_label_other_without_code_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires other_species_code"):
            VerifiedLabel(
                image_path="/fake/path/img.png",
                image_filename="img.png",
                species_code="OTHER",
            )

    def test_verified_label_code_without_other_rejected(self) -> None:
        with pytest.raises(ValidationError, match="only valid when species_code='OTHER'"):
            VerifiedLabel(
                image_path="/fake/path/img.png",
                image_filename="img.png",
                species_code="HOFI",
                other_species_code="CALT",
            )

    def test_verified_label_with_reviewer_notes(self) -> None:
        v = VerifiedLabel(
            image_path="/fake/path/img.png",
            image_filename="img.png",
            species_code="OTHER",
            other_species_code="CALT",
            reviewer_notes="Distinctive rusty undertail, California Towhee.",
        )
        assert "California Towhee" in v.reviewer_notes

    def test_verified_label_tracks_agreement(self) -> None:
        """agreed_with_pre_label=True/False/None lets Layer 3 analyse
        pre-label accuracy without comparing embedded pre_label."""
        v_agreed = VerifiedLabel(
            image_path="/fake/path/img.png",
            image_filename="img.png",
            species_code="HOFI",
            agreed_with_pre_label=True,
        )
        v_corrected = VerifiedLabel(
            image_path="/fake/path/img.png",
            image_filename="img.png",
            species_code="OTHER",
            other_species_code="CALT",
            agreed_with_pre_label=False,
        )
        assert v_agreed.agreed_with_pre_label is True
        assert v_corrected.agreed_with_pre_label is False


class TestBackwardCompat:
    """Real records from the existing pre_labels.jsonl must parse unchanged."""

    def test_real_record_utc_timestamp(self) -> None:
        """Record 1 from Kane's actual pre_labels.jsonl (smoke test image)."""
        record = {
            "image_path": r"C:\Users\Miki\Desktop\Kane\avis\data\smoke_test_images\20260424_183619_406198_cam0.png",
            "image_filename": "20260424_183619_406198_cam0.png",
            "capture_timestamp": "2026-04-24T18:36:19.406198+00:00",
            "observation_timestamp": None,
            "audio_hint": None,
            "audio_confidence": None,
            "llm_response": {
                "bird_visible": False,
                "species_code": "NONE",
                "confidence": 1.0,
                "reasoning": "The feeder is clearly visible with seed and an orange slice, but no bird is present on or around it.",
                "uncertain_between": None,
            },
            "model_name": "gemini-2.5-flash",
            "prompt_version": "v1.0",
            "labeled_at": "2026-04-24T18:55:35.770625+00:00",
            "elapsed_seconds": 3.327999999994063,
        }
        p = PreLabel.model_validate(record)
        assert p.image_filename == "20260424_183619_406198_cam0.png"
        assert p.llm_response.species_code == "NONE"
        # New field defaults to None for pre-v1.1 records
        assert p.llm_response.other_species_code is None

    def test_real_record_pdt_offset_timestamp(self) -> None:
        """Record 2 from Kane's actual pre_labels.jsonl (real capture,
        tzinfo=-07:00 instead of +00:00)."""
        record = {
            "image_path": r"C:\Users\Miki\Desktop\Kane\avis\data\captures\images\20260424_133057_292152_cam1.png",
            "image_filename": "20260424_133057_292152_cam1.png",
            "capture_timestamp": "2026-04-24T06:30:57.292152-07:00",
            "observation_timestamp": "2026-04-24T06:31:00.832133-07:00",
            "audio_hint": None,
            "audio_confidence": None,
            "llm_response": {
                "bird_visible": False,
                "species_code": "NONE",
                "confidence": 0.95,
                "reasoning": "The image clearly shows an empty bird feeder with seed and orange slices. No bird is visible on or around the feeder.",
                "uncertain_between": None,
            },
            "model_name": "gemini-2.5-flash",
            "prompt_version": "v1.0",
            "labeled_at": "2026-04-24T19:29:05.622596-07:00",
            "elapsed_seconds": 2.985000000000582,
        }
        p = PreLabel.model_validate(record)
        assert p.image_filename == "20260424_133057_292152_cam1.png"
        assert p.capture_timestamp is not None
        assert p.llm_response.other_species_code is None
