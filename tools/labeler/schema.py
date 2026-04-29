"""Pydantic schemas for the labeling assistant.

Defines the structured data produced by the pre-labeler and consumed
by downstream stages (review UI, training data export).

Design choices:
- `PreLabelResponse` is what Gemini returns — a flat, LLM-friendly shape
  that LangChain's `with_structured_output` can constrain.
- `PreLabel` is what we PERSIST — `PreLabelResponse` plus filesystem,
  timing, audio context, and prompt-version metadata. This lets us
  reproduce or re-analyse a label run without re-querying Gemini.
- Species codes match the 4-letter codes in configs/species.yaml, plus
  three sentinels: "NONE" (empty feeder), "UNKNOWN" (bird visible but
  Gemini unsure), and "OTHER" (bird visible, species is out-of-vocabulary
  — reviewer-side construct introduced in Layer 2 review UI).
- `VerifiedLabel` is produced by the human-review UI (Layer 2).
  Defining it here keeps schema evolution in one file.

Schema compatibility notes:
- Adding the OTHER sentinel and `other_species_code` field is ADDITIVE.
  The existing 2828 records in pre_labels.jsonl were written by Layer 1
  prompt v1.0 which does not emit OTHER — they parse unchanged.
- `other_species_code` is optional and defaults to None, so records
  without it validate fine.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Species code constants ────────────────────────────────────────────────────

# 20 SD species from configs/species.yaml plus sentinel codes.
# Kept in sync manually — if configs/species.yaml changes, update this list
# and re-run the pre-labeler so Gemini sees the current species set.
KNOWN_SPECIES_CODES: tuple[str, ...] = (
    "HOFI", "MODO", "ANHU", "CAVI", "MOCH",
    "AMRO", "SOSP", "LEGO", "DOWO", "WREN",
    "AMCR", "SPTO", "BLPH", "HOSP", "EUST",
    "WCSP", "HOORI", "WBNU", "OCWA", "YRUM",
)

# Sentinels that may appear in species_code in place of a 4-letter code.
SENTINEL_NO_BIRD = "NONE"      # feeder is empty / no bird visible
SENTINEL_UNKNOWN = "UNKNOWN"   # bird visible but species truly cannot be determined
SENTINEL_OTHER = "OTHER"       # bird visible, species is out-of-vocabulary
                               # (reviewer-side only in prompt v1.0 —
                               # see docs/investigations/labeling-assistant-ui-2026-04-25.md)

SENTINELS: tuple[str, ...] = (SENTINEL_NO_BIRD, SENTINEL_UNKNOWN, SENTINEL_OTHER)
ALLOWED_CODES: tuple[str, ...] = KNOWN_SPECIES_CODES + SENTINELS

# Pattern for validating `other_species_code` — 4 uppercase letters, same
# shape as the AOU codes we already use for known species. Reviewer types
# a candidate code (e.g. "CALT" for California Towhee) when a bird is
# visible but not in the 20-species list.
_OTHER_CODE_PATTERN = re.compile(r"^[A-Z]{4}$")


def _normalise_species_code(v: str) -> str:
    """Strip whitespace, uppercase, return; let the caller validate membership."""
    return v.strip().upper()


def _validate_other_code(v: str | None) -> str | None:
    """Validate the out-of-vocab candidate code format.

    Returns the normalised (uppercase, stripped) code or None. Raises
    ValueError if the code is present but not a valid 4-letter pattern,
    or if it collides with a known species code or sentinel.
    """
    if v is None:
        return None
    normalised = _normalise_species_code(v)
    if not _OTHER_CODE_PATTERN.match(normalised):
        raise ValueError(
            f"other_species_code '{v}' must be 4 uppercase letters "
            f"(AOU-style code, e.g. 'CALT' for California Towhee)."
        )
    if normalised in ALLOWED_CODES:
        raise ValueError(
            f"other_species_code '{normalised}' is already a known species "
            f"or sentinel — use species_code='{normalised}' directly instead."
        )
    return normalised


# ── LLM response shape (what Gemini is asked to return) ──────────────────────


class PreLabelResponse(BaseModel):
    """Structured response Gemini produces for a single image.

    This is the shape passed to `ChatGoogleGenerativeAI.with_structured_output()`.
    Keep this class minimal and LLM-friendly — no nested objects, no
    computed fields. Any post-processing / enrichment happens in
    PreLabel below.

    Note on `other_species_code`: in prompt v1.0 Gemini never emits
    species_code='OTHER' (not instructed to), so this field is always None
    from the pre-labeler. Kept on the schema so reviewer-produced records
    and future v1.1+ LLM responses share one shape.
    """

    bird_visible: bool = Field(
        description="True if a bird is visible in the image. False for empty feeder.",
    )

    species_code: str = Field(
        description=(
            "4-letter species code from the provided list, or 'NONE' if no bird "
            "is visible, or 'UNKNOWN' if a bird is visible but species cannot "
            "be determined with confidence."
        ),
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Visual confidence in the species_code classification. "
            "Express uncertainty honestly — low values are valuable signal. "
            "For 'NONE' this is confidence that the feeder is empty."
        ),
    )

    reasoning: str = Field(
        description=(
            "1-2 sentences describing what you see in the image, and what "
            "visual features support the chosen species_code."
        ),
    )

    uncertain_between: list[str] | None = Field(
        default=None,
        description=(
            "If confidence is below 0.7 and more than one species is plausible, "
            "list the alternative 4-letter codes here. Null otherwise."
        ),
    )

    other_species_code: str | None = Field(
        default=None,
        description=(
            "When species_code='OTHER', a 4-letter candidate code for the "
            "out-of-vocabulary species (e.g. 'CALT' for California Towhee). "
            "None for all in-vocab species, NONE, and UNKNOWN."
        ),
    )

    @field_validator("species_code")
    @classmethod
    def _validate_species_code(cls, v: str) -> str:
        v_upper = _normalise_species_code(v)
        if v_upper not in ALLOWED_CODES:
            raise ValueError(
                f"species_code '{v}' is not in allowed codes. Must be one of "
                f"the 20 SD species codes, 'NONE', 'UNKNOWN', or 'OTHER'."
            )
        return v_upper

    @field_validator("uncertain_between")
    @classmethod
    def _validate_uncertain_between(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        normalised = [_normalise_species_code(s) for s in v]
        invalid = [s for s in normalised if s not in ALLOWED_CODES]
        if invalid:
            raise ValueError(
                f"uncertain_between contains invalid codes: {invalid}. "
                f"All codes must be in the allowed set."
            )
        return normalised

    @field_validator("other_species_code")
    @classmethod
    def _validate_other_species_code(cls, v: str | None) -> str | None:
        return _validate_other_code(v)

    @model_validator(mode="after")
    def _other_code_consistency(self) -> PreLabelResponse:
        """Enforce the OTHER ↔ other_species_code relationship.

        - If species_code == 'OTHER', other_species_code MUST be present.
        - If species_code != 'OTHER', other_species_code MUST be None
          (prevents confused records with both a known code and an OOV hint).
        """
        if self.species_code == SENTINEL_OTHER and self.other_species_code is None:
            raise ValueError(
                "species_code='OTHER' requires other_species_code to be set "
                "(4-letter candidate code for the out-of-vocab species)."
            )
        if self.species_code != SENTINEL_OTHER and self.other_species_code is not None:
            raise ValueError(
                f"other_species_code is only valid when species_code='OTHER', "
                f"but species_code='{self.species_code}'."
            )
        return self


# ── Persisted record (pre_labels.jsonl) ───────────────────────────────────────


class PreLabel(BaseModel):
    """A single pre-labelled image record persisted to pre_labels.jsonl.

    Contains the LLM response plus everything needed to reproduce the
    label run: image path, observation cross-reference, audio hint that
    was shown to Gemini, timestamps, model and prompt identifiers.
    """

    # ── Image identification ──────────────────────────────────────────────────

    image_path: str = Field(
        description="Absolute path to the image file on disk at label time.",
    )

    image_filename: str = Field(
        description="Basename of the image, used for indexing and de-duplication.",
    )

    capture_timestamp: datetime | None = Field(
        default=None,
        description=(
            "Timestamp of the capture, parsed from the image filename. "
            "Null if the filename does not match the expected pattern."
        ),
    )

    # ── Cross-reference to observation ────────────────────────────────────────

    observation_timestamp: datetime | None = Field(
        default=None,
        description=(
            "Timestamp from the matched observations.jsonl record, if any. "
            "Null if no matching observation was found."
        ),
    )

    audio_hint: str | None = Field(
        default=None,
        description=(
            "Species code BirdNET heard within the capture window, if any. "
            "Passed to Gemini as additional context but explicitly marked as "
            "NOT authoritative. Null if no audio detection available."
        ),
    )

    audio_confidence: float | None = Field(
        default=None,
        description="BirdNET confidence for audio_hint. Null if no audio hint.",
    )

    # ── LLM response ──────────────────────────────────────────────────────────

    llm_response: PreLabelResponse = Field(
        description="Structured response from Gemini for this image.",
    )

    # ── Run metadata (for reproducibility) ────────────────────────────────────

    model_name: str = Field(
        description="Identifier of the Gemini model used, e.g. 'gemini-2.5-flash'.",
    )

    prompt_version: str = Field(
        description="Version tag for the prompt template, from prompts.py",
    )

    labeled_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the pre-label was produced.",
    )

    elapsed_seconds: float = Field(
        ge=0.0,
        description="Wall-clock time spent on the Gemini call for this image.",
    )

    # Pydantic configuration — allow datetime serialisation via isoformat.
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v),
        },
    }


# ── Verified label (produced by the human-review UI — Layer 2) ───────────────


class VerifiedLabel(BaseModel):
    """A single verified labelled image record, output of the human-review UI.

    Produced per verification action in the review UI. One record per
    image_filename in the final verified_labels.jsonl — the ReviewStore
    handles the append-then-rewrite semantics to keep that invariant.
    """

    image_path: str = Field(
        description="Absolute path to the image at verification time.",
    )
    image_filename: str = Field(
        description="Basename of the image, used for indexing and dedup.",
    )

    # ── Human-provided label ──────────────────────────────────────────────────

    species_code: str = Field(
        description=(
            "Verified species code — either a known 4-letter code, NONE "
            "(empty feeder), UNKNOWN (cannot determine), or OTHER (bird "
            "visible, species out-of-vocab — requires other_species_code)."
        ),
    )

    other_species_code: str | None = Field(
        default=None,
        description=(
            "When species_code='OTHER', the 4-letter candidate code for the "
            "out-of-vocab species the reviewer identified (e.g. 'CALT')."
        ),
    )

    reviewer_notes: str | None = Field(
        default=None,
        description="Free-text notes from the reviewer. Optional.",
    )

    # ── Provenance ────────────────────────────────────────────────────────────

    pre_label: PreLabel | None = Field(
        default=None,
        description=(
            "The pre-label the reviewer started from, if any. Embedded rather "
            "than referenced so each verified record is self-contained and "
            "can be audited without rejoining to pre_labels.jsonl."
        ),
    )
    agreed_with_pre_label: bool | None = Field(
        default=None,
        description=(
            "True if the human confirmed the pre-label verbatim. False if the "
            "human corrected it. Null if no pre-label was shown to the reviewer."
        ),
    )

    verified_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the verification was recorded.",
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("species_code")
    @classmethod
    def _validate_species_code(cls, v: str) -> str:
        v_upper = _normalise_species_code(v)
        if v_upper not in ALLOWED_CODES:
            raise ValueError(
                f"species_code '{v}' is not a valid species or sentinel."
            )
        return v_upper

    @field_validator("other_species_code")
    @classmethod
    def _validate_other_species_code(cls, v: str | None) -> str | None:
        return _validate_other_code(v)

    @model_validator(mode="after")
    def _other_code_consistency(self) -> VerifiedLabel:
        """Mirror the PreLabelResponse invariant: OTHER ↔ other_species_code."""
        if self.species_code == SENTINEL_OTHER and self.other_species_code is None:
            raise ValueError(
                "species_code='OTHER' requires other_species_code to be set."
            )
        if self.species_code != SENTINEL_OTHER and self.other_species_code is not None:
            raise ValueError(
                f"other_species_code is only valid when species_code='OTHER', "
                f"but species_code='{self.species_code}'."
            )
        return self

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v),
        },
    }
