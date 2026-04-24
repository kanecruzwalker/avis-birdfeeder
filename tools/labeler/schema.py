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
  two sentinels: "NONE" (empty feeder) and "UNKNOWN" (bird visible but
  Gemini unsure).
- `VerifiedLabel` is produced by the future human-review UI (Layer 2).
  Defining it here keeps schema evolution in one file.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

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

# Sentinels the pre-labeler may return when no 4-letter code applies.
SENTINEL_NO_BIRD = "NONE"          # feeder is empty / no bird visible
SENTINEL_UNKNOWN = "UNKNOWN"       # bird visible but species cannot be determined

ALLOWED_CODES: tuple[str, ...] = KNOWN_SPECIES_CODES + (SENTINEL_NO_BIRD, SENTINEL_UNKNOWN)


# ── LLM response shape (what Gemini is asked to return) ──────────────────────


class PreLabelResponse(BaseModel):
    """Structured response Gemini produces for a single image.

    This is the shape passed to `ChatGoogleGenerativeAI.with_structured_output()`.
    Keep this class minimal and LLM-friendly — no nested objects, no
    computed fields. Any post-processing / enrichment happens in
    PreLabel below.
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

    uncertain_between: Optional[list[str]] = Field(
        default=None,
        description=(
            "If confidence is below 0.7 and more than one species is plausible, "
            "list the alternative 4-letter codes here. Null otherwise."
        ),
    )

    @field_validator("species_code")
    @classmethod
    def _validate_species_code(cls, v: str) -> str:
        v_upper = v.strip().upper()
        if v_upper not in ALLOWED_CODES:
            raise ValueError(
                f"species_code '{v}' is not in allowed codes. Must be one of "
                f"the 20 SD species codes, 'NONE', or 'UNKNOWN'."
            )
        return v_upper

    @field_validator("uncertain_between")
    @classmethod
    def _validate_uncertain_between(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is None:
            return None
        normalised = [s.strip().upper() for s in v]
        invalid = [s for s in normalised if s not in ALLOWED_CODES]
        if invalid:
            raise ValueError(
                f"uncertain_between contains invalid codes: {invalid}. "
                f"All codes must be in the allowed set."
            )
        return normalised


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

    capture_timestamp: Optional[datetime] = Field(
        default=None,
        description=(
            "Timestamp of the capture, parsed from the image filename. "
            "Null if the filename does not match the expected pattern."
        ),
    )

    # ── Cross-reference to observation ────────────────────────────────────────

    observation_timestamp: Optional[datetime] = Field(
        default=None,
        description=(
            "Timestamp from the matched observations.jsonl record, if any. "
            "Null if no matching observation was found."
        ),
    )

    audio_hint: Optional[str] = Field(
        default=None,
        description=(
            "Species code BirdNET heard within the capture window, if any. "
            "Passed to Gemini as additional context but explicitly marked as "
            "NOT authoritative. Null if no audio detection available."
        ),
    )

    audio_confidence: Optional[float] = Field(
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
        default_factory=lambda: datetime.now(timezone.utc),
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


# ── Verified label (produced by future human-review UI) ──────────────────────


class VerifiedLabel(BaseModel):
    """A single verified labelled image record, output of the human-review UI.

    Not produced by this PR — defined here so the schema lives in one place
    and the review-UI PR can import it directly.
    """

    image_path: str
    image_filename: str

    # Human-provided ground truth
    species_code: str
    reviewer_notes: Optional[str] = None

    # Provenance
    pre_label: Optional[PreLabel] = Field(
        default=None,
        description="The pre-label the reviewer started from, if any.",
    )
    agreed_with_pre_label: Optional[bool] = Field(
        default=None,
        description=(
            "True if the human confirmed the pre-label. False if the human "
            "corrected it. Null if no pre-label was shown to the reviewer."
        ),
    )

    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("species_code")
    @classmethod
    def _validate_species_code(cls, v: str) -> str:
        v_upper = v.strip().upper()
        if v_upper not in ALLOWED_CODES:
            raise ValueError(
                f"species_code '{v}' is not a valid species or sentinel."
            )
        return v_upper

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v),
        },
    }