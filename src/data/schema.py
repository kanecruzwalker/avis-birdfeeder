"""
src/data/schema.py

Shared data models for the Avis system.

This is the single source of truth for data shapes passed between modules.
Both the audio and visual pipelines produce ClassificationResult objects.
The fusion module consumes two ClassificationResults and produces a BirdObservation.
The agent and notifier consume BirdObservation objects.

No ML dependencies here — this module must import cleanly in any environment.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    """Return current UTC time as a timezone-aware datetime. Use instead of datetime.utcnow()."""
    return datetime.now(UTC)


class Modality(StrEnum):
    """Which sensor produced this classification."""

    AUDIO = "audio"
    VISUAL = "visual"
    FUSED = "fused"


class ClassificationResult(BaseModel):
    """
    Output of a single-modality classifier (audio or visual).

    Produced by:
        src.audio.classify
        src.vision.classify

    Consumed by:
        src.fusion.combiner
    """

    model_config = {"protected_namespaces": ()}  # allow model_ prefixed fields

    species_code: str = Field(
        ...,
        description="Short species identifier, e.g. 'AMRO' for American Robin.",
    )
    common_name: str = Field(
        ...,
        description="Human-readable common name, e.g. 'American Robin'.",
    )
    scientific_name: str = Field(
        ...,
        description="Binomial scientific name, e.g. 'Turdus migratorius'.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score in [0, 1].",
    )
    modality: Modality = Field(
        ...,
        description="Which pipeline produced this result.",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when classification was performed.",
    )
    model_version: str = Field(
        default="unknown",
        description="Identifier for the model that produced this result.",
    )

    @field_validator("species_code")
    @classmethod
    def species_code_uppercase(cls, v: str) -> str:
        """Species codes are always uppercase by convention (e.g. 'AMRO', not 'amro')."""
        return v.upper()


class BirdObservation(BaseModel):
    """
    A confirmed bird observation — the fused output passed to notify and log.

    Produced by:
        src.fusion.combiner

    Consumed by:
        src.notify.notifier
        src.agent.bird_agent (for logging)
    """

    species_code: str = Field(..., description="Fused species identifier.")
    common_name: str = Field(..., description="Human-readable common name.")
    scientific_name: str = Field(..., description="Binomial scientific name.")
    fused_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Combined confidence score after fusion.",
    )
    audio_result: ClassificationResult | None = Field(
        default=None,
        description="Raw audio classification result, if available.",
    )
    visual_result: ClassificationResult | None = Field(
        default=None,
        description="Raw visual classification result, if available.",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the observation.",
    )
    image_path: str | None = Field(
        default=None,
        description="Local path to the captured frame image, if saved.",
    )
    audio_path: str | None = Field(
        default=None,
        description="Local path to the captured audio clip, if saved.",
    )

    @field_validator("species_code")
    @classmethod
    def species_code_uppercase(cls, v: str) -> str:
        return v.upper()

    @property
    def has_both_modalities(self) -> bool:
        """True if both audio and visual results contributed to this observation."""
        return self.audio_result is not None and self.visual_result is not None
