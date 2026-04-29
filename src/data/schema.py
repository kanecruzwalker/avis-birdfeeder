"""Shared data models for the Avis system.

Single source of truth for data shapes passed between modules:
audio and vision pipelines emit ``ClassificationResult``; fusion
consumes two of those and emits ``BirdObservation``; agent and
notifier consume ``BirdObservation``.

No ML dependencies here — this module must import cleanly in any
environment.

Field-evolution rule: optional fields with ``None`` defaults so
existing serialized records keep parsing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    """Return current UTC time as a timezone-aware datetime. Use instead of datetime.utcnow()."""
    return datetime.now(UTC)


# ── Gate reason values ────────────────────────────────────────────────────────
# Stored as plain string constants (not a StrEnum) for maximum backward
# compatibility with serialized BirdObservation records in observations.jsonl.
# New values can be added freely without risk of breaking deserialization of
# older records; renames of existing values require care because existing
# JSONL entries would need to stay readable (either via a compat alias table
# or an explicit migration).
#
# These are the values that may appear in BirdObservation.gate_reason.
# Always import these constants rather than writing the raw string literals
# so the values stay consistent across producers and consumers.

GATE_REASON_NO_BIRD_DETECTED = "no_bird_detected"
"""The bird-presence detector returned no detection; classifier was skipped.
Set by BirdAgent._cycle when both cameras' CaptureResults have gate_passed=False
AND no audio detection is available."""

GATE_REASON_BELOW_CONFIDENCE_THRESHOLD = "below_confidence_threshold"
"""The fused confidence was below the agent's confidence_threshold.
Set by BirdAgent._cycle when the threshold gate suppresses dispatch."""

GATE_REASON_SPECIES_COOLDOWN = "species_cooldown"
"""The species was dispatched within the configured cooldown_seconds window.
Set by BirdAgent._cycle when the cooldown gate suppresses dispatch."""

GATE_REASON_NO_AUDIO_DETECTED = "no_audio_detected"
"""The audio pipeline returned no SD species detection.
Reserved for future use when audio-only pipelines need explicit reason codes."""


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
    camera_index: int | None = Field(
        default=None,
        description=(
            "Camera index that produced this visual result (0=primary, 1=secondary). "
            "None for audio results."
        ),
    )

    @field_validator("species_code")
    @classmethod
    def species_code_uppercase(cls, v: str) -> str:
        """Species codes are always uppercase by convention (e.g. 'AMRO', not 'amro')."""
        return v.upper()


class BirdObservation(BaseModel):
    """A confirmed bird observation — the fused output of the pipeline.

    Produced by ``src.fusion.combiner``; consumed by
    ``src.notify.notifier`` and logged by ``src.agent.bird_agent``.
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

    dispatched: bool = Field(
        default=True,
        description=(
            "Whether this observation triggered a user-facing notification. "
            "True when the observation passed the confidence threshold and "
            "cooldown gates and was dispatched via the notifier (push, email, "
            "webhook). False when the observation was classified but suppressed "
            "because fused_confidence was below the agent threshold, or because "
            "the species was within its cooldown window after a recent dispatch. "
            "Default True preserves backward compatibility: records written "
            "before this field existed only existed because they were dispatched, "
            "so deserializing an older record correctly treats it as dispatched=True. "
            "Use this field to distinguish the 'user saw it' stream from the full "
            "classification stream when analyzing logs."
        ),
    )

    # ── Modality results ──────────────────────────────────────────────────────
    audio_result: ClassificationResult | None = Field(
        default=None,
        description="Raw audio classification result, if available.",
    )
    visual_result: ClassificationResult | None = Field(
        default=None,
        description="Primary camera classification result, if available.",
    )
    visual_result_2: ClassificationResult | None = Field(
        default=None,
        description=(
            "Secondary camera classification result, if available. "
            "Populated when dual-camera capture is active. "
            "The higher-confidence of visual_result and visual_result_2 "
            "is used for fusion; the other is stored here for the record."
        ),
    )

    # ── Timestamps and media paths ────────────────────────────────────────────
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the observation.",
    )
    image_path: str | None = Field(
        default=None,
        description="Local path to the captured frame image from primary camera, if saved.",
    )
    image_path_2: str | None = Field(
        default=None,
        description="Local path to the captured frame image from secondary camera, if saved.",
    )
    image_path_full: str | None = Field(
        default=None,
        description=(
            "Local path to the full uncropped frame from the primary camera, "
            "if saved. Populated only on dispatched observations when the "
            "agent persists the full-frame variant for the dashboard; "
            "suppressed observations have this as None. "
            "The annotated variant (full frame with the YOLO bounding box "
            "drawn over it) is saved alongside on disk when "
            "detection_box is set; its path is derived from this one by "
            "swapping the '_full' stem suffix for '_annotated'."
        ),
    )
    audio_path: str | None = Field(
        default=None,
        description="Local path to the captured audio clip, if saved.",
    )

    detection_mode: str = Field(
        default="fixed_crop",
        description=(
            "Crop strategy active when this observation was made. "
            "'fixed_crop' = static ROI. 'yolo' = YOLOv8s bounding box. "
            "Set by ExperimentOrchestrator for A/B analysis."
        ),
    )

    gate_reason: str | None = Field(
        default=None,
        description=(
            "If this observation was suppressed (not dispatched to users), "
            "why. Populated by BirdAgent._cycle when a gate blocks dispatch. "
            "Stable string constants defined at module scope (GATE_REASON_*). "
            "\n\n"
            "Values (see module-level constants):\n"
            "  None                          — observation dispatched normally,\n"
            "                                  OR legacy record from before the "
            "field existed\n"
            "  'no_bird_detected'            — bird-presence gate returned no "
            "detection; classifier skipped\n"
            "  'below_confidence_threshold'  — fused confidence below agent "
            "threshold\n"
            "  'species_cooldown'            — species dispatched within cooldown "
            "window\n"
            "\n"
            "Relationship to the `dispatched` field:\n"
            "  dispatched=True   → gate_reason MUST be None\n"
            "  dispatched=False  → gate_reason SHOULD be set\n"
            "  Legacy suppressed records pre-dating this field may have "
            "dispatched=False and gate_reason=None; treat these as\n"
            "  'below_confidence_threshold' or 'species_cooldown' based on "
            "context (fused_confidence vs threshold).\n"
            "\n"
            "Backward compatible — old records deserialize with gate_reason=None."
        ),
    )

    # ── Detection + stereo fields ────────────────────────────────────────────
    # Populated by the detection pipeline + StereoEstimator when active;
    # None/False otherwise.
    detection_box: tuple[int, int, int, int] | None = Field(
        default=None,
        description=(
            "Bounding box (x, y, width, height) of the detected bird in the "
            "primary camera frame, when the object detector ran. Required "
            "for per-bird stereo depth estimation."
        ),
    )
    estimated_depth_cm: float | None = Field(
        default=None,
        description=(
            "Stereo depth estimate — distance from cameras to bird in cm. "
            "Requires detection_box to be set and stereo calibration active."
        ),
    )
    estimated_size_cm: float | None = Field(
        default=None,
        description=(
            "Estimated bird body length in cm, derived from stereo disparity "
            "and detection_box dimensions. Useful as an auxiliary feature for "
            "species disambiguation (e.g. House Finch vs Lesser Goldfinch at "
            "similar distances)."
        ),
    )
    stereo_calibrated: bool = Field(
        default=False,
        description=(
            "True if stereo calibration matrices were loaded and applied "
            "when computing depth for this observation."
        ),
    )

    @field_validator("species_code")
    @classmethod
    def species_code_uppercase(cls, v: str) -> str:
        return v.upper()

    @property
    def has_both_modalities(self) -> bool:
        """True if both audio and visual results contributed to this observation."""
        return self.audio_result is not None and self.visual_result is not None

    @property
    def has_dual_camera(self) -> bool:
        """True if both cameras produced classification results for this observation."""
        return self.visual_result is not None and self.visual_result_2 is not None

    @property
    def has_stereo_estimate(self) -> bool:
        """True if stereo depth estimation was performed for this observation."""
        return self.estimated_depth_cm is not None and self.stereo_calibrated
