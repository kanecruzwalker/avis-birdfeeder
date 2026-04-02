"""
src/fusion/combiner.py

Combines audio and visual ClassificationResults into a single BirdObservation.

Phase 5 additions:
    fuse() now accepts an optional visual_result_2 from the secondary camera.
    When both cameras produce results, the higher-confidence visual result
    is used for fusion and the other is stored in BirdObservation.visual_result_2
    for the observation record.

    This is the dual-camera strategy: classify independently from both cameras,
    fuse the best visual result with audio. No stereo geometry is involved here —
    that's Phase 6 (StereoEstimator). This module only handles confidence scores.

Why winner-takes-all for dual camera?
    When Camera 0 sees a House Finch at 0.91 confidence and Camera 1 sees a
    Spotted Towhee at 0.43 confidence, the right answer is House Finch —
    Camera 0 had a better view. Taking the higher-confidence visual result
    naturally handles cases where one camera is occluded or poorly framed.
    When both cameras agree on species, we average their confidences before
    fusing with audio.

Fusion strategies (unchanged from Phase 4):
    "equal":    simple average of confidence scores
    "weighted": configurable per-modality weight (audio=0.55, visual=0.45)
    "max":      take whichever modality has higher confidence

Species conflict resolution (unchanged):
    When audio and visual disagree, higher raw confidence wins regardless of strategy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from src.data.schema import BirdObservation, ClassificationResult, Modality

logger = logging.getLogger(__name__)


class ScoreFuser:
    """
    Fuses audio and visual classification results into a BirdObservation.

    Phase 5: accepts optional visual_result_2 from secondary camera.
    When both visual results are present, selects the best before fusing with audio.

    Usage:
        fuser = ScoreFuser.from_config("configs/thresholds.yaml")
        observation = fuser.fuse(
            audio_result=audio_result,
            visual_result=result_cam0,
            visual_result_2=result_cam1,   # optional, Phase 5 dual camera
        )
    """

    VALID_STRATEGIES = {"equal", "weighted", "max"}

    def __init__(
        self,
        strategy: str = "weighted",
        audio_weight: float = 0.55,
        visual_weight: float = 0.45,
    ) -> None:
        """
        Args:
            strategy:      Fusion strategy. One of "equal", "weighted", or "max".
            audio_weight:  Weight for audio confidence when strategy is "weighted".
            visual_weight: Weight for visual confidence when strategy is "weighted".

        Raises:
            ValueError: If strategy is not recognized or weights don't sum to 1.
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {self.VALID_STRATEGIES}, got '{strategy}'")
        if strategy == "weighted" and abs(audio_weight + visual_weight - 1.0) > 1e-6:
            raise ValueError(
                f"audio_weight + visual_weight must equal 1.0, "
                f"got {audio_weight} + {visual_weight} = {audio_weight + visual_weight}"
            )
        self.strategy = strategy
        self.audio_weight = audio_weight
        self.visual_weight = visual_weight

    @classmethod
    def from_config(cls, config_path: str) -> ScoreFuser:
        """
        Construct a ScoreFuser from configs/thresholds.yaml.

        Args:
            config_path: Path to configs/thresholds.yaml.

        Returns:
            Configured ScoreFuser instance.
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        fusion_cfg = cfg.get("fusion", {})
        strategy = fusion_cfg.get("strategy", "weighted")
        audio_weight = float(fusion_cfg.get("audio_weight", 0.55))
        visual_weight = float(fusion_cfg.get("visual_weight", 0.45))

        logger.info(
            "ScoreFuser loaded | strategy=%s audio_weight=%.2f visual_weight=%.2f",
            strategy,
            audio_weight,
            visual_weight,
        )
        return cls(strategy=strategy, audio_weight=audio_weight, visual_weight=visual_weight)

    def fuse(
        self,
        audio_result: ClassificationResult | None = None,
        visual_result: ClassificationResult | None = None,
        visual_result_2: ClassificationResult | None = None,
    ) -> BirdObservation:
        """
        Combine modality results into a BirdObservation.

        Dual-camera handling (Phase 5):
            When both visual_result and visual_result_2 are provided:
            - If they agree on species: average their confidences, use that as
              the visual confidence going into audio fusion.
            - If they disagree: take the higher-confidence result. The other
              is stored in BirdObservation.visual_result_2 for the record.
            Either way, visual_result_2 is always stored in the observation.

        Single-camera / single-modality graceful degradation (unchanged):
            - Only audio: use audio result directly.
            - Only one visual: use that visual result directly.
            - Both audio and one visual: fuse as Phase 4.

        Args:
            audio_result:    Output from AudioClassifier.predict(), or None.
            visual_result:   Output from primary camera VisualClassifier, or None.
            visual_result_2: Output from secondary camera VisualClassifier, or None.

        Returns:
            BirdObservation with fused species and confidence.

        Raises:
            ValueError: If all three results are None.
        """
        if audio_result is None and visual_result is None and visual_result_2 is None:
            raise ValueError(
                "At least one of audio_result, visual_result, or visual_result_2 "
                "must be provided."
            )

        # ── Dual-camera visual selection ──────────────────────────────────────
        # Resolve two visual results to one before audio fusion.
        # The non-selected result is preserved in the observation as visual_result_2.
        best_visual, secondary_visual = self._select_best_visual(visual_result, visual_result_2)

        # ── Single-modality graceful fallback ─────────────────────────────────
        if audio_result is None and best_visual is not None:
            logger.debug("fuse: audio unavailable — using visual only")
            return self._observation_from_single(best_visual, visual_result_2=secondary_visual)

        if best_visual is None and audio_result is not None:
            logger.debug("fuse: visual unavailable — using audio only")
            return self._observation_from_single(audio_result, visual_result_2=None)

        # ── Both modalities available ─────────────────────────────────────────
        if audio_result.species_code == best_visual.species_code:
            fused_confidence = self._fuse_confidence(
                audio_result.confidence, best_visual.confidence
            )
            winner = audio_result  # both agree — use audio for metadata
            logger.debug(
                "fuse: agreement on %s — fused=%.3f (strategy=%s)",
                winner.species_code,
                fused_confidence,
                self.strategy,
            )
        else:
            # Disagreement — winner takes all by raw confidence
            if audio_result.confidence >= best_visual.confidence:
                winner = audio_result
                fused_confidence = audio_result.confidence
            else:
                winner = best_visual
                fused_confidence = best_visual.confidence
            logger.debug(
                "fuse: disagreement audio=%s(%.3f) visual=%s(%.3f) — winner=%s",
                audio_result.species_code,
                audio_result.confidence,
                best_visual.species_code,
                best_visual.confidence,
                winner.species_code,
            )

        return BirdObservation(
            species_code=winner.species_code,
            common_name=winner.common_name,
            scientific_name=winner.scientific_name,
            fused_confidence=fused_confidence,
            audio_result=audio_result,
            visual_result=best_visual,
            visual_result_2=secondary_visual,
        )

    def _select_best_visual(
        self,
        result_1: ClassificationResult | None,
        result_2: ClassificationResult | None,
    ) -> tuple[ClassificationResult | None, ClassificationResult | None]:
        """
        Select the best visual result from two camera results.

        Returns (best, other) where best goes into audio fusion and
        other is stored as visual_result_2 in the observation.

        When both cameras agree on species: the returned best_result has its
        confidence replaced with the average of both cameras' confidences,
        giving a more stable estimate when both views corroborate each other.
        A new ClassificationResult is constructed to hold the averaged confidence.

        When cameras disagree: the higher-confidence result wins outright.

        Args:
            result_1: Primary camera result, or None.
            result_2: Secondary camera result, or None.

        Returns:
            Tuple of (best_result, secondary_result).
            secondary_result is None if only one camera had a result.
        """
        if result_1 is None and result_2 is None:
            return None, None

        if result_1 is None:
            return result_2, None

        if result_2 is None:
            return result_1, None

        # Both cameras have results
        if result_1.species_code == result_2.species_code:
            # Agreement — average the confidences into a new result
            avg_conf = (result_1.confidence + result_2.confidence) / 2.0
            averaged = ClassificationResult(
                species_code=result_1.species_code,
                common_name=result_1.common_name,
                scientific_name=result_1.scientific_name,
                confidence=avg_conf,
                modality=result_1.modality,
                model_version=result_1.model_version,
                camera_index=result_1.camera_index,  # primary camera index
            )
            logger.debug(
                "Dual camera agreement on %s: cam0=%.3f cam1=%.3f avg=%.3f",
                result_1.species_code,
                result_1.confidence,
                result_2.confidence,
                avg_conf,
            )
            return averaged, result_2
        else:
            # Disagreement — higher confidence wins
            if result_1.confidence >= result_2.confidence:
                logger.debug(
                    "Dual camera disagreement: cam0=%s(%.3f) beats cam1=%s(%.3f)",
                    result_1.species_code,
                    result_1.confidence,
                    result_2.species_code,
                    result_2.confidence,
                )
                return result_1, result_2
            else:
                logger.debug(
                    "Dual camera disagreement: cam1=%s(%.3f) beats cam0=%s(%.3f)",
                    result_2.species_code,
                    result_2.confidence,
                    result_1.species_code,
                    result_1.confidence,
                )
                return result_2, result_1

    def _fuse_confidence(self, audio_conf: float, visual_conf: float) -> float:
        """
        Combine audio and visual confidence scores by configured strategy.

        Args:
            audio_conf:  Confidence score from audio classifier [0, 1].
            visual_conf: Confidence score from visual classifier [0, 1].

        Returns:
            Fused confidence score in [0, 1].
        """
        if self.strategy == "equal":
            return (audio_conf + visual_conf) / 2.0
        elif self.strategy == "weighted":
            return audio_conf * self.audio_weight + visual_conf * self.visual_weight
        elif self.strategy == "max":
            return max(audio_conf, visual_conf)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _observation_from_single(
        self,
        result: ClassificationResult,
        visual_result_2: ClassificationResult | None = None,
    ) -> BirdObservation:
        """
        Build a BirdObservation from a single primary result with no cross-modal fusion.

        Confidence is passed through unchanged. visual_result_2 is attached
        to the observation for the record even in single-modality cases.

        Args:
            result:          Primary ClassificationResult (audio or visual).
            visual_result_2: Secondary camera result to attach, if available.

        Returns:
            BirdObservation wrapping the single result.
        """
        is_audio = result.modality == Modality.AUDIO
        return BirdObservation(
            species_code=result.species_code,
            common_name=result.common_name,
            scientific_name=result.scientific_name,
            fused_confidence=result.confidence,
            audio_result=result if is_audio else None,
            visual_result=result if not is_audio else None,
            visual_result_2=visual_result_2,
        )
