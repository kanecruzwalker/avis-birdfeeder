"""
src/fusion/combiner.py

Combines audio and visual ClassificationResults into a single BirdObservation.

Why a dedicated fusion module?
    Audio and visual classifiers each produce a species prediction with a
    confidence score. Neither is perfect in isolation:
        - Audio alone may misidentify species with similar calls
        - Visual alone may fail in poor lighting or partial occlusion
    Combining both modalities produces a more reliable identification.

Fusion strategies (configurable via configs/thresholds.yaml):
    - "equal":    simple average of confidence scores
    - "weighted": configurable per-modality weight (e.g. audio=0.55, visual=0.45)
    - "max":      take whichever modality has higher confidence

If only one modality is available (e.g. bird is silent), fusion falls back
gracefully to the single available result with its confidence unchanged.

Species conflict resolution:
    When audio and visual classifiers disagree on species, the higher-confidence
    result wins regardless of strategy. Strategy only affects how confidence
    scores are combined when both classifiers agree on species.

    Example: audio says HOFI (0.8), visual says WCSP (0.3).
    The fused result is HOFI at whatever the weighted/equal/max score computes
    to for the winning modality — because you can't meaningfully average
    confidence scores for different species.
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

    Usage:
        fuser = ScoreFuser(strategy="weighted", audio_weight=0.55, visual_weight=0.45)
        observation = fuser.fuse(audio_result, visual_result)
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
        Construct a ScoreFuser from a thresholds config YAML.

        Reads fusion.strategy, fusion.audio_weight, and fusion.visual_weight
        from the YAML file. All other keys are ignored.

        Args:
            config_path: Path to configs/thresholds.yaml.

        Returns:
            Configured ScoreFuser instance.

        Raises:
            FileNotFoundError: If config_path does not exist.
            KeyError: If required fusion keys are missing from the config.
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
            "ScoreFuser loaded from config: strategy=%s audio_weight=%.2f visual_weight=%.2f",
            strategy,
            audio_weight,
            visual_weight,
        )
        return cls(
            strategy=strategy,
            audio_weight=audio_weight,
            visual_weight=visual_weight,
        )

    def fuse(
        self,
        audio_result: ClassificationResult | None = None,
        visual_result: ClassificationResult | None = None,
    ) -> BirdObservation:
        """
        Combine one or both modality results into a BirdObservation.

        Handles graceful degradation:
            - Both provided, same species: fuse confidence scores by strategy
            - Both provided, different species: winner-takes-all by confidence
            - Only audio: use audio result directly (confidence unchanged)
            - Only visual: use visual result directly (confidence unchanged)
            - Neither: raises ValueError

        Args:
            audio_result:  Output from AudioClassifier.predict(), or None.
            visual_result: Output from VisualClassifier.predict(), or None.

        Returns:
            BirdObservation with fused species and confidence.

        Raises:
            ValueError: If both results are None.
        """
        if audio_result is None and visual_result is None:
            raise ValueError("At least one of audio_result or visual_result must be provided.")

        # ── Single-modality graceful fallback ─────────────────────────────────
        if audio_result is None:
            logger.debug("fuse: audio unavailable, using visual result only")
            return self._observation_from_single(visual_result)  # type: ignore[arg-type]
        if visual_result is None:
            logger.debug("fuse: visual unavailable, using audio result only")
            return self._observation_from_single(audio_result)

        # ── Both modalities available ─────────────────────────────────────────
        if audio_result.species_code == visual_result.species_code:
            # Agreement — fuse confidence scores
            fused_confidence = self._fuse_confidence(
                audio_result.confidence, visual_result.confidence
            )
            winner = audio_result  # use audio for species metadata (both agree)
            logger.debug(
                "fuse: both agree on %s — fused confidence=%.3f (strategy=%s)",
                winner.species_code,
                fused_confidence,
                self.strategy,
            )
        else:
            # Disagreement — winner takes all by raw confidence
            if audio_result.confidence >= visual_result.confidence:
                winner = audio_result
            else:
                winner = visual_result
            fused_confidence = winner.confidence
            logger.debug(
                "fuse: disagreement audio=%s(%.3f) visual=%s(%.3f) — winner=%s",
                audio_result.species_code,
                audio_result.confidence,
                visual_result.species_code,
                visual_result.confidence,
                winner.species_code,
            )

        return BirdObservation(
            species_code=winner.species_code,
            common_name=winner.common_name,
            scientific_name=winner.scientific_name,
            fused_confidence=fused_confidence,
            audio_result=audio_result,
            visual_result=visual_result,
        )

    def _fuse_confidence(self, audio_conf: float, visual_conf: float) -> float:
        """
        Combine two confidence scores according to the configured strategy.

        Strategies:
            equal:    arithmetic mean — (a + v) / 2
            weighted: weighted average — a * audio_weight + v * visual_weight
            max:      maximum of the two scores

        Args:
            audio_conf:  Confidence score from the audio classifier [0, 1].
            visual_conf: Confidence score from the visual classifier [0, 1].

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
            # Should never reach here — constructor validates strategy
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _observation_from_single(self, result: ClassificationResult) -> BirdObservation:
        """
        Build a BirdObservation from a single ClassificationResult with no fusion.

        Confidence is passed through unchanged. The audio_result or visual_result
        field on the observation is set based on the modality of the input.

        Args:
            result: A ClassificationResult from either modality.

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
        )
