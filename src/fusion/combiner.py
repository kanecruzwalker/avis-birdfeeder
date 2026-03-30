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
    - "equal":      simple average of confidence scores
    - "weighted":   configurable per-modality weight (e.g. audio=0.6, visual=0.4)
    - "max":        take whichever modality has higher confidence
    - "learned":    a small learned combiner (future phase)

If only one modality is available (e.g. bird is silent), fusion falls back
gracefully to the single available result.

Phase 3 will implement the full fusion logic.
"""

from __future__ import annotations

from src.data.schema import BirdObservation, ClassificationResult


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
            strategy: Fusion strategy. One of "equal", "weighted", or "max".
            audio_weight: Weight for audio confidence when strategy is "weighted".
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

        Args:
            config_path: Path to configs/thresholds.yaml.

        Returns:
            Configured ScoreFuser instance.
        """
        raise NotImplementedError("Implement in Phase 3.")

    def fuse(
        self,
        audio_result: ClassificationResult | None = None,
        visual_result: ClassificationResult | None = None,
    ) -> BirdObservation:
        """
        Combine one or both modality results into a BirdObservation.

        Handles graceful degradation:
            - Both provided: fuse according to strategy
            - Only audio:    use audio result directly (confidence unchanged)
            - Only visual:   use visual result directly (confidence unchanged)
            - Neither:       raises ValueError

        Args:
            audio_result: Output from AudioClassifier.predict(), or None.
            visual_result: Output from VisualClassifier.predict(), or None.

        Returns:
            BirdObservation with fused species and confidence.

        Raises:
            ValueError: If both results are None.
        """
        if audio_result is None and visual_result is None:
            raise ValueError("At least one of audio_result or visual_result must be provided.")

        # Graceful single-modality fallback
        if audio_result is None:
            return self._observation_from_single(visual_result)  # type: ignore[arg-type]
        if visual_result is None:
            return self._observation_from_single(audio_result)

        # Both modalities available — apply fusion strategy
        raise NotImplementedError("Full fusion logic will be implemented in Phase 3.")

    def _observation_from_single(self, result: ClassificationResult) -> BirdObservation:
        """
        Build a BirdObservation from a single ClassificationResult with no fusion.

        Args:
            result: A ClassificationResult from either modality.

        Returns:
            BirdObservation wrapping the single result.
        """
        raise NotImplementedError("Implement in Phase 3.")
