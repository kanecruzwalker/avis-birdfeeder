"""
tests/agent/test_bird_agent.py

Unit tests for BirdAgent.

Strategy:
    - All classifier and notifier calls are mocked — no model weights needed.
    - _cycle() return value and dispatch behavior tested across all scenarios:
      both modalities, audio-only, visual-only, below-threshold, exceptions.
    - from_config() verified to resolve paths and construct all sub-components.
    - No hardware, no real files beyond configs/*.yaml.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.agent.bird_agent import BirdAgent
from src.data.schema import BirdObservation, ClassificationResult, Modality

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(
    species_code: str = "HOFI",
    confidence: float = 0.9,
    modality: Modality = Modality.AUDIO,
) -> ClassificationResult:
    return ClassificationResult(
        species_code=species_code,
        common_name="House Finch",
        scientific_name="Haemorhous mexicanus",
        confidence=confidence,
        modality=modality,
    )


def _make_observation(
    species_code: str = "HOFI",
    fused_confidence: float = 0.9,
) -> BirdObservation:
    return BirdObservation(
        species_code=species_code,
        common_name="House Finch",
        scientific_name="Haemorhous mexicanus",
        fused_confidence=fused_confidence,
    )


def _make_agent(
    audio_result: ClassificationResult | None = None,
    visual_result: ClassificationResult | None = None,
    fused_observation: BirdObservation | None = None,
    confidence_threshold: float = 0.7,
    audio_enabled: bool = True,
    visual_enabled: bool = True,
) -> BirdAgent:
    """Build a BirdAgent with all sub-components mocked."""
    audio_clf = MagicMock()
    audio_clf.predict.return_value = audio_result or _make_result(modality=Modality.AUDIO)

    visual_clf = MagicMock()
    visual_clf.predict.return_value = visual_result or _make_result(modality=Modality.VISUAL)

    fuser = MagicMock()
    fuser.fuse.return_value = fused_observation or _make_observation()

    notifier = MagicMock()

    return BirdAgent(
        audio_classifier=audio_clf if audio_enabled else None,
        visual_classifier=visual_clf if visual_enabled else None,
        fuser=fuser,
        notifier=notifier,
        confidence_threshold=confidence_threshold,
    )


DUMMY_SPEC = np.zeros((128, 282), dtype=np.float32)
DUMMY_FRAME = np.zeros((224, 224, 3), dtype=np.float32)


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestBirdAgentInit:
    def test_stores_threshold(self) -> None:
        agent = _make_agent(confidence_threshold=0.8)
        assert agent.confidence_threshold == 0.8

    def test_raises_if_both_classifiers_none(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            BirdAgent(
                audio_classifier=None,
                visual_classifier=None,
                fuser=MagicMock(),
                notifier=MagicMock(),
            )

    def test_audio_only_is_valid(self) -> None:
        agent = _make_agent(visual_enabled=False)
        assert agent.visual_classifier is None
        assert agent.audio_classifier is not None

    def test_visual_only_is_valid(self) -> None:
        agent = _make_agent(audio_enabled=False)
        assert agent.audio_classifier is None
        assert agent.visual_classifier is not None


# ── _cycle — happy paths ──────────────────────────────────────────────────────

class TestCycleHappyPath:
    def test_returns_observation_above_threshold(self) -> None:
        obs = _make_observation(fused_confidence=0.9)
        agent = _make_agent(fused_observation=obs, confidence_threshold=0.7)
        result = agent._cycle(spectrogram=DUMMY_SPEC, frame=DUMMY_FRAME)
        assert result is not None
        assert result.species_code == "HOFI"

    def test_dispatches_when_above_threshold(self) -> None:
        obs = _make_observation(fused_confidence=0.9)
        agent = _make_agent(fused_observation=obs, confidence_threshold=0.7)
        agent._cycle(spectrogram=DUMMY_SPEC, frame=DUMMY_FRAME)
        agent.notifier.dispatch.assert_called_once_with(obs)

    def test_returns_none_below_threshold(self) -> None:
        obs = _make_observation(fused_confidence=0.4)
        agent = _make_agent(fused_observation=obs, confidence_threshold=0.7)
        result = agent._cycle(spectrogram=DUMMY_SPEC, frame=DUMMY_FRAME)
        assert result is None

    def test_no_dispatch_below_threshold(self) -> None:
        obs = _make_observation(fused_confidence=0.4)
        agent = _make_agent(fused_observation=obs, confidence_threshold=0.7)
        agent._cycle(spectrogram=DUMMY_SPEC, frame=DUMMY_FRAME)
        agent.notifier.dispatch.assert_not_called()

    def test_fuser_called_with_both_results(self) -> None:
        agent = _make_agent()
        agent._cycle(spectrogram=DUMMY_SPEC, frame=DUMMY_FRAME)
        agent.fuser.fuse.assert_called_once()
        call_kwargs = agent.fuser.fuse.call_args.kwargs
        assert call_kwargs["audio_result"] is not None
        assert call_kwargs["visual_result"] is not None


# ── _cycle — single modality ──────────────────────────────────────────────────

class TestCycleSingleModality:
    def test_audio_only_no_frame(self) -> None:
        agent = _make_agent()
        agent._cycle(spectrogram=DUMMY_SPEC, frame=None)
        call_kwargs = agent.fuser.fuse.call_args.kwargs
        assert call_kwargs["audio_result"] is not None
        assert call_kwargs["visual_result"] is None

    def test_visual_only_no_spec(self) -> None:
        agent = _make_agent()
        agent._cycle(spectrogram=None, frame=DUMMY_FRAME)
        call_kwargs = agent.fuser.fuse.call_args.kwargs
        assert call_kwargs["audio_result"] is None
        assert call_kwargs["visual_result"] is not None

    def test_no_inputs_returns_none(self) -> None:
        agent = _make_agent()
        result = agent._cycle(spectrogram=None, frame=None)
        assert result is None
        agent.fuser.fuse.assert_not_called()
        agent.notifier.dispatch.assert_not_called()

    def test_audio_disabled_classifier_skipped(self) -> None:
        agent = _make_agent(audio_enabled=False)
        agent._cycle(spectrogram=DUMMY_SPEC, frame=DUMMY_FRAME)
        call_kwargs = agent.fuser.fuse.call_args.kwargs
        assert call_kwargs["audio_result"] is None


# ── _cycle — exception handling ───────────────────────────────────────────────

class TestCycleExceptionHandling:
    def test_audio_exception_continues_with_visual(self) -> None:
        """If audio classifier raises, visual result still fused."""
        agent = _make_agent()
        agent.audio_classifier.predict.side_effect = RuntimeError("model broken")
        agent._cycle(spectrogram=DUMMY_SPEC, frame=DUMMY_FRAME)
        call_kwargs = agent.fuser.fuse.call_args.kwargs
        assert call_kwargs["audio_result"] is None
        assert call_kwargs["visual_result"] is not None

    def test_visual_exception_continues_with_audio(self) -> None:
        """If visual classifier raises, audio result still fused."""
        agent = _make_agent()
        agent.visual_classifier.predict.side_effect = RuntimeError("model broken")
        agent._cycle(spectrogram=DUMMY_SPEC, frame=DUMMY_FRAME)
        call_kwargs = agent.fuser.fuse.call_args.kwargs
        assert call_kwargs["audio_result"] is not None
        assert call_kwargs["visual_result"] is None

    def test_both_exceptions_returns_none(self) -> None:
        agent = _make_agent()
        agent.audio_classifier.predict.side_effect = RuntimeError("broken")
        agent.visual_classifier.predict.side_effect = RuntimeError("broken")
        result = agent._cycle(spectrogram=DUMMY_SPEC, frame=DUMMY_FRAME)
        assert result is None
        agent.fuser.fuse.assert_not_called()


# ── from_config ───────────────────────────────────────────────────────────────

class TestFromConfig:
    def test_constructs_without_error(self) -> None:
        """from_config() should succeed with real YAML files — no weights needed."""
        agent = BirdAgent.from_config("configs/")
        assert agent.confidence_threshold == pytest.approx(0.7)
        assert agent.loop_interval_seconds == pytest.approx(1.0)

    def test_both_classifiers_present(self) -> None:
        agent = BirdAgent.from_config("configs/")
        assert agent.audio_classifier is not None
        assert agent.visual_classifier is not None

    def test_fuser_and_notifier_present(self) -> None:
        agent = BirdAgent.from_config("configs/")
        assert agent.fuser is not None
        assert agent.notifier is not None

    def test_classifiers_not_loaded(self) -> None:
        """Lazy loading — model weights not on disk yet, _model should be None."""
        agent = BirdAgent.from_config("configs/")
        assert agent.audio_classifier._model is None
        assert agent.visual_classifier._model is None
