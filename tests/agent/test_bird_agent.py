"""
tests/agent/test_bird_agent.py

Unit tests for BirdAgent.

Phase 5 changes from Phase 4:
    - BirdAgent.__init__() now requires audio_capture and vision_capture args.
      Both are mocked in _make_agent() — no hardware required.
    - _cycle() no longer accepts spectrogram/frame arguments. It calls
      capture modules internally. Tests mock capture modules to control
      what _cycle() sees.
    - AudioClassifier.predict() now takes a file path (not a spectrogram array).
      NoBirdDetectedError is the expected exception when no bird is found.
    - ScoreFuser.fuse() now accepts an optional visual_result_2 parameter.
    - Cooldown suppression: repeated same-species dispatches within
      cooldown_seconds are suppressed.

Strategy:
    - All classifiers, capture modules, fuser, and notifier are mocked.
    - _cycle() behavior tested by controlling mock return values.
    - from_config() verified with real YAML files — no weights needed.
    - No hardware, no real audio/image files.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.bird_agent import BirdAgent
from src.audio.classify import NoBirdDetectedError
from src.data.schema import BirdObservation, ClassificationResult, Modality

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(
    species_code: str = "HOFI",
    confidence: float = 0.9,
    modality: Modality = Modality.AUDIO,
    camera_index: int | None = None,
) -> ClassificationResult:
    return ClassificationResult(
        species_code=species_code,
        common_name="House Finch",
        scientific_name="Haemorhous mexicanus",
        confidence=confidence,
        modality=modality,
        camera_index=camera_index,
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


def _make_capture_result(camera_index: int = 0) -> MagicMock:
    """Build a mock CaptureResult with a valid preprocessed frame."""
    import numpy as np

    cr = MagicMock()
    cr.frame = np.zeros((224, 224, 3), dtype=np.float32)
    cr.raw_frame = cr.frame
    cr.camera_index = camera_index
    cr.image_path = Path(f"/tmp/cam{camera_index}.png")
    cr.motion_score = 0.1
    return cr


def _make_agent(
    audio_result: ClassificationResult | None = None,
    visual_result: ClassificationResult | None = None,
    fused_observation: BirdObservation | None = None,
    confidence_threshold: float = 0.7,
    cooldown_seconds: float = 0.0,  # disabled by default in tests
    audio_enabled: bool = True,
    visual_enabled: bool = True,
    audio_capture_returns: Path | None = Path("/tmp/fake.wav"),
    vision_capture_returns: tuple = (None, None),
) -> BirdAgent:
    """
    Build a BirdAgent with all sub-components mocked.

    audio_capture_returns: what AudioCapture.capture_window() returns.
                           Path = audio captured. None = below energy threshold.
    vision_capture_returns: what VisionCapture.capture_frames() returns.
                            Tuple of (primary_CaptureResult|None, secondary|None).
    """
    audio_clf = MagicMock()
    audio_clf.predict.return_value = audio_result or _make_result(modality=Modality.AUDIO)

    visual_clf = MagicMock()
    visual_clf.predict.return_value = visual_result or _make_result(modality=Modality.VISUAL)

    audio_cap = MagicMock()
    audio_cap.capture_window.return_value = audio_capture_returns

    vision_cap = MagicMock()
    vision_cap.capture_frames.return_value = vision_capture_returns

    fuser = MagicMock()
    fuser.fuse.return_value = fused_observation or _make_observation()

    notifier = MagicMock()

    return BirdAgent(
        audio_classifier=audio_clf if audio_enabled else None,
        visual_classifier=visual_clf if visual_enabled else None,
        audio_capture=audio_cap if audio_enabled else None,
        vision_capture=vision_cap if visual_enabled else None,
        fuser=fuser,
        notifier=notifier,
        confidence_threshold=confidence_threshold,
        cooldown_seconds=cooldown_seconds,
    )


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestBirdAgentInit:
    def test_stores_threshold(self) -> None:
        agent = _make_agent(confidence_threshold=0.8)
        assert agent.confidence_threshold == 0.8

    def test_stores_cooldown(self) -> None:
        agent = _make_agent(cooldown_seconds=45.0)
        assert agent.cooldown_seconds == 45.0

    def test_raises_if_both_classifiers_none(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            BirdAgent(
                audio_classifier=None,
                visual_classifier=None,
                audio_capture=None,
                vision_capture=None,
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
        cap = _make_capture_result(camera_index=0)
        agent = _make_agent(
            fused_observation=obs,
            confidence_threshold=0.7,
            audio_capture_returns=Path("/tmp/fake.wav"),
            vision_capture_returns=(cap, None),
        )
        result = agent._cycle()
        assert result is not None
        assert result.species_code == "HOFI"

    def test_dispatches_when_above_threshold(self) -> None:
        obs = _make_observation(fused_confidence=0.9)
        cap = _make_capture_result(camera_index=0)
        agent = _make_agent(
            fused_observation=obs,
            confidence_threshold=0.7,
            audio_capture_returns=Path("/tmp/fake.wav"),
            vision_capture_returns=(cap, None),
        )
        agent._cycle()
        agent.notifier.dispatch.assert_called_once()

    def test_returns_none_below_threshold(self) -> None:
        obs = _make_observation(fused_confidence=0.4)
        cap = _make_capture_result(camera_index=0)
        agent = _make_agent(
            fused_observation=obs,
            confidence_threshold=0.7,
            audio_capture_returns=Path("/tmp/fake.wav"),
            vision_capture_returns=(cap, None),
        )
        result = agent._cycle()
        assert result is None

    def test_no_dispatch_below_threshold(self) -> None:
        obs = _make_observation(fused_confidence=0.4)
        cap = _make_capture_result(camera_index=0)
        agent = _make_agent(
            fused_observation=obs,
            confidence_threshold=0.7,
            audio_capture_returns=Path("/tmp/fake.wav"),
            vision_capture_returns=(cap, None),
        )
        agent._cycle()
        agent.notifier.dispatch.assert_not_called()

    def test_fuser_called_with_audio_and_visual(self) -> None:
        cap = _make_capture_result(camera_index=0)
        agent = _make_agent(
            audio_capture_returns=Path("/tmp/fake.wav"),
            vision_capture_returns=(cap, None),
        )
        agent._cycle()
        agent.fuser.fuse.assert_called_once()
        kwargs = agent.fuser.fuse.call_args.kwargs
        assert kwargs["audio_result"] is not None
        assert kwargs["visual_result"] is not None


# ── _cycle — dual camera ──────────────────────────────────────────────────────


class TestCycleDualCamera:
    def test_fuser_called_with_visual_result_2(self) -> None:
        """Both cameras captured — fuser receives visual_result_2."""
        cap0 = _make_capture_result(camera_index=0)
        cap1 = _make_capture_result(camera_index=1)
        agent = _make_agent(
            audio_capture_returns=None,  # no audio this cycle
            vision_capture_returns=(cap0, cap1),
        )
        agent._cycle()
        kwargs = agent.fuser.fuse.call_args.kwargs
        assert kwargs["visual_result"] is not None
        assert kwargs["visual_result_2"] is not None

    def test_visual_result_2_none_when_secondary_camera_empty(self) -> None:
        """Secondary camera below motion threshold — visual_result_2 is None."""
        cap0 = _make_capture_result(camera_index=0)
        agent = _make_agent(
            audio_capture_returns=None,
            vision_capture_returns=(cap0, None),
        )
        agent._cycle()
        kwargs = agent.fuser.fuse.call_args.kwargs
        assert kwargs["visual_result"] is not None
        assert kwargs["visual_result_2"] is None


# ── _cycle — single modality ──────────────────────────────────────────────────


class TestCycleSingleModality:
    def test_audio_only_no_visual(self) -> None:
        """No camera motion — only audio classified."""
        agent = _make_agent(
            audio_capture_returns=Path("/tmp/fake.wav"),
            vision_capture_returns=(None, None),
        )
        agent._cycle()
        kwargs = agent.fuser.fuse.call_args.kwargs
        assert kwargs["audio_result"] is not None
        assert kwargs["visual_result"] is None

    def test_visual_only_no_audio(self) -> None:
        """Audio below energy threshold — only visual classified."""
        cap = _make_capture_result(camera_index=0)
        agent = _make_agent(
            audio_capture_returns=None,  # energy gate suppressed audio
            vision_capture_returns=(cap, None),
        )
        agent._cycle()
        kwargs = agent.fuser.fuse.call_args.kwargs
        assert kwargs["audio_result"] is None
        assert kwargs["visual_result"] is not None

    def test_no_capture_returns_none(self) -> None:
        """Both gates suppressed — no classifier inputs, no fuse call."""
        agent = _make_agent(
            audio_capture_returns=None,
            vision_capture_returns=(None, None),
        )
        result = agent._cycle()
        assert result is None
        agent.fuser.fuse.assert_not_called()
        agent.notifier.dispatch.assert_not_called()

    def test_audio_disabled_capture_skipped(self) -> None:
        """Audio classifier disabled — capture_window never called."""
        cap = _make_capture_result(camera_index=0)
        agent = _make_agent(
            audio_enabled=False,
            vision_capture_returns=(cap, None),
        )
        agent._cycle()
        assert agent.audio_capture is None


# ── _cycle — exception handling ───────────────────────────────────────────────


class TestCycleExceptionHandling:
    def test_no_bird_detected_error_skips_audio(self) -> None:
        """NoBirdDetectedError from BirdNET — audio result is None, visual continues."""
        cap = _make_capture_result(camera_index=0)
        agent = _make_agent(
            audio_capture_returns=Path("/tmp/fake.wav"),
            vision_capture_returns=(cap, None),
        )
        agent.audio_classifier.predict.side_effect = NoBirdDetectedError("no bird")
        agent._cycle()
        kwargs = agent.fuser.fuse.call_args.kwargs
        assert kwargs["audio_result"] is None
        assert kwargs["visual_result"] is not None

    def test_audio_runtime_error_continues_with_visual(self) -> None:
        """Unexpected audio classifier error — graceful degradation to visual only."""
        cap = _make_capture_result(camera_index=0)
        agent = _make_agent(
            audio_capture_returns=Path("/tmp/fake.wav"),
            vision_capture_returns=(cap, None),
        )
        agent.audio_classifier.predict.side_effect = RuntimeError("model broken")
        agent._cycle()
        kwargs = agent.fuser.fuse.call_args.kwargs
        assert kwargs["audio_result"] is None
        assert kwargs["visual_result"] is not None

    def test_visual_exception_continues_with_audio(self) -> None:
        """Visual classifier error — graceful degradation to audio only."""
        agent = _make_agent(
            audio_capture_returns=Path("/tmp/fake.wav"),
            vision_capture_returns=(_make_capture_result(0), None),
        )
        agent.visual_classifier.predict.side_effect = RuntimeError("model broken")
        agent._cycle()
        kwargs = agent.fuser.fuse.call_args.kwargs
        assert kwargs["audio_result"] is not None
        assert kwargs["visual_result"] is None

    def test_both_classifier_errors_returns_none(self) -> None:
        """Both classifiers fail — no fuse, no dispatch."""
        cap = _make_capture_result(camera_index=0)
        agent = _make_agent(
            audio_capture_returns=Path("/tmp/fake.wav"),
            vision_capture_returns=(cap, None),
        )
        agent.audio_classifier.predict.side_effect = RuntimeError("broken")
        agent.visual_classifier.predict.side_effect = RuntimeError("broken")
        result = agent._cycle()
        assert result is None
        agent.fuser.fuse.assert_not_called()


# ── _cycle — cooldown ─────────────────────────────────────────────────────────


class TestCycleCooldown:
    def test_first_dispatch_not_suppressed(self) -> None:
        """First detection always dispatches regardless of cooldown setting."""
        obs = _make_observation(fused_confidence=0.9)
        cap = _make_capture_result(camera_index=0)
        agent = _make_agent(
            fused_observation=obs,
            cooldown_seconds=30.0,
            audio_capture_returns=Path("/tmp/fake.wav"),
            vision_capture_returns=(cap, None),
        )
        agent._cycle()
        agent.notifier.dispatch.assert_called_once()

    def test_second_dispatch_suppressed_within_cooldown(self) -> None:
        """Same species detected twice rapidly — second dispatch suppressed."""
        obs = _make_observation(fused_confidence=0.9, species_code="HOFI")
        cap = _make_capture_result(camera_index=0)
        agent = _make_agent(
            fused_observation=obs,
            cooldown_seconds=30.0,
            audio_capture_returns=Path("/tmp/fake.wav"),
            vision_capture_returns=(cap, None),
        )
        agent._cycle()  # first — dispatches
        agent._cycle()  # second — suppressed by cooldown
        assert agent.notifier.dispatch.call_count == 1

    def test_different_species_not_suppressed_by_cooldown(self) -> None:
        """Different species detected — not suppressed even if first species on cooldown."""
        cap = _make_capture_result(camera_index=0)

        obs_hofi = _make_observation(fused_confidence=0.9, species_code="HOFI")
        agent = _make_agent(
            fused_observation=obs_hofi,
            cooldown_seconds=30.0,
            audio_capture_returns=Path("/tmp/fake.wav"),
            vision_capture_returns=(cap, None),
        )
        agent._cycle()  # HOFI dispatched

        # Now change fuser to return a different species
        obs_modo = _make_observation(fused_confidence=0.9, species_code="MODO")
        agent.fuser.fuse.return_value = obs_modo
        agent._cycle()  # MODO — different species, not on cooldown

        assert agent.notifier.dispatch.call_count == 2


# ── from_config ───────────────────────────────────────────────────────────────


class TestFromConfig:
    def test_constructs_without_error(self) -> None:
        agent = BirdAgent.from_config("configs/")
        assert agent.confidence_threshold == pytest.approx(0.7)
        assert agent.loop_interval_seconds == pytest.approx(1.0)

    def test_both_classifiers_present(self) -> None:
        agent = BirdAgent.from_config("configs/")
        assert agent.audio_classifier is not None
        assert agent.visual_classifier is not None

    def test_capture_modules_present(self) -> None:
        agent = BirdAgent.from_config("configs/")
        assert agent.audio_capture is not None
        assert agent.vision_capture is not None

    def test_fuser_and_notifier_present(self) -> None:
        agent = BirdAgent.from_config("configs/")
        assert agent.fuser is not None
        assert agent.notifier is not None

    def test_audio_classifier_not_loaded(self) -> None:
        """Lazy loading — BirdNET Analyzer not loaded until first predict()."""
        agent = BirdAgent.from_config("configs/")
        assert agent.audio_classifier._analyzer is None

    def test_visual_classifier_not_loaded(self) -> None:
        """Lazy loading — EfficientNet extractor not loaded until first predict()."""
        agent = BirdAgent.from_config("configs/")
        assert agent.visual_classifier._extractor is None

    def test_cooldown_loaded_from_config(self) -> None:
        agent = BirdAgent.from_config("configs/")
        assert agent.cooldown_seconds == pytest.approx(30.0)


# ── _is_on_cooldown ───────────────────────────────────────────────────────────


class TestCooldownHelper:
    def test_unknown_species_not_on_cooldown(self) -> None:
        agent = _make_agent(cooldown_seconds=30.0)
        assert agent._is_on_cooldown("HOFI") is False

    def test_species_on_cooldown_after_dispatch(self) -> None:
        from datetime import UTC, datetime

        agent = _make_agent(cooldown_seconds=30.0)
        agent._last_dispatch["HOFI"] = datetime.now(UTC)
        assert agent._is_on_cooldown("HOFI") is True

    def test_species_not_on_cooldown_after_expiry(self) -> None:
        from datetime import UTC, datetime, timedelta

        agent = _make_agent(cooldown_seconds=30.0)
        agent._last_dispatch["HOFI"] = datetime.now(UTC) - timedelta(seconds=60)
        assert agent._is_on_cooldown("HOFI") is False
