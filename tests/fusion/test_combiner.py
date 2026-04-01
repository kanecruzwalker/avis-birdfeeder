"""
tests/fusion/test_combiner.py

Unit tests for src/fusion/combiner.py.

All tests are fully synthetic — no model weights, no hardware, no network.

Test groups:
    TestScoreFuserInit        — constructor validation (existing Phase 1 tests)
    TestFuseConfidence        — _fuse_confidence() for all three strategies
    TestFuseSingleModality    — graceful fallback when one modality is None
    TestFuseBothModalities    — full fusion when both results are provided
    TestFuseDisagreement      — winner-takes-all when species codes differ
    TestFromConfig            — from_config() reads thresholds.yaml correctly
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.data.schema import BirdObservation, ClassificationResult, Modality
from src.fusion.combiner import ScoreFuser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    species_code: str = "HOFI",
    common_name: str = "House Finch",
    scientific_name: str = "Haemorhous mexicanus",
    confidence: float = 0.8,
    modality: Modality = Modality.AUDIO,
) -> ClassificationResult:
    """Build a minimal ClassificationResult for testing."""
    return ClassificationResult(
        species_code=species_code,
        common_name=common_name,
        scientific_name=scientific_name,
        confidence=confidence,
        modality=modality,
    )


# ---------------------------------------------------------------------------
# TestScoreFuserInit  (original Phase 1 tests — unchanged)
# ---------------------------------------------------------------------------


class TestScoreFuserInit:
    def test_valid_weighted_strategy(self):
        """Weighted strategy with weights summing to 1.0 should construct."""
        fuser = ScoreFuser(strategy="weighted", audio_weight=0.55, visual_weight=0.45)
        assert fuser.strategy == "weighted"
        assert fuser.audio_weight == 0.55
        assert fuser.visual_weight == 0.45

    def test_valid_equal_strategy(self):
        """Equal strategy should construct without specifying weights."""
        fuser = ScoreFuser(strategy="equal")
        assert fuser.strategy == "equal"

    def test_valid_max_strategy(self):
        """Max strategy should construct without specifying weights."""
        fuser = ScoreFuser(strategy="max")
        assert fuser.strategy == "max"

    def test_invalid_strategy_raises(self):
        """An unrecognized strategy string should raise ValueError."""
        with pytest.raises(ValueError, match="strategy must be one of"):
            ScoreFuser(strategy="unknown_strategy")

    def test_weighted_weights_not_summing_to_one_raises(self):
        """Weighted strategy with weights that don't sum to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="must equal 1.0"):
            ScoreFuser(strategy="weighted", audio_weight=0.6, visual_weight=0.6)

    def test_default_strategy_is_weighted(self):
        """Default strategy should be 'weighted'."""
        fuser = ScoreFuser()
        assert fuser.strategy == "weighted"

    def test_fuse_with_no_results_raises(self):
        """Calling fuse() with both None should raise ValueError."""
        fuser = ScoreFuser()
        with pytest.raises(ValueError, match="At least one"):
            fuser.fuse(audio_result=None, visual_result=None)


# ---------------------------------------------------------------------------
# TestFuseConfidence
# ---------------------------------------------------------------------------


class TestFuseConfidence:
    def test_equal_strategy_averages_scores(self):
        fuser = ScoreFuser(strategy="equal")
        result = fuser._fuse_confidence(0.8, 0.4)
        assert abs(result - 0.6) < 1e-6

    def test_weighted_strategy_applies_weights(self):
        fuser = ScoreFuser(strategy="weighted", audio_weight=0.55, visual_weight=0.45)
        result = fuser._fuse_confidence(0.8, 0.4)
        expected = 0.8 * 0.55 + 0.4 * 0.45
        assert abs(result - expected) < 1e-6

    def test_max_strategy_returns_higher_score(self):
        fuser = ScoreFuser(strategy="max")
        assert fuser._fuse_confidence(0.8, 0.4) == 0.8
        assert fuser._fuse_confidence(0.3, 0.9) == 0.9

    def test_equal_strategy_symmetric(self):
        """equal strategy result must not depend on argument order."""
        fuser = ScoreFuser(strategy="equal")
        assert fuser._fuse_confidence(0.7, 0.3) == fuser._fuse_confidence(0.3, 0.7)

    def test_fused_confidence_in_valid_range(self):
        """All strategies must produce confidence in [0, 1]."""
        for strategy in ["equal", "max"]:
            fuser = ScoreFuser(strategy=strategy)
            result = fuser._fuse_confidence(1.0, 0.0)
            assert 0.0 <= result <= 1.0

    def test_weighted_identical_scores(self):
        """When both scores are equal, all strategies should return that score."""
        for strategy in ["equal", "max"]:
            fuser = ScoreFuser(strategy=strategy)
            assert abs(fuser._fuse_confidence(0.7, 0.7) - 0.7) < 1e-6

        fuser = ScoreFuser(strategy="weighted", audio_weight=0.6, visual_weight=0.4)
        assert abs(fuser._fuse_confidence(0.7, 0.7) - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# TestFuseSingleModality
# ---------------------------------------------------------------------------


class TestFuseSingleModality:
    def test_audio_only_returns_observation(self):
        fuser = ScoreFuser()
        audio = _make_result(modality=Modality.AUDIO, confidence=0.75)
        obs = fuser.fuse(audio_result=audio, visual_result=None)
        assert isinstance(obs, BirdObservation)

    def test_audio_only_confidence_unchanged(self):
        fuser = ScoreFuser()
        audio = _make_result(modality=Modality.AUDIO, confidence=0.75)
        obs = fuser.fuse(audio_result=audio, visual_result=None)
        assert obs.fused_confidence == 0.75

    def test_audio_only_species_preserved(self):
        fuser = ScoreFuser()
        audio = _make_result(species_code="WCSP", modality=Modality.AUDIO, confidence=0.6)
        obs = fuser.fuse(audio_result=audio, visual_result=None)
        assert obs.species_code == "WCSP"

    def test_audio_only_sets_audio_result(self):
        fuser = ScoreFuser()
        audio = _make_result(modality=Modality.AUDIO)
        obs = fuser.fuse(audio_result=audio, visual_result=None)
        assert obs.audio_result is audio
        assert obs.visual_result is None

    def test_visual_only_returns_observation(self):
        fuser = ScoreFuser()
        visual = _make_result(modality=Modality.VISUAL, confidence=0.65)
        obs = fuser.fuse(audio_result=None, visual_result=visual)
        assert isinstance(obs, BirdObservation)

    def test_visual_only_confidence_unchanged(self):
        fuser = ScoreFuser()
        visual = _make_result(modality=Modality.VISUAL, confidence=0.65)
        obs = fuser.fuse(audio_result=None, visual_result=visual)
        assert obs.fused_confidence == 0.65

    def test_visual_only_sets_visual_result(self):
        fuser = ScoreFuser()
        visual = _make_result(modality=Modality.VISUAL)
        obs = fuser.fuse(audio_result=None, visual_result=visual)
        assert obs.visual_result is visual
        assert obs.audio_result is None


# ---------------------------------------------------------------------------
# TestFuseBothModalities
# ---------------------------------------------------------------------------


class TestFuseBothModalities:
    def test_agreement_returns_observation(self):
        fuser = ScoreFuser(strategy="weighted")
        audio = _make_result(species_code="HOFI", confidence=0.8, modality=Modality.AUDIO)
        visual = _make_result(species_code="HOFI", confidence=0.6, modality=Modality.VISUAL)
        obs = fuser.fuse(audio_result=audio, visual_result=visual)
        assert isinstance(obs, BirdObservation)

    def test_agreement_species_code_correct(self):
        fuser = ScoreFuser()
        audio = _make_result(species_code="HOFI", confidence=0.8, modality=Modality.AUDIO)
        visual = _make_result(species_code="HOFI", confidence=0.6, modality=Modality.VISUAL)
        obs = fuser.fuse(audio_result=audio, visual_result=visual)
        assert obs.species_code == "HOFI"

    def test_agreement_weighted_confidence(self):
        fuser = ScoreFuser(strategy="weighted", audio_weight=0.55, visual_weight=0.45)
        audio = _make_result(species_code="HOFI", confidence=0.8, modality=Modality.AUDIO)
        visual = _make_result(species_code="HOFI", confidence=0.6, modality=Modality.VISUAL)
        obs = fuser.fuse(audio_result=audio, visual_result=visual)
        expected = 0.8 * 0.55 + 0.6 * 0.45
        assert abs(obs.fused_confidence - expected) < 1e-6

    def test_agreement_equal_confidence(self):
        fuser = ScoreFuser(strategy="equal")
        audio = _make_result(species_code="HOFI", confidence=0.8, modality=Modality.AUDIO)
        visual = _make_result(species_code="HOFI", confidence=0.6, modality=Modality.VISUAL)
        obs = fuser.fuse(audio_result=audio, visual_result=visual)
        assert abs(obs.fused_confidence - 0.7) < 1e-6

    def test_agreement_max_confidence(self):
        fuser = ScoreFuser(strategy="max")
        audio = _make_result(species_code="HOFI", confidence=0.8, modality=Modality.AUDIO)
        visual = _make_result(species_code="HOFI", confidence=0.6, modality=Modality.VISUAL)
        obs = fuser.fuse(audio_result=audio, visual_result=visual)
        assert obs.fused_confidence == 0.8

    def test_both_results_preserved_on_observation(self):
        fuser = ScoreFuser()
        audio = _make_result(species_code="HOFI", confidence=0.8, modality=Modality.AUDIO)
        visual = _make_result(species_code="HOFI", confidence=0.6, modality=Modality.VISUAL)
        obs = fuser.fuse(audio_result=audio, visual_result=visual)
        assert obs.audio_result is audio
        assert obs.visual_result is visual

    def test_has_both_modalities_true(self):
        fuser = ScoreFuser()
        audio = _make_result(species_code="HOFI", modality=Modality.AUDIO)
        visual = _make_result(species_code="HOFI", modality=Modality.VISUAL)
        obs = fuser.fuse(audio_result=audio, visual_result=visual)
        assert obs.has_both_modalities is True


# ---------------------------------------------------------------------------
# TestFuseDisagreement
# ---------------------------------------------------------------------------


class TestFuseDisagreement:
    def test_higher_audio_confidence_wins(self):
        fuser = ScoreFuser()
        audio = _make_result(species_code="HOFI", confidence=0.9, modality=Modality.AUDIO)
        visual = _make_result(
            species_code="WCSP",
            confidence=0.4,
            modality=Modality.VISUAL,
            common_name="White-crowned Sparrow",
            scientific_name="Zonotrichia leucophrys",
        )
        obs = fuser.fuse(audio_result=audio, visual_result=visual)
        assert obs.species_code == "HOFI"

    def test_higher_visual_confidence_wins(self):
        fuser = ScoreFuser()
        audio = _make_result(species_code="HOFI", confidence=0.3, modality=Modality.AUDIO)
        visual = _make_result(
            species_code="WCSP",
            confidence=0.9,
            modality=Modality.VISUAL,
            common_name="White-crowned Sparrow",
            scientific_name="Zonotrichia leucophrys",
        )
        obs = fuser.fuse(audio_result=audio, visual_result=visual)
        assert obs.species_code == "WCSP"

    def test_disagreement_winner_confidence_used(self):
        """Winner's raw confidence is used, not a blended score."""
        fuser = ScoreFuser()
        audio = _make_result(species_code="HOFI", confidence=0.9, modality=Modality.AUDIO)
        visual = _make_result(
            species_code="WCSP",
            confidence=0.4,
            modality=Modality.VISUAL,
            common_name="White-crowned Sparrow",
            scientific_name="Zonotrichia leucophrys",
        )
        obs = fuser.fuse(audio_result=audio, visual_result=visual)
        assert obs.fused_confidence == 0.9

    def test_disagreement_both_results_still_on_observation(self):
        """Both raw results are preserved even when there is disagreement."""
        fuser = ScoreFuser()
        audio = _make_result(species_code="HOFI", confidence=0.9, modality=Modality.AUDIO)
        visual = _make_result(
            species_code="WCSP",
            confidence=0.4,
            modality=Modality.VISUAL,
            common_name="White-crowned Sparrow",
            scientific_name="Zonotrichia leucophrys",
        )
        obs = fuser.fuse(audio_result=audio, visual_result=visual)
        assert obs.audio_result is audio
        assert obs.visual_result is visual


# ---------------------------------------------------------------------------
# TestFromConfig
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_loads_strategy_from_yaml(self, tmp_path: Path):
        cfg = {"fusion": {"strategy": "equal", "audio_weight": 0.5, "visual_weight": 0.5}}
        config_file = tmp_path / "thresholds.yaml"
        config_file.write_text(yaml.dump(cfg))
        fuser = ScoreFuser.from_config(str(config_file))
        assert fuser.strategy == "equal"

    def test_loads_weights_from_yaml(self, tmp_path: Path):
        cfg = {"fusion": {"strategy": "weighted", "audio_weight": 0.7, "visual_weight": 0.3}}
        config_file = tmp_path / "thresholds.yaml"
        config_file.write_text(yaml.dump(cfg))
        fuser = ScoreFuser.from_config(str(config_file))
        assert abs(fuser.audio_weight - 0.7) < 1e-6
        assert abs(fuser.visual_weight - 0.3) < 1e-6

    def test_raises_on_missing_config(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ScoreFuser.from_config(str(tmp_path / "nonexistent.yaml"))

    def test_defaults_when_fusion_key_absent(self, tmp_path: Path):
        """If fusion key is missing, should fall back to defaults."""
        cfg = {"agent": {"confidence_threshold": 0.7}}
        config_file = tmp_path / "thresholds.yaml"
        config_file.write_text(yaml.dump(cfg))
        fuser = ScoreFuser.from_config(str(config_file))
        assert fuser.strategy == "weighted"
        assert abs(fuser.audio_weight - 0.55) < 1e-6
