"""
tests/fusion/test_combiner.py

Unit tests for src/fusion/combiner.py

Tests that don't require model weights or hardware — specifically the
ScoreFuser constructor validation and strategy configuration logic.

Full fusion tests (requiring ClassificationResult inputs) will be added
in Phase 3 when the fusion logic is implemented.
"""

import pytest

from src.fusion.combiner import ScoreFuser


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
