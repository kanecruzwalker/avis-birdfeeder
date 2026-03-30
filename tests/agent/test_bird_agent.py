"""
tests/agent/test_bird_agent.py

Unit tests for src/agent/bird_agent.py

Tests agent construction and configuration without triggering any
hardware access, model loading, or the actual run loop.
"""

from src.agent.bird_agent import BirdAgent


class TestBirdAgentInit:
    def test_default_construction(self):
        """BirdAgent should construct with sensible defaults."""
        agent = BirdAgent()
        assert agent.audio_enabled is True
        assert agent.visual_enabled is True
        assert agent.confidence_threshold == 0.7
        assert agent.loop_interval_seconds == 1.0
        assert agent._running is False

    def test_custom_confidence_threshold(self):
        """Custom confidence threshold should be stored correctly."""
        agent = BirdAgent(confidence_threshold=0.85)
        assert agent.confidence_threshold == 0.85

    def test_audio_disabled(self):
        """Agent should allow disabling the audio pipeline."""
        agent = BirdAgent(audio_enabled=False)
        assert agent.audio_enabled is False
        assert agent.visual_enabled is True

    def test_visual_disabled(self):
        """Agent should allow disabling the visual pipeline."""
        agent = BirdAgent(visual_enabled=False)
        assert agent.visual_enabled is False
        assert agent.audio_enabled is True

    def test_stop_sets_running_false(self):
        """stop() should set _running to False."""
        agent = BirdAgent()
        agent._running = True
        agent.stop()
        assert agent._running is False
