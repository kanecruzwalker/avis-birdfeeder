"""
src/agent/bird_agent.py

The central orchestrator for the Avis system.

The BirdAgent runs a continuous perception-decision-action loop:

    PERCEIVE  →  DETECT  →  CLASSIFY  →  FUSE  →  ACT

    1. Perceive:  Continuously monitors camera feed and microphone input.
    2. Detect:    Determines whether a bird is likely present (motion + audio energy).
    3. Classify:  Triggers audio and visual classification pipelines in parallel.
    4. Fuse:      Combines classifier outputs into a single BirdObservation.
    5. Act:       Dispatches the observation via the notifier, saves media.

Design principles:
    - The agent owns the loop. All other modules are called by the agent.
    - No other module imports from agent (see ARCHITECTURE.md dependency rules).
    - The agent is configurable via YAML — loop interval, confidence thresholds,
      which modalities are active, etc.
    - Graceful degradation: if one modality fails (e.g. mic unplugged), the
      agent continues with the other.

Phase 3 will implement the detection logic (motion + audio energy thresholds).
Phase 4 will wire in the real classifiers.
Phase 5 will add parallel execution for audio + visual pipelines.

Run directly:
    python -m src.agent.bird_agent
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class BirdAgent:
    """
    Main agentic loop for the Avis birdfeeder system.

    Usage:
        agent = BirdAgent.from_config("configs/")
        agent.run()
    """

    def __init__(
        self,
        loop_interval_seconds: float = 1.0,
        audio_enabled: bool = True,
        visual_enabled: bool = True,
        confidence_threshold: float = 0.7,
    ) -> None:
        """
        Args:
            loop_interval_seconds: Seconds to wait between perception cycles
                when no bird is detected. Reduces CPU load during idle periods.
            audio_enabled: Whether the audio pipeline is active. Set False if
                mic is unavailable or being debugged independently.
            visual_enabled: Whether the visual pipeline is active.
            confidence_threshold: Minimum fused confidence to trigger a
                notification. Predictions below this are discarded.
        """
        self.loop_interval_seconds = loop_interval_seconds
        self.audio_enabled = audio_enabled
        self.visual_enabled = visual_enabled
        self.confidence_threshold = confidence_threshold
        self._running = False

        logger.info(
            "BirdAgent initialized | "
            f"audio={audio_enabled} visual={visual_enabled} "
            f"threshold={confidence_threshold} interval={loop_interval_seconds}s"
        )

    @classmethod
    def from_config(cls, config_dir: str | Path) -> BirdAgent:
        """
        Construct a BirdAgent from the configs/ directory.

        Reads:
            configs/thresholds.yaml  → confidence_threshold, loop_interval
            configs/paths.yaml       → model paths, log path

        Args:
            config_dir: Path to the configs/ directory.

        Returns:
            Fully configured BirdAgent instance.
        """
        raise NotImplementedError("Implement in Phase 3.")

    def run(self) -> None:
        """
        Start the main perception-decision-action loop.

        Runs until interrupted (KeyboardInterrupt) or stop() is called.
        """
        self._running = True
        logger.info("BirdAgent starting main loop.")
        try:
            while self._running:
                self._cycle()
                time.sleep(self.loop_interval_seconds)
        except KeyboardInterrupt:
            logger.info("BirdAgent interrupted by user.")
        finally:
            self._running = False
            logger.info("BirdAgent stopped.")

    def stop(self) -> None:
        """
        Signal the agent loop to stop after the current cycle completes.
        Safe to call from another thread.
        """
        self._running = False

    def _cycle(self) -> None:
        """
        Execute one perception-decision-action cycle.

        Steps:
            1. Detect bird presence (motion / audio energy)
            2. If detected: capture audio window + camera frame
            3. Run classifiers (audio and/or visual)
            4. Fuse results
            5. If fused confidence >= threshold: dispatch via notifier
        """
        # Phase 3: implement detection
        # Phase 4: wire in classifiers
        # Phase 5: parallelize audio + visual capture/classify
        logger.debug("Agent cycle — classifiers not yet wired in.")


def main() -> None:
    """Entry point when running the agent directly: python -m src.agent.bird_agent"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    agent = BirdAgent()
    agent.run()


if __name__ == "__main__":
    main()
