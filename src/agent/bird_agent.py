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

Phase 4 wires in AudioClassifier and VisualClassifier. Capture hardware
(mic + camera) and parallel execution are Phase 5.

Run directly:
    python -m src.agent.bird_agent
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import yaml

from src.audio.classify import AudioClassifier
from src.data.schema import BirdObservation
from src.fusion.combiner import ScoreFuser
from src.notify.notifier import Notifier
from src.vision.classify import VisualClassifier

logger = logging.getLogger(__name__)


class BirdAgent:
    """
    Main agentic loop for the Avis birdfeeder system.

    In Phase 4 the agent accepts pre-captured numpy arrays (spectrogram +
    frame) passed directly to _cycle(). Phase 5 will add live capture via
    src.audio.capture and src.vision.capture, replacing the stub inputs.

    Usage:
        agent = BirdAgent.from_config("configs/")
        agent.run()
    """

    def __init__(
        self,
        audio_classifier: AudioClassifier | None,
        visual_classifier: VisualClassifier | None,
        fuser: ScoreFuser,
        notifier: Notifier,
        loop_interval_seconds: float = 1.0,
        confidence_threshold: float = 0.7,
    ) -> None:
        """
        Args:
            audio_classifier: Loaded AudioClassifier, or None if audio disabled.
            visual_classifier: Loaded VisualClassifier, or None if visual disabled.
            fuser: ScoreFuser instance for combining classifier outputs.
            notifier: Notifier instance for dispatching observations.
            loop_interval_seconds: Seconds to wait between cycles during idle.
            confidence_threshold: Minimum fused confidence to trigger notification.
        """
        if audio_classifier is None and visual_classifier is None:
            raise ValueError("At least one of audio_classifier or visual_classifier must be set.")

        self.audio_classifier = audio_classifier
        self.visual_classifier = visual_classifier
        self.fuser = fuser
        self.notifier = notifier
        self.loop_interval_seconds = loop_interval_seconds
        self.confidence_threshold = confidence_threshold
        self._running = False

        active = []
        if audio_classifier:
            active.append("audio")
        if visual_classifier:
            active.append("visual")

        logger.info(
            "BirdAgent initialized | modalities=%s threshold=%.2f interval=%.1fs",
            "+".join(active),
            confidence_threshold,
            loop_interval_seconds,
        )

    @classmethod
    def from_config(cls, config_dir: str | Path) -> BirdAgent:
        """
        Construct a BirdAgent from the configs/ directory.

        Reads:
            configs/thresholds.yaml  → confidence_threshold, loop_interval_seconds
            configs/paths.yaml       → model paths, log paths (via sub-components)

        Both classifiers are constructed but weights are loaded lazily on first
        predict() call — from_config() itself is fast and does not touch disk
        beyond reading YAML.

        Args:
            config_dir: Path to the configs/ directory.

        Returns:
            Fully configured BirdAgent instance.
        """
        config_dir = Path(config_dir)
        thresholds_path = config_dir / "thresholds.yaml"
        paths_path = config_dir / "paths.yaml"

        with thresholds_path.open() as f:
            thresholds = yaml.safe_load(f)

        confidence_threshold = thresholds["agent"]["confidence_threshold"]
        loop_interval_seconds = thresholds["agent"]["loop_interval_seconds"]

        audio_classifier = AudioClassifier.from_config(str(paths_path))
        visual_classifier = VisualClassifier.from_config(str(paths_path))
        fuser = ScoreFuser.from_config(str(thresholds_path))
        notifier = Notifier.from_config(
            notify_config_path=str(config_dir / "notify.yaml"),
            paths_config_path=str(paths_path),
        )

        logger.info(
            "BirdAgent.from_config | config_dir=%s threshold=%.2f",
            config_dir,
            confidence_threshold,
        )

        return cls(
            audio_classifier=audio_classifier,
            visual_classifier=visual_classifier,
            fuser=fuser,
            notifier=notifier,
            loop_interval_seconds=loop_interval_seconds,
            confidence_threshold=confidence_threshold,
        )

    def run(self) -> None:
        """
        Start the main perception-decision-action loop.

        Runs until interrupted (KeyboardInterrupt) or stop() is called.
        In Phase 4 each cycle receives stub inputs (zeros) since live capture
        is not yet wired. Phase 5 replaces stubs with real capture calls.
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

    def _cycle(
        self,
        spectrogram: np.ndarray | None = None,
        frame: np.ndarray | None = None,
    ) -> BirdObservation | None:
        """
        Execute one perception-decision-action cycle.

        Phase 4: accepts pre-captured numpy arrays for offline / notebook use.
        Phase 5: spectrogram and frame will be captured live from hardware;
                 this signature will remain but defaults will come from capture.

        Steps:
            1. Run available classifiers on provided inputs.
            2. Fuse results (handles single-modality gracefully via ScoreFuser).
            3. If fused confidence >= threshold: dispatch via notifier.
            4. Return the BirdObservation (or None if below threshold).

        Args:
            spectrogram: Float32 (n_mels, time_frames) mel spectrogram, or None
                         to skip audio classification this cycle.
            frame: Float32 (224, 224, 3) ImageNet-normalized frame, or None
                   to skip visual classification this cycle.

        Returns:
            BirdObservation if confidence >= threshold, else None.
        """
        audio_result = None
        visual_result = None

        # Audio classification
        if self.audio_classifier is not None and spectrogram is not None:
            try:
                audio_result = self.audio_classifier.predict(spectrogram)
                logger.debug(
                    "Audio: %s (%.3f)", audio_result.species_code, audio_result.confidence
                )
            except Exception:
                logger.exception("Audio classifier failed — skipping audio this cycle.")

        # Visual classification
        if self.visual_classifier is not None and frame is not None:
            try:
                visual_result = self.visual_classifier.predict(frame)
                logger.debug(
                    "Visual: %s (%.3f)", visual_result.species_code, visual_result.confidence
                )
            except Exception:
                logger.exception("Visual classifier failed — skipping visual this cycle.")

        # Nothing to fuse
        if audio_result is None and visual_result is None:
            logger.debug("Agent cycle — no classifier inputs available.")
            return None

        # Fuse
        observation = self.fuser.fuse(
            audio_result=audio_result,
            visual_result=visual_result,
        )

        # Threshold gate
        if observation.fused_confidence < self.confidence_threshold:
            logger.debug(
                "Observation below threshold: %s %.3f < %.3f",
                observation.species_code,
                observation.fused_confidence,
                self.confidence_threshold,
            )
            return None

        # Dispatch
        self.notifier.dispatch(observation)
        logger.info(
            "Observation dispatched: %s (fused=%.3f)",
            observation.species_code,
            observation.fused_confidence,
        )
        return observation
