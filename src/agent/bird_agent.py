"""
src/agent/bird_agent.py

The central orchestrator for the Avis system.

The BirdAgent runs a continuous perception-decision-action loop:

    PERCEIVE  →  DETECT  →  CLASSIFY  →  FUSE  →  ACT

    1. Perceive:  Captures audio windows and camera frames continuously.
    2. Detect:    Energy gate (audio) and motion gate (vision) filter empty cycles.
    3. Classify:  Audio → BirdNET. Both camera frames → EfficientNet + LogReg.
    4. Fuse:      Combines all available results into a single BirdObservation.
    5. Act:       Dispatches via notifier, saves media paths to observation.

Phase 5 changes from Phase 4:
    - _cycle() no longer accepts stub numpy arrays. Live capture is handled
      internally via AudioCapture and VisionCapture.
    - AudioClassifier.predict() now takes a WAV file path (not a spectrogram array).
    - VisualClassifier.predict() takes a preprocessed (224, 224, 3) frame array.
    - Both cameras are classified each cycle. ScoreFuser receives visual_result_2.
    - BirdObservation media paths (audio_path, image_path, image_path_2) are
      populated from the capture modules before dispatch.
    - Cooldown suppression: repeated observations of the same species within
      cooldown_seconds are suppressed to prevent notification spam.

Design principles (unchanged):
    - The agent owns the loop. All other modules are called by the agent.
    - No other module imports from agent (see ARCHITECTURE.md dependency rules).
    - Graceful degradation: if one modality or one camera fails, the agent
      continues with whatever is available.
    - All tunable values come from YAML configs — nothing hardcoded.

Run directly:
    python -m src.agent.bird_agent
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.audio.capture import AudioCapture
from src.audio.classify import AudioClassifier, NoBirdDetectedError
from src.data.schema import BirdObservation
from src.fusion.combiner import ScoreFuser
from src.notify.notifier import Notifier
from src.vision.capture import VisionCapture
from src.vision.classify import VisualClassifier

logger = logging.getLogger(__name__)


class BirdAgent:
    """
    Main agentic loop for the Avis birdfeeder system.

    Phase 5: Live capture from Fifine USB mic and dual Pi Camera Module 3 sensors.
    Audio classification via BirdNET (file-based).
    Visual classification via frozen EfficientNet + LogReg (frame-based).
    Dual camera results fused by ScoreFuser with visual_result_2 support.

    Usage:
        agent = BirdAgent.from_config("configs/")
        agent.run()         # blocks until KeyboardInterrupt
        agent.stop()        # call from another thread to stop gracefully
    """

    def __init__(
        self,
        audio_classifier: AudioClassifier | None,
        visual_classifier: VisualClassifier | None,
        audio_capture: AudioCapture | None,
        vision_capture: VisionCapture | None,
        fuser: ScoreFuser,
        notifier: Notifier,
        loop_interval_seconds: float = 1.0,
        confidence_threshold: float = 0.7,
        cooldown_seconds: float = 30.0,
    ) -> None:
        """
        Args:
            audio_classifier:      Loaded AudioClassifier, or None if audio disabled.
            visual_classifier:     Loaded VisualClassifier, or None if visual disabled.
            audio_capture:         AudioCapture instance, or None if audio disabled.
            vision_capture:        VisionCapture instance, or None if visual disabled.
            fuser:                 ScoreFuser for combining classifier outputs.
            notifier:              Notifier for dispatching observations.
            loop_interval_seconds: Seconds to wait between cycles during idle.
            confidence_threshold:  Minimum fused confidence to trigger notification.
            cooldown_seconds:      Suppress repeat notifications for same species
                                   within this window. Prevents spam when a bird
                                   lingers at the feeder.
        """
        if audio_classifier is None and visual_classifier is None:
            raise ValueError("At least one of audio_classifier or visual_classifier must be set.")

        self.audio_classifier = audio_classifier
        self.visual_classifier = visual_classifier
        self.audio_capture = audio_capture
        self.vision_capture = vision_capture
        self.fuser = fuser
        self.notifier = notifier
        self.loop_interval_seconds = loop_interval_seconds
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self._running = False

        # Cooldown tracking: species_code → last dispatch datetime (UTC)
        self._last_dispatch: dict[str, datetime] = {}

        active = []
        if audio_classifier:
            active.append("audio")
        if visual_classifier:
            active.append("visual(dual-camera)")

        logger.info(
            "BirdAgent initialized | modalities=%s threshold=%.2f " "interval=%.1fs cooldown=%.0fs",
            "+".join(active),
            confidence_threshold,
            loop_interval_seconds,
            cooldown_seconds,
        )

    @classmethod
    def from_config(cls, config_dir: str | Path) -> BirdAgent:
        """
        Construct a BirdAgent from the configs/ directory.

        Reads:
            configs/thresholds.yaml  → confidence_threshold, loop_interval_seconds,
                                       cooldown_seconds
            configs/paths.yaml       → model paths, capture output dirs (via sub-components)
            configs/hardware.yaml    → device indices, capture parameters (via sub-components)
            configs/notify.yaml      → notification channel config (via Notifier)
            configs/species.yaml     → species lookup (via classifiers)

        All sub-components use lazy loading — model weights are not loaded until
        the first predict() call. from_config() itself is fast.

        Args:
            config_dir: Path to the configs/ directory.

        Returns:
            Fully configured BirdAgent instance.
        """
        config_dir = Path(config_dir)
        thr_path = config_dir / "thresholds.yaml"
        paths_path = config_dir / "paths.yaml"
        notify_path = config_dir / "notify.yaml"

        with thr_path.open() as f:
            thresholds = yaml.safe_load(f)

        confidence_threshold = thresholds["agent"]["confidence_threshold"]
        loop_interval_seconds = thresholds["agent"]["loop_interval_seconds"]
        cooldown_seconds = thresholds["agent"].get("cooldown_seconds", 30.0)

        audio_classifier = AudioClassifier.from_config(str(paths_path))
        audio_capture = AudioCapture.from_config(str(config_dir))
        vision_capture = VisionCapture.from_config(str(config_dir))
        # Share Hailo VDevice between YOLO detector and EfficientNet extractor
        visual_classifier = VisualClassifier.from_config(
            str(paths_path),
            shared_vdevice=vision_capture.get_shared_vdevice(),
        )

        fuser = ScoreFuser.from_config(str(thr_path))
        notifier = Notifier.from_config(
            notify_config_path=str(notify_path),
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
            audio_capture=audio_capture,
            vision_capture=vision_capture,
            fuser=fuser,
            notifier=notifier,
            loop_interval_seconds=loop_interval_seconds,
            confidence_threshold=confidence_threshold,
            cooldown_seconds=cooldown_seconds,
        )

    def run(self) -> None:
        """
        Start the main perception-decision-action loop.

        Runs until interrupted (KeyboardInterrupt) or stop() is called.
        Cleans up camera hardware on exit via VisionCapture.stop().
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
            if self.vision_capture is not None:
                self.vision_capture.stop()
            logger.info("BirdAgent stopped.")

    def stop(self) -> None:
        """
        Signal the agent loop to stop after the current cycle completes.
        Safe to call from another thread.
        """
        self._running = False

    def _cycle(self) -> BirdObservation | None:
        """
        Execute one full perception-decision-action cycle.

        Phase 5 cycle steps:
            1. Capture audio window → energy gate → save WAV → get file path
            2. Capture frames from both cameras → motion gate → save PNGs
            3. Classify audio (BirdNET on WAV file path) if captured
            4. Classify primary camera frame if captured
            5. Classify secondary camera frame if captured
            6. Fuse all available results
            7. Apply confidence threshold gate
            8. Apply cooldown gate (suppress repeated same-species notifications)
            9. Populate media paths on observation
            10. Dispatch via notifier

        Returns:
            BirdObservation if dispatched, else None.
        """
        audio_result = None
        visual_result = None
        visual_result_2 = None
        audio_path = None
        image_path = None
        image_path_2 = None

        # ── Step 1: Audio capture ─────────────────────────────────────────────
        if self.audio_capture is not None and self.audio_classifier is not None:
            try:
                audio_path = self.audio_capture.capture_window()
                # capture_window() returns None if below energy threshold
            except Exception:
                logger.exception("Audio capture failed — skipping audio this cycle.")

        # ── Step 2: Visual capture (both cameras) ─────────────────────────────
        capture_primary = None
        capture_secondary = None
        if self.vision_capture is not None:
            try:
                capture_primary, capture_secondary = self.vision_capture.capture_frames()
            except Exception:
                logger.exception("Vision capture failed — skipping visual this cycle.")

        if capture_primary is not None:
            image_path = capture_primary.image_path

        if capture_secondary is not None:
            image_path_2 = capture_secondary.image_path

        # ── Step 3: Audio classification ──────────────────────────────────────
        if audio_path is not None and self.audio_classifier is not None:
            try:
                audio_result = self.audio_classifier.predict(audio_path)
                logger.debug("Audio: %s (%.3f)", audio_result.species_code, audio_result.confidence)
            except NoBirdDetectedError as exc:
                logger.debug("Audio: no SD bird detected — %s", exc)
            except Exception:
                logger.exception("Audio classifier failed — skipping audio this cycle.")

        # ── Step 4: Primary camera classification ─────────────────────────────
        if capture_primary is not None and self.visual_classifier is not None:
            try:
                visual_result = self.visual_classifier.predict(
                    capture_primary.frame,
                    camera_index=capture_primary.camera_index,
                )
                logger.debug(
                    "Visual cam0: %s (%.3f)",
                    visual_result.species_code,
                    visual_result.confidence,
                )
            except Exception:
                logger.exception("Visual classifier (cam0) failed — skipping this cycle.")

        # ── Step 5: Secondary camera classification ───────────────────────────
        if capture_secondary is not None and self.visual_classifier is not None:
            try:
                visual_result_2 = self.visual_classifier.predict(
                    capture_secondary.frame,
                    camera_index=capture_secondary.camera_index,
                )
                logger.debug(
                    "Visual cam1: %s (%.3f)",
                    visual_result_2.species_code,
                    visual_result_2.confidence,
                )
            except Exception:
                logger.exception("Visual classifier (cam1) failed — skipping cam1 this cycle.")

        # ── Nothing to fuse ───────────────────────────────────────────────────
        if audio_result is None and visual_result is None and visual_result_2 is None:
            logger.debug("Agent cycle — no classifier inputs available.")
            return None

        # ── Step 6: Fuse ──────────────────────────────────────────────────────
        observation = self.fuser.fuse(
            audio_result=audio_result,
            visual_result=visual_result,
            visual_result_2=visual_result_2,
        )

        # ── Step 7: Confidence threshold gate ─────────────────────────────────
        if observation.fused_confidence < self.confidence_threshold:
            logger.debug(
                "Below threshold: %s %.3f < %.3f",
                observation.species_code,
                observation.fused_confidence,
                self.confidence_threshold,
            )
            return None

        # ── Step 8: Cooldown gate ─────────────────────────────────────────────
        if self._is_on_cooldown(observation.species_code):
            logger.debug(
                "Cooldown active for %s — suppressing notification.",
                observation.species_code,
            )
            return None

        # ── Step 9: Populate media paths ──────────────────────────────────────
        # Pydantic models are immutable by default — rebuild with media paths set.
        observation = observation.model_copy(
            update={
                "audio_path": str(audio_path) if audio_path else None,
                "image_path": str(image_path) if image_path else None,
                "image_path_2": str(image_path_2) if image_path_2 else None,
            }
        )

        # ── Step 10: Dispatch ─────────────────────────────────────────────────
        self.notifier.dispatch(observation)
        self._last_dispatch[observation.species_code] = datetime.now(UTC)

        logger.info(
            "Observation dispatched: %s (fused=%.3f) audio=%s cam0=%s cam1=%s",
            observation.species_code,
            observation.fused_confidence,
            "✓" if audio_result else "–",
            "✓" if visual_result else "–",
            "✓" if visual_result_2 else "–",
        )
        return observation

    def _is_on_cooldown(self, species_code: str) -> bool:
        """
        Check whether a species is within the notification cooldown window.

        A species is on cooldown if it was dispatched within the last
        cooldown_seconds seconds. This prevents notification spam when a
        bird lingers at the feeder across multiple agent cycles.

        Args:
            species_code: The species to check.

        Returns:
            True if the species should be suppressed this cycle.
        """
        last = self._last_dispatch.get(species_code)
        if last is None:
            return False

        elapsed = (datetime.now(UTC) - last).total_seconds()
        return elapsed < self.cooldown_seconds
