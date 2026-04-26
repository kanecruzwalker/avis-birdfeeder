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
from src.data.schema import (
    GATE_REASON_BELOW_CONFIDENCE_THRESHOLD,
    GATE_REASON_NO_BIRD_DETECTED,
    GATE_REASON_SPECIES_COOLDOWN,
    BirdObservation,
)
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

        Phase 8 cycle steps (Branch 2 added the bird-presence gate):
            1. Capture audio window → energy gate → save WAV → get file path
            2. Capture frames from both cameras → motion gate → bird-presence
               gate → save PNGs. Each CaptureResult carries gate_passed +
               gate_reason populated by VisionCapture.
            3. Classify audio (BirdNET on WAV file path) if captured.
            4. Classify primary camera frame if capture present AND gate_passed.
            5. Classify secondary camera frame if capture present AND gate_passed.
            6. If both cameras gated out AND no audio → log gate-suppressed
               observation (gate_reason="no_bird_detected") and exit. This is
               the Phase 8 fix — previously these frames were silently classified
               as the nearest-species-in-feature-space, producing noise dispatches.
            7. Fuse all available results.
            8. Populate media paths, then confidence threshold gate
               (suppressed observations logged with
               gate_reason="below_confidence_threshold").
            9. Cooldown gate (suppressed observations logged with
               gate_reason="species_cooldown").
           10. Dispatch via notifier (only if both gates pass).

        PR #51 introduced notifier.log_suppressed() so below-threshold and
        cooldown-suppressed observations are preserved in observations.jsonl.
        Branch 2 extends this with the gate_reason field, so each suppressed
        record now carries the specific reason for suppression (enabling the
        pre/post-gate ablation analysis in the Phase 8 report).

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

        # ── Step 4: Primary camera classification (gated) ─────────────────────
        # Only classify if the capture exists AND its bird-presence gate passed.
        # Getattr is used for backward-compat with CaptureResult objects from
        # pre-Branch-2 code paths (or tests) that don't set gate_passed.
        # Default True means: without a gate, behave as before — always classify.
        primary_gate_passed = capture_primary is not None and getattr(
            capture_primary, "gate_passed", True
        )
        if primary_gate_passed and self.visual_classifier is not None:
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

        # ── Step 5: Secondary camera classification (gated) ───────────────────
        secondary_gate_passed = capture_secondary is not None and getattr(
            capture_secondary, "gate_passed", True
        )
        if secondary_gate_passed and self.visual_classifier is not None:
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

        # ── Step 6: Gate-only suppression path (Phase 8 fix) ─────────────────
        # If both cameras produced captures but BOTH had their bird-presence
        # gates block the frame, AND we got no audio detection either, the
        # whole cycle is a "motion but no bird" event. Rather than silently
        # returning None (losing the record) or running the classifier on
        # empty scene (producing noise), we log a gate-suppressed observation
        # with the appropriate reason. This is the core Phase 8 fix.
        any_capture = capture_primary is not None or capture_secondary is not None
        any_gate_passed = primary_gate_passed or secondary_gate_passed
        if any_capture and not any_gate_passed and audio_result is None:
            # Prefer the primary camera's gate_reason; fall back to secondary.
            gate_reason = (
                getattr(capture_primary, "gate_reason", None)
                or getattr(capture_secondary, "gate_reason", None)
                or GATE_REASON_NO_BIRD_DETECTED
            )
            self._log_gate_suppressed(
                gate_reason=gate_reason,
                image_path=image_path,
                image_path_2=image_path_2,
                audio_path=audio_path,
            )
            return None

        # ── Nothing to fuse ───────────────────────────────────────────────────
        if audio_result is None and visual_result is None and visual_result_2 is None:
            logger.debug("Agent cycle — no classifier inputs available.")
            return None

        # ── Step 7: Fuse ──────────────────────────────────────────────────────
        observation = self.fuser.fuse(
            audio_result=audio_result,
            visual_result=visual_result,
            visual_result_2=visual_result_2,
        )

        # ── Step 8: Confidence threshold gate ─────────────────────────────────
        # Populate media paths BEFORE gate checks so suppressed observations
        # retain image/audio references for analysis.
        # Propagate detection_mode from whichever camera capture produced the result.
        # CaptureResult.detection_mode reflects the actual crop strategy used
        # (fixed_crop or yolo). Without this propagation, BirdObservation always
        # records "fixed_crop" regardless of which mode ran — blocks A/B analysis.
        detection_mode = "fixed_crop"  # safe default
        if capture_primary is not None and getattr(capture_primary, "detection_mode", None):
            detection_mode = capture_primary.detection_mode
        elif capture_secondary is not None and getattr(capture_secondary, "detection_mode", None):
            detection_mode = capture_secondary.detection_mode

        observation = observation.model_copy(
            update={
                "audio_path": str(audio_path) if audio_path else None,
                "image_path": str(image_path) if image_path else None,
                "image_path_2": str(image_path_2) if image_path_2 else None,
                "detection_mode": detection_mode,
            }
        )

        if observation.fused_confidence < self.confidence_threshold:
            observation = observation.model_copy(
                update={"gate_reason": GATE_REASON_BELOW_CONFIDENCE_THRESHOLD}
            )
            logger.debug(
                "Below threshold: %s %.3f < %.3f — logging as suppressed.",
                observation.species_code,
                observation.fused_confidence,
                self.confidence_threshold,
            )
            self.notifier.log_suppressed(observation)
            return None

        # ── Step 9: Cooldown gate ─────────────────────────────────────────────
        if self._is_on_cooldown(observation.species_code):
            observation = observation.model_copy(
                update={"gate_reason": GATE_REASON_SPECIES_COOLDOWN}
            )
            logger.debug(
                "Cooldown active for %s — logging as suppressed.",
                observation.species_code,
            )
            self.notifier.log_suppressed(observation)
            return None

        # ── Step 10: Dispatch ────────────────────────────────────────────────
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

    def _log_gate_suppressed(
        self,
        gate_reason: str,
        image_path: Path | str | None,
        image_path_2: Path | str | None,
        audio_path: Path | str | None,
    ) -> None:
        """
        Log a gate-suppressed observation — motion fired but the bird-presence
        gate rejected the frame AND no audio detection filled the gap.

        Without this method, motion-triggered empty-feeder frames would either
        be silently skipped (losing the data for ablation analysis) or
        dispatched as whatever the classifier confidently-but-wrongly picked
        from scene texture. This was the root Phase 8 failure mode documented
        in docs/investigations/hailo-2026-04-22.md.

        We synthesize a minimal BirdObservation with species_code="NONE",
        fused_confidence=0.0, and gate_reason populated. The record preserves
        the image/audio paths so researchers reviewing the log can correlate
        gate-suppressed entries with the saved frames that produced them.

        The sentinel species_code="NONE" is deliberate. Schema-wise, the
        validator requires a non-empty uppercase code; "NONE" satisfies that
        while being unambiguous as a not-a-species marker. Consumers filtering
        observations.jsonl for real detections should filter on either
        gate_reason is None or species_code != "NONE".

        Args:
            gate_reason:  Why the observation was suppressed. Typically
                          GATE_REASON_NO_BIRD_DETECTED from schema constants.
            image_path:   Primary camera image path if saved.
            image_path_2: Secondary camera image path if saved.
            audio_path:   Audio capture path if any (usually None on
                          gate-suppressed cycles).
        """
        observation = BirdObservation(
            species_code="NONE",
            common_name="(no bird detected)",
            scientific_name="",
            fused_confidence=0.0,
            dispatched=False,
            gate_reason=gate_reason,
            audio_path=str(audio_path) if audio_path else None,
            image_path=str(image_path) if image_path else None,
            image_path_2=str(image_path_2) if image_path_2 else None,
        )
        self.notifier.log_suppressed(observation)
        logger.debug(
            "Gate-suppressed cycle logged | reason=%s image=%s audio=%s",
            gate_reason,
            image_path,
            audio_path,
        )

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
