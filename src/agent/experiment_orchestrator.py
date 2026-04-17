"""
src/agent/experiment_orchestrator.py

The top-level agentic orchestrator for the Avis system.

ExperimentOrchestrator wraps BirdAgent and adds three autonomous behaviours
that satisfy the course's agentic + experimental rubric requirements:

1. Autonomous startup
   On Pi boot (via systemd), ExperimentOrchestrator boots the full system,
   opens cameras, waits for the hardware to warm up, then dispatches a
   "Avis is live" Pushover notification. No human interaction required
   after the power button is pressed.

2. A/B detection mode switching
   Every window_minutes (configurable in hardware.yaml), the orchestrator
   autonomously switches detection_mode between "fixed_crop" and "yolo".
   It tags every BirdObservation.detection_mode before dispatch so the
   observation log records which strategy produced each detection.
   After each window completes it builds an ExperimentWindowReport and
   optionally pushes a brief A/B summary.

3. Daily species summary
   At summary_hour UTC (configurable, default 20:00 = 8pm) the orchestrator
   builds a DailySummaryReport from the day's observations, writes it to
   logs/daily_summaries/YYYY-MM-DD.md + .json, and pushes a condensed
   summary to Pushover. The push message is designed to read well on a
   phone lock screen: species count, top visitor, confidence, A/B delta.

Design principles:
    - Orchestrator WRAPS BirdAgent — it does not extend or modify it.
      BirdAgent._cycle() is called unchanged. The orchestrator adds the
      outer timing loop and post-cycle bookkeeping.
    - BirdAgent remains independently testable with no orchestrator dependency.
    - detection_mode switching works by calling VisionCapture's config at
      runtime — no restart required. The mode toggle is a pure in-memory
      change; hardware.yaml committed defaults are unaffected.
    - All timing is based on wall-clock UTC. No cron, no scheduling library.
    - If the daily summary push fails (network down, Pushover unavailable),
      the .md/.json files are still written and the agent continues running.

Config keys consumed from hardware.yaml (under orchestrator:):
    window_minutes:      A/B window duration in minutes (default: 30)
    ab_modes:            List of modes to cycle through (default: [fixed_crop, yolo])
    summary_hour_utc:    Hour (0-23 UTC) to dispatch daily summary (default: 20)
    startup_delay_seconds: Seconds to wait after camera open before "live" push (default: 10)
    push_window_summaries: Whether to push after each A/B window (default: false)
    daily_summaries_dir: Where to write .md/.json daily reports (default: logs/daily_summaries)

Run via systemd (Pi boot):
    See scripts/avis.service — ExperimentOrchestrator.main() is the entry point.

Run directly (development / demo):
    python -m src.agent.experiment_orchestrator
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import yaml

from src.agent.bird_agent import BirdAgent
from src.notify.report_builder import ReportBuilder
from src.agent.bird_analyst_agent import BirdAnalystAgent  # noqa: F401


logger = logging.getLogger(__name__)

# Sentinel used when a/b mode list isn't configured
_DEFAULT_AB_MODES = ["fixed_crop", "yolo"]


class ExperimentOrchestrator:
    """
    Agentic top-level orchestrator for Avis.

    Wraps BirdAgent and autonomously manages:
        - Boot notification ("Avis is live")
        - Periodic A/B detection mode switching
        - Post-window experiment summaries
        - Daily species summary dispatch

    Usage (normal boot):
        orchestrator = ExperimentOrchestrator.from_config("configs/")
        orchestrator.run()   # blocks indefinitely, handles KeyboardInterrupt

    Usage (test / single-cycle):
        orchestrator = ExperimentOrchestrator(agent, notifier, builder, ...)
        orchestrator._run_cycle()   # one tick, returns immediately
    """

    def __init__(
        self,
        agent: BirdAgent,
        report_builder: ReportBuilder,
        analyst=None,
        window_minutes: float = 30.0,
        ab_modes: list[str] | None = None,
        summary_hour_utc: int = 20,
        startup_delay_seconds: float = 10.0,
        push_window_summaries: bool = False,
        daily_summaries_dir: str | Path = "logs/daily_summaries",
    ) -> None:
        """
        Args:
            agent:                  Fully configured BirdAgent instance.
            report_builder:         ReportBuilder pointed at observations.jsonl.
            window_minutes:         How long each A/B detection mode runs before
                                    switching. Lower = more switching, less data per
                                    window. 30 minutes is a reasonable default for
                                    a busy feeder; increase for sparse bird activity.
            ab_modes:               Ordered list of detection modes to cycle through.
                                    ["fixed_crop", "yolo"] by default. Extend to
                                    ["fixed_crop", "yolo", "fixed_crop"] to spend
                                    more time on the baseline.
            summary_hour_utc:       UTC hour (0-23) when the daily summary is built
                                    and dispatched. Default 20 = 8pm UTC = noon PDT.
            startup_delay_seconds:  Seconds between camera open and "live" push.
                                    Gives the background model time to initialise.
            push_window_summaries:  If True, push a brief A/B window summary to
                                    Pushover after each mode window completes.
                                    Useful during active experimentation; leave False
                                    for day-to-day operation to avoid notification fatigue.
            daily_summaries_dir:    Directory where daily .md/.json reports are written.
        """
        self.agent = agent
        self.report_builder = report_builder
        self.window_minutes = window_minutes
        self.ab_modes = ab_modes if ab_modes else _DEFAULT_AB_MODES
        self.summary_hour_utc = summary_hour_utc
        self.startup_delay_seconds = startup_delay_seconds
        self.push_window_summaries = push_window_summaries
        self.daily_summaries_dir = Path(daily_summaries_dir)
        self.analyst = analyst

        self._running = False

        # A/B state
        self._current_mode_idx: int = 0
        self._window_start: datetime = datetime.now(UTC)

        # Daily summary state — track which date we last summarised
        self._last_summary_date: date | None = None

        logger.info(
            "ExperimentOrchestrator initialized | "
            "window=%.0fmin ab_modes=%s summary_hour=%02d:00 UTC",
            window_minutes,
            self.ab_modes,
            summary_hour_utc,
        )

    @classmethod
    def from_config(cls, config_dir: str | Path) -> ExperimentOrchestrator:
        """
        Construct ExperimentOrchestrator from configs/ directory.

        Reads hardware.yaml for orchestrator config and builds BirdAgent
        and ReportBuilder from the same configs/ directory.

        The orchestrator section in hardware.yaml is entirely optional —
        all keys have sensible defaults so the system works out of the
        box without any config changes.

        Args:
            config_dir: Path to the configs/ directory.

        Returns:
            Fully configured ExperimentOrchestrator.
        """
        config_dir = Path(config_dir)
        hw_path = config_dir / "hardware.yaml"
        paths_path = config_dir / "paths.yaml"

        # Read orchestrator config block (entirely optional in hardware.yaml)
        orchestrator_cfg: dict = {}
        if hw_path.exists():
            with hw_path.open() as f:
                hw = yaml.safe_load(f)
            orchestrator_cfg = hw.get("orchestrator", {})

        window_minutes = float(orchestrator_cfg.get("window_minutes", 30.0))
        ab_modes = orchestrator_cfg.get("ab_modes", _DEFAULT_AB_MODES)
        summary_hour_utc = int(orchestrator_cfg.get("summary_hour_utc", 20))
        startup_delay_seconds = float(orchestrator_cfg.get("startup_delay_seconds", 10.0))
        push_window_summaries = bool(orchestrator_cfg.get("push_window_summaries", False))
        daily_summaries_dir = orchestrator_cfg.get("daily_summaries_dir", "logs/daily_summaries")

        # Build BirdAgent — all its config comes from the same configs/ directory
        agent = BirdAgent.from_config(str(config_dir))

        # Build ReportBuilder — needs path to observations log
        observations_path = "logs/observations.jsonl"
        if paths_path.exists():
            with paths_path.open() as f:
                paths_cfg = yaml.safe_load(f)
            observations_path = paths_cfg.get("logs", {}).get("observations", observations_path)

        report_builder = ReportBuilder(observations_path=observations_path)

        analyst = None
        llm_cfg = hw.get("llm", {}) if hw_path.exists() else {}
        if llm_cfg.get("enabled", True):
            try:
                analyst = BirdAnalystAgent.from_config(str(config_dir))
            except Exception as exc:
                import logging as _log
                _log.getLogger(__name__).warning("BirdAnalystAgent init failed: %s", exc)

        logger.info(
            "ExperimentOrchestrator.from_config | config_dir=%s "
            "window=%.0fmin ab_modes=%s",
            config_dir,
            window_minutes,
            ab_modes,
        )

        return cls(
            agent=agent,
            report_builder=report_builder,
            window_minutes=window_minutes,
            ab_modes=ab_modes,
            summary_hour_utc=summary_hour_utc,
            startup_delay_seconds=startup_delay_seconds,
            push_window_summaries=push_window_summaries,
            daily_summaries_dir=str(daily_summaries_dir),
            analyst = analyst,        


        )

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the orchestrated agent loop.

        Sequence on first call:
            1. Set initial detection mode on VisionCapture
            2. Wait startup_delay_seconds for background model to warm up
            3. Push "Avis is live" notification
            4. Enter main loop (calls _run_cycle() on each BirdAgent tick)

        The loop runs until KeyboardInterrupt or stop() is called.
        Cleans up via BirdAgent (which calls VisionCapture.stop()) on exit.
        """
        self._running = True
        self._boot_time = datetime.now(UTC)
        self._window_start = datetime.now(UTC)

        # ── Boot sequence ─────────────────────────────────────────────────────
        logger.info("ExperimentOrchestrator boot sequence starting.")

        # Apply initial detection mode
        initial_mode = self.ab_modes[self._current_mode_idx]
        self._apply_detection_mode(initial_mode)

        # Give picamera2 and background model a moment to initialise
        logger.info(
            "Waiting %.0fs for hardware warm-up before live notification...",
            self.startup_delay_seconds,
        )
        time.sleep(self.startup_delay_seconds)

        # "Avis is live" push — first human-visible signal after boot
        self._push_startup_notification(initial_mode)

        # ── Main loop ─────────────────────────────────────────────────────────
        logger.info("Entering main orchestration loop.")
        try:
            while self._running:
                self._run_cycle()
                time.sleep(self.agent.loop_interval_seconds)
        except KeyboardInterrupt:
            logger.info("ExperimentOrchestrator interrupted by user.")
        finally:
            self._running = False
            # BirdAgent.run() normally handles cleanup, but since we're
            # calling _cycle() directly we clean up here instead.
            if self.agent.vision_capture is not None:
                self.agent.vision_capture.stop()
            logger.info("ExperimentOrchestrator stopped.")

    def stop(self) -> None:
        """Signal the orchestration loop to stop. Safe to call from another thread."""
        self._running = False
        self.agent.stop()


    # ── Internal loop tick ────────────────────────────────────────────────────

    def _run_cycle(self):
        """
        One orchestration tick.
    
        LLM path (when analyst.llm_available):
            Call analyst.advise() → execute AnalystDecision returned.
            The LLM decides whether to switch modes, push, or generate a report.
    
        Fallback path (when LLM unavailable):
            Original fixed-schedule logic runs unchanged — A/B window timer,
            daily summary at summary_hour_utc, mode rotation.
    
        Either path ends with BirdAgent._cycle() running — the feeder detection
        loop always executes regardless of which decision path ran above it.
        """
        now = datetime.now(UTC)
    
        if self.analyst is not None and self.analyst.llm_available:
            # ── LLM path ──────────────────────────────────────────────────────
            window_elapsed = (now - self._window_start).total_seconds() / 60
            uptime = (now - self._boot_time).total_seconds() if hasattr(self, '_boot_time') else 0.0
    
            decision = self.analyst.advise(
                vision_capture=self.agent.vision_capture,
                notifier=self.agent.notifier,
                current_mode=self.current_detection_mode(),
                uptime_seconds=uptime,
                window_elapsed_minutes=window_elapsed,
                window_total_minutes=self.window_minutes,
            )
    
            if decision is not None:
                # Execute what the agent decided
                if decision.switch_mode and decision.switch_mode != self.current_detection_mode():
                    self._apply_detection_mode(decision.switch_mode)
                    self._window_start = now  # reset window on LLM-driven switch
    
                if decision.generate_report:
                    self._fire_daily_summary(now)
    
                if decision.push_message and decision.push_message != "[pushed by agent]":
                    # Agent set a message but didn't call push_notification directly
                    self._push_text(decision.push_message)
            else:
                # LLM returned None this cycle — run fallback for this tick
                self._fallback_cycle(now)
        else:
            # ── Fallback path — fixed schedule ────────────────────────────────
            self._fallback_cycle(now)
    
        # Agent detection loop always runs regardless of decision path
        self.agent._cycle()
    
    
    def _fallback_cycle(self, now):
        """
        Fixed-schedule fallback logic — runs when LLM is unavailable.
    
        This is the original _run_cycle logic extracted into its own method
        so the LLM path can call it as a fallback without code duplication.
        """
        window_elapsed = (now - self._window_start).total_seconds() / 60
        if window_elapsed >= self.window_minutes and len(self.ab_modes) > 1:
            self._rotate_detection_mode(window_end=now)
    
        if self._should_fire_daily_summary(now):
            self._fire_daily_summary(now)



    # ── A/B mode management ───────────────────────────────────────────────────

    def _rotate_detection_mode(self, window_end: datetime) -> None:
        """
        Switch to the next detection mode in the rotation.

        Builds an ExperimentWindowReport for the window that just ended,
        optionally pushes it, then sets the new mode on VisionCapture.

        Args:
            window_end: The UTC time the window ended (used for the report).
        """
        completed_mode = self.ab_modes[self._current_mode_idx]

        # Build report for the window that just ended
        try:
            report = self.report_builder.build_window_report(
                mode=completed_mode,
                window_start=self._window_start,
                window_end=window_end,
            )
            logger.info(
                "A/B window complete | mode=%s detections=%d species=%d "
                "mean_conf=%.3f duration=%.0fmin",
                completed_mode,
                report.detections,
                report.unique_species,
                report.mean_confidence,
                report.duration_minutes,
            )
            if self.push_window_summaries:
                self._push_text(report.to_push_message())
        except Exception:
            logger.exception("Failed to build window report for mode=%s", completed_mode)

        # Advance to next mode
        self._current_mode_idx = (self._current_mode_idx + 1) % len(self.ab_modes)
        new_mode = self.ab_modes[self._current_mode_idx]
        self._apply_detection_mode(new_mode)
        self._window_start = window_end

    def _apply_detection_mode(self, mode: str) -> None:
        """
        Apply a detection mode to VisionCapture at runtime.

        Changes are in-memory only — hardware.yaml is not modified.
        This means the committed default (fixed_crop) is preserved and
        the Pi always boots into the safe baseline before switching.

        Args:
            mode: "fixed_crop" or "yolo"
        """
        if self.agent.vision_capture is None:
            logger.debug("No VisionCapture available — detection_mode not applied.")
            return

        previous = getattr(self.agent.vision_capture, "detection_mode", "unknown")
        self.agent.vision_capture.detection_mode = mode

        if previous != mode:
            logger.info("Detection mode → %s (was %s)", mode, previous)

    def current_detection_mode(self) -> str:
        """Return the detection mode currently active on VisionCapture."""
        if self.agent.vision_capture is None:
            return self.ab_modes[self._current_mode_idx]
        return getattr(self.agent.vision_capture, "detection_mode", "fixed_crop")

    # ── Daily summary ─────────────────────────────────────────────────────────

    def _should_fire_daily_summary(self, now: datetime) -> bool:
        """
        Return True if the daily summary should fire right now.

        Fires once per day when the UTC hour matches summary_hour_utc
        and we haven't already summarised today.
        """
        if now.hour != self.summary_hour_utc:
            return False
        today = now.date()
        if self._last_summary_date == today:
            return False
        return True

    def _fire_daily_summary(self, now: datetime) -> None:
        today = now.date()
        logger.info("Firing daily summary for %s", today.isoformat())

        try:
            report = self.report_builder.build_daily_summary(for_date=today)

            md_path, json_path = self.report_builder.write_daily_summary(
                report=report,
                output_dir=self.daily_summaries_dir,
            )
            logger.info("Daily summary written → %s, %s", md_path, json_path)

            # Set the date BEFORE pushing — push failure must not block this
            self._last_summary_date = today

            # Push is best-effort — failure is logged but never prevents the date being recorded
            try:
                self._push_text(report.to_push_message())
            except Exception:
                logger.warning("Daily summary push failed — files already written.")

        except Exception:
            logger.exception("Daily summary failed for %s", today.isoformat())

    # ── Notifications ─────────────────────────────────────────────────────────

    def _push_startup_notification(self, initial_mode: str) -> None:
        """
        Push the "Avis is live" boot notification.

        Includes the initial detection mode and species count so the
        user knows the system is ready and how it's configured.
        """
        now_str = datetime.now(UTC).strftime("%H:%M UTC")
        message = (
            f"🐦 Avis is live — {now_str}\n"
            f"Mode: {initial_mode} | "
            f"A/B window: {self.window_minutes:.0f}min | "
            f"Daily summary: {self.summary_hour_utc:02d}:00 UTC"
        )
        logger.info("Sending startup notification: %s", message)
        self._push_text(message)

    def _push_text(self, message: str) -> None:
        """
        Push a plain text message via the agent's Notifier.

        Uses Notifier._push() directly — this bypasses the observation
        dispatch path (which formats species detections) so the message
        text is sent verbatim as the notification body.

        Fails silently — orchestrator push failures are logged but never
        crash the agent loop.
        """
        try:
            notifier = self.agent.notifier
            # Only attempt push if the push channel is enabled
            if not getattr(notifier, "push_enabled", False):
                logger.debug("Push channel disabled — skipping orchestrator notification.")
                return
            notifier._push_text(message)
        except AttributeError:
            # _push_text may not exist if Notifier hasn't implemented it yet —
            # log and continue rather than crashing the loop.
            logger.debug("Notifier._push_text not available — skipping push.")
        except Exception:
            logger.exception("Orchestrator push notification failed.")


# ── Module entry point ────────────────────────────────────────────────────────


def main() -> None:
    """
    Entry point for Pi boot via systemd.

    Configures logging to stdout (systemd captures this to journald),
    builds the orchestrator from configs/, and starts the main loop.

    Called by scripts/avis.service:
        ExecStart = python -m src.agent.experiment_orchestrator
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Avis ExperimentOrchestrator starting from systemd.")

    orchestrator = ExperimentOrchestrator.from_config("configs")
    orchestrator.run()


if __name__ == "__main__":
    main()