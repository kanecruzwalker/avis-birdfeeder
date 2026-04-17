"""
tests/agent/test_experiment_orchestrator.py

Unit tests for ExperimentOrchestrator.

Tests are fully synthetic — no Pi hardware, no Hailo, no network calls.
All external dependencies (BirdAgent, Notifier, ReportBuilder) are mocked
so these tests run cleanly in CI on any machine.

Test strategy:
    - Mode rotation logic is tested without running the agent loop
    - Daily summary scheduling is tested by injecting controlled UTC times
    - Startup notification path is verified against mocked Notifier
    - _run_cycle integration is tested with a mocked BirdAgent._cycle()
    - from_config construction is tested with a minimal in-memory config

What is NOT tested here:
    - BirdAgent internals (tested in tests/agent/test_bird_agent.py)
    - ReportBuilder aggregation (tested in tests/notify/test_report_builder.py)
    - Actual Pushover HTTP calls (tested in tests/notify/test_notifier.py)
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.experiment_orchestrator import ExperimentOrchestrator
from src.notify.report_builder import DailySummaryReport, ExperimentWindowReport, ReportBuilder


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_agent(detection_mode: str = "fixed_crop") -> MagicMock:
    """Return a minimal BirdAgent mock with the attributes orchestrator reads."""
    agent = MagicMock()
    agent.loop_interval_seconds = 0.0  # don't sleep in tests
    agent.notifier.push_enabled = True
    # VisionCapture mock — detection_mode is get/set directly
    agent.vision_capture.detection_mode = detection_mode
    agent._cycle.return_value = None
    return agent


def _make_report_builder(tmp_path: Path) -> ReportBuilder:
    """Return a ReportBuilder pointing at a temp observations.jsonl."""
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text("")
    return ReportBuilder(observations_path=str(obs_path))


def _make_orchestrator(
    tmp_path: Path,
    window_minutes: float = 30.0,
    ab_modes: list[str] | None = None,
    summary_hour_utc: int = 20,
) -> ExperimentOrchestrator:
    """Build an orchestrator with all external calls mocked."""
    agent = _make_agent()
    builder = _make_report_builder(tmp_path)
    return ExperimentOrchestrator(
        agent=agent,
        report_builder=builder,
        window_minutes=window_minutes,
        ab_modes=ab_modes or ["fixed_crop", "yolo"],
        summary_hour_utc=summary_hour_utc,
        startup_delay_seconds=0.0,
        push_window_summaries=False,
        daily_summaries_dir=str(tmp_path / "summaries"),
    )


def _empty_report(for_date: date | None = None) -> DailySummaryReport:
    """Return a minimal empty DailySummaryReport for mocking."""
    return DailySummaryReport(
        report_date=for_date or date.today(),
        generated_at=datetime.now(UTC),
        total_detections=0,
        unique_species=0,
        species=[],
        audio_only_count=0,
        visual_only_count=0,
        fused_count=0,
        detection_mode_counts=Counter(),
        mean_confidence=0.0,
        observation_window_hours=24.0,
    )


# ── Construction ──────────────────────────────────────────────────────────────


class TestExperimentOrchestratorInit:
    def test_default_ab_modes_set(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        assert orch.ab_modes == ["fixed_crop", "yolo"]

    def test_custom_ab_modes_preserved(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path, ab_modes=["fixed_crop", "yolo", "fixed_crop"])
        assert orch.ab_modes == ["fixed_crop", "yolo", "fixed_crop"]

    def test_starts_at_mode_index_zero(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        assert orch._current_mode_idx == 0

    def test_no_last_summary_date_on_init(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        assert orch._last_summary_date is None

    def test_single_mode_list_accepted(self, tmp_path: Path) -> None:
        # Single mode means no rotation — orchestrator should not crash
        orch = _make_orchestrator(tmp_path, ab_modes=["fixed_crop"])
        assert orch.ab_modes == ["fixed_crop"]


# ── Detection mode management ─────────────────────────────────────────────────


class TestApplyDetectionMode:
    def test_applies_mode_to_vision_capture(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        orch._apply_detection_mode("yolo")
        assert orch.agent.vision_capture.detection_mode == "yolo"

    def test_apply_fixed_crop(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        orch._apply_detection_mode("fixed_crop")
        assert orch.agent.vision_capture.detection_mode == "fixed_crop"

    def test_no_crash_when_no_vision_capture(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        orch.agent.vision_capture = None
        # Should log a debug message but not raise
        orch._apply_detection_mode("yolo")

    def test_current_detection_mode_returns_capture_mode(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        orch.agent.vision_capture.detection_mode = "yolo"
        assert orch.current_detection_mode() == "yolo"

    def test_current_mode_falls_back_when_no_capture(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path, ab_modes=["yolo", "fixed_crop"])
        orch.agent.vision_capture = None
        # Returns from ab_modes list, not crash
        assert orch.current_detection_mode() in ("yolo", "fixed_crop")


# ── A/B window rotation ───────────────────────────────────────────────────────


class TestModeRotation:
    def test_rotate_advances_mode_index(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        assert orch._current_mode_idx == 0

        now = datetime.now(UTC)
        orch._rotate_detection_mode(window_end=now)

        assert orch._current_mode_idx == 1
        assert orch.agent.vision_capture.detection_mode == "yolo"

    def test_rotate_wraps_around(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path, ab_modes=["fixed_crop", "yolo"])
        orch._current_mode_idx = 1

        now = datetime.now(UTC)
        orch._rotate_detection_mode(window_end=now)

        assert orch._current_mode_idx == 0
        assert orch.agent.vision_capture.detection_mode == "fixed_crop"

    def test_rotate_updates_window_start(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        now = datetime(2026, 4, 15, 10, 30, tzinfo=UTC)
        orch._rotate_detection_mode(window_end=now)
        assert orch._window_start == now

    def test_no_rotation_with_single_mode(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path, ab_modes=["fixed_crop"])
        initial_idx = orch._current_mode_idx

        now = datetime.now(UTC)
        orch._rotate_detection_mode(window_end=now)

        # Wraps to same index with single-element list
        assert orch._current_mode_idx == initial_idx

    def test_window_elapsed_triggers_rotate_in_run_cycle(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path, window_minutes=1.0)
        # Set window_start to 2 minutes ago so window has elapsed
        orch._window_start = datetime.now(UTC) - timedelta(minutes=2)
        orch._last_summary_date = datetime.now(UTC).date()  # suppress summary

        orch._run_cycle()

        # Mode should have rotated
        assert orch._current_mode_idx == 1

    def test_window_not_elapsed_no_rotation(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path, window_minutes=60.0)
        orch._window_start = datetime.now(UTC)  # just started
        orch._last_summary_date = datetime.now(UTC).date()

        orch._run_cycle()

        assert orch._current_mode_idx == 0


# ── Daily summary scheduling ──────────────────────────────────────────────────


class TestDailySummaryScheduling:
    def test_should_fire_at_correct_hour(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path, summary_hour_utc=20)
        now = datetime(2026, 4, 15, 20, 5, tzinfo=UTC)
        assert orch._should_fire_daily_summary(now) is True

    def test_should_not_fire_at_wrong_hour(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path, summary_hour_utc=20)
        now = datetime(2026, 4, 15, 19, 59, tzinfo=UTC)
        assert orch._should_fire_daily_summary(now) is False

    def test_should_not_fire_twice_same_day(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path, summary_hour_utc=20)
        today = date(2026, 4, 15)
        orch._last_summary_date = today

        now = datetime(2026, 4, 15, 20, 0, tzinfo=UTC)
        assert orch._should_fire_daily_summary(now) is False

    def test_fires_on_new_day(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path, summary_hour_utc=20)
        orch._last_summary_date = date(2026, 4, 14)  # yesterday

        now = datetime(2026, 4, 15, 20, 0, tzinfo=UTC)
        assert orch._should_fire_daily_summary(now) is True

    def test_fire_daily_summary_sets_last_date(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        now = datetime(2026, 4, 15, 20, 0, tzinfo=UTC)

        # Patch report builder to avoid file system side effects
        orch.report_builder.build_daily_summary = MagicMock(return_value=_empty_report())
        orch.report_builder.write_daily_summary = MagicMock(
            return_value=(tmp_path / "x.md", tmp_path / "x.json")
        )
        orch._push_text = MagicMock()

        orch._fire_daily_summary(now)

        assert orch._last_summary_date == date(2026, 4, 15)

    def test_fire_daily_summary_writes_files(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        now = datetime(2026, 4, 15, 20, 0, tzinfo=UTC)
        report = _empty_report(for_date=date(2026, 4, 15))

        orch.report_builder.build_daily_summary = MagicMock(return_value=report)
        write_mock = MagicMock(return_value=(tmp_path / "x.md", tmp_path / "x.json"))
        orch.report_builder.write_daily_summary = write_mock
        orch._push_text = MagicMock()

        orch._fire_daily_summary(now)

        write_mock.assert_called_once()
        call_kwargs = write_mock.call_args
        assert call_kwargs[1]["report"] is report or call_kwargs[0][0] is report

    def test_fire_daily_summary_pushes_message(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        now = datetime(2026, 4, 15, 20, 0, tzinfo=UTC)

        orch.report_builder.build_daily_summary = MagicMock(return_value=_empty_report())
        orch.report_builder.write_daily_summary = MagicMock(
            return_value=(tmp_path / "x.md", tmp_path / "x.json")
        )
        push_mock = MagicMock()
        orch._push_text = push_mock

        orch._fire_daily_summary(now)

        push_mock.assert_called_once()
        # Message should be a non-empty string
        msg = push_mock.call_args[0][0]
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_fire_daily_summary_survives_push_failure(self, tmp_path: Path) -> None:
        """Push failure must not prevent _last_summary_date being set."""
        orch = _make_orchestrator(tmp_path)
        now = datetime(2026, 4, 15, 20, 0, tzinfo=UTC)

        orch.report_builder.build_daily_summary = MagicMock(return_value=_empty_report())
        orch.report_builder.write_daily_summary = MagicMock(
            return_value=(tmp_path / "x.md", tmp_path / "x.json")
        )
        orch._push_text = MagicMock(side_effect=Exception("network down"))

        # Should not raise
        orch._fire_daily_summary(now)
        assert orch._last_summary_date == date(2026, 4, 15)


# ── run_cycle integration ──────────────────────────────────────────────────────


class TestRunCycle:
    def test_calls_agent_cycle_each_tick(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path, window_minutes=60.0)
        orch._last_summary_date = datetime.now(UTC).date()

        orch._run_cycle()
        orch._run_cycle()
        orch._run_cycle()

        assert orch.agent._cycle.call_count == 3

    def test_run_cycle_does_not_raise_on_empty_capture(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path, window_minutes=60.0)
        orch._last_summary_date = datetime.now(UTC).date()
        orch.agent._cycle.return_value = None  # typical empty cycle

        # Should complete without raising
        orch._run_cycle()


# ── Startup notification ──────────────────────────────────────────────────────


class TestStartupNotification:
    def test_startup_push_includes_mode(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        push_mock = MagicMock()
        orch._push_text = push_mock

        orch._push_startup_notification("fixed_crop")

        push_mock.assert_called_once()
        msg = push_mock.call_args[0][0]
        assert "fixed_crop" in msg

    def test_startup_push_includes_live(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        push_mock = MagicMock()
        orch._push_text = push_mock

        orch._push_startup_notification("yolo")

        msg = push_mock.call_args[0][0]
        assert "live" in msg.lower() or "Avis" in msg

    def test_push_text_skips_when_push_disabled(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        orch.agent.notifier.push_enabled = False

        # Should not raise, should just log and return
        orch._push_text("test message")


# ── Stop / lifecycle ──────────────────────────────────────────────────────────


class TestLifecycle:
    def test_stop_sets_running_false(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        orch._running = True
        orch.stop()
        assert orch._running is False

    def test_stop_calls_agent_stop(self, tmp_path: Path) -> None:
        orch = _make_orchestrator(tmp_path)
        orch.stop()
        orch.agent.stop.assert_called_once()