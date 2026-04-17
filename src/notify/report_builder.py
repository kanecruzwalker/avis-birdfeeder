"""
src/notify/report_builder.py

Builds structured daily and A/B experiment summary reports from the
observations log (logs/observations.jsonl).

Why a separate module from notifier.py?
    Notifier's job is dispatch — it takes a single observation and routes it
    to configured channels. ReportBuilder's job is aggregation — it reads many
    observations and synthesises them into a human + machine readable report.
    Keeping them separate means Notifier stays simple and testable, and
    ReportBuilder can be used independently (notebooks, scripts, orchestrator).

Report types:
    DailySummaryReport  — one per day. Total detections, unique species,
                          top species, per-species counts, A/B breakdown
                          if ExperimentOrchestrator was running.
    ExperimentWindowReport — one per A/B window. Compares fixed_crop vs yolo
                             confidence distributions and detection counts for
                             the window duration just completed.

Output formats:
    .md file  — human readable, committed to logs/daily_summaries/
    .json file — machine readable, same directory, same stem
    Pushover  — condensed single-message summary pushed via Notifier._push()

The Pushover message is intentionally brief — one or two lines that read well
on a phone lock screen. Full detail lives in the .md/.json files.

As the push service grows to support richer payloads (image, audio, chart
attachment), ReportBuilder is where those attachments are assembled before
being handed to Notifier for dispatch.

Config keys consumed:
    paths.yaml: logs.observations, logs.agent
    notify.yaml: channels.push (report push respects same toggle as detections)
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import mean, stdev
from typing import Any

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class SpeciesSummary:
    """Aggregated stats for one species within a report window."""

    code: str
    common_name: str
    count: int
    mean_confidence: float
    max_confidence: float
    first_seen: datetime
    last_seen: datetime
    detection_modes: Counter = field(default_factory=Counter)

    @property
    def mode_breakdown(self) -> str:
        """Human-readable mode breakdown, e.g. 'fixed_crop×8 yolo×4'."""
        parts = [f"{mode}×{n}" for mode, n in sorted(self.detection_modes.items())]
        return " ".join(parts) if parts else "unknown"


@dataclass
class DailySummaryReport:
    """
    All observations for a single calendar day (local Pi time).

    Produced by ReportBuilder.build_daily_summary().
    Consumed by ExperimentOrchestrator for end-of-day dispatch.
    """

    report_date: date
    generated_at: datetime
    total_detections: int
    unique_species: int
    species: list[SpeciesSummary]  # sorted by count descending
    audio_only_count: int
    visual_only_count: int
    fused_count: int
    detection_mode_counts: Counter  # {"fixed_crop": N, "yolo": M}
    mean_confidence: float
    observation_window_hours: float  # how many hours of data this covers

    @property
    def top_species(self) -> SpeciesSummary | None:
        """Species with the highest detection count today."""
        return self.species[0] if self.species else None

    def to_markdown(self) -> str:
        """
        Render the report as a Markdown document.

        Suitable for writing to logs/daily_summaries/YYYY-MM-DD.md.
        Written in a style that works as both a dev log entry and a
        course deliverable appendix.
        """
        lines: list[str] = []
        d = self.report_date

        lines.append(f"# Avis Daily Summary — {d.isoformat()}")
        lines.append(f"*Generated {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}*")
        lines.append("")

        lines.append("## Overview")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total detections | {self.total_detections} |")
        lines.append(f"| Unique species | {self.unique_species} |")
        lines.append(f"| Mean confidence | {self.mean_confidence:.1%} |")
        lines.append(f"| Observation window | {self.observation_window_hours:.1f} hours |")
        lines.append(f"| Audio-only | {self.audio_only_count} |")
        lines.append(f"| Visual-only | {self.visual_only_count} |")
        lines.append(f"| Fused (audio + visual) | {self.fused_count} |")
        lines.append("")

        if self.detection_mode_counts:
            lines.append("## Detection Mode Breakdown")
            for mode, count in sorted(self.detection_mode_counts.items()):
                pct = count / self.total_detections * 100 if self.total_detections else 0
                lines.append(f"- **{mode}**: {count} detections ({pct:.0f}%)")
            lines.append("")

        if self.species:
            lines.append("## Species Detected")
            lines.append(
                "| Species | Count | Mean Conf | Max Conf | First Seen | Mode Breakdown |"
            )
            lines.append("|---------|-------|-----------|----------|------------|----------------|")
            for s in self.species:
                first = s.first_seen.strftime("%H:%M")
                lines.append(
                    f"| {s.common_name} ({s.code}) "
                    f"| {s.count} "
                    f"| {s.mean_confidence:.1%} "
                    f"| {s.max_confidence:.1%} "
                    f"| {first} "
                    f"| {s.mode_breakdown} |"
                )
            lines.append("")

        if not self.species:
            lines.append("*No detections today.*")
            lines.append("")

        lines.append("---")
        lines.append("*Avis Agentic Birdfeeder — CS 450 Spring 2026*")

        return "\n".join(lines)

    def to_push_message(self) -> str:
        """
        Condensed one-to-two line summary suitable for a Pushover notification.

        Designed to be readable on a phone lock screen without unlocking.
        """
        if not self.total_detections:
            return f"🐦 Avis daily summary ({self.report_date}): no detections today."

        top = self.top_species
        top_str = f"{top.common_name} ×{top.count}" if top else "—"
        mode_str = ""
        if len(self.detection_mode_counts) > 1:
            fc = self.detection_mode_counts.get("fixed_crop", 0)
            yo = self.detection_mode_counts.get("yolo", 0)
            mode_str = f" | crop:{fc} yolo:{yo}"

        return (
            f"🐦 {self.report_date}: {self.total_detections} detections, "
            f"{self.unique_species} species. "
            f"Top: {top_str}. "
            f"Avg conf: {self.mean_confidence:.0%}{mode_str}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict for machine-readable output."""
        return {
            "report_date": self.report_date.isoformat(),
            "generated_at": self.generated_at.isoformat(),
            "total_detections": self.total_detections,
            "unique_species": self.unique_species,
            "mean_confidence": round(self.mean_confidence, 4),
            "observation_window_hours": round(self.observation_window_hours, 2),
            "audio_only_count": self.audio_only_count,
            "visual_only_count": self.visual_only_count,
            "fused_count": self.fused_count,
            "detection_mode_counts": dict(self.detection_mode_counts),
            "species": [
                {
                    "code": s.code,
                    "common_name": s.common_name,
                    "count": s.count,
                    "mean_confidence": round(s.mean_confidence, 4),
                    "max_confidence": round(s.max_confidence, 4),
                    "first_seen": s.first_seen.isoformat(),
                    "last_seen": s.last_seen.isoformat(),
                    "detection_modes": dict(s.detection_modes),
                }
                for s in self.species
            ],
        }


@dataclass
class ExperimentWindowReport:
    """
    Comparison of fixed_crop vs yolo for a single A/B window.

    A window is the period between two detection_mode switches by
    ExperimentOrchestrator. The report captures what each mode achieved
    during its slice of that window.

    Produced by ReportBuilder.build_window_report().
    """

    window_start: datetime
    window_end: datetime
    mode: str  # which mode was active for this window
    detections: int
    unique_species: int
    mean_confidence: float
    confidences: list[float]  # raw list for distribution analysis

    @property
    def duration_minutes(self) -> float:
        return (self.window_end - self.window_start).total_seconds() / 60

    @property
    def detections_per_hour(self) -> float:
        hours = self.duration_minutes / 60
        return self.detections / hours if hours > 0 else 0.0

    @property
    def confidence_std(self) -> float:
        return stdev(self.confidences) if len(self.confidences) > 1 else 0.0

    def to_push_message(self) -> str:
        """Single-line window summary for Pushover."""
        return (
            f"🔬 A/B window ({self.mode}): "
            f"{self.detections} detections, "
            f"{self.unique_species} species, "
            f"avg conf {self.mean_confidence:.0%} "
            f"over {self.duration_minutes:.0f} min"
        )


# ── Builder ───────────────────────────────────────────────────────────────────


class ReportBuilder:
    """
    Reads observations.jsonl and builds structured summary reports.

    Usage:
        builder = ReportBuilder(observations_path="logs/observations.jsonl")
        report  = builder.build_daily_summary(for_date=date.today())
        md_text = report.to_markdown()
        push_msg = report.to_push_message()

    The builder is stateless — each call reads the log file fresh.
    This is intentional: the log file is the source of truth and may be
    written by a separate process (agent loop) while the builder runs.

    For high-frequency reads on Pi (e.g. every 30 minutes), reading only
    the last N lines is sufficient and avoids loading the full log. The
    builder uses a lightweight line-scan approach rather than loading all
    observations into memory.
    """

    def __init__(self, observations_path: str | Path) -> None:
        """
        Args:
            observations_path: Path to logs/observations.jsonl.
                               Does not need to exist yet — build methods
                               return empty reports if the file is absent.
        """
        self.observations_path = Path(observations_path)

    def _read_observations(
        self,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Read and parse observations from the JSONL log.

        Args:
            since: Only include observations at or after this UTC timestamp.
            until: Only include observations before this UTC timestamp.

        Returns:
            List of parsed observation dicts within the time window.
            Malformed lines are skipped with a warning.
        """
        if not self.observations_path.exists():
            logger.debug("Observations log not found at %s — returning empty.", self.observations_path)
            return []

        results = []
        with self.observations_path.open() as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obs = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON on line %d", line_num)
                    continue

                # Parse timestamp — handle both naive and aware ISO strings
                raw_ts = obs.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(raw_ts)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                except (ValueError, TypeError):
                    logger.warning("Skipping observation with unparseable timestamp: %r", raw_ts)
                    continue

                if since is not None and ts < since:
                    continue
                if until is not None and ts >= until:
                    continue

                obs["_parsed_timestamp"] = ts
                results.append(obs)

        return results

    def build_daily_summary(self, for_date: date | None = None) -> DailySummaryReport:
        """
        Build a DailySummaryReport for a given calendar day.

        Observations are bucketed by UTC date. For a Pi running in a
        single timezone, UTC is consistent — the report date matches
        what the agent logged.

        Args:
            for_date: The calendar date to summarise. Defaults to today (UTC).

        Returns:
            DailySummaryReport. If no observations exist for the date,
            returns an empty report with zero counts.
        """
        if for_date is None:
            for_date = datetime.now(UTC).date()

        day_start = datetime(for_date.year, for_date.month, for_date.day, tzinfo=UTC)
        day_end = day_start + timedelta(days=1)

        observations = self._read_observations(since=day_start, until=day_end)
        return self._aggregate(observations, for_date, day_start, day_end)

    def build_recent_summary(self, hours: float = 24.0) -> DailySummaryReport:
        """
        Build a summary report for the last N hours of observations.

        Used by ExperimentOrchestrator for periodic status pushes even
        when a full calendar day hasn't elapsed (e.g. the Pi was just
        powered on and has 2 hours of data).

        Args:
            hours: How many hours back to look.

        Returns:
            DailySummaryReport covering the window. report_date is today.
        """
        until = datetime.now(UTC)
        since = until - timedelta(hours=hours)
        observations = self._read_observations(since=since, until=until)
        return self._aggregate(observations, until.date(), since, until)

    def build_window_report(
        self,
        mode: str,
        window_start: datetime,
        window_end: datetime,
    ) -> ExperimentWindowReport:
        """
        Build an ExperimentWindowReport for a single A/B detection mode window.

        Args:
            mode:         The detection_mode that was active ("fixed_crop"|"yolo").
            window_start: UTC start of the window.
            window_end:   UTC end of the window.

        Returns:
            ExperimentWindowReport with detection stats for this window.
        """
        observations = self._read_observations(since=window_start, until=window_end)
        # Filter to only observations matching this mode
        mode_obs = [o for o in observations if o.get("detection_mode", "fixed_crop") == mode]

        confidences = [float(o.get("fused_confidence", 0.0)) for o in mode_obs]
        species_codes = {o.get("species_code", "") for o in mode_obs}

        return ExperimentWindowReport(
            window_start=window_start,
            window_end=window_end,
            mode=mode,
            detections=len(mode_obs),
            unique_species=len(species_codes),
            mean_confidence=mean(confidences) if confidences else 0.0,
            confidences=confidences,
        )

    def _aggregate(
        self,
        observations: list[dict[str, Any]],
        report_date: date,
        window_start: datetime,
        window_end: datetime,
    ) -> DailySummaryReport:
        """
        Internal aggregation: turns a list of observation dicts into a report.

        Grouped by species_code. Confidence scores, timestamps, and detection
        mode counts are all accumulated per-species and globally.
        """
        # Per-species buckets
        species_counts: Counter = Counter()
        species_confidences: dict[str, list[float]] = defaultdict(list)
        species_first_seen: dict[str, datetime] = {}
        species_last_seen: dict[str, datetime] = {}
        species_names: dict[str, str] = {}
        species_modes: dict[str, Counter] = defaultdict(Counter)

        audio_only = visual_only = fused = 0
        all_confidences: list[float] = []
        mode_counts: Counter = Counter()

        for obs in observations:
            code = obs.get("species_code", "UNKNOWN").upper()
            conf = float(obs.get("fused_confidence", 0.0))
            ts = obs["_parsed_timestamp"]
            mode = obs.get("detection_mode", "fixed_crop")

            species_counts[code] += 1
            species_confidences[code].append(conf)
            species_names[code] = obs.get("common_name", code)
            species_modes[code][mode] += 1
            all_confidences.append(conf)
            mode_counts[mode] += 1

            if code not in species_first_seen or ts < species_first_seen[code]:
                species_first_seen[code] = ts
            if code not in species_last_seen or ts > species_last_seen[code]:
                species_last_seen[code] = ts

            # Modality accounting
            has_audio = obs.get("audio_result") is not None
            has_visual = obs.get("visual_result") is not None
            if has_audio and has_visual:
                fused += 1
            elif has_audio:
                audio_only += 1
            else:
                visual_only += 1

        # Build per-species summaries, sorted by count descending
        species_summaries = sorted(
            [
                SpeciesSummary(
                    code=code,
                    common_name=species_names[code],
                    count=count,
                    mean_confidence=mean(species_confidences[code]),
                    max_confidence=max(species_confidences[code]),
                    first_seen=species_first_seen[code],
                    last_seen=species_last_seen[code],
                    detection_modes=species_modes[code],
                )
                for code, count in species_counts.items()
            ],
            key=lambda s: s.count,
            reverse=True,
        )

        window_hours = (window_end - window_start).total_seconds() / 3600

        return DailySummaryReport(
            report_date=report_date,
            generated_at=datetime.now(UTC),
            total_detections=len(observations),
            unique_species=len(species_counts),
            species=species_summaries,
            audio_only_count=audio_only,
            visual_only_count=visual_only,
            fused_count=fused,
            detection_mode_counts=mode_counts,
            mean_confidence=mean(all_confidences) if all_confidences else 0.0,
            observation_window_hours=window_hours,
        )

    def write_daily_summary(
        self,
        report: DailySummaryReport,
        output_dir: str | Path,
    ) -> tuple[Path, Path]:
        """
        Write a daily summary report to disk as both .md and .json.

        Files are named YYYY-MM-DD.md and YYYY-MM-DD.json.
        Parent directory is created if it does not exist.

        Args:
            report:     The DailySummaryReport to write.
            output_dir: Directory to write files into (logs/daily_summaries/).

        Returns:
            Tuple of (md_path, json_path) for the two written files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = report.report_date.isoformat()
        md_path = output_dir / f"{stem}.md"
        json_path = output_dir / f"{stem}.json"

        md_path.write_text(report.to_markdown(), encoding="utf-8")
        json_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        logger.info(
            "Daily summary written | md=%s json=%s detections=%d species=%d",
            md_path.name,
            json_path.name,
            report.total_detections,
            report.unique_species,
        )
        return md_path, json_path
