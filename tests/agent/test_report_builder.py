"""
tests/notify/test_report_builder.py

Unit tests for ReportBuilder, DailySummaryReport, and ExperimentWindowReport.

All tests use synthetic JSONL data written to tmp_path — no real observation
log or network calls required. Tests run cleanly in CI.

Coverage:
    - JSONL parsing (valid, malformed, missing file)
    - Time-window filtering (since/until)
    - Per-species aggregation (counts, confidence, first/last seen)
    - Modality accounting (audio-only, visual-only, fused)
    - Detection mode counting
    - DailySummaryReport.to_markdown() structure
    - DailySummaryReport.to_push_message() length and content
    - DailySummaryReport.to_dict() roundtrip
    - ExperimentWindowReport stats (duration, detections_per_hour)
    - write_daily_summary() file creation and content
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

from src.notify.report_builder import (
    DailySummaryReport,
    ReportBuilder,
    SpeciesSummary,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _obs(
    species_code: str = "HOFI",
    common_name: str = "House Finch",
    fused_confidence: float = 0.85,
    timestamp: datetime | None = None,
    detection_mode: str = "fixed_crop",
    has_audio: bool = True,
    has_visual: bool = True,
) -> dict:
    """Build a minimal observation dict matching the JSONL schema."""
    ts = timestamp or datetime(2026, 4, 15, 10, 0, 0, tzinfo=UTC)
    return {
        "species_code": species_code,
        "common_name": common_name,
        "scientific_name": "Haemorhous mexicanus",
        "fused_confidence": fused_confidence,
        "timestamp": ts.isoformat(),
        "detection_mode": detection_mode,
        "audio_result": {"species_code": species_code} if has_audio else None,
        "visual_result": {"species_code": species_code} if has_visual else None,
    }


def _write_obs(path: Path, observations: list[dict]) -> None:
    """Write a list of observation dicts to a JSONL file."""
    with path.open("w") as f:
        for obs in observations:
            f.write(json.dumps(obs) + "\n")


def _make_builder(tmp_path: Path, observations: list[dict] | None = None) -> ReportBuilder:
    """Create a ReportBuilder with optional pre-written observations."""
    obs_path = tmp_path / "observations.jsonl"
    if observations is not None:
        _write_obs(obs_path, observations)
    else:
        obs_path.write_text("")
    return ReportBuilder(observations_path=str(obs_path))


# ── ReportBuilder._read_observations ─────────────────────────────────────────


class TestReadObservations:
    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, [])
        result = builder._read_observations()
        assert result == []

    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        builder = ReportBuilder(observations_path=str(tmp_path / "nonexistent.jsonl"))
        result = builder._read_observations()
        assert result == []

    def test_reads_single_observation(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, [_obs()])
        result = builder._read_observations()
        assert len(result) == 1
        assert result[0]["species_code"] == "HOFI"

    def test_reads_multiple_observations(self, tmp_path: Path) -> None:
        obs = [_obs("HOFI"), _obs("MODO"), _obs("BLPH")]
        builder = _make_builder(tmp_path, obs)
        result = builder._read_observations()
        assert len(result) == 3

    def test_skips_malformed_json_line(self, tmp_path: Path) -> None:
        obs_path = tmp_path / "observations.jsonl"
        obs_path.write_text(
            json.dumps(_obs("HOFI")) + "\n"
            + "not valid json\n"
            + json.dumps(_obs("MODO")) + "\n"
        )
        builder = ReportBuilder(observations_path=str(obs_path))
        result = builder._read_observations()
        # Bad line skipped, valid lines kept
        assert len(result) == 2

    def test_skips_observation_with_bad_timestamp(self, tmp_path: Path) -> None:
        bad = _obs("HOFI")
        bad["timestamp"] = "not-a-date"
        good = _obs("MODO")
        builder = _make_builder(tmp_path, [bad, good])
        result = builder._read_observations()
        assert len(result) == 1
        assert result[0]["species_code"] == "MODO"

    def test_filters_by_since(self, tmp_path: Path) -> None:
        early = _obs("HOFI", timestamp=datetime(2026, 4, 15, 6, 0, tzinfo=UTC))
        late = _obs("MODO", timestamp=datetime(2026, 4, 15, 14, 0, tzinfo=UTC))
        builder = _make_builder(tmp_path, [early, late])
        since = datetime(2026, 4, 15, 10, 0, tzinfo=UTC)
        result = builder._read_observations(since=since)
        assert len(result) == 1
        assert result[0]["species_code"] == "MODO"

    def test_filters_by_until(self, tmp_path: Path) -> None:
        early = _obs("HOFI", timestamp=datetime(2026, 4, 15, 6, 0, tzinfo=UTC))
        late = _obs("MODO", timestamp=datetime(2026, 4, 15, 14, 0, tzinfo=UTC))
        builder = _make_builder(tmp_path, [early, late])
        until = datetime(2026, 4, 15, 10, 0, tzinfo=UTC)
        result = builder._read_observations(until=until)
        assert len(result) == 1
        assert result[0]["species_code"] == "HOFI"

    def test_parsed_timestamp_is_aware(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, [_obs()])
        result = builder._read_observations()
        ts = result[0]["_parsed_timestamp"]
        assert ts.tzinfo is not None


# ── build_daily_summary ───────────────────────────────────────────────────────


class TestBuildDailySummary:
    def test_empty_log_returns_zero_counts(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, [])
        report = builder.build_daily_summary(for_date=date(2026, 4, 15))
        assert report.total_detections == 0
        assert report.unique_species == 0
        assert report.mean_confidence == 0.0

    def test_counts_detections_correctly(self, tmp_path: Path) -> None:
        obs = [
            _obs("HOFI", timestamp=datetime(2026, 4, 15, 10, 0, tzinfo=UTC)),
            _obs("HOFI", timestamp=datetime(2026, 4, 15, 11, 0, tzinfo=UTC)),
            _obs("MODO", timestamp=datetime(2026, 4, 15, 12, 0, tzinfo=UTC)),
        ]
        builder = _make_builder(tmp_path, obs)
        report = builder.build_daily_summary(for_date=date(2026, 4, 15))
        assert report.total_detections == 3
        assert report.unique_species == 2

    def test_excludes_other_day(self, tmp_path: Path) -> None:
        yesterday = _obs("HOFI", timestamp=datetime(2026, 4, 14, 10, 0, tzinfo=UTC))
        today = _obs("MODO", timestamp=datetime(2026, 4, 15, 10, 0, tzinfo=UTC))
        builder = _make_builder(tmp_path, [yesterday, today])
        report = builder.build_daily_summary(for_date=date(2026, 4, 15))
        assert report.total_detections == 1
        assert report.species[0].code == "MODO"

    def test_species_sorted_by_count_descending(self, tmp_path: Path) -> None:
        obs = [
            _obs("HOFI", timestamp=datetime(2026, 4, 15, 10, 0, tzinfo=UTC)),
            _obs("HOFI", timestamp=datetime(2026, 4, 15, 11, 0, tzinfo=UTC)),
            _obs("HOFI", timestamp=datetime(2026, 4, 15, 12, 0, tzinfo=UTC)),
            _obs("MODO", timestamp=datetime(2026, 4, 15, 13, 0, tzinfo=UTC)),
        ]
        builder = _make_builder(tmp_path, obs)
        report = builder.build_daily_summary(for_date=date(2026, 4, 15))
        assert report.species[0].code == "HOFI"
        assert report.species[0].count == 3
        assert report.species[1].code == "MODO"

    def test_top_species_property(self, tmp_path: Path) -> None:
        obs = [
            _obs("HOFI", timestamp=datetime(2026, 4, 15, 10, 0, tzinfo=UTC)),
            _obs("HOFI", timestamp=datetime(2026, 4, 15, 11, 0, tzinfo=UTC)),
            _obs("MODO", timestamp=datetime(2026, 4, 15, 12, 0, tzinfo=UTC)),
        ]
        builder = _make_builder(tmp_path, obs)
        report = builder.build_daily_summary(for_date=date(2026, 4, 15))
        assert report.top_species is not None
        assert report.top_species.code == "HOFI"

    def test_top_species_none_when_empty(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, [])
        report = builder.build_daily_summary(for_date=date(2026, 4, 15))
        assert report.top_species is None

    def test_mean_confidence_computed(self, tmp_path: Path) -> None:
        obs = [
            _obs("HOFI", fused_confidence=0.80, timestamp=datetime(2026, 4, 15, 10, 0, tzinfo=UTC)),
            _obs("HOFI", fused_confidence=0.90, timestamp=datetime(2026, 4, 15, 11, 0, tzinfo=UTC)),
        ]
        builder = _make_builder(tmp_path, obs)
        report = builder.build_daily_summary(for_date=date(2026, 4, 15))
        assert abs(report.mean_confidence - 0.85) < 1e-6

    def test_detection_mode_counts(self, tmp_path: Path) -> None:
        obs = [
            _obs("HOFI", detection_mode="fixed_crop", timestamp=datetime(2026, 4, 15, 10, 0, tzinfo=UTC)),
            _obs("HOFI", detection_mode="yolo", timestamp=datetime(2026, 4, 15, 11, 0, tzinfo=UTC)),
            _obs("MODO", detection_mode="yolo", timestamp=datetime(2026, 4, 15, 12, 0, tzinfo=UTC)),
        ]
        builder = _make_builder(tmp_path, obs)
        report = builder.build_daily_summary(for_date=date(2026, 4, 15))
        assert report.detection_mode_counts["fixed_crop"] == 1
        assert report.detection_mode_counts["yolo"] == 2

    def test_modality_counting_fused(self, tmp_path: Path) -> None:
        obs = [_obs("HOFI", has_audio=True, has_visual=True,
                    timestamp=datetime(2026, 4, 15, 10, 0, tzinfo=UTC))]
        builder = _make_builder(tmp_path, obs)
        report = builder.build_daily_summary(for_date=date(2026, 4, 15))
        assert report.fused_count == 1
        assert report.audio_only_count == 0
        assert report.visual_only_count == 0

    def test_modality_counting_audio_only(self, tmp_path: Path) -> None:
        obs = [_obs("HOFI", has_audio=True, has_visual=False,
                    timestamp=datetime(2026, 4, 15, 10, 0, tzinfo=UTC))]
        builder = _make_builder(tmp_path, obs)
        report = builder.build_daily_summary(for_date=date(2026, 4, 15))
        assert report.audio_only_count == 1
        assert report.fused_count == 0

    def test_modality_counting_visual_only(self, tmp_path: Path) -> None:
        obs = [_obs("HOFI", has_audio=False, has_visual=True,
                    timestamp=datetime(2026, 4, 15, 10, 0, tzinfo=UTC))]
        builder = _make_builder(tmp_path, obs)
        report = builder.build_daily_summary(for_date=date(2026, 4, 15))
        assert report.visual_only_count == 1
        assert report.fused_count == 0


# ── build_window_report ───────────────────────────────────────────────────────


class TestBuildWindowReport:
    def test_counts_only_matching_mode(self, tmp_path: Path) -> None:
        start = datetime(2026, 4, 15, 10, 0, tzinfo=UTC)
        end = datetime(2026, 4, 15, 10, 30, tzinfo=UTC)
        obs = [
            _obs("HOFI", detection_mode="fixed_crop", timestamp=datetime(2026, 4, 15, 10, 5, tzinfo=UTC)),
            _obs("MODO", detection_mode="yolo", timestamp=datetime(2026, 4, 15, 10, 10, tzinfo=UTC)),
            _obs("BLPH", detection_mode="fixed_crop", timestamp=datetime(2026, 4, 15, 10, 20, tzinfo=UTC)),
        ]
        builder = _make_builder(tmp_path, obs)
        report = builder.build_window_report(mode="fixed_crop", window_start=start, window_end=end)
        assert report.detections == 2
        assert report.unique_species == 2

    def test_empty_window_returns_zero(self, tmp_path: Path) -> None:
        start = datetime(2026, 4, 15, 10, 0, tzinfo=UTC)
        end = datetime(2026, 4, 15, 10, 30, tzinfo=UTC)
        builder = _make_builder(tmp_path, [])
        report = builder.build_window_report(mode="yolo", window_start=start, window_end=end)
        assert report.detections == 0
        assert report.mean_confidence == 0.0

    def test_duration_minutes(self, tmp_path: Path) -> None:
        start = datetime(2026, 4, 15, 10, 0, tzinfo=UTC)
        end = datetime(2026, 4, 15, 10, 30, tzinfo=UTC)
        builder = _make_builder(tmp_path, [])
        report = builder.build_window_report(mode="fixed_crop", window_start=start, window_end=end)
        assert report.duration_minutes == 30.0

    def test_detections_per_hour(self, tmp_path: Path) -> None:
        start = datetime(2026, 4, 15, 10, 0, tzinfo=UTC)
        end = datetime(2026, 4, 15, 10, 30, tzinfo=UTC)
        obs = [
            _obs("HOFI", detection_mode="fixed_crop", timestamp=datetime(2026, 4, 15, 10, 5, tzinfo=UTC)),
            _obs("MODO", detection_mode="fixed_crop", timestamp=datetime(2026, 4, 15, 10, 15, tzinfo=UTC)),
        ]
        builder = _make_builder(tmp_path, obs)
        report = builder.build_window_report(mode="fixed_crop", window_start=start, window_end=end)
        assert report.detections == 2
        assert abs(report.detections_per_hour - 4.0) < 0.1  # 2 per 30min = 4/hr

    def test_push_message_contains_mode(self, tmp_path: Path) -> None:
        start = datetime(2026, 4, 15, 10, 0, tzinfo=UTC)
        end = datetime(2026, 4, 15, 10, 30, tzinfo=UTC)
        builder = _make_builder(tmp_path, [])
        report = builder.build_window_report(mode="yolo", window_start=start, window_end=end)
        msg = report.to_push_message()
        assert "yolo" in msg


# ── DailySummaryReport rendering ──────────────────────────────────────────────


class TestDailySummaryReportRendering:
    def _make_report(self, detections: int = 5) -> DailySummaryReport:
        species = [
            SpeciesSummary(
                code="HOFI",
                common_name="House Finch",
                count=detections,
                mean_confidence=0.87,
                max_confidence=0.95,
                first_seen=datetime(2026, 4, 15, 8, 0, tzinfo=UTC),
                last_seen=datetime(2026, 4, 15, 17, 0, tzinfo=UTC),
                detection_modes=Counter({"fixed_crop": 3, "yolo": 2}),
            )
        ]
        return DailySummaryReport(
            report_date=date(2026, 4, 15),
            generated_at=datetime(2026, 4, 15, 20, 0, tzinfo=UTC),
            total_detections=detections,
            unique_species=1,
            species=species,
            audio_only_count=1,
            visual_only_count=1,
            fused_count=3,
            detection_mode_counts=Counter({"fixed_crop": 3, "yolo": 2}),
            mean_confidence=0.87,
            observation_window_hours=24.0,
        )

    def test_to_markdown_contains_date(self) -> None:
        report = self._make_report()
        md = report.to_markdown()
        assert "2026-04-15" in md

    def test_to_markdown_contains_species(self) -> None:
        report = self._make_report()
        md = report.to_markdown()
        assert "House Finch" in md

    def test_to_markdown_contains_detection_count(self) -> None:
        report = self._make_report(detections=7)
        md = report.to_markdown()
        assert "7" in md

    def test_to_markdown_contains_mode_breakdown(self) -> None:
        report = self._make_report()
        md = report.to_markdown()
        assert "fixed_crop" in md
        assert "yolo" in md

    def test_empty_report_markdown_handles_gracefully(self) -> None:
        report = DailySummaryReport(
            report_date=date(2026, 4, 15),
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
        md = report.to_markdown()
        assert "No detections" in md

    def test_to_push_message_non_empty(self) -> None:
        report = self._make_report()
        msg = report.to_push_message()
        assert isinstance(msg, str)
        assert len(msg) > 10

    def test_to_push_message_contains_species_count(self) -> None:
        report = self._make_report()
        msg = report.to_push_message()
        assert "1" in msg  # unique_species = 1

    def test_to_push_message_empty_report(self) -> None:
        report = DailySummaryReport(
            report_date=date(2026, 4, 15),
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
        msg = report.to_push_message()
        assert "no detections" in msg.lower()

    def test_to_dict_roundtrip(self) -> None:
        report = self._make_report()
        d = report.to_dict()
        assert d["total_detections"] == report.total_detections
        assert d["unique_species"] == report.unique_species
        assert d["report_date"] == "2026-04-15"
        assert len(d["species"]) == 1
        assert d["species"][0]["code"] == "HOFI"

    def test_mode_breakdown_string(self) -> None:
        s = SpeciesSummary(
            code="HOFI",
            common_name="House Finch",
            count=5,
            mean_confidence=0.87,
            max_confidence=0.95,
            first_seen=datetime.now(UTC),
            last_seen=datetime.now(UTC),
            detection_modes=Counter({"fixed_crop": 3, "yolo": 2}),
        )
        assert "fixed_crop" in s.mode_breakdown
        assert "yolo" in s.mode_breakdown


# ── write_daily_summary ───────────────────────────────────────────────────────


class TestWriteDailySummary:
    def test_creates_md_file(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, [])
        report = DailySummaryReport(
            report_date=date(2026, 4, 15),
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
        md_path, json_path = builder.write_daily_summary(report, tmp_path / "summaries")
        assert md_path.exists()
        assert md_path.suffix == ".md"

    def test_creates_json_file(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, [])
        report = DailySummaryReport(
            report_date=date(2026, 4, 15),
            generated_at=datetime.now(UTC),
            total_detections=2,
            unique_species=1,
            species=[],
            audio_only_count=0,
            visual_only_count=0,
            fused_count=2,
            detection_mode_counts=Counter({"fixed_crop": 2}),
            mean_confidence=0.85,
            observation_window_hours=24.0,
        )
        md_path, json_path = builder.write_daily_summary(report, tmp_path / "summaries")
        assert json_path.exists()
        assert json_path.suffix == ".json"

    def test_json_is_parseable(self, tmp_path: Path) -> None:
        import json as _json
        builder = _make_builder(tmp_path, [])
        report = DailySummaryReport(
            report_date=date(2026, 4, 15),
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
        _, json_path = builder.write_daily_summary(report, tmp_path / "summaries")
        data = _json.loads(json_path.read_text())
        assert data["report_date"] == "2026-04-15"

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, [])
        report = DailySummaryReport(
            report_date=date(2026, 4, 15),
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
        output_dir = tmp_path / "new" / "nested" / "dir"
        assert not output_dir.exists()
        builder.write_daily_summary(report, output_dir)
        assert output_dir.exists()

    def test_files_named_by_date(self, tmp_path: Path) -> None:
        builder = _make_builder(tmp_path, [])
        report = DailySummaryReport(
            report_date=date(2026, 4, 15),
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
        md_path, json_path = builder.write_daily_summary(report, tmp_path / "summaries")
        assert "2026-04-15" in md_path.name
        assert "2026-04-15" in json_path.name
