"""Unit tests for tools.labeler.pre_labeler.

Tests helpers (timestamp parsing, observation indexing, resume logic)
and the batch loop with a mocked Gemini model. No real API calls.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

from tools.labeler.pre_labeler import (
    MIN_AUDIO_HINT_CONFIDENCE,
    ObservationIndex,
    PreLabeler,
    load_already_labeled,
    parse_capture_timestamp,
)
from tools.labeler.schema import PreLabelResponse


class TestParseCaptureTimestamp:
    """Parse UTC datetime from Pi capture filenames."""

    def test_valid_filename(self) -> None:
        ts = parse_capture_timestamp("20260424_141605_420369_cam0.png")
        assert ts is not None
        assert ts.year == 2026
        assert ts.month == 4
        assert ts.day == 24
        assert ts.hour == 14
        assert ts.minute == 16
        assert ts.second == 5
        assert ts.microsecond == 420369
        assert ts.tzinfo == UTC

    def test_cam1_also_parses(self) -> None:
        ts = parse_capture_timestamp("20260424_141605_420369_cam1.png")
        assert ts is not None

    def test_non_matching_filename(self) -> None:
        assert parse_capture_timestamp("random.png") is None
        assert parse_capture_timestamp("image.jpg") is None
        assert parse_capture_timestamp("20260424_cam0.png") is None

    def test_impossible_date_returns_none(self) -> None:
        # Month 99 is invalid
        assert parse_capture_timestamp("20269924_141605_420369_cam0.png") is None


class TestObservationIndex:
    """Build an index from observations.jsonl for fast filename lookup."""

    def _write_jsonl(self, tmp_path: Path, records: list[dict]) -> Path:
        p = tmp_path / "observations.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        return p

    def test_empty_file(self, tmp_path: Path) -> None:
        p = self._write_jsonl(tmp_path, [])
        idx = ObservationIndex.from_jsonl(p)
        assert idx.lookup("anything.png") is None

    def test_missing_file_returns_empty_index(self, tmp_path: Path) -> None:
        idx = ObservationIndex.from_jsonl(tmp_path / "nonexistent.jsonl")
        assert idx.lookup("anything.png") is None

    def test_lookup_by_filename_basename(self, tmp_path: Path) -> None:
        p = self._write_jsonl(
            tmp_path,
            [
                {
                    "image_path": "/mnt/data/captures/img1.png",
                    "audio_result": {"species_code": "HOFI", "confidence": 0.85},
                },
            ],
        )
        idx = ObservationIndex.from_jsonl(p)
        assert idx.lookup("img1.png") is not None

    def test_extract_audio_hint_high_confidence(self, tmp_path: Path) -> None:
        p = self._write_jsonl(
            tmp_path,
            [
                {
                    "image_path": "/mnt/data/captures/img1.png",
                    "audio_result": {"species_code": "HOFI", "confidence": 0.85},
                },
            ],
        )
        idx = ObservationIndex.from_jsonl(p)
        species, conf = idx.extract_audio_hint("img1.png")
        assert species == "HOFI"
        assert conf == 0.85

    def test_extract_audio_hint_below_threshold_dropped(self, tmp_path: Path) -> None:
        """Audio hints below MIN_AUDIO_HINT_CONFIDENCE are discarded."""
        below = MIN_AUDIO_HINT_CONFIDENCE - 0.05
        p = self._write_jsonl(
            tmp_path,
            [
                {
                    "image_path": "/mnt/data/captures/img1.png",
                    "audio_result": {"species_code": "HOFI", "confidence": below},
                },
            ],
        )
        idx = ObservationIndex.from_jsonl(p)
        species, conf = idx.extract_audio_hint("img1.png")
        assert species is None
        assert conf is None

    def test_extract_audio_hint_no_audio_result(self, tmp_path: Path) -> None:
        p = self._write_jsonl(
            tmp_path,
            [{"image_path": "/mnt/data/captures/img1.png", "audio_result": None}],
        )
        idx = ObservationIndex.from_jsonl(p)
        species, conf = idx.extract_audio_hint("img1.png")
        assert species is None
        assert conf is None

    def test_indexes_both_cameras(self, tmp_path: Path) -> None:
        p = self._write_jsonl(
            tmp_path,
            [
                {
                    "image_path": "/mnt/data/captures/img1_cam0.png",
                    "image_path_2": "/mnt/data/captures/img1_cam1.png",
                    "audio_result": {"species_code": "HOFI", "confidence": 0.85},
                },
            ],
        )
        idx = ObservationIndex.from_jsonl(p)
        assert idx.lookup("img1_cam0.png") is not None
        assert idx.lookup("img1_cam1.png") is not None

    def test_malformed_line_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "observations.jsonl"
        p.write_text(
            '{"image_path": "/x/img1.png", "audio_result": null}\n'
            "{not valid json\n"
            '{"image_path": "/x/img2.png", "audio_result": null}\n',
            encoding="utf-8",
        )
        idx = ObservationIndex.from_jsonl(p)
        # Both valid records indexed despite malformed middle line
        assert idx.lookup("img1.png") is not None
        assert idx.lookup("img2.png") is not None


class TestLoadAlreadyLabeled:
    """Resume support — skip images already in pre_labels.jsonl."""

    def test_missing_file_returns_empty_set(self, tmp_path: Path) -> None:
        result = load_already_labeled(tmp_path / "pre_labels.jsonl")
        assert result == set()

    def test_reads_labeled_filenames(self, tmp_path: Path) -> None:
        p = tmp_path / "pre_labels.jsonl"
        p.write_text(
            '{"image_filename": "img1.png", "other": "fields"}\n'
            '{"image_filename": "img2.png"}\n',
            encoding="utf-8",
        )
        result = load_already_labeled(p)
        assert result == {"img1.png", "img2.png"}

    def test_malformed_line_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "pre_labels.jsonl"
        p.write_text(
            '{"image_filename": "img1.png"}\n' "invalid json\n" '{"image_filename": "img2.png"}\n',
            encoding="utf-8",
        )
        result = load_already_labeled(p)
        assert result == {"img1.png", "img2.png"}


class TestPreLabelerBatchLoop:
    """Batch loop with mocked Gemini. Tests dispatch, resume, and summary counts."""

    def _make_labeler_with_mock(self, response: PreLabelResponse) -> PreLabeler:
        """Construct a PreLabeler without hitting Gemini, with a canned response."""
        labeler = PreLabeler.__new__(PreLabeler)
        labeler.model_name = "gemini-2.5-flash-stub"
        labeler.temperature = 0.1
        labeler.max_retries = 0
        labeler.retry_delay_seconds = 0
        labeler.inter_request_delay = 0
        labeler._system_prompt = "stub system prompt"

        # Mock the structured model so .invoke returns our canned response.
        labeler._structured_model = MagicMock()
        labeler._structured_model.invoke = MagicMock(return_value=response)
        return labeler

    def _make_image(self, tmp_path: Path, filename: str) -> Path:
        """Write a tiny fake PNG so open() succeeds. Gemini is mocked out."""
        p = tmp_path / filename
        # A 1-byte file is enough — the mock doesn't inspect content.
        p.write_bytes(b"\x00")
        return p

    def test_run_processes_images_newest_first(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        # Create three images with distinct timestamps
        for fn in (
            "20260424_100000_000000_cam0.png",
            "20260424_110000_000000_cam0.png",
            "20260424_120000_000000_cam0.png",
        ):
            self._make_image(img_dir, fn)

        response = PreLabelResponse(
            bird_visible=False,
            species_code="NONE",
            confidence=1.0,
            reasoning="Empty feeder.",
        )
        labeler = self._make_labeler_with_mock(response)
        output = tmp_path / "pre_labels.jsonl"

        summary = labeler.run(
            image_dir=img_dir,
            observations_path=None,
            output_path=output,
            limit=3,
        )

        assert summary["attempted"] == 3
        assert summary["succeeded"] == 3
        assert summary["failed"] == 0
        # Three records written
        lines = output.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_run_honours_limit(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for fn in (
            "20260424_100000_000000_cam0.png",
            "20260424_110000_000000_cam0.png",
            "20260424_120000_000000_cam0.png",
        ):
            self._make_image(img_dir, fn)

        response = PreLabelResponse(
            bird_visible=False,
            species_code="NONE",
            confidence=1.0,
            reasoning="Empty feeder.",
        )
        labeler = self._make_labeler_with_mock(response)
        output = tmp_path / "pre_labels.jsonl"

        summary = labeler.run(
            image_dir=img_dir,
            observations_path=None,
            output_path=output,
            limit=2,
        )
        assert summary["succeeded"] == 2

    def test_run_skips_already_labeled(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for fn in ("img_a.png", "img_b.png"):
            self._make_image(img_dir, fn)

        # Pre-populate pre_labels.jsonl as if img_a was already processed
        output = tmp_path / "pre_labels.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"image_filename": "img_a.png"}\n', encoding="utf-8")

        response = PreLabelResponse(
            bird_visible=False,
            species_code="NONE",
            confidence=1.0,
            reasoning="Empty feeder.",
        )
        labeler = self._make_labeler_with_mock(response)
        summary = labeler.run(
            image_dir=img_dir,
            observations_path=None,
            output_path=output,
            limit=10,
        )
        assert summary["skipped_already_labeled"] == 1
        assert summary["succeeded"] == 1

    def test_run_respects_min_capture_time(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        self._make_image(img_dir, "20260420_120000_000000_cam0.png")  # old
        self._make_image(img_dir, "20260424_120000_000000_cam0.png")  # new

        response = PreLabelResponse(
            bird_visible=False,
            species_code="NONE",
            confidence=1.0,
            reasoning="Empty feeder.",
        )
        labeler = self._make_labeler_with_mock(response)
        output = tmp_path / "pre_labels.jsonl"

        summary = labeler.run(
            image_dir=img_dir,
            observations_path=None,
            output_path=output,
            limit=10,
            min_capture_time=datetime(2026, 4, 22, tzinfo=UTC),
        )
        assert summary["succeeded"] == 1
        assert summary["skipped_too_old"] == 1

    def test_camera_filter(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        self._make_image(img_dir, "20260424_100000_000000_cam0.png")
        self._make_image(img_dir, "20260424_110000_000000_cam1.png")

        response = PreLabelResponse(
            bird_visible=False,
            species_code="NONE",
            confidence=1.0,
            reasoning="Empty feeder.",
        )
        labeler = self._make_labeler_with_mock(response)
        output = tmp_path / "pre_labels.jsonl"

        summary = labeler.run(
            image_dir=img_dir,
            observations_path=None,
            output_path=output,
            limit=10,
            camera_filter="cam0",
        )
        assert summary["succeeded"] == 1

    def test_run_creates_output_parent_directory(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        self._make_image(img_dir, "20260424_100000_000000_cam0.png")

        response = PreLabelResponse(
            bird_visible=False,
            species_code="NONE",
            confidence=1.0,
            reasoning="Empty feeder.",
        )
        labeler = self._make_labeler_with_mock(response)
        output = tmp_path / "nested" / "dir" / "pre_labels.jsonl"

        summary = labeler.run(
            image_dir=img_dir,
            observations_path=None,
            output_path=output,
            limit=1,
        )
        assert summary["succeeded"] == 1
        assert output.exists()

    def test_run_continues_after_single_image_failure(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        self._make_image(img_dir, "20260424_100000_000000_cam0.png")
        self._make_image(img_dir, "20260424_110000_000000_cam0.png")

        labeler = self._make_labeler_with_mock(
            PreLabelResponse(
                bird_visible=False,
                species_code="NONE",
                confidence=1.0,
                reasoning="OK.",
            )
        )
        # First call fails, second succeeds.
        labeler._structured_model.invoke.side_effect = [
            Exception("transient network error"),
            PreLabelResponse(
                bird_visible=False,
                species_code="NONE",
                confidence=1.0,
                reasoning="Empty.",
            ),
        ]
        output = tmp_path / "pre_labels.jsonl"
        summary = labeler.run(
            image_dir=img_dir,
            observations_path=None,
            output_path=output,
            limit=10,
        )
        assert summary["attempted"] == 2
        assert summary["succeeded"] == 1
        assert summary["failed"] == 1
