"""Unit tests for src.web.observation_store.

Covers:
    - ID derivation and parsing — round-trip + format guard
    - Cache invalidation tied to mtime
    - Missing-file handling (returns empty list)
    - Malformed lines logged and skipped
    - total / total_dispatched / latest / latest_dispatched
    - query() filters: from / to / species / dispatched / cursor / limit
    - get() — happy path, malformed ID, missing record
    - Naive timestamps treated as UTC
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.data.schema import BirdObservation, ClassificationResult, Modality
from src.web.observation_store import (
    ObservationNotFound,
    ObservationStore,
    _id_for,
    _parse_id,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_obs(
    *,
    species_code: str = "HOFI",
    common_name: str = "House Finch",
    scientific_name: str = "Haemorhous mexicanus",
    fused_confidence: float = 0.9,
    dispatched: bool = True,
    timestamp: datetime | None = None,
    detection_mode: str = "fixed_crop",
    gate_reason: str | None = None,
) -> BirdObservation:
    """Construct a minimal-but-valid BirdObservation for tests."""
    return BirdObservation(
        species_code=species_code,
        common_name=common_name,
        scientific_name=scientific_name,
        fused_confidence=fused_confidence,
        dispatched=dispatched,
        visual_result=ClassificationResult(
            modality=Modality.VISUAL,
            species_code=species_code,
            common_name=common_name,
            scientific_name=scientific_name,
            confidence=fused_confidence,
            timestamp=timestamp or datetime.now(UTC),
        ),
        timestamp=timestamp or datetime.now(UTC),
        detection_mode=detection_mode,
        gate_reason=gate_reason,
    )


def write_jsonl(path: Path, observations: list[BirdObservation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for o in observations:
            fh.write(json.dumps(o.model_dump(mode="json")) + "\n")


@pytest.fixture
def empty_store(tmp_path: Path) -> ObservationStore:
    """Store pointing at a non-existent file."""
    return ObservationStore(tmp_path / "missing.jsonl")


@pytest.fixture
def populated_store(tmp_path: Path) -> tuple[ObservationStore, list[BirdObservation]]:
    """Store pointing at a file with five observations across two species."""
    base = datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC)
    obs = [
        make_obs(species_code="HOFI", timestamp=base, fused_confidence=0.95),
        make_obs(
            species_code="CALT",
            timestamp=base + timedelta(minutes=1),
            fused_confidence=0.88,
        ),
        # Suppressed observation — dispatched=False
        make_obs(
            species_code="HOFI",
            timestamp=base + timedelta(minutes=2),
            fused_confidence=0.4,
            dispatched=False,
            gate_reason="below_confidence_threshold",
        ),
        make_obs(
            species_code="CALT",
            timestamp=base + timedelta(minutes=3),
            fused_confidence=0.91,
            detection_mode="yolo",
        ),
        make_obs(
            species_code="HOFI",
            timestamp=base + timedelta(minutes=4),
            fused_confidence=0.97,
            detection_mode="yolo",
        ),
    ]
    path = tmp_path / "observations.jsonl"
    write_jsonl(path, obs)
    return ObservationStore(path), obs


# ── ID derivation ────────────────────────────────────────────────────────────


class TestIdDerivation:
    def test_round_trip(self):
        ts = datetime(2026, 4, 28, 12, 34, 56, 123456, tzinfo=UTC)
        obs = make_obs(timestamp=ts)
        ident = _id_for(obs)
        parsed = _parse_id(ident)
        assert parsed == ts

    def test_id_format_is_yyyymmdd(self):
        ts = datetime(2026, 4, 28, 12, 34, 56, 123456, tzinfo=UTC)
        ident = _id_for(make_obs(timestamp=ts))
        assert ident == "20260428T123456123456"

    def test_naive_timestamp_treated_as_utc(self):
        ts_naive = datetime(2026, 4, 28, 12, 34, 56, 123456)
        ident = _id_for(make_obs(timestamp=ts_naive))
        assert ident == "20260428T123456123456"

    def test_parse_id_rejects_garbage(self):
        assert _parse_id("not-an-id") is None
        assert _parse_id("") is None
        assert _parse_id("20260428T1234561234567") is None  # one digit too many
        assert _parse_id("abcdefghT123456abcdef") is None

    def test_parse_id_rejects_invalid_date(self):
        # February 30 doesn't exist
        assert _parse_id("20260230T120000000000") is None


# ── Empty / missing file ─────────────────────────────────────────────────────


class TestEmptyOrMissing:
    def test_total_zero_when_missing(self, empty_store):
        assert empty_store.total() == 0

    def test_latest_none_when_missing(self, empty_store):
        assert empty_store.latest() is None

    def test_latest_dispatched_none_when_missing(self, empty_store):
        assert empty_store.latest_dispatched() is None

    def test_query_returns_empty_when_missing(self, empty_store):
        records, cursor = empty_store.query()
        assert records == []
        assert cursor is None

    def test_file_mtime_none_when_missing(self, empty_store):
        assert empty_store.file_mtime() is None

    def test_get_raises_when_missing(self, empty_store):
        with pytest.raises(ObservationNotFound):
            empty_store.get("20260428T120000000000")


# ── Caching / mtime ──────────────────────────────────────────────────────────


class TestCacheInvalidation:
    def test_reload_when_file_changes(self, tmp_path):
        path = tmp_path / "observations.jsonl"
        first = make_obs(timestamp=datetime(2026, 4, 28, 12, 0, tzinfo=UTC))
        write_jsonl(path, [first])
        store = ObservationStore(path)
        assert store.total() == 1

        # Append another record and bump mtime so the store reloads.
        second = make_obs(timestamp=datetime(2026, 4, 28, 12, 1, tzinfo=UTC))
        write_jsonl(path, [first, second])
        os.utime(path, (time.time() + 1, time.time() + 1))
        assert store.total() == 2

    def test_no_reload_when_file_unchanged(self, tmp_path):
        path = tmp_path / "observations.jsonl"
        write_jsonl(path, [make_obs()])
        store = ObservationStore(path)
        store.total()  # warm
        first_mtime = store._cached_mtime
        store.total()  # same call — should be a noop
        assert store._cached_mtime == first_mtime

    def test_force_reload_via_reload_method(self, tmp_path):
        path = tmp_path / "observations.jsonl"
        write_jsonl(path, [make_obs()])
        store = ObservationStore(path)
        store.total()  # warm
        store.reload()
        assert store._cached_mtime is None


# ── Malformed lines ──────────────────────────────────────────────────────────


class TestMalformedLines:
    def test_skips_invalid_json(self, tmp_path, caplog):
        path = tmp_path / "observations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        valid = make_obs()
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(valid.model_dump(mode="json")) + "\n")
            fh.write("not valid json\n")
            fh.write("\n")  # blank line — also tolerated
            fh.write(json.dumps(valid.model_dump(mode="json")) + "\n")
        store = ObservationStore(path)
        assert store.total() == 2

    def test_skips_schema_violations(self, tmp_path):
        path = tmp_path / "observations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        valid = make_obs()
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(valid.model_dump(mode="json")) + "\n")
            # Missing required fields:
            fh.write(json.dumps({"species_code": "HOFI"}) + "\n")
        store = ObservationStore(path)
        # The malformed entry is logged & skipped, valid one survives.
        assert store.total() == 1


# ── Counts and latest ────────────────────────────────────────────────────────


class TestCounts:
    def test_total_counts_all_records(self, populated_store):
        store, obs = populated_store
        assert store.total() == len(obs)

    def test_total_dispatched_excludes_suppressed(self, populated_store):
        store, obs = populated_store
        expected = sum(1 for o in obs if o.dispatched)
        assert store.total_dispatched() == expected

    def test_latest_is_newest_overall(self, populated_store):
        store, obs = populated_store
        newest = max(obs, key=lambda o: o.timestamp)
        assert store.latest().timestamp == newest.timestamp

    def test_latest_dispatched_is_newest_dispatched(self, populated_store):
        store, obs = populated_store
        dispatched = [o for o in obs if o.dispatched]
        newest = max(dispatched, key=lambda o: o.timestamp)
        assert store.latest_dispatched().timestamp == newest.timestamp


# ── Query filters ────────────────────────────────────────────────────────────


class TestQueryFilters:
    def test_default_dispatched_only(self, populated_store):
        store, obs = populated_store
        records, _ = store.query()
        assert all(r.dispatched for r in records)

    def test_dispatched_false_returns_only_suppressed(self, populated_store):
        store, _ = populated_store
        records, _ = store.query(dispatched=False)
        assert records and all(not r.dispatched for r in records)

    def test_dispatched_none_returns_both(self, populated_store):
        store, obs = populated_store
        records, _ = store.query(dispatched=None, limit=100)
        assert len(records) == len(obs)

    def test_species_filter_uppercases(self, populated_store):
        store, _ = populated_store
        upper, _ = store.query(species="HOFI", dispatched=None, limit=100)
        lower, _ = store.query(species="hofi", dispatched=None, limit=100)
        assert [r.timestamp for r in upper] == [r.timestamp for r in lower]

    def test_species_filter_excludes_others(self, populated_store):
        store, _ = populated_store
        records, _ = store.query(species="CALT", dispatched=None, limit=100)
        assert records and all(r.species_code == "CALT" for r in records)

    def test_from_ts_inclusive_lower_bound(self, populated_store):
        store, obs = populated_store
        cutoff = obs[2].timestamp  # third observation
        records, _ = store.query(from_ts=cutoff, dispatched=None, limit=100)
        assert all(r.timestamp >= cutoff for r in records)

    def test_to_ts_inclusive_upper_bound(self, populated_store):
        store, obs = populated_store
        cutoff = obs[2].timestamp
        records, _ = store.query(to_ts=cutoff, dispatched=None, limit=100)
        assert all(r.timestamp <= cutoff for r in records)

    def test_results_newest_first(self, populated_store):
        store, _ = populated_store
        records, _ = store.query(dispatched=None, limit=100)
        timestamps = [r.timestamp for r in records]
        assert timestamps == sorted(timestamps, reverse=True)


# ── Pagination ───────────────────────────────────────────────────────────────


class TestPagination:
    def test_cursor_round_trip(self, populated_store):
        store, _ = populated_store
        page1, cursor = store.query(dispatched=None, limit=2)
        assert len(page1) == 2
        assert cursor is not None
        page2, _ = store.query(dispatched=None, limit=2, cursor=cursor)
        # Page 2 starts strictly older than the last item on page 1
        assert page2[0].timestamp < page1[-1].timestamp

    def test_no_cursor_when_page_not_full(self, populated_store):
        store, _ = populated_store
        records, cursor = store.query(dispatched=None, limit=100)
        assert cursor is None  # all results fit, no further page

    def test_unrecognized_cursor_returns_empty_page(self, populated_store):
        store, _ = populated_store
        records, cursor = store.query(cursor="not-an-id")
        assert records == []
        assert cursor is None


# ── get() lookup ─────────────────────────────────────────────────────────────


class TestGet:
    def test_get_by_id_returns_record(self, populated_store):
        store, obs = populated_store
        target = obs[0]
        ident = _id_for(target)
        result = store.get(ident)
        assert result.timestamp == target.timestamp
        assert result.species_code == target.species_code

    def test_get_unknown_raises(self, populated_store):
        store, _ = populated_store
        with pytest.raises(ObservationNotFound):
            store.get("20210101T000000000000")  # date with no record

    def test_get_malformed_id_raises(self, populated_store):
        store, _ = populated_store
        with pytest.raises(ObservationNotFound):
            store.get("garbage")
