"""Unit tests for tools.labeler.ui.review_store.

Covers:
- load(): reading pre_labels.jsonl and verified_labels.jsonl cleanly
- species_summary() and coverage() on the landing page
- next_unverified() and get_review_item()
- record_verification() happy path (append)
- record_verification() correction path (atomic rewrite)
- record_verification() concurrency conflict (HTTP 409 path)
- force_overwrite=True acceptance
- image_path() traversal guard

All tests use temp dirs — no network, no fixtures outside this file.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tools.labeler.schema import PreLabel, PreLabelResponse, VerifiedLabel
from tools.labeler.ui.review_store import (
    ConcurrencyConflict,
    PreLabelNotFound,
    ReviewStore,
    ReviewStoreError,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_pre_label(
    filename: str,
    species: str = "HOFI",
    confidence: float = 0.9,
    other_code: str | None = None,
) -> PreLabel:
    """Build a PreLabel record for test fixtures."""
    return PreLabel(
        image_path=f"/test/images/{filename}",
        image_filename=filename,
        llm_response=PreLabelResponse(
            bird_visible=species not in ("NONE",),
            species_code=species,
            confidence=confidence,
            reasoning="test",
            other_species_code=other_code,
        ),
        model_name="gemini-2.5-flash",
        prompt_version="v1.0",
        elapsed_seconds=2.5,
    )


def _write_pre_labels(path: Path, records: list[PreLabel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


@pytest.fixture
def store_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return (pre_labels_path, verified_labels_path, images_dir)."""
    pre = tmp_path / "labels" / "pre_labels.jsonl"
    verified = tmp_path / "labels" / "verified_labels.jsonl"
    images = tmp_path / "images"
    images.mkdir()
    return pre, verified, images


# ── Load tests ───────────────────────────────────────────────────────────────


class TestLoad:
    def test_load_populates_indices(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(
            pre_path,
            [
                _make_pre_label("a.png", "HOFI"),
                _make_pre_label("b.png", "NONE"),
                _make_pre_label("c.png", "HOFI"),
            ],
        )
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        summary = {row["species_code"]: row for row in store.species_summary()}
        assert summary["HOFI"]["total"] == 2
        assert summary["NONE"]["total"] == 1

        cov = store.coverage()
        assert cov["total_pre_labels"] == 3
        assert cov["total_verified"] == 0
        assert cov["remaining"] == 3

    def test_load_missing_pre_labels_file_raises(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        store = ReviewStore(pre_path, verified_path, images_dir)
        with pytest.raises(ReviewStoreError, match="not found"):
            store.load()

    def test_load_missing_verified_file_is_ok(self, store_paths):
        """First run — verified_labels.jsonl doesn't exist. Should load fine."""
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        assert store.coverage()["total_verified"] == 0

    def test_load_skips_malformed_lines(self, store_paths, caplog):
        pre_path, verified_path, images_dir = store_paths
        pre_path.parent.mkdir(parents=True, exist_ok=True)
        valid = _make_pre_label("good.png").model_dump_json()
        with pre_path.open("w", encoding="utf-8") as fh:
            fh.write("not valid json\n")
            fh.write(valid + "\n")
            fh.write('{"partial": "json"}\n')  # parses as JSON but fails validation
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        assert store.coverage()["total_pre_labels"] == 1
        # Caplog confirms we skipped — not silently
        assert any("malformed" in rec.message.lower() for rec in caplog.records)

    def test_load_is_idempotent(self, store_paths):
        """Calling load() twice produces the same state."""
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(
            pre_path,
            [_make_pre_label("a.png"), _make_pre_label("b.png", "NONE")],
        )
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        first = store.coverage()
        store.load()
        second = store.coverage()
        assert first == second

    def test_verified_with_duplicate_filenames_takes_most_recent(self, store_paths):
        """verified_labels.jsonl is append-only under normal ops, so the
        same filename can appear twice. Most-recent verified_at wins."""
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png", "HOFI")])

        old = VerifiedLabel(
            image_path="/test/a.png",
            image_filename="a.png",
            species_code="HOFI",
            verified_at=datetime(2026, 4, 25, 10, 0, tzinfo=UTC),
        )
        new = VerifiedLabel(
            image_path="/test/a.png",
            image_filename="a.png",
            species_code="HOSP",  # corrected
            verified_at=datetime(2026, 4, 25, 11, 0, tzinfo=UTC),
        )
        verified_path.parent.mkdir(parents=True, exist_ok=True)
        with verified_path.open("w", encoding="utf-8") as fh:
            fh.write(old.model_dump_json() + "\n")
            fh.write(new.model_dump_json() + "\n")

        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        verified_list = store.list_verified()
        assert len(verified_list) == 1
        assert verified_list[0].species_code == "HOSP"


# ── next_unverified / get_review_item ────────────────────────────────────────


class TestNextUnverified:
    def test_returns_first_unverified(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(
            pre_path,
            [
                _make_pre_label("a.png", "HOFI"),
                _make_pre_label("b.png", "HOFI"),
            ],
        )
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        item = store.next_unverified()
        assert item is not None
        assert item.pre_label.image_filename == "a.png"
        assert item.client_load_time.tzinfo is not None

    def test_skips_already_verified(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(
            pre_path,
            [_make_pre_label("a.png"), _make_pre_label("b.png")],
        )
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        # Verify a.png first
        store.record_verification(
            VerifiedLabel(
                image_path="/test/a.png",
                image_filename="a.png",
                species_code="HOFI",
            ),
            client_load_time=datetime.now(UTC),
        )

        item = store.next_unverified()
        assert item is not None
        assert item.pre_label.image_filename == "b.png"

    def test_returns_none_when_all_verified(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        store.record_verification(
            VerifiedLabel(
                image_path="/test/a.png",
                image_filename="a.png",
                species_code="HOFI",
            ),
            client_load_time=datetime.now(UTC),
        )
        assert store.next_unverified() is None

    def test_species_filter(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(
            pre_path,
            [
                _make_pre_label("a.png", "HOFI"),
                _make_pre_label("b.png", "NONE"),
                _make_pre_label("c.png", "HOFI"),
            ],
        )
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        item = store.next_unverified(species_filter="NONE")
        assert item is not None
        assert item.pre_label.image_filename == "b.png"

    def test_species_filter_unknown_returns_none(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png", "HOFI")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        assert store.next_unverified(species_filter="SPTO") is None


class TestGetReviewItem:
    def test_loads_specific_image(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        item = store.get_review_item("a.png")
        assert item.pre_label.image_filename == "a.png"
        assert item.already_verified is None

    def test_reopened_verified_image_exposes_existing(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png", "HOFI")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        store.record_verification(
            VerifiedLabel(
                image_path="/test/a.png",
                image_filename="a.png",
                species_code="HOFI",
            ),
            client_load_time=datetime.now(UTC),
        )
        item = store.get_review_item("a.png")
        assert item.already_verified is not None
        assert item.already_verified.species_code == "HOFI"

    def test_unknown_filename_raises(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        with pytest.raises(PreLabelNotFound):
            store.get_review_item("nope.png")


# ── record_verification: happy path ──────────────────────────────────────────


class TestRecordVerificationAppend:
    def test_first_verify_appends_to_file(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png", "HOFI")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        store.record_verification(
            VerifiedLabel(
                image_path="/test/a.png",
                image_filename="a.png",
                species_code="HOFI",
            ),
            client_load_time=datetime.now(UTC),
        )

        records = _read_jsonl(verified_path)
        assert len(records) == 1
        assert records[0]["image_filename"] == "a.png"
        assert records[0]["species_code"] == "HOFI"

    def test_verify_updates_bucket_count(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(
            pre_path,
            [_make_pre_label("a.png", "HOFI"), _make_pre_label("b.png", "HOFI")],
        )
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        store.record_verification(
            VerifiedLabel(
                image_path="/test/a.png",
                image_filename="a.png",
                species_code="HOFI",
            ),
            client_load_time=datetime.now(UTC),
        )

        summary = {r["species_code"]: r for r in store.species_summary()}
        assert summary["HOFI"]["verified"] == 1
        assert summary["HOFI"]["remaining"] == 1

    def test_verify_with_other_sentinel_persists(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png", "MOCH")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        store.record_verification(
            VerifiedLabel(
                image_path="/test/a.png",
                image_filename="a.png",
                species_code="OTHER",
                other_species_code="CALT",
                agreed_with_pre_label=False,
            ),
            client_load_time=datetime.now(UTC),
        )

        records = _read_jsonl(verified_path)
        assert records[0]["species_code"] == "OTHER"
        assert records[0]["other_species_code"] == "CALT"
        assert records[0]["agreed_with_pre_label"] is False

    def test_verify_unknown_filename_raises(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        with pytest.raises(PreLabelNotFound):
            store.record_verification(
                VerifiedLabel(
                    image_path="/test/nope.png",
                    image_filename="nope.png",
                    species_code="HOFI",
                ),
                client_load_time=datetime.now(UTC),
            )


# ── record_verification: correction path (atomic rewrite) ────────────────────


class TestRecordVerificationCorrection:
    def test_correction_rewrites_file_to_single_record(self, store_paths):
        """Verify an image, then re-verify (correction) — file should end up
        with one record per image_filename, not two."""
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png", "HOFI")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        # First verification
        store.record_verification(
            VerifiedLabel(
                image_path="/test/a.png",
                image_filename="a.png",
                species_code="HOFI",
            ),
            client_load_time=datetime.now(UTC),
        )

        # File has 1 record
        assert len(_read_jsonl(verified_path)) == 1

        # Wait a tick so the first verified_at is strictly earlier
        # (Python datetime resolution is fine-grained but be defensive)
        past = datetime.now(UTC) + timedelta(seconds=1)

        # Correction: client load time is AFTER the existing record's
        # verified_at → accepted as correction
        store.record_verification(
            VerifiedLabel(
                image_path="/test/a.png",
                image_filename="a.png",
                species_code="HOSP",  # corrected
            ),
            client_load_time=past,
        )

        records = _read_jsonl(verified_path)
        assert (
            len(records) == 1
        ), "Correction should rewrite the file to a single record, not append"
        assert records[0]["species_code"] == "HOSP"

    def test_correction_does_not_bump_bucket_count(self, store_paths):
        """Correcting a verified label shouldn't double-count the verified
        tally on the bucket."""
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png", "HOFI")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        store.record_verification(
            VerifiedLabel(
                image_path="/test/a.png",
                image_filename="a.png",
                species_code="HOFI",
            ),
            client_load_time=datetime.now(UTC),
        )
        first = store.coverage()["total_verified"]

        past = datetime.now(UTC) + timedelta(seconds=1)
        store.record_verification(
            VerifiedLabel(
                image_path="/test/a.png",
                image_filename="a.png",
                species_code="HOSP",
            ),
            client_load_time=past,
        )
        second = store.coverage()["total_verified"]
        assert first == second == 1


# ── record_verification: concurrency conflict ────────────────────────────────


class TestConcurrencyConflict:
    def test_stale_client_raises_conflict(self, store_paths):
        """Client A loaded at T0, client B loaded+verified at T1.
        Client A now tries to verify — should conflict."""
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png", "HOFI")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        # Client A loads at T0 (now)
        client_a_load_time = datetime.now(UTC)

        # Client B loads and verifies at T1 (slightly later)
        # record_verification() stamps server-side verified_at at call time,
        # which will be > client_a_load_time
        import time as _time

        _time.sleep(0.01)  # ensure strictly later wall-clock

        store.record_verification(
            VerifiedLabel(
                image_path="/test/a.png",
                image_filename="a.png",
                species_code="HOFI",  # B says HOFI
            ),
            client_load_time=datetime.now(UTC),
        )

        # Client A now tries to verify with its stale load time
        with pytest.raises(ConcurrencyConflict) as exc_info:
            store.record_verification(
                VerifiedLabel(
                    image_path="/test/a.png",
                    image_filename="a.png",
                    species_code="HOSP",  # A says HOSP — disagrees!
                ),
                client_load_time=client_a_load_time,
            )
        # The existing record is surfaced to the client
        assert exc_info.value.existing.species_code == "HOFI"

    def test_force_overwrite_bypasses_conflict(self, store_paths):
        """After seeing the 409, client can retry with force_overwrite=True."""
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png", "HOFI")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        client_a_load_time = datetime.now(UTC)
        import time as _time

        _time.sleep(0.01)

        # Client B verifies first
        store.record_verification(
            VerifiedLabel(
                image_path="/test/a.png",
                image_filename="a.png",
                species_code="HOFI",
            ),
            client_load_time=datetime.now(UTC),
        )

        # Client A force-overwrites
        result = store.record_verification(
            VerifiedLabel(
                image_path="/test/a.png",
                image_filename="a.png",
                species_code="HOSP",
            ),
            client_load_time=client_a_load_time,
            force_overwrite=True,
        )
        assert result.species_code == "HOSP"

        # On-disk should be single record with HOSP
        records = _read_jsonl(verified_path)
        assert len(records) == 1
        assert records[0]["species_code"] == "HOSP"


# ── list_verified ────────────────────────────────────────────────────────────


class TestListVerified:
    def test_returns_empty_when_none(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        assert store.list_verified() == []

    def test_returns_all_verified_sorted_most_recent_first(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(
            pre_path,
            [
                _make_pre_label("a.png", "HOFI"),
                _make_pre_label("b.png", "HOFI"),
                _make_pre_label("c.png", "NONE"),
            ],
        )
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        import time as _time

        for filename, species in [("a.png", "HOFI"), ("b.png", "HOFI"), ("c.png", "NONE")]:
            store.record_verification(
                VerifiedLabel(
                    image_path=f"/test/{filename}",
                    image_filename=filename,
                    species_code=species,
                ),
                client_load_time=datetime.now(UTC),
            )
            _time.sleep(0.005)

        verified = store.list_verified()
        assert len(verified) == 3
        # Most-recent-first order
        assert verified[0].image_filename == "c.png"
        assert verified[-1].image_filename == "a.png"

    def test_species_filter(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(
            pre_path,
            [_make_pre_label("a.png", "HOFI"), _make_pre_label("b.png", "NONE")],
        )
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        for filename, species in [("a.png", "HOFI"), ("b.png", "NONE")]:
            store.record_verification(
                VerifiedLabel(
                    image_path=f"/test/{filename}",
                    image_filename=filename,
                    species_code=species,
                ),
                client_load_time=datetime.now(UTC),
            )

        hofi_only = store.list_verified(species_filter="HOFI")
        assert len(hofi_only) == 1
        assert hofi_only[0].species_code == "HOFI"


# ── image_path() security ────────────────────────────────────────────────────


class TestImagePath:
    def test_valid_filename_resolves(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png")])
        (images_dir / "a.png").write_bytes(b"fake")
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        resolved = store.image_path("a.png")
        assert resolved.name == "a.png"

    def test_traversal_rejected(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        with pytest.raises(ReviewStoreError):
            store.image_path("../etc/passwd")

    def test_absolute_path_rejected(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        with pytest.raises(ReviewStoreError):
            store.image_path("/etc/passwd")

    def test_backslash_rejected(self, store_paths):
        pre_path, verified_path, images_dir = store_paths
        _write_pre_labels(pre_path, [_make_pre_label("a.png")])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()
        with pytest.raises(ReviewStoreError):
            store.image_path("..\\windows\\system32")


# ── Thread safety smoke ──────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_verifies_no_corruption(self, store_paths):
        """Hammer the store from multiple threads. Each thread verifies a
        different image. The final JSONL should have exactly N valid records."""
        pre_path, verified_path, images_dir = store_paths
        filenames = [f"img_{i:03d}.png" for i in range(20)]
        _write_pre_labels(pre_path, [_make_pre_label(f, "HOFI") for f in filenames])
        store = ReviewStore(pre_path, verified_path, images_dir)
        store.load()

        def verify_one(filename: str) -> None:
            store.record_verification(
                VerifiedLabel(
                    image_path=f"/test/{filename}",
                    image_filename=filename,
                    species_code="HOFI",
                ),
                client_load_time=datetime.now(UTC),
            )

        threads = [threading.Thread(target=verify_one, args=(f,)) for f in filenames]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        records = _read_jsonl(verified_path)
        assert len(records) == 20
        # All records parsed cleanly
        for rec in records:
            assert rec["species_code"] == "HOFI"
