"""In-memory store and file I/O layer for the labeling-assistant review UI.

ReviewStore is the single source of truth for:
- Loading pre_labels.jsonl into an indexed view
- Writing verified_labels.jsonl (append-only + atomic rewrite on correction)
- Serving the group-by-species landing page data
- Optimistic concurrency via client_load_time stamps

Thread safety
-------------
The UI runs as a single FastAPI app with a single uvicorn worker by design —
this is a dev tool, not a high-traffic service. We still guard file writes
and index mutations with an RLock so concurrent requests from different
tabs / devices can't corrupt state.

Failure semantics
-----------------
- Malformed JSONL lines are logged and skipped during load (resilient).
- File writes are atomic on correction: write temp file, fsync, rename.
- Append writes are line-buffered and flushed immediately.
- The store is a pure wrapper over JSONL files — no hidden database,
  no cache beyond the in-memory index. Delete the JSONL to reset state.

Author: Kane Cruz-Walker, 2026-04-25 (Layer 2 review UI PR)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from ..schema import PreLabel, VerifiedLabel

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────


class ReviewStoreError(Exception):
    """Base class for ReviewStore errors."""


class PreLabelNotFound(ReviewStoreError):
    """Raised when a verify action references an unknown image_filename."""


class ConcurrencyConflict(ReviewStoreError):
    """Raised when a verify action would clobber a newer verified record.

    Carries the existing verified record so the caller can surface it to
    the client for explicit overwrite confirmation. The HTTP route turns
    this into a 409 response.
    """

    def __init__(self, existing: VerifiedLabel):
        super().__init__(
            f"Image '{existing.image_filename}' was verified as "
            f"'{existing.species_code}' at {existing.verified_at.isoformat()} "
            f"after this client's load."
        )
        self.existing = existing


# ── Group-by-species view ─────────────────────────────────────────────────────


class SpeciesBucket:
    """Lightweight summary of one species row on the landing page.

    Exposed as a plain dict to keep the HTTP layer simple — not a Pydantic
    model because it's derived data that never gets persisted.
    """

    __slots__ = ("species_code", "total", "verified", "image_filenames")

    def __init__(self, species_code: str) -> None:
        self.species_code: str = species_code
        self.total: int = 0
        self.verified: int = 0
        # Preserve insertion order so "next" returns images in the order
        # they were pre-labelled (chronologically newest first, since the
        # pre-labeler runs newest-first).
        self.image_filenames: list[str] = []

    def to_dict(self) -> dict:
        return {
            "species_code": self.species_code,
            "total": self.total,
            "verified": self.verified,
            "remaining": self.total - self.verified,
            "coverage": (self.verified / self.total) if self.total else 0.0,
        }


# ── ReviewStore ───────────────────────────────────────────────────────────────


class ReviewStore:
    """Read pre-labels, write verified labels, index by species."""

    def __init__(
        self,
        pre_labels_path: Path,
        verified_labels_path: Path,
        images_dir: Path,
    ) -> None:
        """Initialise the store. Call `load()` before serving traffic.

        Args:
            pre_labels_path: path to the existing pre_labels.jsonl (read-only).
            verified_labels_path: where to append/rewrite verified_labels.jsonl.
            images_dir: directory containing the capture PNGs. Served by the
                UI via a `/image/{filename}` route. The store doesn't read the
                images itself; it only validates that the filenames index here
                exist on disk when asked.
        """
        self.pre_labels_path = pre_labels_path
        self.verified_labels_path = verified_labels_path
        self.images_dir = images_dir

        # In-memory indices, populated by load().
        self._pre_labels: OrderedDict[str, PreLabel] = OrderedDict()
        self._verified: dict[str, VerifiedLabel] = {}
        self._species_buckets: dict[str, SpeciesBucket] = {}

        # Protects both indices and file writes.
        self._lock = threading.RLock()

        # Set once after first successful load() so repeated load() calls
        # on the same instance are safe.
        self._loaded = False

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Read both JSONL files into memory. Safe to call multiple times."""
        with self._lock:
            self._pre_labels.clear()
            self._verified.clear()
            self._species_buckets.clear()

            self._load_pre_labels()
            self._load_verified_labels()
            self._rebuild_species_buckets()
            self._loaded = True

        logger.info(
            "ReviewStore loaded | pre_labels=%d verified=%d species=%d",
            len(self._pre_labels),
            len(self._verified),
            len(self._species_buckets),
        )

    def _load_pre_labels(self) -> None:
        if not self.pre_labels_path.exists():
            raise ReviewStoreError(
                f"pre_labels.jsonl not found at {self.pre_labels_path}. "
                f"Run `python -m tools.labeler` to generate it first."
            )

        skipped = 0
        with self.pre_labels_path.open("rb") as fh:
            for line_num, raw in enumerate(fh, start=1):
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    pre_label = PreLabel.model_validate_json(line)
                except (json.JSONDecodeError, ValidationError) as exc:
                    logger.warning(
                        "Skipping malformed pre_labels.jsonl line %d: %s",
                        line_num,
                        exc,
                    )
                    skipped += 1
                    continue
                # If the same filename appears twice (shouldn't happen with
                # the pre-labeler's resume logic, but defensive), the LAST
                # record wins — most recent re-labeling.
                self._pre_labels[pre_label.image_filename] = pre_label

        if skipped:
            logger.warning("Skipped %d malformed pre-label lines during load", skipped)

    def _load_verified_labels(self) -> None:
        if not self.verified_labels_path.exists():
            # First run — no verified labels yet. Normal.
            return

        # verified_labels.jsonl is append-only under normal operation, so
        # a filename may appear more than once (e.g. after a correction we
        # haven't compacted yet). Most recent verified_at wins.
        skipped = 0
        with self.verified_labels_path.open("rb") as fh:
            for line_num, raw in enumerate(fh, start=1):
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    verified = VerifiedLabel.model_validate_json(line)
                except (json.JSONDecodeError, ValidationError) as exc:
                    logger.warning(
                        "Skipping malformed verified_labels.jsonl line %d: %s",
                        line_num,
                        exc,
                    )
                    skipped += 1
                    continue

                existing = self._verified.get(verified.image_filename)
                if existing is None or verified.verified_at > existing.verified_at:
                    self._verified[verified.image_filename] = verified

        if skipped:
            logger.warning(
                "Skipped %d malformed verified-label lines during load",
                skipped,
            )

    def _rebuild_species_buckets(self) -> None:
        """Group pre-labels by species_code for the landing page."""
        for pre in self._pre_labels.values():
            code = pre.llm_response.species_code
            bucket = self._species_buckets.setdefault(code, SpeciesBucket(code))
            bucket.total += 1
            bucket.image_filenames.append(pre.image_filename)
            if pre.image_filename in self._verified:
                bucket.verified += 1

    # ── Landing-page queries ─────────────────────────────────────────────────

    def species_summary(self) -> list[dict]:
        """Return the group-by-species view for the landing page.

        Sorted by total descending (most-populous species first), so the
        reviewer sees NONE / SOSP / HOFI etc. at the top — matching
        the natural distribution on the existing 2828 records.
        """
        self._ensure_loaded()
        with self._lock:
            rows = [b.to_dict() for b in self._species_buckets.values()]
        rows.sort(key=lambda r: r["total"], reverse=True)
        return rows

    def coverage(self) -> dict:
        """Overall verification coverage numbers."""
        self._ensure_loaded()
        with self._lock:
            total = len(self._pre_labels)
            verified = len(self._verified)
        return {
            "total_pre_labels": total,
            "total_verified": verified,
            "remaining": total - verified,
            "coverage": (verified / total) if total else 0.0,
        }

    # ── Image dispatch (next / by filename / listing) ────────────────────────

    def next_unverified(
        self,
        species_filter: str | None = None,
    ) -> ReviewItem | None:
        """Return the next unverified pre-label, optionally species-scoped.

        Returns None when there's nothing left to review in that filter.
        The response includes a `client_load_time` that the client echoes
        back on verify for optimistic-concurrency conflict detection.
        """
        self._ensure_loaded()
        with self._lock:
            candidates: list[str]
            if species_filter:
                bucket = self._species_buckets.get(species_filter.upper())
                if bucket is None:
                    return None
                candidates = bucket.image_filenames
            else:
                candidates = list(self._pre_labels.keys())

            for filename in candidates:
                if filename not in self._verified:
                    pre = self._pre_labels[filename]
                    return ReviewItem(
                        pre_label=pre,
                        already_verified=None,
                        client_load_time=datetime.now(UTC),
                    )
            return None

    def get_review_item(self, image_filename: str) -> ReviewItem:
        """Load a specific image for review (e.g. re-opening a verified one)."""
        self._ensure_loaded()
        with self._lock:
            pre = self._pre_labels.get(image_filename)
            if pre is None:
                raise PreLabelNotFound(
                    f"No pre-label found for '{image_filename}'."
                )
            existing = self._verified.get(image_filename)
        return ReviewItem(
            pre_label=pre,
            already_verified=existing,
            client_load_time=datetime.now(UTC),
        )

    def list_verified(
        self,
        species_filter: str | None = None,
    ) -> list[VerifiedLabel]:
        """Return all verified labels, optionally filtered by species.

        Sorted most-recent-first so the verified view shows recent work
        at the top.
        """
        self._ensure_loaded()
        with self._lock:
            all_verified = list(self._verified.values())

        if species_filter:
            filt = species_filter.upper()
            all_verified = [v for v in all_verified if v.species_code == filt]

        all_verified.sort(key=lambda v: v.verified_at, reverse=True)
        return all_verified

    # ── Verify (write) ───────────────────────────────────────────────────────

    def record_verification(
        self,
        verified: VerifiedLabel,
        client_load_time: datetime,
        force_overwrite: bool = False,
    ) -> VerifiedLabel:
        """Persist a verification, respecting optimistic concurrency.

        Four-case conflict resolution:

        1. No existing verified record for this image  → append.
        2. Existing record, `verified_at < client_load_time`  → accept as
           correction (reviewer re-opened a previously-verified image).
        3. Existing record, `verified_at > client_load_time`, `force_overwrite=False`
           → raise ConcurrencyConflict. Client will prompt user.
        4. Existing record, `verified_at > client_load_time`, `force_overwrite=True`
           → accept as explicit overwrite.

        Writes:
        - Case 1: append one line to verified_labels.jsonl (fast path).
        - Cases 2 and 4: rewrite entire file atomically (preserves the
          one-record-per-image invariant in on-disk state).

        Returns the persisted VerifiedLabel (with verified_at pinned to
        server-current-time, not whatever the client sent).
        """
        self._ensure_loaded()

        # Stamp server-side verified_at so the client can't forge timestamps.
        verified = verified.model_copy(
            update={"verified_at": datetime.now(UTC)}
        )

        with self._lock:
            if verified.image_filename not in self._pre_labels:
                raise PreLabelNotFound(
                    f"No pre-label for '{verified.image_filename}' — cannot verify."
                )

            existing = self._verified.get(verified.image_filename)

            if existing is None:
                # Case 1: append-only fast path
                self._append_to_file(verified)
                self._verified[verified.image_filename] = verified
                self._bump_bucket_verified(verified.image_filename)
                return verified

            # There's an existing verification. Correction vs conflict?
            if existing.verified_at <= client_load_time:
                # Case 2: correction — reviewer explicitly re-opened this image.
                self._rewrite_with_replacement(verified)
                self._verified[verified.image_filename] = verified
                # bucket count doesn't change — was verified, still verified
                return verified

            # existing.verified_at > client_load_time: someone else wrote this
            # after our client loaded. Conflict.
            if not force_overwrite:
                raise ConcurrencyConflict(existing)

            # Case 4: explicit overwrite
            self._rewrite_with_replacement(verified)
            self._verified[verified.image_filename] = verified
            return verified

    # ── File-write primitives (caller must hold self._lock) ──────────────────

    def _append_to_file(self, verified: VerifiedLabel) -> None:
        """Append one record to verified_labels.jsonl. Fast path."""
        self.verified_labels_path.parent.mkdir(parents=True, exist_ok=True)
        line = verified.model_dump_json()
        # Open in binary to control newline handling explicitly — avoids
        # Windows CRLF surprises that would break JSONL parsing downstream.
        with self.verified_labels_path.open("ab") as fh:
            fh.write(line.encode("utf-8") + b"\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _rewrite_with_replacement(self, verified: VerifiedLabel) -> None:
        """Rewrite verified_labels.jsonl atomically, swapping in the new record.

        Writes to a sibling temp file in the same directory (so rename is
        atomic on the same filesystem), fsyncs, then renames over the
        original. On any failure the original file is untouched.
        """
        self.verified_labels_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the new in-memory view first so the write is just one pass.
        updated_records: list[VerifiedLabel] = []
        for filename, existing in self._verified.items():
            if filename == verified.image_filename:
                updated_records.append(verified)
            else:
                updated_records.append(existing)
        # If somehow the replacement filename wasn't in the existing set
        # (shouldn't happen — caller checks), include it.
        if verified.image_filename not in self._verified:
            updated_records.append(verified)

        # Write to temp file in same dir so rename stays on same fs.
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            suffix=".jsonl.tmp",
            dir=self.verified_labels_path.parent,
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                for record in updated_records:
                    fh.write(record.model_dump_json().encode("utf-8") + b"\n")
                fh.flush()
                os.fsync(fh.fileno())
            # Atomic rename (os.replace works cross-platform on Windows too
            # as of Python 3.3+, unlike os.rename which fails if dest exists).
            os.replace(tmp_path, self.verified_labels_path)
        except Exception:
            # Best-effort cleanup of the temp file on failure
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            raise

    def _bump_bucket_verified(self, image_filename: str) -> None:
        pre = self._pre_labels.get(image_filename)
        if pre is None:
            return
        bucket = self._species_buckets.get(pre.llm_response.species_code)
        if bucket:
            bucket.verified += 1

    # ── Misc ─────────────────────────────────────────────────────────────────

    def image_path(self, image_filename: str) -> Path:
        """Return the on-disk path for an image, validating containment.

        The UI's image-serving route calls this to translate a filename in
        the URL to an absolute path. Guards against path traversal by
        ensuring the result stays inside `self.images_dir`.
        """
        if "/" in image_filename or "\\" in image_filename or ".." in image_filename:
            raise ReviewStoreError(f"Invalid image filename: {image_filename!r}")
        candidate = (self.images_dir / image_filename).resolve()
        base = self.images_dir.resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ReviewStoreError(
                f"Image path escapes images_dir: {image_filename!r}"
            ) from exc
        return candidate

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise ReviewStoreError(
                "ReviewStore.load() must be called before querying."
            )


# ── ReviewItem ────────────────────────────────────────────────────────────────


class ReviewItem:
    """Wrapper returned by the store for routes to serialise.

    Not a Pydantic model — the route layer composes its own response body
    so the HTTP shape is decoupled from the internal storage shape.
    """

    __slots__ = ("pre_label", "already_verified", "client_load_time")

    def __init__(
        self,
        pre_label: PreLabel,
        already_verified: VerifiedLabel | None,
        client_load_time: datetime,
    ) -> None:
        self.pre_label = pre_label
        self.already_verified = already_verified
        self.client_load_time = client_load_time
