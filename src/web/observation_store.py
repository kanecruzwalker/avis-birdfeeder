"""Read-only adapter over ``logs/observations.jsonl``.

The dashboard never writes to ``observations.jsonl`` — that file is
the agent's source of truth and the dashboard just reads it. This
module loads the file into memory, parses each line as a
``BirdObservation``, and serves filtered / paginated views to the
route layer.

Caching
-------
The store keeps the parsed records in memory and reloads only when
the file's mtime changes. With the agent appending one record per
detection, mtime ticks every time the file grows, so the cache stays
fresh without polling the disk on every request.

At the projected scale (~21k records today, target ~100k before we
revisit) full-scan filters are fine — a 100k-record list with a
species + date filter is well under 100ms in Python.

ID convention
-------------
``BirdObservation`` has no ``id`` field. We derive a URL-safe ID
from the timestamp: ``YYYYMMDDTHHMMSSffffff`` (UTC, no separators,
sortable). Round-trip is exact because timestamps are unique per
record (the agent writes serially).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from src.data.schema import BirdObservation

logger = logging.getLogger(__name__)


# ── ID derivation ─────────────────────────────────────────────────────────────


_ID_PATTERN = re.compile(r"^\d{8}T\d{12}$")  # YYYYMMDDTHHMMSSffffff


def _id_for(obs: BirdObservation) -> str:
    """URL-safe stable ID for an observation, derived from timestamp.

    Format: ``YYYYMMDDTHHMMSSffffff`` in UTC. Reversible — the route
    handler parses the ID back into a datetime and looks up the
    matching record.
    """
    ts = obs.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    else:
        ts = ts.astimezone(UTC)
    return ts.strftime("%Y%m%dT%H%M%S%f")


def _parse_id(observation_id: str) -> datetime | None:
    """Parse an ID back into a UTC datetime. Returns ``None`` if the
    ID isn't in the expected format — callers map that to a 404."""
    if not _ID_PATTERN.match(observation_id):
        return None
    try:
        return datetime.strptime(observation_id, "%Y%m%dT%H%M%S%f").replace(tzinfo=UTC)
    except ValueError:
        return None


# ── Errors ────────────────────────────────────────────────────────────────────


class ObservationNotFound(LookupError):
    """Raised when a lookup-by-id misses. Routes turn this into a 404."""


# ── Store ─────────────────────────────────────────────────────────────────────


class ObservationStore:
    """In-memory cache of ``observations.jsonl``.

    Thread-safe: a single lock guards the parse + cache update path.
    Reads after the cache is warm are lock-free in the common case
    where mtime hasn't changed.

    Args:
        observations_path: Path to the JSONL file. May not exist at
            construction time — an agent that hasn't dispatched yet
            won't have created it. Reads return an empty list until
            the file appears.
    """

    def __init__(self, observations_path: Path) -> None:
        self._path = observations_path
        self._lock = threading.Lock()
        self._cached_mtime: float | None = None
        self._records: list[BirdObservation] = []
        # Newest-first order is preserved by sorting after each load.
        # observations.jsonl is append-only so records on disk are
        # mostly in chronological order, but we don't rely on that.

    # ── Cache management ──────────────────────────────────────────────────────

    def _load_if_changed(self) -> None:
        """Reload the file if mtime has changed since the last load.

        Skips work entirely when the file hasn't moved. If the file
        doesn't exist, the cache is left as an empty list — the
        dashboard renders cleanly against a fresh deploy.
        """
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            with self._lock:
                self._cached_mtime = None
                self._records = []
            return

        if mtime == self._cached_mtime:
            return

        with self._lock:
            # Re-check inside the lock in case another thread already
            # reloaded while we were waiting.
            try:
                mtime = self._path.stat().st_mtime
            except FileNotFoundError:
                self._cached_mtime = None
                self._records = []
                return
            if mtime == self._cached_mtime:
                return

            records = list(self._iter_parse(self._path))
            # Newest first — the dashboard's recent / timeline views
            # expect this order and we save the route handlers from
            # re-sorting on every request.
            records.sort(key=lambda o: o.timestamp, reverse=True)
            self._records = records
            self._cached_mtime = mtime
            logger.info(
                "ObservationStore loaded %d records from %s",
                len(records),
                self._path,
            )

    @staticmethod
    def _iter_parse(path: Path) -> Iterator[BirdObservation]:
        """Yield BirdObservations from a JSONL file. Bad lines are
        logged and skipped rather than aborting the load — a single
        malformed record shouldn't break the dashboard."""
        with path.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    yield BirdObservation.model_validate(payload)
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning(
                        "Skipping malformed observation at %s:%d — %s",
                        path,
                        lineno,
                        exc,
                    )

    # ── Public API ────────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Force a fresh read on the next query (test hook)."""
        with self._lock:
            self._cached_mtime = None

    def total(self) -> int:
        """Count of all records in the file (dispatched + suppressed)."""
        self._load_if_changed()
        return len(self._records)

    def total_dispatched(self) -> int:
        """Count of dispatched records — the 'user-visible' total."""
        self._load_if_changed()
        return sum(1 for o in self._records if o.dispatched)

    def latest(self) -> BirdObservation | None:
        """Most recent record by timestamp, or ``None`` when empty."""
        self._load_if_changed()
        return self._records[0] if self._records else None

    def latest_dispatched(self) -> BirdObservation | None:
        """Most recent dispatched record, or ``None``."""
        self._load_if_changed()
        for o in self._records:
            if o.dispatched:
                return o
        return None

    def file_mtime(self) -> float | None:
        """Underlying file's mtime, or ``None`` if it doesn't exist.

        Exposed so /api/status can derive an agent_status heuristic
        without re-stat'ing the file in the route handler.
        """
        try:
            return self._path.stat().st_mtime
        except FileNotFoundError:
            return None

    def get(self, observation_id: str) -> BirdObservation:
        """Look up a single observation by its derived ID.

        Raises:
            ObservationNotFound: if the ID is malformed or no record
                has the matching timestamp.
        """
        self._load_if_changed()
        target = _parse_id(observation_id)
        if target is None:
            raise ObservationNotFound(observation_id)
        for o in self._records:
            ts = o.timestamp if o.timestamp.tzinfo else o.timestamp.replace(tzinfo=UTC)
            if ts.astimezone(UTC) == target:
                return o
        raise ObservationNotFound(observation_id)

    def query(
        self,
        *,
        from_ts: datetime | None = None,
        to_ts: datetime | None = None,
        species: str | None = None,
        dispatched: bool | None = True,
        limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[BirdObservation], str | None]:
        """Filtered + paginated view of the records.

        Args:
            from_ts: keep records with ``timestamp >= from_ts``.
            to_ts: keep records with ``timestamp <= to_ts``.
            species: 4-letter code (case-insensitive). ``None`` = any.
            dispatched: ``True`` = dispatched only (the default,
                matching the API contract). ``False`` = suppressed
                only. ``None`` = both.
            limit: max records to return. Caller should clamp to a
                sensible upper bound; the store doesn't second-guess.
            cursor: ID returned in the previous page's
                ``next_cursor``. Records older than the cursor's
                timestamp are returned.

        Returns:
            ``(records, next_cursor)``. ``next_cursor`` is ``None``
            when there's no further page (i.e., fewer than ``limit``
            records remain).
        """
        self._load_if_changed()

        if species is not None:
            species = species.upper()

        cursor_ts: datetime | None = None
        if cursor is not None:
            cursor_ts = _parse_id(cursor)
            if cursor_ts is None:
                # An unrecognized cursor returns an empty page rather
                # than 500'ing — easier on the client.
                return [], None

        out: list[BirdObservation] = []
        for o in self._records:
            ts = o.timestamp if o.timestamp.tzinfo else o.timestamp.replace(tzinfo=UTC)
            ts_utc = ts.astimezone(UTC)

            if cursor_ts is not None and ts_utc >= cursor_ts:
                continue
            if from_ts is not None and ts_utc < from_ts:
                continue
            if to_ts is not None and ts_utc > to_ts:
                continue
            if species is not None and o.species_code != species:
                continue
            if dispatched is True and not o.dispatched:
                continue
            if dispatched is False and o.dispatched:
                continue

            out.append(o)
            if len(out) >= limit:
                break

        next_cursor = _id_for(out[-1]) if len(out) >= limit else None
        return out, next_cursor


# Public — routes import these
__all__ = [
    "ObservationNotFound",
    "ObservationStore",
    "_id_for",
    "_parse_id",
]
