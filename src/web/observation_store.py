"""Read-only adapter over ``logs/observations.jsonl``.

The dashboard never writes to ``observations.jsonl`` — that's the
agent's source of truth. This module loads the file into memory,
parses each line as a :class:`BirdObservation`, and serves filtered
+ paginated views to the route layer. The cache reloads when the
file's mtime changes.

ID convention: ``BirdObservation`` has no ``id`` field, so the URL
ID is the timestamp formatted as ``YYYYMMDDTHHMMSSffffff`` (UTC,
sortable, reversible — the agent writes serially, so timestamps
are unique per record).
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


_ID_PATTERN = re.compile(r"^\d{8}T\d{12}$")  # YYYYMMDDTHHMMSSffffff


def _id_for(obs: BirdObservation) -> str:
    """URL-safe stable ID derived from the observation's UTC timestamp."""
    ts = obs.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    else:
        ts = ts.astimezone(UTC)
    return ts.strftime("%Y%m%dT%H%M%S%f")


def _parse_id(observation_id: str) -> datetime | None:
    """Parse an ID back to UTC datetime. ``None`` on bad format."""
    if not _ID_PATTERN.match(observation_id):
        return None
    try:
        return datetime.strptime(observation_id, "%Y%m%dT%H%M%S%f").replace(tzinfo=UTC)
    except ValueError:
        return None


def _to_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


class ObservationNotFound(LookupError):
    """Raised on lookup-by-id miss. Routes turn this into a 404."""


class ObservationStore:
    """In-memory cache of ``observations.jsonl`` with mtime invalidation.

    Thread-safe via a single lock. Reads are O(N) over the cached
    list — fine at the projected scale (~100k records ceiling
    before we reach for an index).
    """

    def __init__(self, observations_path: Path) -> None:
        self._path = observations_path
        self._lock = threading.Lock()
        self._cached_mtime: float | None = None
        self._records: list[BirdObservation] = []

    def _load_if_changed(self) -> None:
        """Reload the file when mtime has changed since the last load."""
        with self._lock:
            try:
                mtime = self._path.stat().st_mtime
            except FileNotFoundError:
                self._cached_mtime = None
                self._records = []
                return
            if mtime == self._cached_mtime:
                return
            records = sorted(
                self._iter_parse(self._path),
                key=lambda o: o.timestamp,
                reverse=True,
            )
            self._records = records
            self._cached_mtime = mtime
            logger.info("ObservationStore loaded %d records from %s", len(records), self._path)

    @staticmethod
    def _iter_parse(path: Path) -> Iterator[BirdObservation]:
        """Yield observations from JSONL. Bad lines are logged and skipped."""
        with path.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    yield BirdObservation.model_validate(json.loads(line))
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning(
                        "Skipping malformed observation at %s:%d — %s", path, lineno, exc
                    )

    # ── Public API ────────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Force a fresh read on the next query (test hook)."""
        with self._lock:
            self._cached_mtime = None

    def total(self) -> int:
        self._load_if_changed()
        return len(self._records)

    def total_dispatched(self) -> int:
        self._load_if_changed()
        return sum(1 for o in self._records if o.dispatched)

    def latest(self) -> BirdObservation | None:
        self._load_if_changed()
        return self._records[0] if self._records else None

    def latest_dispatched(self) -> BirdObservation | None:
        self._load_if_changed()
        return next((o for o in self._records if o.dispatched), None)

    def file_mtime(self) -> float | None:
        """Underlying file's mtime, or ``None`` if it doesn't exist.

        Exposed so /api/status can derive its agent_status heuristic
        without re-stat'ing the file in the route handler.
        """
        try:
            return self._path.stat().st_mtime
        except FileNotFoundError:
            return None

    def get(self, observation_id: str) -> BirdObservation:
        """Look up by derived ID. Raises :class:`ObservationNotFound`."""
        self._load_if_changed()
        target = _parse_id(observation_id)
        if target is None:
            raise ObservationNotFound(observation_id)
        for o in self._records:
            if _to_utc(o.timestamp) == target:
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
        """Filtered + paginated view, newest first.

        Args:
            from_ts: keep records with ``timestamp >= from_ts``.
            to_ts: keep records with ``timestamp <= to_ts``.
            species: 4-letter code (case-insensitive). ``None`` = any.
            dispatched: ``True`` (default) returns dispatched only;
                ``False`` returns suppressed only; ``None`` returns both.
            limit: max records this page. Caller clamps the upper bound.
            cursor: ID from a previous page's ``next_cursor``.

        Returns:
            ``(records, next_cursor)``. ``next_cursor`` is ``None``
            when no further page exists.
        """
        self._load_if_changed()

        if species is not None:
            species = species.upper()

        cursor_ts: datetime | None = None
        if cursor is not None:
            cursor_ts = _parse_id(cursor)
            if cursor_ts is None:
                # Unrecognized cursor → empty page rather than 500.
                return [], None

        out: list[BirdObservation] = []
        for o in self._records:
            ts_utc = _to_utc(o.timestamp)
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


__all__ = [
    "ObservationNotFound",
    "ObservationStore",
    "_id_for",
    "_parse_id",
]
