"""Observation list + detail endpoints.

``GET /api/observations``
    Filtered, paginated view of ``observations.jsonl``. Newest first
    (matches what the SPA's recent / timeline / gallery views all
    expect). Pagination is cursor-based: each response includes a
    ``next_cursor`` that's an opaque string the client passes back
    on the next call.

``GET /api/observations/{id}``
    Single observation by its derived ID. ID format is the
    timestamp formatted as ``YYYYMMDDTHHMMSSffffff`` (UTC) — the
    store builds and parses these. Returns 404 when the ID is
    malformed or doesn't match any record.

Both endpoints sit behind the token wall — same dependency as
``/api/status``.

The response models here mirror ``BirdObservation`` from
``src.data.schema`` but exist as a dedicated wire-shape so we can
add presentation-only fields later (``id`` is the obvious one)
without leaking them back into the agent's data model.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.data.schema import BirdObservation, ClassificationResult

from ..auth import RequireToken
from ..observation_store import (
    ObservationNotFound,
    ObservationStore,
    _id_for,
)

# /api/observations* sits behind the same auth wall as /api/status.
router = APIRouter(
    prefix="/api",
    tags=["observations"],
    dependencies=[RequireToken],
)


# ── Response shapes ──────────────────────────────────────────────────────────
#
# Wire shape ≈ BirdObservation + an `id` field. Defining it
# explicitly (instead of model_dump'ing BirdObservation) keeps the
# wire contract independent of internal schema churn.


class ObservationOut(BaseModel):
    """A single observation as the dashboard sees it.

    Adds ``id`` (the URL-safe timestamp ID) on top of every
    BirdObservation field.
    """

    id: str
    species_code: str
    common_name: str
    scientific_name: str
    fused_confidence: float
    dispatched: bool
    audio_result: ClassificationResult | None = None
    visual_result: ClassificationResult | None = None
    visual_result_2: ClassificationResult | None = None
    timestamp: datetime
    image_path: str | None = None
    image_path_2: str | None = None
    audio_path: str | None = None
    detection_mode: str = "fixed_crop"
    gate_reason: str | None = None
    detection_box: list[int] | None = None
    estimated_depth_cm: float | None = None
    estimated_size_cm: float | None = None
    stereo_calibrated: bool = False

    @classmethod
    def from_record(cls, obs: BirdObservation) -> ObservationOut:
        return cls(id=_id_for(obs), **obs.model_dump())


class ObservationListOut(BaseModel):
    """Paginated list response.

    ``next_cursor`` is ``None`` when the page wasn't full — i.e.,
    there are no more records matching the query.
    """

    items: list[ObservationOut]
    next_cursor: str | None = Field(
        default=None,
        description=(
            "Opaque pagination token. Pass back as ``cursor`` on the "
            "next request to fetch records older than the last item "
            "in this page. ``None`` when no further page exists."
        ),
    )
    count: int = Field(
        description="Number of items in this page (not the global total).",
    )


# ── Limit clamp ──────────────────────────────────────────────────────────────

# The SPA's recent / timeline / gallery views default to 50 items
# per fetch. 500 is the upper bound — anything bigger gets clamped
# rather than rejected so a sloppy client doesn't 422 the user.
_DEFAULT_LIMIT = 50
_MAX_LIMIT = 500


# ── Helpers ──────────────────────────────────────────────────────────────────


def _store(request: Request) -> ObservationStore:
    return request.app.state.observation_store


# ── Routes ───────────────────────────────────────────────────────────────────


@router.get("/observations", response_model=ObservationListOut)
def list_observations(
    request: Request,
    from_ts: Annotated[
        datetime | None,
        Query(
            alias="from",
            description=(
                "ISO 8601 timestamp. Only observations at or after "
                "this moment are returned. Naive timestamps are "
                "treated as UTC."
            ),
        ),
    ] = None,
    to_ts: Annotated[
        datetime | None,
        Query(
            alias="to",
            description=(
                "ISO 8601 timestamp. Only observations at or before this moment are returned."
            ),
        ),
    ] = None,
    species: Annotated[
        str | None,
        Query(
            description=(
                "4-letter species code (case-insensitive). e.g., ``HOFI`` for House Finch."
            ),
            min_length=1,
            max_length=8,
        ),
    ] = None,
    dispatched: Annotated[
        bool | None,
        Query(
            description=(
                "``true`` (default) returns only dispatched "
                "observations — the user-visible stream. ``false`` "
                "returns only suppressed observations. Omit the "
                "param entirely (or pass any value other than "
                "true/false) for the full stream."
            ),
        ),
    ] = True,
    limit: Annotated[
        int,
        Query(
            ge=1,
            description=(
                "Max records per page. Clamped to 500 — larger "
                "values are silently reduced rather than rejected."
            ),
        ),
    ] = _DEFAULT_LIMIT,
    cursor: Annotated[
        str | None,
        Query(
            description=(
                "Pagination cursor from a previous response's "
                "``next_cursor``. An unrecognized cursor returns an "
                "empty page rather than 400 — keeps the client "
                "robust to ID format changes."
            ),
        ),
    ] = None,
) -> ObservationListOut:
    """Filtered + paginated observation list. Newest first."""
    capped_limit = min(limit, _MAX_LIMIT)

    # Normalize tz-naive bounds to UTC. The agent writes UTC, the
    # store compares in UTC, but a curl user might paste a naive
    # ISO string. Treat that as UTC rather than 422'ing.
    if from_ts is not None and from_ts.tzinfo is None:
        from_ts = from_ts.replace(tzinfo=UTC)
    if to_ts is not None and to_ts.tzinfo is None:
        to_ts = to_ts.replace(tzinfo=UTC)

    records, next_cursor = _store(request).query(
        from_ts=from_ts,
        to_ts=to_ts,
        species=species,
        dispatched=dispatched,
        limit=capped_limit,
        cursor=cursor,
    )
    items = [ObservationOut.from_record(r) for r in records]
    return ObservationListOut(
        items=items,
        next_cursor=next_cursor,
        count=len(items),
    )


@router.get("/observations/{observation_id}", response_model=ObservationOut)
def get_observation(observation_id: str, request: Request) -> ObservationOut:
    """Look up a single observation by ID. 404 on miss."""
    try:
        record = _store(request).get(observation_id)
    except ObservationNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"observation not found: {observation_id}",
        ) from exc
    return ObservationOut.from_record(record)
