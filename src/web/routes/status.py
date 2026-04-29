"""Status + health endpoints.

``/health``       — unauthenticated liveness check (used by Tailscale
                    and any external monitor)
``/api/status``   — authenticated, returns dashboard uptime, agent
                    status, and a few headline counts derived from
                    ``observations.jsonl``

The agent and dashboard run as separate systemd units, so the
dashboard can't ask the agent "are you alive?" directly. Instead we
check how recently ``observations.jsonl`` was updated:

    < 60s ago    →  "live"
    < 10min ago  →  "idle"
    older / missing →  "stale"

That's a heuristic, not a contract — empty feeders look the same as
crashed agents to this check. Good enough for the dashboard's status
chip; not good enough for paging.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..auth import RequireToken
from ..observation_store import ObservationStore

# ── Routers ───────────────────────────────────────────────────────────────────

# /health stays unauthenticated. Mounted on its own router so we
# don't accidentally attach RequireToken to it later.
public_router = APIRouter(tags=["public"])

# /api/status sits behind the token wall.
status_router = APIRouter(prefix="/api", tags=["status"], dependencies=[RequireToken])


# ── Heuristic thresholds ──────────────────────────────────────────────────────

_LIVE_THRESHOLD_SECONDS = 60
_IDLE_THRESHOLD_SECONDS = 10 * 60


# ── App-state accessors ──────────────────────────────────────────────────────


def _store(request: Request) -> ObservationStore:
    return request.app.state.observation_store


def _start_time(request: Request) -> float:
    """Monotonic-ish start time recorded by the factory.

    Stored as a Unix timestamp (``time.time()``) so the difference
    against ``time.time()`` here gives uptime in seconds.
    """
    return request.app.state.start_time


# ── Response models ──────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class StatusResponse(BaseModel):
    """Body of ``GET /api/status``."""

    service: str
    version: str
    uptime_seconds: float
    total_observations: int
    total_dispatched: int
    last_observation_at: datetime | None
    last_dispatched_at: datetime | None
    current_mode: str | None = Field(
        default=None,
        description=(
            "Most recent observation's ``detection_mode`` "
            "(``fixed_crop`` or ``yolo``). ``None`` when the file "
            "is empty."
        ),
    )
    agent_status: str = Field(
        description=(
            "Heuristic from observations.jsonl mtime: ``live`` "
            "(<60s), ``idle`` (<10min), or ``stale``."
        ),
    )


# ── Routes ───────────────────────────────────────────────────────────────────


@public_router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Unauthenticated liveness check.

    Returns just status, service, and version. No observation data,
    no token state — auth on every other route is what actually
    protects the dashboard.
    """
    return HealthResponse(
        status="ok",
        service="avis-web",
        version=request.app.version,
    )


@status_router.get("/status", response_model=StatusResponse)
def status(request: Request) -> StatusResponse:
    """Dashboard status snapshot.

    Pulls counts and the most-recent record from the observation
    store. Cheap because the store caches by mtime — repeat calls
    don't re-read the file.
    """
    store = _store(request)
    latest = store.latest()
    latest_dispatched = store.latest_dispatched()
    mtime = store.file_mtime()
    now = time.time()

    if mtime is None:
        agent_status = "stale"
    elif now - mtime < _LIVE_THRESHOLD_SECONDS:
        agent_status = "live"
    elif now - mtime < _IDLE_THRESHOLD_SECONDS:
        agent_status = "idle"
    else:
        agent_status = "stale"

    return StatusResponse(
        service="avis-web",
        version=request.app.version,
        uptime_seconds=max(0.0, now - _start_time(request)),
        total_observations=store.total(),
        total_dispatched=store.total_dispatched(),
        last_observation_at=_to_utc(latest.timestamp) if latest else None,
        last_dispatched_at=_to_utc(latest_dispatched.timestamp) if latest_dispatched else None,
        current_mode=latest.detection_mode if latest else None,
        agent_status=agent_status,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _to_utc(ts: datetime) -> datetime:
    """Return ``ts`` in UTC. Handles naive datetimes by assuming UTC,
    matching the schema's ``_utcnow`` default."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)
