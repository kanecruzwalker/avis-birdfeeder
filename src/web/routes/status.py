"""Status + health endpoints.

``/health``      — unauthenticated liveness check (Tailscale, monitors)
``/api/status``  — authenticated; uptime, agent status heuristic, counts

The agent and dashboard run as separate systemd units, so the
dashboard can't ask the agent "are you alive?" directly. We fall
back to the freshness of ``observations.jsonl``:

    < 60s ago        →  "live"
    < 10min ago      →  "idle"
    older / missing  →  "stale"

Heuristic, not a contract — empty feeders look the same as crashed
agents to this check. Good enough for a status chip; not for paging.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..auth import RequireToken
from ..observation_store import ObservationStore

# One router; auth is per-route. /health stays public.
router = APIRouter(tags=["status"])


_LIVE_THRESHOLD_SECONDS = 60
_IDLE_THRESHOLD_SECONDS = 10 * 60


def _store(request: Request) -> ObservationStore:
    return request.app.state.observation_store


def _to_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


# ── Response models ──────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class StatusResponse(BaseModel):
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
            "(``fixed_crop`` or ``yolo``); ``None`` when the file is empty."
        ),
    )
    agent_status: str = Field(
        description="``live`` (<60s), ``idle`` (<10min), or ``stale``.",
    )


# ── Routes ───────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Unauthenticated liveness check."""
    return HealthResponse(status="ok", service="avis-web", version=request.app.version)


@router.get("/api/status", response_model=StatusResponse, dependencies=[RequireToken])
def status(request: Request) -> StatusResponse:
    """Dashboard status snapshot — counts, uptime, agent freshness."""
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
        uptime_seconds=max(0.0, now - request.app.state.start_time),
        total_observations=store.total(),
        total_dispatched=store.total_dispatched(),
        last_observation_at=_to_utc(latest.timestamp) if latest else None,
        last_dispatched_at=_to_utc(latest_dispatched.timestamp) if latest_dispatched else None,
        current_mode=latest.detection_mode if latest else None,
        agent_status=agent_status,
    )
