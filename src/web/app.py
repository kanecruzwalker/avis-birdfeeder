"""FastAPI app assembly for the Avis web dashboard.

The factory builds an app, attaches a shared :class:`ObservationStore`,
records a start time on app state, and mounts route modules from
``src.web.routes``. Auth (``AVIS_WEB_TOKEN``) is enforced on every
route except ``/health``.

The factory takes explicit configuration so tests can build isolated
apps without touching the filesystem or the production agent.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .box_cache import BoxCache
from .observation_store import ObservationStore
from .routes import chat as chat_routes
from .routes import images as images_routes
from .routes import observations as observations_routes
from .routes import pages as pages_routes
from .routes import status as status_routes
from .routes import stream as stream_routes
from .stream_buffer import StreamBuffer

if TYPE_CHECKING:
    # Type-only import; the real class pulls in google-genai + langchain
    # at module import, which we avoid in this factory so the dashboard
    # boots without those dependencies installed.
    from src.agent.bird_analyst_agent import BirdAnalystAgent

logger = logging.getLogger(__name__)


# Default location of the agent's observation log, resolved relative
# to cwd (the systemd ``WorkingDirectory`` on the Pi). The CLI in
# ``__main__`` overrides this from ``configs/paths.yaml`` or a flag.
_DEFAULT_OBSERVATIONS_PATH = Path("logs/observations.jsonl")

# Default per-frame wait on /api/stream subscriptions. Production
# publishes ~5fps so 5s is well past steady-state — it's the cap on
# how long an idle stream pins a worker thread when the publisher
# stalls. Tests override via ``app.state.stream_wait_timeout`` so
# the sync streaming generator returns control promptly between
# condvar waits.
_DEFAULT_STREAM_WAIT_TIMEOUT_SECONDS = 5.0


def create_app(
    observations_path: Path | None = None,
    stream_buffer: StreamBuffer | None = None,
    box_cache: BoxCache | None = None,
    analyst: BirdAnalystAgent | Any | None = None,
) -> FastAPI:
    """Build and configure a FastAPI app for the web dashboard.

    Args:
        observations_path: Path to the agent's ``observations.jsonl``.
            The store reads lazily and reloads on mtime change — the
            dashboard never writes to it. ``None`` falls back to
            ``logs/observations.jsonl`` relative to cwd.
        stream_buffer: Optional shared :class:`StreamBuffer` populated
            by the agent's ``VisionCapture``. When set, ``/api/stream``
            and ``/api/frame`` serve from it; when ``None``, those
            endpoints return 503.
        box_cache: Optional shared :class:`BoxCache`. The agent updates
            it after each YOLO classification; ``VisionCapture`` reads
            it at publish time and overlays the box on preview frames.
            The dashboard process stashes it on app state for symmetry
            with the stream buffer and for any future per-route
            consumer (e.g. a status chip showing the current detection).
        analyst: Optional :class:`BirdAnalystAgent`. Drives ``POST
            /api/ask``; when ``None``, the chat endpoint returns 503.
            Tests inject a mock with the same ``answer()`` shape so the
            route can be exercised without the LLM.
    """
    if observations_path is None:
        observations_path = _DEFAULT_OBSERVATIONS_PATH

    app = FastAPI(
        title="Avis web dashboard",
        description=(
            "Pi-hosted dashboard for the Avis birdfeeder. Token-auth, "
            "served over Tailscale (or ngrok for short demos). Doesn't "
            "write to anything the agent reads."
        ),
        version="0.1.0",
        # /docs and /redoc off — every route is auth-protected; exposing
        # the schema to unauthenticated callers doesn't help.
        docs_url=None,
        redoc_url=None,
    )

    app.state.observation_store = ObservationStore(observations_path)
    app.state.start_time = time.time()
    app.state.stream_buffer = stream_buffer
    app.state.stream_wait_timeout = _DEFAULT_STREAM_WAIT_TIMEOUT_SECONDS
    app.state.box_cache = box_cache
    app.state.analyst = analyst

    app.include_router(pages_routes.router)  # GET /
    app.include_router(status_routes.router)  # /health + /api/status
    app.include_router(observations_routes.router)  # /api/observations*
    app.include_router(stream_routes.router)  # /api/stream + /api/frame
    app.include_router(images_routes.router)  # /api/observations/{id}/image/*
    app.include_router(chat_routes.router)  # /api/ask

    # /static/* serves the SPA bundle (CSS, JS, fonts). No auth — the
    # bundle is a generic dashboard shell with no secrets. Mount last
    # so the explicit /api/* and / routes win on path collisions.
    pages_routes.STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/static",
        StaticFiles(directory=str(pages_routes.STATIC_DIR)),
        name="static",
    )

    logger.info(
        "Avis web dashboard app created (version=%s, observations_path=%s)",
        app.version,
        observations_path,
    )
    return app
