"""FastAPI app assembly for the Avis web dashboard.

The factory builds an app, attaches the shared ``ObservationStore``,
records a start time on app state, and mounts route modules from
``src.web.routes``. Auth (``AVIS_WEB_TOKEN``) is enforced on every
route except ``/health``.

Roadmap of route mounts (PR boundaries from
``docs/investigations/web-dashboard-2026-04-28.md``):

    PR 2 — ``/api/status``, ``/api/observations``, ``/api/observations/{id}``
    PR 3 — ``/api/stream``, ``/api/frame`` (MJPEG)
    PR 4 — ``/api/observations/{id}/image/{cropped|annotated|full}``
    PR 5 — box-cache annotated stream
    PR 6 — ``GET /``  (HTML SPA)
    PR 7 — timeline / gallery / detail views
    PR 8 — ``POST /api/ask``

The factory takes explicit configuration so tests can build isolated
apps without touching the filesystem or the production agent.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI

from .observation_store import ObservationStore
from .routes import images as images_routes
from .routes import observations as observations_routes
from .routes import status as status_routes
from .routes import stream as stream_routes
from .stream_buffer import StreamBuffer

logger = logging.getLogger(__name__)


# Default location of the agent's observation log. Resolved relative
# to the project root (the cwd when ``python -m src.web`` is run from
# the repo, or the systemd ``WorkingDirectory`` on the Pi). The CLI
# in ``__main__`` overrides this when the operator passes
# ``--observations-path`` or when ``configs/paths.yaml`` resolves
# differently.
_DEFAULT_OBSERVATIONS_PATH = Path("logs/observations.jsonl")


# ── App factory ───────────────────────────────────────────────────────────────


def create_app(
    observations_path: Path | None = None,
    stream_buffer: StreamBuffer | None = None,
) -> FastAPI:
    """Build and configure a FastAPI app for the web dashboard.

    Args:
        observations_path: Path to the agent's ``observations.jsonl``.
            The store reads this file lazily and reloads when its
            mtime changes — the dashboard never writes to it. Tests
            point this at a tmp file. ``None`` falls back to
            ``logs/observations.jsonl`` relative to cwd.
        stream_buffer: Optional shared :class:`StreamBuffer` produced
            by the agent's ``VisionCapture``. When set, ``/api/stream``
            and ``/api/frame`` serve live preview frames from it; when
            ``None``, those endpoints return 503. The dashboard runs
            as its own systemd unit independent of the agent, so in
            production the buffer is wired in only when both run in
            the same process (a future PR adds the cross-process
            bridge for the split-process layout).

    Returns:
        A configured FastAPI app. Auth (``AVIS_WEB_TOKEN``) is enforced
        on every route except ``/health``; the token must be set in the
        environment by the time a request arrives, or routes will
        return a 500 from the auth dependency.
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
        # /docs and /redoc are off — every route is auth-protected and
        # exposing the schema to unauthenticated callers doesn't help.
        # Tests still reach FastAPI internals via the test client.
        docs_url=None,
        redoc_url=None,
    )

    # ── Shared state ──────────────────────────────────────────────────────────

    # The store is stashed on app.state so route handlers can pull it
    # via Request.app.state without a global. Tests override the path
    # by passing observations_path; in production the CLI passes the
    # resolved value from configs/paths.yaml.
    app.state.observation_store = ObservationStore(observations_path)

    # Used by /api/status to compute uptime. time.time() is fine here
    # — uptime accuracy to the second is plenty for a status chip.
    app.state.start_time = time.time()

    # Live preview ring buffer (PR 3). May be None when the dashboard
    # runs without an in-process VisionCapture; the /api/stream and
    # /api/frame routes 503 in that case rather than failing the
    # whole factory. Tests pass an explicit StreamBuffer instance
    # (or a thin mock) to exercise the routes.
    app.state.stream_buffer = stream_buffer

    # /api/stream uses this as the per-frame wait timeout on its
    # buffer subscription. Default is comfortable for the production
    # 5fps publish cadence; tests override to a short value so the
    # streaming generator returns control promptly when the test
    # client exits the response context (the threadpool task can
    # only cancel cleanly between condvar waits).
    app.state.stream_wait_timeout = 5.0

    # ── Routes ────────────────────────────────────────────────────────────────

    # status_routes owns /health (unauth) and /api/status (auth).
    app.include_router(status_routes.router)

    # observations_routes owns /api/observations and /api/observations/{id}.
    app.include_router(observations_routes.router)

    # stream_routes owns /api/stream and /api/frame. Both endpoints
    # 503 when app.state.stream_buffer is None, so it's always safe
    # to mount.
    app.include_router(stream_routes.router)

    # images_routes owns /api/observations/{id}/image/{variant} (PR 4).
    # Reads the same ObservationStore + on-disk capture files; nothing
    # extra to wire up.
    app.include_router(images_routes.router)

    logger.info(
        "Avis web dashboard app created (version=%s, observations_path=%s)",
        app.version,
        observations_path,
    )
    return app
