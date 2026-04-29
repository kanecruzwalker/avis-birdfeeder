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
from .routes import observations as observations_routes
from .routes import status as status_routes

logger = logging.getLogger(__name__)


# Default location of the agent's observation log. Resolved relative
# to the project root (the cwd when ``python -m src.web`` is run from
# the repo, or the systemd ``WorkingDirectory`` on the Pi). The CLI
# in ``__main__`` overrides this when the operator passes
# ``--observations-path`` or when ``configs/paths.yaml`` resolves
# differently.
_DEFAULT_OBSERVATIONS_PATH = Path("logs/observations.jsonl")


# ── App factory ───────────────────────────────────────────────────────────────


def create_app(observations_path: Path | None = None) -> FastAPI:
    """Build and configure a FastAPI app for the web dashboard.

    Args:
        observations_path: Path to the agent's ``observations.jsonl``.
            The store reads this file lazily and reloads when its
            mtime changes — the dashboard never writes to it. Tests
            point this at a tmp file. ``None`` falls back to
            ``logs/observations.jsonl`` relative to cwd.

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

    # ── Routes ────────────────────────────────────────────────────────────────

    # status_routes owns /health (unauth) and /api/status (auth).
    app.include_router(status_routes.public_router)
    app.include_router(status_routes.status_router)

    # observations_routes owns /api/observations and /api/observations/{id}.
    app.include_router(observations_routes.router)

    logger.info(
        "Avis web dashboard app created (version=%s, observations_path=%s)",
        app.version,
        observations_path,
    )
    return app
