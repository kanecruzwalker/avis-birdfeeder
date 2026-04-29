"""FastAPI app assembly for the Avis web dashboard.

PR 1 scope: token middleware ready, /health unauthenticated, every
other route auth-protected. Routes that consume observation data
land in PR 2 (status, observations, observation detail).

The factory is a function (not a module-level app) so tests can
build isolated apps, and the CLI in ``__main__`` can construct the
app after it has confirmed the token at startup.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from .auth import RequireToken  # noqa: F401  -- imported so PR 2 callers can re-use

logger = logging.getLogger(__name__)


# ---- App factory -----------------------------------------------------------


def create_app() -> FastAPI:
    """Build and configure a FastAPI app for the web dashboard.

    Returns:
        A configured FastAPI app. Auth (``AVIS_WEB_TOKEN``) is
        enforced on every route except ``/health``; the token must
        be set in the environment by the time a request arrives, or
        protected routes return 500 from the auth dependency.
    """
    app = FastAPI(
        title="Avis web dashboard",
        description=(
            "Pi-hosted dashboard for the Avis birdfeeder. Token-auth, "
            "served over Tailscale (or ngrok for short demos). Doesn't "
            "write to anything the agent reads."
        ),
        version="0.1.0",
        # /docs and /redoc are off -- every route is auth-protected
        # and exposing the schema to unauthenticated callers doesn't
        # help. Tests still reach FastAPI internals via the test
        # client.
        docs_url=None,
        redoc_url=None,
    )

    # ---- Routes ------------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        """Unauthenticated liveness check.

        For Tailscale liveness probes and external monitors. The
        rest of the API is token-protected; /health is the only
        public hole.
        """
        return {
            "status": "ok",
            "service": "avis-web",
            "version": app.version,
        }

    logger.info("Avis web dashboard app created (version=%s)", app.version)
    return app
