"""FastAPI app assembly for the labeling-assistant review UI.

The `create_app` factory takes explicit configuration so tests can build
isolated apps with temp directories and stub stores. The `__main__`
module wraps it for the production-ish CLI path with .env loading.

Layout served by the app:
    GET  /              — landing page (group-by-species)
    GET  /review        — review screen (one image at a time)
    GET  /verified      — verified-labels list view
    GET  /health        — unauthenticated liveness check

    GET  /api/species         — allowed codes + sentinels
    GET  /api/summary         — group-by-species data
    GET  /api/coverage        — overall coverage numbers
    GET  /api/next            — next unverified pre-label
    GET  /api/review/{f}      — specific pre-label by filename
    POST /api/verify          — submit a verification
    GET  /api/verified        — list verified labels

    GET  /image/{filename}    — serve a capture image (auth-required)

    GET  /static/*            — JS/CSS for the SPA
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .review_store import ReviewStore
from .routes import api_router, image_router, page_router

logger = logging.getLogger(__name__)


# ── Default paths (relative to repo root) ─────────────────────────────────────

# These are the conventional locations for a fresh checkout. The CLI lets
# the operator override any of them, so a fork that wants different paths
# (or runs the UI from a non-root cwd) doesn't need to touch this file.
DEFAULT_PRE_LABELS_PATH = Path("data/labels/pre_labels.jsonl")
DEFAULT_VERIFIED_LABELS_PATH = Path("data/labels/verified_labels.jsonl")
DEFAULT_IMAGES_DIR = Path("data/captures/images")


# ── App factory ───────────────────────────────────────────────────────────────


def create_app(
    pre_labels_path: Path | None = None,
    verified_labels_path: Path | None = None,
    images_dir: Path | None = None,
    static_dir: Path | None = None,
    templates_dir: Path | None = None,
    autoload: bool = True,
) -> FastAPI:
    """Build and configure a FastAPI app for the review UI.

    Args:
        pre_labels_path: where to read pre-labels from. Default: data/labels/pre_labels.jsonl.
        verified_labels_path: where to write verifications. Default: data/labels/verified_labels.jsonl.
        images_dir: directory of capture PNGs. Default: data/captures/images/.
        static_dir: where the JS/CSS bundle lives. Default: tools/labeler/ui/static/.
        templates_dir: Jinja2 template dir. Default: tools/labeler/ui/templates/.
        autoload: if True, calls store.load() during construction so the
            app is ready to serve traffic immediately. Tests pass False
            to control timing.

    Returns:
        A configured FastAPI app. Auth (AVIS_WEB_TOKEN) is enforced on
        all routes except /health and /static/*; the token must be set
        in the environment by the time a request arrives, or routes will
        return a 500 from the auth dependency.
    """
    pre_labels_path = pre_labels_path or DEFAULT_PRE_LABELS_PATH
    verified_labels_path = verified_labels_path or DEFAULT_VERIFIED_LABELS_PATH
    images_dir = images_dir or DEFAULT_IMAGES_DIR

    # Default static + templates dirs live next to this file so the app
    # works regardless of cwd.
    here = Path(__file__).resolve().parent
    static_dir = static_dir or (here / "static")
    templates_dir = templates_dir or (here / "templates")

    app = FastAPI(
        title="Avis labeling-assistant review UI",
        description=(
            "Internal dev tool for verifying Layer 1 pre-labels. Single "
            "user, token-authenticated, not for public deployment."
        ),
        version="0.1.0",
        # No automatic /docs in prod since this is auth-protected. Tests
        # can still reach the FastAPI internals via the test client.
        docs_url=None,
        redoc_url=None,
    )

    # Build the store. We construct it eagerly so auth-related errors
    # (missing pre_labels.jsonl, etc.) surface at startup instead of on
    # the first request.
    store = ReviewStore(
        pre_labels_path=pre_labels_path,
        verified_labels_path=verified_labels_path,
        images_dir=images_dir,
    )
    if autoload:
        store.load()
    app.state.review_store = store

    # Templates and static files. Both directories may not exist yet
    # during early-stage development; we create them lazily so the
    # server can start even before the frontend bundle has been written.
    templates_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    app.state.templates = Jinja2Templates(directory=str(templates_dir))
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Wire up routers
    app.include_router(api_router)
    app.include_router(image_router)
    app.include_router(page_router)

    # Public endpoints (no auth)
    @app.get("/health", tags=["public"])
    def health() -> dict:
        """Unauthenticated liveness check.

        Returns store coverage as a side effect so a monitor can confirm
        the JSONL files are readable without making an authenticated call.
        Does NOT expose any verified-label content.
        """
        try:
            cov = store.coverage()
            return {
                "status": "ok",
                "total_pre_labels": cov["total_pre_labels"],
                "total_verified": cov["total_verified"],
            }
        except Exception as exc:  # noqa: BLE001 — health check must not throw
            logger.exception("Health check failed")
            return {"status": "degraded", "error": str(exc)}

    logger.info(
        "Review UI app created | pre_labels=%s verified=%s images=%s",
        pre_labels_path,
        verified_labels_path,
        images_dir,
    )
    return app
