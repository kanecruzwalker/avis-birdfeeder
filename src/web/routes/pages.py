"""HTML shell + static asset routing for the dashboard SPA.

The dashboard is a single-page vanilla-JS app. ``GET /`` returns the
HTML shell, JS bundle, and CSS; everything else (live preview,
observation list, image variants, chat) is fetched from the
already-mounted ``/api/*`` routers.

Why the shell is unauthenticated
--------------------------------
The HTML/JS/CSS contain no secrets. The token wall sits in front of
``/api/*`` and the image / stream endpoints. The user's first visit
arrives with ``?token=<X>`` in the URL; the SPA reads it, persists
to ``localStorage``, then replaces the URL with the bare path
(``history.replaceState``) so the token doesn't linger in the
browser history bar or get accidentally screenshot. Subsequent
fetches add ``X-Avis-Token``; the MJPEG ``<img>`` is the one place
the token has to ride the URL (browsers won't set custom headers
on image requests).

Why a dedicated module
----------------------
Static-asset mounting and the root route both need to live somewhere
the factory can wire up. Putting them next to the API routers keeps
``src/web/app.py`` purely about composition.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

# Static assets (HTML, CSS, JS, fonts, view modules) live next to this
# package. The factory resolves the absolute path at startup so test
# clients work regardless of cwd.
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"


router = APIRouter(tags=["pages"])


@router.get("/", response_class=FileResponse)
def index() -> FileResponse:
    """Serve the dashboard HTML shell.

    No auth — the shell carries no secrets, and the SPA's first
    boot needs to read ``?token=`` from the URL before it can
    authenticate any API call.
    """
    return FileResponse(
        _INDEX_HTML,
        media_type="text/html",
        headers={
            # The shell can be cached briefly but the JS bundle has
            # no version hash yet — keep it short so a deployed
            # update lands without waiting for cache eviction.
            "Cache-Control": "no-cache",
        },
    )


# Re-exported for the factory's StaticFiles mount.
STATIC_DIR = _STATIC_DIR


__all__ = ["STATIC_DIR", "router"]
