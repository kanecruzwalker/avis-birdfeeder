"""Token authentication for the Avis web dashboard.

Thin re-export of :mod:`src.util.web_auth`, which is shared with
``tools.labeler.ui.auth``. Keeping a module at this path means
existing call sites (``from src.web.auth import RequireToken``)
keep working; the actual logic lives in one place.

See the shared module for token-transport conventions, configuration
errors, and FastAPI integration details.
"""

from __future__ import annotations

from src.util.web_auth import (
    AuthConfigError,
    RequireToken,
    get_configured_token,
    require_token,
)

__all__ = [
    "AuthConfigError",
    "RequireToken",
    "get_configured_token",
    "require_token",
]
