"""Token authentication for the labeling-assistant review UI.

Thin re-export of :mod:`src.util.web_auth`, which is shared with
:mod:`src.web.auth`. Both surfaces use the same ``AVIS_WEB_TOKEN``
secret and the same header / query-param transport, so the logic
lives in one place.

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
