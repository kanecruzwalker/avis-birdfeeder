"""Shared bearer-token auth for the Avis web surfaces.

Both the labeling-assistant review UI (``tools.labeler.ui``) and
the Pi-hosted dashboard (``src.web``) sit behind the same single
shared token (``AVIS_WEB_TOKEN``). This module owns the auth logic;
the two surfaces re-export the public names.

Token transport
---------------
- HTTP header ``X-Avis-Token: <token>`` — preferred, used by all
  JSON API calls.
- Query parameter ``?token=<token>`` — required for browser-loaded
  resources that can't set custom headers (e.g. ``<img>`` tags
  fetching MJPEG streams or PNG images), and for shared-link
  convenience on initial page load.

Comparison uses :func:`hmac.compare_digest` to prevent timing
attacks. Token absence and token mismatch produce indistinguishable
401 responses; the server log distinguishes them for debugging.

The unauthenticated escape hatch is per-route: callers attach
``RequireToken`` (or ``Depends(require_token)``) only to routes
that should be protected. ``/health`` is conventionally exempt.

Per-user accounts and role-based access are out of scope; one
shared token is enough for the small set of trusted users.
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import Depends, HTTPException, Request, status

logger = logging.getLogger(__name__)


# ── Token resolution ──────────────────────────────────────────────────────────

_TOKEN_ENV_VAR = "AVIS_WEB_TOKEN"
_HEADER_NAME = "X-Avis-Token"
_QUERY_PARAM_NAME = "token"
_MIN_TOKEN_LENGTH = 16


class AuthConfigError(RuntimeError):
    """Raised at startup if ``AVIS_WEB_TOKEN`` is missing or invalid.

    Distinct from the 401-on-bad-token path so a misconfigured
    server fails loudly instead of silently letting requests
    through.
    """


def get_configured_token() -> str:
    """Read ``AVIS_WEB_TOKEN`` from the environment.

    Called both at startup (fail-fast on missing token) and on
    every protected request (env is the source of truth — if the
    operator unsets the token at runtime, requests start failing
    immediately rather than running on a stale cache).

    Raises:
        AuthConfigError: token unset, empty/whitespace, or shorter
            than 16 characters (~96 bits at base64).
    """
    raw = os.environ.get(_TOKEN_ENV_VAR, "").strip()
    if not raw:
        raise AuthConfigError(
            f"{_TOKEN_ENV_VAR} is not set. The Avis web surfaces require a token "
            f"for all API and image routes. Generate one with:\n"
            f'    python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
            f"and add it to your .env as {_TOKEN_ENV_VAR}=<value>."
        )
    if len(raw) < _MIN_TOKEN_LENGTH:
        raise AuthConfigError(
            f"{_TOKEN_ENV_VAR} is too short (need at least {_MIN_TOKEN_LENGTH} characters)."
        )
    return raw


# ── FastAPI dependency ────────────────────────────────────────────────────────


def _extract_token(request: Request) -> str | None:
    """Header first, then query string. ``None`` if neither set."""
    header_value = request.headers.get(_HEADER_NAME)
    if header_value:
        return header_value.strip()
    query_value = request.query_params.get(_QUERY_PARAM_NAME)
    if query_value:
        return query_value.strip()
    return None


def require_token(request: Request) -> None:
    """FastAPI dependency: 401 any request without a valid token.

    Use as a route-level dependency::

        @router.get("/api/something", dependencies=[Depends(require_token)])
        def something(): ...

    or attach to an entire APIRouter at construction::

        router = APIRouter(dependencies=[Depends(require_token)])

    Routes that should be public simply omit the dependency.
    """
    expected = get_configured_token()
    provided = _extract_token(request)
    if provided is None:
        # Don't reveal missing-vs-wrong: same 401 either way.
        # The path is logged at debug for operator triage.
        logger.debug("Auth: no token provided on %s", request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not hmac.compare_digest(provided, expected):
        logger.warning(
            "Auth: invalid token on %s (provided length=%d)",
            request.url.path,
            len(provided),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Re-export ``Depends(require_token)`` so callers can write
# ``[RequireToken]`` in route ``dependencies=`` rather than the
# verbose ``Depends(require_token)`` everywhere.
RequireToken = Depends(require_token)


__all__ = [
    "AuthConfigError",
    "RequireToken",
    "get_configured_token",
    "require_token",
]
