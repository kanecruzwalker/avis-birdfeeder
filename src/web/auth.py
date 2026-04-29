"""Token authentication for the Avis web dashboard.

Copy of ``tools/labeler/ui/auth.py`` per the design decision in
``docs/investigations/web-dashboard-2026-04-28.md``: the dashboard
uses the same auth as the labeler UI.

Both read the same ``AVIS_WEB_TOKEN`` env var, so there's only one
secret to manage. The file is duplicated rather than imported from
``tools.labeler.ui.auth`` so ``src/web/`` doesn't pull anything in
from ``tools/`` (same rule as the rest of ``src/``).

Token transport:
    - HTTP header ``X-Avis-Token: <token>`` — preferred, used by all
      JSON API calls from the SPA.
    - Query parameter ``?token=<token>`` — required for the MJPEG
      stream (browsers can't set custom headers on ``<img>`` tags) and
      for shared-link convenience on initial page load.

Comparison uses :func:`hmac.compare_digest` to prevent timing-based
token discovery. Token absence and token mismatch produce
indistinguishable 401 responses.

Token-exempt routes:
    - ``/health`` — for monitoring and Tailscale liveness checks.

Per-user accounts and role-based access are out of scope per the
investigation doc — one shared token is enough for Kane, Dan, and a
few invited friends.
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


class AuthConfigError(RuntimeError):
    """Raised at app startup if AVIS_WEB_TOKEN is missing or invalid.

    Distinct from the 401-on-bad-token path so the operator sees a
    clear server-side error instead of silently letting requests
    through.
    """


def get_configured_token() -> str:
    """Read AVIS_WEB_TOKEN from the environment.

    Raises :class:`AuthConfigError` if the token is missing or empty.
    Called at startup so a misconfigured server fails fast instead of
    accepting unauthenticated requests.
    """
    raw = os.environ.get(_TOKEN_ENV_VAR, "").strip()
    if not raw:
        raise AuthConfigError(
            f"{_TOKEN_ENV_VAR} is not set. The web dashboard requires a token "
            f"for all API and image routes. Generate one with:\n"
            f'    python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
            f"and add it to your .env as {_TOKEN_ENV_VAR}=<value>."
        )
    if len(raw) < 16:
        # 16 chars of token_urlsafe gives roughly 96 bits of entropy.
        # Less than that is too easy to brute-force across the public
        # internet (the dashboard may be exposed on Tailscale, but the
        # token also rides query strings on shared demo links).
        raise AuthConfigError(f"{_TOKEN_ENV_VAR} is too short (need at least 16 characters).")
    return raw


# ── FastAPI dependency ────────────────────────────────────────────────────────


def _extract_token(request: Request) -> str | None:
    """Pull the token from header first, then query string. Returns
    ``None`` if neither is present."""
    header_value = request.headers.get(_HEADER_NAME)
    if header_value:
        return header_value.strip()
    query_value = request.query_params.get(_QUERY_PARAM_NAME)
    if query_value:
        return query_value.strip()
    return None


def require_token(request: Request) -> None:
    """FastAPI dependency that 401s any request without a valid token.

    Use as a route dependency::

        @router.get("/api/something", dependencies=[Depends(require_token)])
        def something(): ...

    Or attach to an entire APIRouter at construction::

        router = APIRouter(dependencies=[Depends(require_token)])

    ``/health`` and other public routes simply omit this dependency.
    """
    expected = get_configured_token()
    provided = _extract_token(request)
    if provided is None:
        # Don't reveal whether the token is missing vs wrong — same 401
        # both ways. Logged distinctly server-side for debugging.
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


# Re-export Depends so route modules can ``from .auth import RequireToken``
# and use it as ``[RequireToken]`` in ``dependencies=``. Slightly nicer
# than the verbose ``Depends(require_token)`` everywhere.
RequireToken = Depends(require_token)
