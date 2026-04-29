"""Token authentication for the labeling-assistant review UI.

Single-user dev tool with optional Tailscale-exposure for phone use,
so we use a shared bearer token rather than per-user accounts. The
token is a randomly-generated string set in the .env via AVIS_WEB_TOKEN.

Token transport:
- HTTP header `X-Avis-Token: <token>` — preferred, used by all API calls
- Query parameter `?token=<token>` — used for image URLs (browsers can't
  set custom headers on `<img>` tags), and for initial page load via
  shared-link convenience.

Comparison uses `hmac.compare_digest` to prevent timing-based token
discovery. Token absence and token mismatch produce indistinguishable
401 responses.

Token-exempt routes:
- `/health` — for monitoring and Tailscale liveness checks

This module is intentionally small. Anything more sophisticated
(per-user accounts, role-based access) is out of scope per the
investigation doc.
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

    Distinct from the 401-on-bad-token path so the operator sees a clear
    server-side error instead of silently letting requests through.
    """


def get_configured_token() -> str:
    """Read AVIS_WEB_TOKEN from the environment.

    Raises AuthConfigError if the token is missing or empty. Called once
    at app startup so a misconfigured server fails fast instead of
    accepting unauthenticated requests.
    """
    raw = os.environ.get(_TOKEN_ENV_VAR, "").strip()
    if not raw:
        raise AuthConfigError(
            f"{_TOKEN_ENV_VAR} is not set. The labeling UI requires a token "
            f"for all API and image routes. Generate one with:\n"
            f"    python -c \"import secrets; print(secrets.token_urlsafe(32))\"\n"
            f"and add it to your .env as {_TOKEN_ENV_VAR}=<value>."
        )
    if len(raw) < 16:
        # 16 chars of token_urlsafe gives roughly 96 bits of entropy. Less
        # than that is too easy to brute-force across the public internet.
        raise AuthConfigError(
            f"{_TOKEN_ENV_VAR} is too short (need at least 16 characters)."
        )
    return raw


# ── FastAPI dependency ────────────────────────────────────────────────────────


def _extract_token(request: Request) -> str | None:
    """Pull the token from header first, then query string. Returns None if
    neither is present."""
    header_value = request.headers.get(_HEADER_NAME)
    if header_value:
        return header_value.strip()
    query_value = request.query_params.get(_QUERY_PARAM_NAME)
    if query_value:
        return query_value.strip()
    return None


def require_token(request: Request) -> None:
    """FastAPI dependency that 401s any request without a valid token.

    Use as a route dependency:
        @router.get("/api/something", dependencies=[Depends(require_token)])
        def something(): ...

    Or attach to an entire APIRouter at construction:
        router = APIRouter(dependencies=[Depends(require_token)])

    /health and other public routes simply omit this dependency.
    """
    expected = get_configured_token()
    provided = _extract_token(request)
    if provided is None:
        # Don't reveal whether the token is missing vs wrong — same 401 both
        # ways. Logged distinctly server-side for debugging.
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


# Re-export Depends so route modules can `from .auth import RequireToken`
# and use it as `[RequireToken]` in dependencies=. Slightly nicer than
# the verbose `Depends(require_token)` everywhere.
RequireToken = Depends(require_token)
