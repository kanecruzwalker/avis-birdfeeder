"""Unit tests for tools.labeler.ui.auth.

Covers:
- get_configured_token() — present, missing, too-short
- require_token() — header path, query path, missing, wrong
- /health endpoint exemption (verified via the actual app in test_routes.py)
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tools.labeler.ui.auth import (
    AuthConfigError,
    RequireToken,
    get_configured_token,
)

# ── get_configured_token ─────────────────────────────────────────────────────


class TestGetConfiguredToken:
    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("AVIS_WEB_TOKEN", "abcdefghij1234567890")
        assert get_configured_token() == "abcdefghij1234567890"

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("AVIS_WEB_TOKEN", "   abcdefghij1234567890  ")
        assert get_configured_token() == "abcdefghij1234567890"

    def test_missing_raises(self, monkeypatch):
        monkeypatch.delenv("AVIS_WEB_TOKEN", raising=False)
        with pytest.raises(AuthConfigError, match="not set"):
            get_configured_token()

    def test_empty_raises(self, monkeypatch):
        monkeypatch.setenv("AVIS_WEB_TOKEN", "")
        with pytest.raises(AuthConfigError, match="not set"):
            get_configured_token()

    def test_whitespace_only_raises(self, monkeypatch):
        monkeypatch.setenv("AVIS_WEB_TOKEN", "   ")
        with pytest.raises(AuthConfigError, match="not set"):
            get_configured_token()

    def test_too_short_raises(self, monkeypatch):
        monkeypatch.setenv("AVIS_WEB_TOKEN", "short")  # 5 chars
        with pytest.raises(AuthConfigError, match="too short"):
            get_configured_token()


# ── require_token via a tiny FastAPI app ─────────────────────────────────────


def _build_test_app() -> FastAPI:
    """A minimal FastAPI app exposing one protected endpoint plus /health,
    so we can exercise the auth dependency end-to-end.

    Uses /protected and /health rather than the full review-UI routes
    because we want tight, focused tests of just the auth path.
    """
    app = FastAPI()

    @app.get("/protected", dependencies=[RequireToken])
    def protected():
        return {"status": "ok"}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


class TestRequireToken:
    @pytest.fixture(autouse=True)
    def _set_token(self, monkeypatch):
        monkeypatch.setenv("AVIS_WEB_TOKEN", "valid-token-1234567890")

    def test_no_token_returns_401(self):
        client = TestClient(_build_test_app())
        r = client.get("/protected")
        assert r.status_code == 401
        assert "Missing" in r.json()["detail"]

    def test_wrong_token_in_header_returns_401(self):
        client = TestClient(_build_test_app())
        r = client.get("/protected", headers={"X-Avis-Token": "wrong"})
        assert r.status_code == 401
        assert "Invalid" in r.json()["detail"]

    def test_correct_token_in_header_returns_200(self):
        client = TestClient(_build_test_app())
        r = client.get("/protected", headers={"X-Avis-Token": "valid-token-1234567890"})
        assert r.status_code == 200

    def test_correct_token_in_query_returns_200(self):
        client = TestClient(_build_test_app())
        r = client.get("/protected?token=valid-token-1234567890")
        assert r.status_code == 200

    def test_header_takes_precedence_over_query(self):
        """If both are present and disagree, header wins (per RFC convention).
        Should still pass when the header is correct."""
        client = TestClient(_build_test_app())
        r = client.get(
            "/protected?token=wrong",
            headers={"X-Avis-Token": "valid-token-1234567890"},
        )
        assert r.status_code == 200

    def test_health_does_not_require_token(self):
        client = TestClient(_build_test_app())
        r = client.get("/health")
        assert r.status_code == 200

    def test_token_with_surrounding_whitespace_in_header_accepted(self):
        """Some clients send tokens with stray whitespace. Strip them."""
        client = TestClient(_build_test_app())
        r = client.get("/protected", headers={"X-Avis-Token": "  valid-token-1234567890  "})
        assert r.status_code == 200

    def test_returns_www_authenticate_header_on_401(self):
        """Standards-compliance — 401 should include WWW-Authenticate so
        clients know what scheme to use."""
        client = TestClient(_build_test_app())
        r = client.get("/protected")
        assert r.status_code == 401
        assert r.headers.get("WWW-Authenticate") == "Bearer"

    def test_misconfigured_server_returns_5xx(self, monkeypatch):
        """If AVIS_WEB_TOKEN isn't set when a request arrives, the auth
        dependency raises AuthConfigError. FastAPI surfaces unhandled
        exceptions as 500."""
        monkeypatch.delenv("AVIS_WEB_TOKEN", raising=False)
        client = TestClient(_build_test_app(), raise_server_exceptions=False)
        r = client.get("/protected", headers={"X-Avis-Token": "anything"})
        assert r.status_code == 500
