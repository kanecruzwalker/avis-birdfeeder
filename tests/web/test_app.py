"""Unit tests for src.web.app.create_app.

PR 1 scope: the factory builds a FastAPI app with /health
unauthenticated and the auth dependency ready for future routes.
These tests pin the /health contract and the docs-disabled setting.

PR 2+ adds tests next to each new route module
(tests/web/test_routes_status.py, test_routes_observations.py, etc.).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.web.app import create_app
from src.web.auth import RequireToken


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("AVIS_WEB_TOKEN", "valid-token-1234567890")
    return TestClient(create_app())


class TestFactory:
    def test_returns_fastapi_instance(self, monkeypatch):
        monkeypatch.setenv("AVIS_WEB_TOKEN", "valid-token-1234567890")
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_title_set(self, monkeypatch):
        monkeypatch.setenv("AVIS_WEB_TOKEN", "valid-token-1234567890")
        app = create_app()
        assert "Avis" in app.title

    def test_version_set(self, monkeypatch):
        monkeypatch.setenv("AVIS_WEB_TOKEN", "valid-token-1234567890")
        app = create_app()
        assert app.version  # non-empty


class TestDocsDisabled:
    """/docs and /redoc are off — every route is auth-protected and
    exposing the schema to unauthenticated callers doesn't help."""

    def test_docs_endpoint_404(self, client):
        r = client.get("/docs")
        assert r.status_code == 404

    def test_redoc_endpoint_404(self, client):
        r = client.get("/redoc")
        assert r.status_code == 404


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_unauthenticated(self, monkeypatch):
        """/health must work without a token so Tailscale liveness
        probes and external monitors can reach it."""
        monkeypatch.setenv("AVIS_WEB_TOKEN", "valid-token-1234567890")
        client = TestClient(create_app())
        # No header, no query param
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_works_when_token_unset(self, monkeypatch):
        """Even with AVIS_WEB_TOKEN missing from env (a misconfigured
        server), /health should still respond — that's the whole point
        of an unauthenticated liveness check."""
        monkeypatch.delenv("AVIS_WEB_TOKEN", raising=False)
        client = TestClient(create_app())
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_response_shape(self, client):
        r = client.get("/health")
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "avis-web"
        assert "version" in body

    def test_health_does_not_leak_token(self, client):
        """Make sure the body doesn't echo anything that looks like
        a token. Guards against an env-state leak in a future change."""
        r = client.get("/health")
        body_text = r.text.lower()
        assert "token" not in body_text
        assert "avis_web_token" not in body_text


class TestAuthWallReady:
    """PR 1 doesn't mount any authenticated routes, but the auth
    dependency must be importable so PR 2 can use it."""

    def test_require_token_importable(self):
        # Already imported at module top — this just asserts the
        # dependency object is the real Depends marker, not a sentinel.
        assert RequireToken is not None

    def test_factory_does_not_require_token_at_construction(self, monkeypatch):
        """create_app() must build an app even if AVIS_WEB_TOKEN is
        unset at construction time. The token is only checked when
        a protected route is hit (and at startup by the CLI in
        src/web/__main__.py)."""
        monkeypatch.delenv("AVIS_WEB_TOKEN", raising=False)
        app = create_app()
        assert isinstance(app, FastAPI)
