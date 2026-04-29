"""Unit tests for src.web.routes.pages + the StaticFiles mount.

Covers:
    - GET / serves the dashboard HTML shell (no auth required)
    - GET / has the right content-type and contains expected SPA hooks
    - GET /static/styles.css serves CSS (no auth required)
    - GET /static/app.js serves JS (no auth required)
    - GET /static/views/live.js + recent.js serve as ES modules
    - GET /static/missing.css 404s
    - The shell is reachable when AVIS_WEB_TOKEN is unset (the SPA's
      first boot needs to render so the user can paste the token URL)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.web.app import create_app

TOKEN = "valid-token-1234567890"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)
    app = create_app(observations_path=tmp_path / "observations.jsonl")
    return TestClient(app)


@pytest.fixture
def client_without_token(tmp_path, monkeypatch):
    """The shell must render even when no token is configured —
    that's the operator's first-boot experience."""
    monkeypatch.delenv("AVIS_WEB_TOKEN", raising=False)
    app = create_app(observations_path=tmp_path / "observations.jsonl")
    return TestClient(app)


# ── Shell ────────────────────────────────────────────────────────────────────


class TestIndex:
    def test_get_root_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")

    def test_shell_contains_spa_hooks(self, client):
        r = client.get("/")
        body = r.text
        assert "<body" in body and 'data-view="live"' in body
        assert "/static/styles.css" in body
        assert "/static/app.js" in body

    def test_shell_has_all_view_sections(self, client):
        """All five views must be wired into the shell so the router
        finds them when their hash route activates. This catches the
        regression where a view module was added but the section
        wasn't (hash route would no-op silently)."""
        body = client.get("/").text
        for view in ("live", "recent", "timeline", "gallery", "detail"):
            assert f'data-view="{view}"' in body, f"missing section for {view}"

    def test_shell_navigation_includes_all_views(self, client):
        """Topbar links must exist for every primary view so users
        can switch between them without typing the hash."""
        body = client.get("/").text
        for view in ("live", "recent", "timeline", "gallery"):
            assert f'data-nav="{view}"' in body, f"missing nav link for {view}"

    def test_root_does_not_require_token(self, client):
        # No token header — shell still renders.
        r = client.get("/")
        assert r.status_code == 200

    def test_root_renders_without_configured_token(self, client_without_token):
        # Even with AVIS_WEB_TOKEN unset, the shell must render so the
        # operator can read the "no access token" message.
        r = client_without_token.get("/")
        assert r.status_code == 200

    def test_root_no_cache_header(self, client):
        r = client.get("/")
        assert "no-cache" in r.headers.get("cache-control", "").lower()


# ── Static assets ────────────────────────────────────────────────────────────


class TestStatic:
    def test_styles_served(self, client):
        r = client.get("/static/styles.css")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/css")
        # Sanity: the six theme tokens we ship.
        assert "warm-light" in r.text
        assert "warm-dark" in r.text
        assert "pollen-light" in r.text
        assert "mono-dark" in r.text

    def test_app_js_served(self, client):
        r = client.get("/static/app.js")
        assert r.status_code == 200
        # Browsers tolerate either application/javascript or text/javascript.
        ct = r.headers["content-type"]
        assert "javascript" in ct
        assert "mountLive" in r.text  # imported from views/live.js

    def test_view_modules_served(self, client):
        view_paths = (
            "/static/views/live.js",
            "/static/views/recent.js",
            "/static/views/timeline.js",
            "/static/views/gallery.js",
            "/static/views/detail.js",
        )
        for path in view_paths:
            r = client.get(path)
            assert r.status_code == 200, f"{path} not served"
            assert "javascript" in r.headers["content-type"]

    def test_static_does_not_require_token(self, client):
        # The bundle is public (no secrets); the API behind it is what
        # enforces auth.
        r = client.get("/static/styles.css")
        assert r.status_code == 200

    def test_missing_static_404s(self, client):
        r = client.get("/static/nope.css")
        assert r.status_code == 404


# ── Boundary: the shell does not leak API access ─────────────────────────────


class TestAuthBoundary:
    def test_api_still_requires_token_when_shell_is_public(self, client):
        # The shell at / is unauthenticated, but /api/* must still 401.
        r = client.get("/api/observations")
        assert r.status_code == 401
