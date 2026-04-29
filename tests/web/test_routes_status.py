"""Unit tests for src.web.routes.status.

Covers:
    - /health works without auth (already pinned in test_app.py;
      duplicated here for one-stop reading)
    - /api/status requires a token
    - response shape
    - agent_status heuristic at all three thresholds
    - uptime is non-negative
    - empty observations file produces nulls cleanly
    - current_mode reflects the latest record's detection_mode
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.data.schema import BirdObservation, ClassificationResult, Modality
from src.web.app import create_app

TOKEN = "valid-token-1234567890"


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_obs(
    *,
    species_code: str = "HOFI",
    fused_confidence: float = 0.9,
    dispatched: bool = True,
    timestamp: datetime | None = None,
    detection_mode: str = "fixed_crop",
) -> BirdObservation:
    ts = timestamp or datetime.now(UTC)
    return BirdObservation(
        species_code=species_code,
        common_name="House Finch",
        scientific_name="Haemorhous mexicanus",
        fused_confidence=fused_confidence,
        dispatched=dispatched,
        visual_result=ClassificationResult(
            modality=Modality.VISUAL,
            species_code=species_code,
            common_name="House Finch",
            scientific_name="Haemorhous mexicanus",
            confidence=fused_confidence,
            timestamp=ts,
        ),
        timestamp=ts,
        detection_mode=detection_mode,
    )


def write_jsonl(path: Path, observations: list[BirdObservation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for o in observations:
            fh.write(json.dumps(o.model_dump(mode="json")) + "\n")


def set_mtime(path: Path, seconds_ago: float) -> None:
    """Backdate a file's mtime by ``seconds_ago``."""
    target = time.time() - seconds_ago
    os.utime(path, (target, target))


@pytest.fixture
def empty_app(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)
    return create_app(observations_path=tmp_path / "missing.jsonl")


@pytest.fixture
def populated_app(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)
    obs = [
        make_obs(timestamp=datetime(2026, 4, 28, 12, 0, tzinfo=UTC)),
        make_obs(
            timestamp=datetime(2026, 4, 28, 12, 1, tzinfo=UTC),
            dispatched=False,
            fused_confidence=0.3,
        ),
        make_obs(
            timestamp=datetime(2026, 4, 28, 12, 2, tzinfo=UTC),
            detection_mode="yolo",
        ),
    ]
    path = tmp_path / "observations.jsonl"
    write_jsonl(path, obs)
    return create_app(observations_path=path), path


# ── /health ──────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_no_auth(self, empty_app):
        client = TestClient(empty_app)
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "avis-web"
        assert "version" in body


# ── /api/status auth ─────────────────────────────────────────────────────────


class TestStatusAuth:
    def test_missing_token_401(self, empty_app):
        client = TestClient(empty_app)
        r = client.get("/api/status")
        assert r.status_code == 401

    def test_wrong_token_401(self, empty_app):
        client = TestClient(empty_app)
        r = client.get("/api/status", headers={"X-Avis-Token": "nope"})
        assert r.status_code == 401

    def test_token_in_header(self, empty_app):
        client = TestClient(empty_app)
        r = client.get("/api/status", headers={"X-Avis-Token": TOKEN})
        assert r.status_code == 200

    def test_token_in_query(self, empty_app):
        client = TestClient(empty_app)
        r = client.get(f"/api/status?token={TOKEN}")
        assert r.status_code == 200


# ── /api/status shape ────────────────────────────────────────────────────────


class TestStatusShape:
    def test_required_fields_present(self, populated_app):
        app, _ = populated_app
        r = TestClient(app).get("/api/status", headers={"X-Avis-Token": TOKEN})
        assert r.status_code == 200
        body = r.json()
        for field in (
            "service",
            "version",
            "uptime_seconds",
            "total_observations",
            "total_dispatched",
            "last_observation_at",
            "last_dispatched_at",
            "current_mode",
            "agent_status",
        ):
            assert field in body

    def test_uptime_non_negative(self, populated_app):
        app, _ = populated_app
        r = TestClient(app).get("/api/status", headers={"X-Avis-Token": TOKEN})
        assert r.json()["uptime_seconds"] >= 0

    def test_counts_match_file(self, populated_app):
        app, _ = populated_app
        body = TestClient(app).get("/api/status", headers={"X-Avis-Token": TOKEN}).json()
        assert body["total_observations"] == 3
        assert body["total_dispatched"] == 2

    def test_current_mode_reflects_latest(self, populated_app):
        app, _ = populated_app
        body = TestClient(app).get("/api/status", headers={"X-Avis-Token": TOKEN}).json()
        # Newest record was detection_mode="yolo"
        assert body["current_mode"] == "yolo"

    def test_empty_file_produces_nulls(self, empty_app):
        body = TestClient(empty_app).get("/api/status", headers={"X-Avis-Token": TOKEN}).json()
        assert body["total_observations"] == 0
        assert body["total_dispatched"] == 0
        assert body["last_observation_at"] is None
        assert body["last_dispatched_at"] is None
        assert body["current_mode"] is None


# ── agent_status heuristic ───────────────────────────────────────────────────


class TestAgentStatusHeuristic:
    def test_stale_when_file_missing(self, empty_app):
        body = TestClient(empty_app).get("/api/status", headers={"X-Avis-Token": TOKEN}).json()
        assert body["agent_status"] == "stale"

    def test_live_when_recent(self, populated_app):
        app, path = populated_app
        # File was just written by the fixture, so mtime is current.
        body = TestClient(app).get("/api/status", headers={"X-Avis-Token": TOKEN}).json()
        assert body["agent_status"] == "live"

    def test_idle_between_one_and_ten_minutes(self, populated_app):
        app, path = populated_app
        set_mtime(path, seconds_ago=120)  # 2 minutes ago
        body = TestClient(app).get("/api/status", headers={"X-Avis-Token": TOKEN}).json()
        assert body["agent_status"] == "idle"

    def test_stale_after_ten_minutes(self, populated_app):
        app, path = populated_app
        set_mtime(path, seconds_ago=15 * 60)  # 15 minutes ago
        body = TestClient(app).get("/api/status", headers={"X-Avis-Token": TOKEN}).json()
        assert body["agent_status"] == "stale"
