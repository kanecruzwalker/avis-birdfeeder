"""Unit tests for src.web.routes.observations.

Covers:
    - /api/observations and /api/observations/{id} require auth
    - default dispatched=true filter
    - species filter (case-insensitive)
    - from / to date range
    - limit clamps at 500
    - cursor pagination round-trip
    - 404 on missing or malformed ID
    - JSON shape (id field present, fields mirror BirdObservation)
    - empty list when filters miss everything
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from src.data.schema import BirdObservation, ClassificationResult, Modality
from src.web.app import create_app
from src.web.observation_store import _id_for

TOKEN = "valid-token-1234567890"
HEADERS = {"X-Avis-Token": TOKEN}


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_obs(
    *,
    species_code: str = "HOFI",
    common_name: str = "House Finch",
    scientific_name: str = "Haemorhous mexicanus",
    fused_confidence: float = 0.9,
    dispatched: bool = True,
    timestamp: datetime | None = None,
    detection_mode: str = "fixed_crop",
) -> BirdObservation:
    ts = timestamp or datetime.now(UTC)
    return BirdObservation(
        species_code=species_code,
        common_name=common_name,
        scientific_name=scientific_name,
        fused_confidence=fused_confidence,
        dispatched=dispatched,
        visual_result=ClassificationResult(
            modality=Modality.VISUAL,
            species_code=species_code,
            common_name=common_name,
            scientific_name=scientific_name,
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


@pytest.fixture
def populated(tmp_path, monkeypatch):
    """Six observations: two species, mixed dispatched/suppressed,
    one minute apart so timestamps don't collide."""
    monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)
    base = datetime(2026, 4, 28, 12, 0, tzinfo=UTC)
    obs = [
        make_obs(species_code="HOFI", timestamp=base),
        make_obs(species_code="CALT", timestamp=base + timedelta(minutes=1)),
        make_obs(
            species_code="HOFI",
            timestamp=base + timedelta(minutes=2),
            dispatched=False,
            fused_confidence=0.4,
        ),
        make_obs(species_code="CALT", timestamp=base + timedelta(minutes=3)),
        make_obs(species_code="HOFI", timestamp=base + timedelta(minutes=4)),
        make_obs(
            species_code="CALT",
            timestamp=base + timedelta(minutes=5),
            detection_mode="yolo",
        ),
    ]
    path = tmp_path / "observations.jsonl"
    write_jsonl(path, obs)
    app = create_app(observations_path=path)
    return TestClient(app), obs


# ── Auth wall ────────────────────────────────────────────────────────────────


class TestAuth:
    def test_list_requires_token(self, populated):
        client, _ = populated
        assert client.get("/api/observations").status_code == 401

    def test_detail_requires_token(self, populated):
        client, obs = populated
        ident = _id_for(obs[0])
        assert client.get(f"/api/observations/{ident}").status_code == 401


# ── /api/observations list ───────────────────────────────────────────────────


class TestListDefaults:
    def test_default_returns_dispatched_only(self, populated):
        client, obs = populated
        r = client.get("/api/observations", headers=HEADERS)
        assert r.status_code == 200
        body = r.json()
        # Five of six are dispatched
        assert body["count"] == 5
        assert all(item["dispatched"] for item in body["items"])

    def test_response_shape(self, populated):
        client, _ = populated
        body = client.get("/api/observations", headers=HEADERS).json()
        assert set(body.keys()) == {"items", "next_cursor", "count"}
        first = body["items"][0]
        # id is the dashboard's addition; everything else mirrors BirdObservation
        assert "id" in first
        assert "species_code" in first
        assert "fused_confidence" in first
        assert "timestamp" in first
        assert "detection_mode" in first

    def test_newest_first(self, populated):
        client, _ = populated
        body = client.get("/api/observations", headers=HEADERS).json()
        timestamps = [item["timestamp"] for item in body["items"]]
        assert timestamps == sorted(timestamps, reverse=True)


# ── Filters ──────────────────────────────────────────────────────────────────


class TestFilters:
    def test_species_filter(self, populated):
        client, _ = populated
        body = client.get("/api/observations?species=HOFI", headers=HEADERS).json()
        assert body["count"] >= 1
        assert all(i["species_code"] == "HOFI" for i in body["items"])

    def test_species_filter_case_insensitive(self, populated):
        client, _ = populated
        upper = client.get("/api/observations?species=HOFI", headers=HEADERS).json()
        lower = client.get("/api/observations?species=hofi", headers=HEADERS).json()
        assert upper == lower

    def test_dispatched_false_returns_only_suppressed(self, populated):
        client, _ = populated
        body = client.get("/api/observations?dispatched=false", headers=HEADERS).json()
        assert body["count"] == 1
        assert all(not i["dispatched"] for i in body["items"])

    def test_from_filter_inclusive(self, populated):
        client, obs = populated
        cutoff = obs[3].timestamp
        body = client.get(
            f"/api/observations?from={quote(cutoff.isoformat())}",
            headers=HEADERS,
        ).json()
        assert body["count"] >= 1
        for item in body["items"]:
            assert datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")) >= cutoff

    def test_to_filter_inclusive(self, populated):
        client, obs = populated
        cutoff = obs[1].timestamp
        body = client.get(
            f"/api/observations?to={quote(cutoff.isoformat())}",
            headers=HEADERS,
        ).json()
        assert body["count"] >= 1
        for item in body["items"]:
            assert datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")) <= cutoff

    def test_filters_can_miss_everything(self, populated):
        client, _ = populated
        body = client.get("/api/observations?species=ZZZZ", headers=HEADERS).json()
        assert body["count"] == 0
        assert body["items"] == []
        assert body["next_cursor"] is None


# ── Pagination ───────────────────────────────────────────────────────────────


class TestPagination:
    def test_cursor_round_trip(self, populated):
        client, _ = populated
        page1 = client.get("/api/observations?limit=2", headers=HEADERS).json()
        assert page1["count"] == 2
        assert page1["next_cursor"] is not None

        page2 = client.get(
            f"/api/observations?limit=2&cursor={page1['next_cursor']}",
            headers=HEADERS,
        ).json()
        # Page 2 starts strictly older than page 1's tail
        assert page2["items"][0]["timestamp"] < page1["items"][-1]["timestamp"]

    def test_limit_clamped_to_max(self, populated):
        client, _ = populated
        # Way over 500 — should not 422, just clamp
        r = client.get("/api/observations?limit=10000", headers=HEADERS)
        assert r.status_code == 200

    def test_limit_below_one_rejected(self, populated):
        client, _ = populated
        r = client.get("/api/observations?limit=0", headers=HEADERS)
        assert r.status_code == 422


# ── /api/observations/{id} ───────────────────────────────────────────────────


class TestDetail:
    def test_detail_happy_path(self, populated):
        client, obs = populated
        target = obs[0]
        ident = _id_for(target)
        body = client.get(f"/api/observations/{ident}", headers=HEADERS).json()
        assert body["id"] == ident
        assert body["species_code"] == target.species_code
        assert body["fused_confidence"] == target.fused_confidence

    def test_detail_unknown_id_404(self, populated):
        client, _ = populated
        r = client.get("/api/observations/20210101T000000000000", headers=HEADERS)
        assert r.status_code == 404

    def test_detail_malformed_id_404(self, populated):
        client, _ = populated
        r = client.get("/api/observations/garbage", headers=HEADERS)
        assert r.status_code == 404

    def test_detail_returns_suppressed_records(self, populated):
        """The list endpoint defaults to dispatched-only, but the detail
        endpoint should return any record by ID — including suppressed
        ones — so the SPA's detail view works regardless of how the user
        got to it."""
        client, obs = populated
        suppressed = next(o for o in obs if not o.dispatched)
        ident = _id_for(suppressed)
        r = client.get(f"/api/observations/{ident}", headers=HEADERS)
        assert r.status_code == 200
        assert r.json()["dispatched"] is False
