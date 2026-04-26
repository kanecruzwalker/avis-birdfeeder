"""Integration tests for tools.labeler.ui.routes via FastAPI TestClient.

Each test builds an isolated FastAPI app against temp pre_labels.jsonl /
verified_labels.jsonl files. We exercise the HTTP layer end-to-end
including auth, validation, status codes, and JSON shapes — but the
actual store behavior is covered by test_review_store.py, so these
tests focus on translation between HTTP and the store.

Token used everywhere: the `_token` and `_app` fixtures.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from tools.labeler.schema import PreLabel, PreLabelResponse
from tools.labeler.ui.server import create_app

_TOKEN = "test-token-1234567890abcdef"


def _make_pre(filename: str, species: str = "HOFI") -> PreLabel:
    return PreLabel(
        image_path=f"/test/{filename}",
        image_filename=filename,
        llm_response=PreLabelResponse(
            bird_visible=species != "NONE",
            species_code=species,
            confidence=0.9,
            reasoning="test reasoning",
        ),
        model_name="gemini-2.5-flash",
        prompt_version="v1.0",
        elapsed_seconds=2.5,
    )


def _write_pre_labels(path: Path, records: list[PreLabel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")


@pytest.fixture
def setup_app(tmp_path: Path, monkeypatch) -> Iterator[tuple[TestClient, Path, Path]]:
    """Build an isolated app + client and yield (client, pre_path, images_dir).

    Uses monkeypatch to set AVIS_WEB_TOKEN so the auth layer is happy.
    """
    monkeypatch.setenv("AVIS_WEB_TOKEN", _TOKEN)
    pre_path = tmp_path / "labels" / "pre_labels.jsonl"
    verified_path = tmp_path / "labels" / "verified_labels.jsonl"
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    # Default fixture has 3 pre-labels
    _write_pre_labels(
        pre_path,
        [
            _make_pre("a.png", "HOFI"),
            _make_pre("b.png", "HOFI"),
            _make_pre("c.png", "NONE"),
        ],
    )
    # Real PNG bytes for the image-serving test
    (images_dir / "a.png").write_bytes(b"fake png data")
    (images_dir / "b.png").write_bytes(b"fake png data b")
    (images_dir / "c.png").write_bytes(b"fake png data c")
    app = create_app(
        pre_labels_path=pre_path,
        verified_labels_path=verified_path,
        images_dir=images_dir,
    )
    client = TestClient(app)
    yield client, pre_path, images_dir


def _auth_get(client: TestClient, url: str) -> Response:
    return client.get(url, headers={"X-Avis-Token": _TOKEN})


def _auth_post(client: TestClient, url: str, json: dict) -> Response:
    return client.post(url, json=json, headers={"X-Avis-Token": _TOKEN})


# ── /health is open ──────────────────────────────────────────────────────────


class TestHealth:
    def test_health_no_auth(self, setup_app):
        client, _, _ = setup_app
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["total_pre_labels"] == 3
        assert body["total_verified"] == 0


# ── /api/species ──────────────────────────────────────────────────────────────


class TestSpeciesEndpoint:
    def test_returns_known_and_sentinels(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/species")
        assert r.status_code == 200
        data = r.json()
        assert len(data["known"]) == 20
        assert data["sentinels"] == ["NONE", "UNKNOWN", "OTHER"]
        assert "all" in data
        assert len(data["all"]) == 23


# ── /api/summary and /api/coverage ───────────────────────────────────────────


class TestSummary:
    def test_buckets_sorted_by_total_descending(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/summary")
        assert r.status_code == 200
        species = r.json()["species"]
        assert species[0]["species_code"] == "HOFI"
        assert species[0]["total"] == 2
        assert species[1]["species_code"] == "NONE"
        assert species[1]["total"] == 1

    def test_includes_overall_coverage(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/summary")
        cov = r.json()["coverage"]
        assert cov["total_pre_labels"] == 3
        assert cov["total_verified"] == 0
        assert cov["remaining"] == 3


class TestCoverage:
    def test_returns_coverage_dict(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/coverage")
        assert r.status_code == 200
        data = r.json()
        assert set(data.keys()) >= {"total_pre_labels", "total_verified", "remaining"}


# ── /api/next ────────────────────────────────────────────────────────────────


class TestNext:
    def test_returns_first_unverified(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/next")
        assert r.status_code == 200
        body = r.json()
        assert body["image_filename"] == "a.png"
        assert body["pre_label_species"] == "HOFI"
        assert body["image_url"] == "/image/a.png"
        # The optimistic-concurrency stamp the client must echo back
        assert "client_load_time" in body
        # Has not been verified yet
        assert body["already_verified_species"] is None

    def test_species_filter_routes_correctly(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/next?species=NONE")
        assert r.status_code == 200
        assert r.json()["image_filename"] == "c.png"
        assert r.json()["pre_label_species"] == "NONE"

    def test_empty_queue_returns_404(self, tmp_path, monkeypatch):
        """When all images have been verified, /api/next returns 404 with
        the queue_empty code so the client can show a 'done!' state.

        Builds its own isolated app rather than using setup_app since we
        need a single pre-label not three.
        """
        monkeypatch.setenv("AVIS_WEB_TOKEN", _TOKEN)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        pre_path = scratch / "labels" / "pre_labels.jsonl"
        _write_pre_labels(pre_path, [_make_pre("only.png", "HOFI")])
        verified_path = scratch / "labels" / "verified_labels.jsonl"
        images_dir = scratch / "images"
        images_dir.mkdir()
        app = create_app(
            pre_labels_path=pre_path,
            verified_labels_path=verified_path,
            images_dir=images_dir,
        )
        client = TestClient(app)

        # Verify it
        r = client.get("/api/next", headers={"X-Avis-Token": _TOKEN})
        load_time = r.json()["client_load_time"]
        client.post(
            "/api/verify",
            headers={"X-Avis-Token": _TOKEN},
            json={
                "image_filename": "only.png",
                "species_code": "HOFI",
                "client_load_time": load_time,
            },
        )

        # Now next should be empty
        r = client.get("/api/next", headers={"X-Avis-Token": _TOKEN})
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "queue_empty"

    def test_unknown_species_filter_returns_404(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/next?species=ZZZZ")
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "queue_empty"


# ── /api/review/{filename} ───────────────────────────────────────────────────


class TestReviewSpecific:
    def test_loads_named_image(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/review/c.png")
        assert r.status_code == 200
        assert r.json()["image_filename"] == "c.png"
        assert r.json()["pre_label_species"] == "NONE"

    def test_unknown_filename_returns_404(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/review/nope.png")
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "not_found"

    def test_already_verified_image_includes_existing(self, setup_app):
        """Re-opening a verified image should include the existing record
        so the UI can pre-populate the form."""
        client, _, _ = setup_app
        # First, verify a.png
        r = _auth_get(client, "/api/next")
        load_time = r.json()["client_load_time"]
        _auth_post(
            client,
            "/api/verify",
            {
                "image_filename": "a.png",
                "species_code": "HOFI",
                "client_load_time": load_time,
            },
        )
        # Now re-open it
        r = _auth_get(client, "/api/review/a.png")
        body = r.json()
        assert body["already_verified_species"] == "HOFI"
        assert body["already_verified_at"] is not None


# ── /api/verify happy paths ──────────────────────────────────────────────────


class TestVerifyHappyPath:
    def test_first_verify_returns_persisted_record(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/next")
        load_time = r.json()["client_load_time"]

        r = _auth_post(
            client,
            "/api/verify",
            {
                "image_filename": "a.png",
                "species_code": "HOFI",
                "client_load_time": load_time,
                "agreed_with_pre_label": True,
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["code"] == "ok"
        assert body["verified"]["image_filename"] == "a.png"
        assert body["verified"]["species_code"] == "HOFI"
        assert body["verified"]["agreed_with_pre_label"] is True

    def test_verify_with_other_sentinel(self, setup_app):
        """The CALT case: pre-labeled MOCH, reviewer corrects to OTHER+CALT."""
        client, _, _ = setup_app
        r = _auth_get(client, "/api/next")
        load_time = r.json()["client_load_time"]

        r = _auth_post(
            client,
            "/api/verify",
            {
                "image_filename": "a.png",
                "species_code": "OTHER",
                "other_species_code": "CALT",
                "client_load_time": load_time,
                "agreed_with_pre_label": False,
                "reviewer_notes": "California Towhee — out-of-vocab.",
            },
        )
        assert r.status_code == 200
        verified = r.json()["verified"]
        assert verified["species_code"] == "OTHER"
        assert verified["other_species_code"] == "CALT"

    def test_verify_with_uppercase_normalisation(self, setup_app):
        """Lowercase species codes should be accepted (schema normalises)."""
        client, _, _ = setup_app
        r = _auth_get(client, "/api/next")
        load_time = r.json()["client_load_time"]

        r = _auth_post(
            client,
            "/api/verify",
            {
                "image_filename": "a.png",
                "species_code": "hofi",  # lowercase
                "client_load_time": load_time,
            },
        )
        assert r.status_code == 200
        assert r.json()["verified"]["species_code"] == "HOFI"


# ── /api/verify validation errors ────────────────────────────────────────────


class TestVerifyValidation:
    def test_other_without_other_species_code_returns_422(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/next")
        load_time = r.json()["client_load_time"]

        r = _auth_post(
            client,
            "/api/verify",
            {
                "image_filename": "a.png",
                "species_code": "OTHER",
                # other_species_code missing
                "client_load_time": load_time,
            },
        )
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "validation"

    def test_invalid_species_code_returns_422(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/next")
        load_time = r.json()["client_load_time"]

        r = _auth_post(
            client,
            "/api/verify",
            {
                "image_filename": "a.png",
                "species_code": "ZZZZ",
                "client_load_time": load_time,
            },
        )
        assert r.status_code == 422

    def test_unknown_image_filename_returns_404(self, setup_app):
        client, _, _ = setup_app
        r = _auth_post(
            client,
            "/api/verify",
            {
                "image_filename": "nonexistent.png",
                "species_code": "HOFI",
                "client_load_time": datetime.now(UTC).isoformat(),
            },
        )
        assert r.status_code == 404


# ── /api/verify concurrency ─────────────────────────────────────────────────


class TestVerifyConcurrency:
    def test_stale_client_gets_409(self, setup_app):
        """Client A loads at T0, Client B loads+verifies at T1.
        Client A's verify with stale T0 → 409 with existing record."""
        client, _, _ = setup_app

        # Client A loads
        r = _auth_get(client, "/api/next")
        load_time_a = r.json()["client_load_time"]

        # Client B loads
        r = _auth_get(client, "/api/next")
        load_time_b = r.json()["client_load_time"]

        # Client B verifies first (later wall-clock time)
        import time

        time.sleep(0.01)
        _auth_post(
            client,
            "/api/verify",
            {
                "image_filename": "a.png",
                "species_code": "HOFI",
                "client_load_time": load_time_b,
            },
        )

        # Client A tries to verify with stale T0 → conflict
        r = _auth_post(
            client,
            "/api/verify",
            {
                "image_filename": "a.png",
                "species_code": "HOSP",  # disagrees!
                "client_load_time": load_time_a,
            },
        )
        assert r.status_code == 409
        body = r.json()["detail"]
        assert body["code"] == "conflict"
        assert body["existing"]["species_code"] == "HOFI"

    def test_force_overwrite_after_409(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/next")
        load_time_a = r.json()["client_load_time"]
        r = _auth_get(client, "/api/next")
        load_time_b = r.json()["client_load_time"]

        import time

        time.sleep(0.01)
        _auth_post(
            client,
            "/api/verify",
            {
                "image_filename": "a.png",
                "species_code": "HOFI",
                "client_load_time": load_time_b,
            },
        )

        # Client A retries with force_overwrite=True
        r = _auth_post(
            client,
            "/api/verify",
            {
                "image_filename": "a.png",
                "species_code": "HOSP",
                "client_load_time": load_time_a,
                "force_overwrite": True,
            },
        )
        assert r.status_code == 200
        assert r.json()["verified"]["species_code"] == "HOSP"


# ── /api/verified ────────────────────────────────────────────────────────────


class TestVerifiedList:
    def test_empty_when_nothing_verified(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/verified")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 0
        assert body["records"] == []

    def test_returns_records_after_verification(self, setup_app):
        client, _, _ = setup_app
        r = _auth_get(client, "/api/next")
        load_time = r.json()["client_load_time"]
        _auth_post(
            client,
            "/api/verify",
            {
                "image_filename": "a.png",
                "species_code": "HOFI",
                "client_load_time": load_time,
            },
        )
        r = _auth_get(client, "/api/verified")
        body = r.json()
        assert body["total"] == 1
        assert body["records"][0]["image_filename"] == "a.png"
        assert body["records"][0]["species_code"] == "HOFI"
        assert "verified_at" in body["records"][0]

    def test_species_filter_works(self, setup_app):
        client, _, _ = setup_app
        # Verify both a (HOFI) and c (NONE)
        for filename, species in [("a.png", "HOFI"), ("c.png", "NONE")]:
            r = _auth_get(client, f"/api/review/{filename}")
            load_time = r.json()["client_load_time"]
            _auth_post(
                client,
                "/api/verify",
                {
                    "image_filename": filename,
                    "species_code": species,
                    "client_load_time": load_time,
                },
            )

        r = _auth_get(client, "/api/verified?species=HOFI")
        body = r.json()
        assert body["total"] == 1
        assert body["records"][0]["species_code"] == "HOFI"


# ── /image/{filename} ────────────────────────────────────────────────────────


class TestImageServing:
    def test_serves_image_with_token(self, setup_app):
        client, _, _ = setup_app
        r = client.get(f"/image/a.png?token={_TOKEN}")
        assert r.status_code == 200
        assert r.content == b"fake png data"
        assert "Cache-Control" in r.headers

    def test_serves_image_with_header_token(self, setup_app):
        client, _, _ = setup_app
        r = client.get("/image/a.png", headers={"X-Avis-Token": _TOKEN})
        assert r.status_code == 200

    def test_image_requires_auth(self, setup_app):
        client, _, _ = setup_app
        r = client.get("/image/a.png")
        assert r.status_code == 401

    def test_missing_image_returns_404(self, setup_app):
        client, _, _ = setup_app
        r = client.get(f"/image/missing.png?token={_TOKEN}")
        assert r.status_code == 404
        assert r.json()["detail"]["code"] == "image_missing"

    def test_traversal_attempt_blocked(self, setup_app):
        """Path traversal via URL-encoded ../ should fail. Whether it 400s
        (store rejected the name) or 404s (FastAPI rejected the route)
        the client sees a non-200."""
        client, _, _ = setup_app
        r = client.get(f"/image/..%2Fetc%2Fpasswd?token={_TOKEN}")
        assert r.status_code in (400, 404)


# ── HTML page routes ─────────────────────────────────────────────────────────


class TestPageRoutes:
    """These tests just confirm the page routes exist and require auth.
    The actual rendered HTML is exercised by manual testing — there's no
    rendering value in unit-testing Jinja templates here.

    Note: the templates dir starts empty in fresh checkouts. Until the
    frontend chunk lands, the page routes will return 500 on a
    TemplateNotFound. We test the auth gate instead.
    """

    def test_index_requires_auth(self, setup_app):
        client, _, _ = setup_app
        r = client.get("/")
        assert r.status_code == 401

    def test_verified_page_requires_auth(self, setup_app):
        client, _, _ = setup_app
        r = client.get("/verified")
        assert r.status_code == 401

    def test_review_page_requires_auth(self, setup_app):
        client, _, _ = setup_app
        r = client.get("/review")
        assert r.status_code == 401
