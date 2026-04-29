"""Unit tests for src.web.routes.images.

Covers:
    - all three variants (cropped, full, annotated) require auth
    - 200 + correct bytes when files exist on disk
    - 404 when the variant's path field is None on the observation
    - 404 when the path is set but the file is missing on disk
    - 404 when the observation ID is unknown / malformed
    - 422 when the variant is not one of cropped/full/annotated
    - annotated path derivation (swap _full → _annotated stem suffix)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.data.schema import BirdObservation, ClassificationResult, Modality
from src.web.app import create_app
from src.web.observation_store import _id_for

TOKEN = "valid-token-1234567890"
HEADERS = {"X-Avis-Token": TOKEN}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_obs(
    *,
    image_path: Path | None = None,
    image_path_full: Path | None = None,
    detection_mode: str = "fixed_crop",
    timestamp: datetime,
) -> BirdObservation:
    return BirdObservation(
        species_code="HOFI",
        common_name="House Finch",
        scientific_name="Haemorhous mexicanus",
        fused_confidence=0.9,
        dispatched=True,
        visual_result=ClassificationResult(
            modality=Modality.VISUAL,
            species_code="HOFI",
            common_name="House Finch",
            scientific_name="Haemorhous mexicanus",
            confidence=0.9,
            timestamp=timestamp,
        ),
        timestamp=timestamp,
        image_path=str(image_path) if image_path else None,
        image_path_full=str(image_path_full) if image_path_full else None,
        detection_mode=detection_mode,
    )


def _write_jsonl(path: Path, observations: list[BirdObservation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for o in observations:
            fh.write(json.dumps(o.model_dump(mode="json")) + "\n")


def _write_png(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def populated(tmp_path, monkeypatch):
    """Three observations spanning the path/disk-presence matrix:

    - ``all_three``: cropped + full + annotated all exist on disk
      (yolo mode with a detection box).
    - ``cropped_only``: image_path set + on disk; image_path_full is
      None on the record (full + annotated should 404).
    - ``ghost``: both path fields set but neither file exists on disk
      (every variant should 404 with a "missing on disk" detail).
    """
    monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)

    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    base = datetime(2026, 4, 28, 12, 0, tzinfo=UTC)

    # --- all_three: full set on disk ---
    cropped_a = images_dir / "20260428_120000_000000_cam0.png"
    full_a = images_dir / "20260428_120000_000000_cam0_full.png"
    annotated_a = images_dir / "20260428_120000_000000_cam0_annotated.png"
    _write_png(cropped_a, b"cropped_a_bytes")
    _write_png(full_a, b"full_a_bytes")
    _write_png(annotated_a, b"annotated_a_bytes")
    obs_all = _make_obs(
        image_path=cropped_a,
        image_path_full=full_a,
        detection_mode="yolo",
        timestamp=base,
    )

    # --- cropped_only: no full path on the record ---
    cropped_b = images_dir / "20260428_120100_000000_cam0.png"
    _write_png(cropped_b, b"cropped_b_bytes")
    obs_cropped_only = _make_obs(
        image_path=cropped_b,
        timestamp=base + timedelta(minutes=1),
    )

    # --- ghost: paths set on the record but no files on disk ---
    obs_ghost = _make_obs(
        image_path=images_dir / "ghost_cam0.png",
        image_path_full=images_dir / "ghost_cam0_full.png",
        timestamp=base + timedelta(minutes=2),
    )

    jsonl = tmp_path / "observations.jsonl"
    _write_jsonl(jsonl, [obs_all, obs_cropped_only, obs_ghost])

    app = create_app(observations_path=jsonl)
    client = TestClient(app)
    return client, {
        "all_three": obs_all,
        "cropped_only": obs_cropped_only,
        "ghost": obs_ghost,
    }


# ── Auth wall ────────────────────────────────────────────────────────────────


class TestAuth:
    @pytest.mark.parametrize("variant", ["cropped", "full", "annotated"])
    def test_image_requires_token(self, populated, variant):
        client, obs_map = populated
        ident = _id_for(obs_map["all_three"])
        r = client.get(f"/api/observations/{ident}/image/{variant}")
        assert r.status_code == 401


# ── Happy path ───────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_cropped_returns_file_bytes(self, populated):
        client, obs_map = populated
        ident = _id_for(obs_map["all_three"])
        r = client.get(
            f"/api/observations/{ident}/image/cropped",
            headers=HEADERS,
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/png")
        assert r.content == b"cropped_a_bytes"

    def test_full_returns_file_bytes(self, populated):
        client, obs_map = populated
        ident = _id_for(obs_map["all_three"])
        r = client.get(
            f"/api/observations/{ident}/image/full",
            headers=HEADERS,
        )
        assert r.status_code == 200
        assert r.content == b"full_a_bytes"

    def test_annotated_returns_file_bytes(self, populated):
        """Annotated path is derived from image_path_full by swapping
        the ``_full`` stem suffix for ``_annotated``. The fixture
        writes that file to disk, so the route should resolve and
        serve it."""
        client, obs_map = populated
        ident = _id_for(obs_map["all_three"])
        r = client.get(
            f"/api/observations/{ident}/image/annotated",
            headers=HEADERS,
        )
        assert r.status_code == 200
        assert r.content == b"annotated_a_bytes"


# ── Missing — None path field ────────────────────────────────────────────────


class TestMissingPathField:
    def test_full_404_when_image_path_full_is_none(self, populated):
        client, obs_map = populated
        ident = _id_for(obs_map["cropped_only"])
        r = client.get(
            f"/api/observations/{ident}/image/full",
            headers=HEADERS,
        )
        assert r.status_code == 404

    def test_annotated_404_when_image_path_full_is_none(self, populated):
        """Annotated derives from image_path_full — when that's None
        the annotated path is unresolvable too."""
        client, obs_map = populated
        ident = _id_for(obs_map["cropped_only"])
        r = client.get(
            f"/api/observations/{ident}/image/annotated",
            headers=HEADERS,
        )
        assert r.status_code == 404


# ── Missing — file not on disk ───────────────────────────────────────────────


class TestMissingOnDisk:
    @pytest.mark.parametrize("variant", ["cropped", "full", "annotated"])
    def test_404_when_file_missing(self, populated, variant):
        client, obs_map = populated
        ident = _id_for(obs_map["ghost"])
        r = client.get(
            f"/api/observations/{ident}/image/{variant}",
            headers=HEADERS,
        )
        assert r.status_code == 404


# ── Bad input ────────────────────────────────────────────────────────────────


class TestBadInput:
    def test_unknown_observation_id_404(self, populated):
        client, _ = populated
        r = client.get(
            "/api/observations/20210101T000000000000/image/cropped",
            headers=HEADERS,
        )
        assert r.status_code == 404

    def test_malformed_observation_id_404(self, populated):
        client, _ = populated
        r = client.get(
            "/api/observations/garbage/image/cropped",
            headers=HEADERS,
        )
        assert r.status_code == 404

    def test_unknown_variant_422(self, populated):
        """The route's variant param is a Literal — FastAPI rejects
        anything outside cropped/full/annotated with a 422 before
        the handler runs."""
        client, obs_map = populated
        ident = _id_for(obs_map["all_three"])
        r = client.get(
            f"/api/observations/{ident}/image/sideways",
            headers=HEADERS,
        )
        assert r.status_code == 422
