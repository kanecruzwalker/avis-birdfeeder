"""Observation image-variant endpoints (PR 4).

``GET /api/observations/{id}/image/{variant}``
    Where ``{variant}`` is one of ``cropped``, ``full``, ``annotated``.
    Returns the corresponding PNG file from the agent's capture
    directory.

Path resolution per variant:

    cropped    — ``observation.image_path``. Always saved by
                 ``VisionCapture._process_frame`` when motion gate
                 fires, so this is populated for every observation
                 (dispatched and suppressed) that had a capture.

    full       — ``observation.image_path_full``. Only saved on
                 dispatched observations after PR 4 (the agent's
                 ``_save_dispatch_image_variants`` helper writes it
                 right before the notifier hands off). Suppressed
                 records and pre-PR-4 records have this as ``None`` →
                 404.

    annotated  — derived from ``image_path_full`` by swapping the
                 ``_full`` stem suffix for ``_annotated``. Only
                 actually written to disk when YOLO mode produced a
                 box, so the file may be missing even when
                 ``image_path_full`` is set → 404.

404 cases (all return 404 with a descriptive ``detail``):
    - Observation ID malformed or unknown.
    - The variant's path field on the observation is ``None``.
    - The path is set but the file is missing on disk (e.g., the
      capture directory was rotated out from under the dashboard).

Auth: same token wall as the rest of ``/api/*``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from src.data.schema import BirdObservation

from ..auth import RequireToken
from ..observation_store import ObservationNotFound, ObservationStore

ImageVariant = Literal["cropped", "full", "annotated"]


router = APIRouter(
    prefix="/api",
    tags=["images"],
    dependencies=[RequireToken],
)


def _store(request: Request) -> ObservationStore:
    return request.app.state.observation_store


def _path_for_variant(
    observation: BirdObservation,
    variant: ImageVariant,
) -> Path | None:
    """Resolve the on-disk path for a variant, or ``None`` if the
    observation doesn't carry one.

    ``annotated`` is derived rather than stored — see module docstring
    for the rationale.
    """
    if variant == "cropped":
        return Path(observation.image_path) if observation.image_path else None
    if variant == "full":
        return Path(observation.image_path_full) if observation.image_path_full else None
    # annotated
    if not observation.image_path_full:
        return None
    full = Path(observation.image_path_full)
    # The full-frame stem ends with "_full" by the agent's
    # ``_save_dispatch_image_variants`` convention. If that
    # convention drifts, the annotated route should fail loud
    # rather than silently serve the wrong file — so we 404.
    suffix_marker = "_full"
    if not full.stem.endswith(suffix_marker):
        return None
    annotated_stem = full.stem[: -len(suffix_marker)] + "_annotated"
    return full.with_name(annotated_stem + full.suffix)


@router.get("/observations/{observation_id}/image/{variant}")
def get_observation_image(
    observation_id: str,
    variant: ImageVariant,
    request: Request,
) -> FileResponse:
    """Serve one of the three image variants for an observation."""
    try:
        observation = _store(request).get(observation_id)
    except ObservationNotFound as exc:
        raise HTTPException(
            status_code=404,
            detail=f"observation not found: {observation_id}",
        ) from exc

    path = _path_for_variant(observation, variant)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"image variant not available: {variant}",
        )
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"image file missing on disk: {path.name}",
        )

    # The agent saves PNG (see VisionCapture._save_frame and
    # BirdAgent._save_dispatch_image_variants). Hardcoding the
    # media type is fine while that's the convention; if the
    # agent ever switches to JPEG the file extension will reflect
    # it and we can derive media_type from path.suffix here.
    return FileResponse(path, media_type="image/png")
