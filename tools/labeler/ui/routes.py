"""HTTP routes for the labeling-assistant review UI.

Two route groups:
- API routes (`/api/...`) — JSON in/out, used by the SPA frontend
- Page routes (`/`, `/verified`) — return rendered HTML
- Image route (`/image/{filename}`) — serves capture images

All routes are protected by `RequireToken` from auth.py except the
public `/health` endpoint defined in server.py.

Design:
- Routes are pure translation between HTTP shapes and ReviewStore calls.
  No business logic lives here. If a function looks too long, the
  business logic should probably move into the store.
- Pydantic request/response models are defined inline near the routes
  that use them — they're tiny and tightly coupled to one endpoint
  each, so a separate models.py would be over-engineering.
- ConcurrencyConflict is the one place we deliberately diverge from the
  store's exception model: it gets surfaced as HTTP 409 with the
  existing record so the client can prompt for overwrite.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Path as PathParam, Query, Request, status
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from ..schema import KNOWN_SPECIES_CODES, SENTINELS, VerifiedLabel
from .auth import RequireToken
from .review_store import (
    ConcurrencyConflict,
    PreLabelNotFound,
    ReviewStore,
    ReviewStoreError,
)

logger = logging.getLogger(__name__)


# ── App-state accessors ──────────────────────────────────────────────────────

# The ReviewStore and Jinja2Templates instances live on the FastAPI app's
# state object (set up in server.py). Routes reach for them via these
# helpers so the dependency is explicit and easy to mock in tests.


def _store(request: Request) -> ReviewStore:
    return request.app.state.review_store


def _templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates


# ── Request / response models ─────────────────────────────────────────────────


class VerifyRequest(BaseModel):
    """Body of POST /api/verify."""

    image_filename: str
    species_code: str
    other_species_code: Optional[str] = None
    reviewer_notes: Optional[str] = None
    agreed_with_pre_label: Optional[bool] = None
    client_load_time: datetime = Field(
        description=(
            "ISO 8601 timestamp returned in the /api/next or /api/review/{f} "
            "response. Echoed back here so the server can detect concurrent "
            "writes from another client / device."
        ),
    )
    force_overwrite: bool = Field(
        default=False,
        description=(
            "Set true on retry after seeing a 409 Conflict, signalling that "
            "the user has acknowledged the existing record and chooses to "
            "overwrite it."
        ),
    )


class ReviewItemResponse(BaseModel):
    """Body returned by /api/next and /api/review/{filename}.

    Flattens the PreLabel into the review-screen-relevant fields so the
    frontend doesn't have to reach through nested objects. The
    `image_url` is the route the frontend should use for `<img src=...>`.
    """

    image_filename: str
    image_url: str
    capture_timestamp: Optional[datetime]

    pre_label_species: str
    pre_label_confidence: float
    pre_label_reasoning: str
    pre_label_uncertain_between: Optional[list[str]]

    audio_hint: Optional[str]
    audio_confidence: Optional[float]

    already_verified_species: Optional[str] = None
    already_verified_other_species: Optional[str] = None
    already_verified_at: Optional[datetime] = None

    client_load_time: datetime


# ── Routers ──────────────────────────────────────────────────────────────────

# All API routes carry the RequireToken dependency at the router level so
# we can't forget it on a new endpoint. Page and image routes carry it the
# same way on their own routers.

api_router = APIRouter(prefix="/api", dependencies=[RequireToken])
page_router = APIRouter(dependencies=[RequireToken])
image_router = APIRouter(dependencies=[RequireToken])


# ── /api/species — allowed codes for frontend dropdown ────────────────────────


@api_router.get("/species")
def list_species() -> dict:
    """Return the allowed species codes + sentinels for UI dropdown rendering.

    Returns a dict with three keys:
    - `known`: 20 4-letter species codes from configs/species.yaml
    - `sentinels`: NONE, UNKNOWN, OTHER
    - `all`: convenience union, in sentinel-last order

    Frontend uses `known` for the quick-correct row and `all` for the full
    correction picker.
    """
    return {
        "known": list(KNOWN_SPECIES_CODES),
        "sentinels": list(SENTINELS),
        "all": list(KNOWN_SPECIES_CODES) + list(SENTINELS),
    }


# ── /api/summary and /api/coverage — landing-page data ───────────────────────


@api_router.get("/summary")
def species_summary(request: Request) -> dict:
    """Group-by-species view for the landing page.

    Returns:
        { "species": [ { species_code, total, verified, remaining, coverage }, ... ],
          "coverage": { total_pre_labels, total_verified, remaining, coverage } }
    """
    store = _store(request)
    return {
        "species": store.species_summary(),
        "coverage": store.coverage(),
    }


@api_router.get("/coverage")
def coverage(request: Request) -> dict:
    """Just the overall coverage numbers — useful for header status badges."""
    store = _store(request)
    return store.coverage()


# ── /api/next and /api/review/{filename} — fetch image to review ─────────────


@api_router.get("/next", response_model=ReviewItemResponse)
def next_review_item(
    request: Request,
    species: Optional[str] = Query(
        default=None,
        description="Optional species_code to filter to (e.g. 'MOCH').",
    ),
    token: Optional[str] = Query(
        default=None,
        description=(
            "Auth token. Accepted in query string for shareable links; "
            "X-Avis-Token header is preferred for API calls."
        ),
        include_in_schema=False,
    ),
) -> ReviewItemResponse:
    """Hand the next unverified pre-label to the client.

    404 with a structured `code: queue_empty` body when no unverified
    images remain in the requested filter.
    """
    store = _store(request)
    item = store.next_unverified(species_filter=species)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "queue_empty",
                "message": (
                    f"No unverified pre-labels left in filter='{species or 'all'}'."
                ),
            },
        )
    return _to_review_response(item)


@api_router.get("/review/{filename}", response_model=ReviewItemResponse)
def review_specific(
    request: Request,
    filename: str = PathParam(..., description="image_filename to review."),
    token: Optional[str] = Query(default=None, include_in_schema=False),
) -> ReviewItemResponse:
    """Open a specific image (e.g. when re-opening a verified record)."""
    store = _store(request)
    try:
        item = store.get_review_item(filename)
    except PreLabelNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": str(exc)},
        ) from exc
    return _to_review_response(item)


def _to_review_response(item) -> ReviewItemResponse:
    """Translate a store ReviewItem into the HTTP response shape."""
    pre = item.pre_label
    resp = ReviewItemResponse(
        image_filename=pre.image_filename,
        image_url=f"/image/{pre.image_filename}",
        capture_timestamp=pre.capture_timestamp,
        pre_label_species=pre.llm_response.species_code,
        pre_label_confidence=pre.llm_response.confidence,
        pre_label_reasoning=pre.llm_response.reasoning,
        pre_label_uncertain_between=pre.llm_response.uncertain_between,
        audio_hint=pre.audio_hint,
        audio_confidence=pre.audio_confidence,
        client_load_time=item.client_load_time,
    )
    if item.already_verified is not None:
        resp.already_verified_species = item.already_verified.species_code
        resp.already_verified_other_species = item.already_verified.other_species_code
        resp.already_verified_at = item.already_verified.verified_at
    return resp


# ── /api/verify — submit a verified label ────────────────────────────────────


@api_router.post("/verify")
def verify(
    request: Request,
    payload: VerifyRequest = Body(...),
) -> dict:
    """Persist a verification.

    Status codes:
    - 200: written successfully
    - 404: image_filename is not in pre_labels.jsonl
    - 409: optimistic-concurrency conflict — body includes the existing
            record so the client can prompt the user for overwrite

    Body on 200 is the persisted record (server-stamped verified_at).
    Body on 409 is { "code": "conflict", "existing": {VerifiedLabel} }.
    """
    store = _store(request)

    # Build the VerifiedLabel from the request. Pydantic validates the
    # OTHER ↔ other_species_code invariants here, before we touch disk.
    pre = None
    try:
        pre_item = store.get_review_item(payload.image_filename)
        pre = pre_item.pre_label
    except PreLabelNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": str(exc)},
        ) from exc

    try:
        verified = VerifiedLabel(
            image_path=pre.image_path,
            image_filename=pre.image_filename,
            species_code=payload.species_code,
            other_species_code=payload.other_species_code,
            reviewer_notes=payload.reviewer_notes,
            pre_label=pre,
            agreed_with_pre_label=payload.agreed_with_pre_label,
        )
    except ValueError as exc:
        # Pydantic validation errors (e.g. OTHER without other_species_code)
        # surface here. 422 is the conventional code for "well-formed JSON
        # that fails business rules."
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "validation", "message": str(exc)},
        ) from exc

    try:
        persisted = store.record_verification(
            verified=verified,
            client_load_time=payload.client_load_time,
            force_overwrite=payload.force_overwrite,
        )
    except ConcurrencyConflict as exc:
        # 409 with the existing record — the client uses this to render
        # an "overwrite or cancel" prompt. The next request will set
        # force_overwrite=True if the user chooses overwrite.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "conflict",
                "message": str(exc),
                "existing": exc.existing.model_dump(mode="json"),
            },
        ) from exc
    except PreLabelNotFound as exc:
        # Race: pre-label disappeared between our check and the store
        # call. Vanishingly unlikely with a single-worker server but
        # handled cleanly.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": str(exc)},
        ) from exc

    return {
        "code": "ok",
        "verified": persisted.model_dump(mode="json"),
    }


# ── /api/verified — list of verified labels ──────────────────────────────────


@api_router.get("/verified")
def list_verified(
    request: Request,
    species: Optional[str] = Query(
        default=None, description="Optional species_code filter."
    ),
    limit: int = Query(default=200, ge=1, le=2000),
) -> dict:
    """Return verified labels, most-recent-first, optionally filtered.

    `limit` caps the response size for big datasets — the frontend can
    paginate by re-querying with different filters as needed. Default
    200 is enough to fill a typical desktop scroll view.
    """
    store = _store(request)
    records = store.list_verified(species_filter=species)
    truncated = records[:limit]
    return {
        "total": len(records),
        "returned": len(truncated),
        "records": [
            {
                "image_filename": r.image_filename,
                "image_url": f"/image/{r.image_filename}",
                "species_code": r.species_code,
                "other_species_code": r.other_species_code,
                "agreed_with_pre_label": r.agreed_with_pre_label,
                "reviewer_notes": r.reviewer_notes,
                "verified_at": r.verified_at.isoformat(),
            }
            for r in truncated
        ],
    }


# ── /image/{filename} — serve capture images ─────────────────────────────────


@image_router.get("/image/{filename}")
def get_image(
    request: Request,
    filename: str = PathParam(..., description="Capture image filename."),
    token: Optional[str] = Query(default=None, include_in_schema=False),
) -> FileResponse:
    """Serve a capture image. Auth-protected.

    The token in the query string is necessary for `<img src=...>` because
    browsers can't set custom headers on those tags. Auth dependency runs
    before us and accepts both header and query forms.

    Path traversal is guarded by the store's image_path() helper.
    """
    store = _store(request)
    try:
        path = store.image_path(filename)
    except ReviewStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_filename", "message": str(exc)},
        ) from exc

    if not path.exists() or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "image_missing",
                "message": (
                    f"Image '{filename}' not on disk in configured images_dir."
                ),
            },
        )

    return FileResponse(
        path=path,
        media_type="image/png",
        headers={
            # Cache aggressively — capture images are immutable once
            # written. Saves bandwidth on phone over Tailscale.
            "Cache-Control": "public, max-age=3600",
        },
    )


# ── HTML page routes ─────────────────────────────────────────────────────────


@page_router.get("/", response_class=HTMLResponse)
def index_page(request: Request) -> HTMLResponse:
    """Landing page — group-by-species buckets."""
    templates = _templates(request)
    # Pull the token off the request so we can inline it into the rendered
    # page; the SPA uses it for subsequent fetch() calls and image URLs.
    token = request.headers.get("X-Avis-Token") or request.query_params.get("token", "")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "token": token,
            "view": "summary",
        },
    )


@page_router.get("/review", response_class=HTMLResponse)
def review_page(
    request: Request,
    species: Optional[str] = Query(default=None),
) -> HTMLResponse:
    """Review screen — single image + verify controls."""
    templates = _templates(request)
    token = request.headers.get("X-Avis-Token") or request.query_params.get("token", "")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "token": token,
            "view": "review",
            "species_filter": species or "",
        },
    )


@page_router.get("/verified", response_class=HTMLResponse)
def verified_page(request: Request) -> HTMLResponse:
    """Verified-list view."""
    templates = _templates(request)
    token = request.headers.get("X-Avis-Token") or request.query_params.get("token", "")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "token": token,
            "view": "verified",
        },
    )