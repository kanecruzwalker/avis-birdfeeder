"""Live preview stream endpoints.

``GET /api/stream``
    Multipart MJPEG stream. Browsers consume this directly via
    ``<img src="/api/stream?token=...">``. Each part is a JPEG
    frame produced by the capture loop (see
    :class:`src.web.stream_buffer.StreamBuffer`).

``GET /api/frame``
    Single most-recent JPEG. A polling fallback for clients that
    do not want a long-lived MJPEG connection (or for embedding
    one frame in a static HTML response).

Both endpoints sit behind the same auth wall as the rest of
``/api/*``. The MJPEG stream specifically requires the token in the
query string -- ``<img>`` requests cannot set custom headers, so the
``X-Avis-Token`` route is unavailable for that case. The token is
already exposed to the user who has it, so no additional leakage
risk over the page that renders the ``<img>``.

The dashboard process and the agent process may be the same process
or two separate ones (the production pi has them as independent
systemd units). When they are separate, ``app.state.stream_buffer``
is ``None`` from the dashboard's perspective and both endpoints
return 503; cross-process bridging is a later PR.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse

from ..auth import RequireToken
from ..stream_buffer import StreamBuffer, SubscriberLimitExceeded

logger = logging.getLogger(__name__)


# /api/stream + /api/frame share the same auth wall as the other
# /api routes. RequireToken handles both header and query-param
# token transport, so the MJPEG-needs-?token=... case Just Works.
router = APIRouter(prefix="/api", tags=["stream"], dependencies=[RequireToken])


# ---- Wire format constants --------------------------------------------------

# Boundary string for multipart/x-mixed-replace. Any token works;
# we pick a constant so tests can match it exactly. Browsers do not
# care what the boundary is as long as the response advertises it.
_BOUNDARY = "avis-frame-boundary"

# Default per-frame wait timeout. Bounds how long an idle stream
# holds a threadpool worker if the publisher stalls. The browser
# closes the connection cleanly when the iterator ends; on user
# focus it typically reloads the <img> and reconnects. Override per
# app via ``app.state.stream_wait_timeout`` (tests use a short
# value so the streaming generator returns control promptly when
# the test client exits the context).
_DEFAULT_STREAM_WAIT_TIMEOUT_SECONDS = 5.0


def _wait_timeout(request: Request) -> float:
    return getattr(
        request.app.state,
        "stream_wait_timeout",
        _DEFAULT_STREAM_WAIT_TIMEOUT_SECONDS,
    )


# ---- App-state accessors ----------------------------------------------------


def _buffer(request: Request) -> StreamBuffer | None:
    """Pull the StreamBuffer off app.state (or None if not wired).

    The factory always sets ``app.state.stream_buffer`` (possibly to
    ``None``); using ``getattr`` guards against an older app that
    predates the field.
    """
    return getattr(request.app.state, "stream_buffer", None)


# ---- Routes -----------------------------------------------------------------


@router.get(
    "/frame",
    response_class=Response,
    responses={
        200: {
            "description": "Latest preview JPEG.",
            "content": {"image/jpeg": {}},
        },
        503: {"description": "Preview not configured or no frame published yet."},
    },
)
def latest_frame(request: Request) -> Response:
    """Return the most-recent JPEG snapshot.

    Returns 503 in two cases:
    - the dashboard process has no StreamBuffer wired in (``None`` on
      app.state), e.g., dashboard running standalone without the
      agent;
    - the buffer exists but is empty (the capture loop has not run a
      cycle yet, or has not published any frames).

    The two cases are distinguishable by the ``detail`` string for
    operator debugging but produce the same status code -- a polling
    client should treat both as "no preview right now, retry later".
    """
    buf = _buffer(request)
    if buf is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="preview stream not configured",
        )
    frame = buf.latest()
    if frame is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="no preview frame available yet",
        )
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@router.get(
    "/stream",
    responses={
        200: {
            "description": "Multipart MJPEG stream.",
            "content": {"multipart/x-mixed-replace": {}},
        },
        503: {"description": "Preview not configured or subscriber cap reached."},
    },
)
def mjpeg_stream(request: Request) -> StreamingResponse:
    """Multipart MJPEG live stream.

    Wire format (one part per frame, repeated until the client
    disconnects)::

        --avis-frame-boundary\\r\\n
        Content-Type: image/jpeg\\r\\n
        Content-Length: N\\r\\n
        \\r\\n
        <N bytes of JPEG>\\r\\n

    The first part is the most-recent frame already in the buffer
    (if any) so the viewer's ``<img>`` does not flash blank for
    ~200ms. Subsequent parts come from the subscription, which
    yields each newly-published frame.

    Subscriber cap: configured on the buffer (default 5). Past the
    cap, this endpoint returns 503 with ``Retry-After: 10`` so the
    client can back off rather than hammer the Pi.
    """
    buf = _buffer(request)
    if buf is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="preview stream not configured",
        )

    try:
        subscription = buf.subscribe(timeout=_wait_timeout(request))
    except SubscriberLimitExceeded as exc:
        # Outbound bandwidth on a Pi-class device is the practical
        # ceiling on simultaneous viewers; refuse cleanly rather than
        # degrade existing streams.
        logger.info("Stream subscriber limit reached: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Retry-After": "10"},
        ) from exc

    initial = buf.latest()

    def generator() -> Iterator[bytes]:
        try:
            if initial is not None:
                yield _frame_chunk(initial)
            for frame in subscription:
                yield _frame_chunk(frame)
        finally:
            # Always release the subscriber slot, even if the client
            # disconnected mid-yield (StreamingResponse will close
            # the generator on disconnect).
            subscription.close()

    return StreamingResponse(
        generator(),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        headers={
            # MJPEG streams must not be cached -- the response is
            # technically infinite. Some intermediaries get confused
            # without explicit no-store.
            "Cache-Control": "no-store, no-cache, must-revalidate",
            # Many browsers buffer multipart responses unless the
            # connection is explicitly told to flush. close vs
            # keep-alive doesn't matter here in practice but we set
            # it to make the intent explicit.
            "Connection": "close",
        },
    )


# ---- Helpers ----------------------------------------------------------------


def _frame_chunk(jpeg: bytes) -> bytes:
    """Wrap a JPEG payload as a single multipart part.

    Format spec is loose -- both ``\\r\\n`` and ``\\n`` line endings
    are widely accepted. We use CRLF to match the HTTP convention
    most browsers' MJPEG parsers were written for; the original
    Mozilla server-push spec uses CRLF.
    """
    header = (
        f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
    ).encode("ascii")
    return header + jpeg + b"\r\n"
