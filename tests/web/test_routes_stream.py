"""Unit tests for src.web.routes.stream.

Covers:
    - both endpoints require auth (header or query token)
    - /api/frame: 200 with image/jpeg when buffer is warm
    - /api/frame: 503 when buffer is None or empty
    - /api/stream: 200 with multipart/x-mixed-replace content-type
    - /api/stream: includes the boundary in the response
    - /api/stream: yields the latest frame as the first part
    - /api/stream: subscribers count clears after the response closes
    - /api/stream: 503 with Retry-After when subscriber cap reached
    - query-string token works for both endpoints (browsers can't
      send custom headers on <img>, so the stream endpoint MUST
      accept ?token=)
"""

from __future__ import annotations

import time
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.web.app import create_app
from src.web.stream_buffer import StreamBuffer

TOKEN = "valid-token-1234567890"
HEADERS = {"X-Avis-Token": TOKEN}


# A tiny, valid JPEG so the wire-format checks have something real
# to wrap. Generated once at import so each test does not pay the
# encode cost.
def _tiny_jpeg() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(123, 45, 67)).save(buf, format="JPEG", quality=75)
    return buf.getvalue()


JPEG_A = _tiny_jpeg()


# ---- Fixtures --------------------------------------------------------------


def _make_client(*, stream_buffer):
    """Build a TestClient with a tight stream wait timeout.

    The streaming generator uses a sync condvar wait, which can only
    return control between waits. The default 5s production timeout
    would make tests that open /api/stream block up to 5s after the
    test client closes the response. 0.3s is short enough to keep
    tests snappy and long enough to exercise the wait/notify path.
    """
    app = create_app(stream_buffer=stream_buffer)
    app.state.stream_wait_timeout = 0.3
    return TestClient(app)


@pytest.fixture
def warm_buffer() -> StreamBuffer:
    """Buffer with one frame already published.

    Mirrors the steady state during a normal capture loop -- at
    least one frame is in the ring before any subscriber connects.
    """
    buf = StreamBuffer(capacity=4, max_subscribers=2)
    buf.publish(JPEG_A)
    return buf


@pytest.fixture
def empty_buffer() -> StreamBuffer:
    return StreamBuffer(capacity=4, max_subscribers=2)


@pytest.fixture
def warm_client(monkeypatch, warm_buffer) -> TestClient:
    monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)
    return _make_client(stream_buffer=warm_buffer)


@pytest.fixture
def empty_client(monkeypatch, empty_buffer) -> TestClient:
    monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)
    return _make_client(stream_buffer=empty_buffer)


@pytest.fixture
def no_buffer_client(monkeypatch) -> TestClient:
    """Dashboard process without a wired-in StreamBuffer.

    Production layout: avis-web.service runs separately from
    avis.service, so the dashboard's app.state.stream_buffer is
    None. Routes must 503 cleanly in that case rather than crash.
    """
    monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)
    return _make_client(stream_buffer=None)


# ---- Auth ------------------------------------------------------------------


class TestAuth:
    def test_frame_requires_token(self, warm_client):
        assert warm_client.get("/api/frame").status_code == 401

    def test_stream_requires_token(self, warm_client):
        assert warm_client.get("/api/stream").status_code == 401

    def test_frame_accepts_query_token(self, warm_client):
        r = warm_client.get(f"/api/frame?token={TOKEN}")
        assert r.status_code == 200

    def test_stream_accepts_query_token(self, warm_client):
        # <img src="/api/stream?token=..."> is the production case;
        # this contract MUST hold or the SPA cannot render the live
        # preview.
        with warm_client.stream("GET", f"/api/stream?token={TOKEN}") as r:
            assert r.status_code == 200


# ---- /api/frame ------------------------------------------------------------


class TestFrame:
    def test_frame_returns_200_with_jpeg(self, warm_client):
        r = warm_client.get("/api/frame", headers=HEADERS)
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert r.content == JPEG_A

    def test_frame_returns_no_cache_header(self, warm_client):
        r = warm_client.get("/api/frame", headers=HEADERS)
        cc = r.headers.get("cache-control", "").lower()
        assert "no-store" in cc

    def test_frame_503_when_empty(self, empty_client):
        r = empty_client.get("/api/frame", headers=HEADERS)
        assert r.status_code == 503

    def test_frame_503_when_no_buffer(self, no_buffer_client):
        r = no_buffer_client.get("/api/frame", headers=HEADERS)
        assert r.status_code == 503
        # Different detail string than empty so an operator looking
        # at logs can tell them apart.
        assert "configured" in r.json()["detail"].lower()

    def test_frame_returns_latest(self, monkeypatch):
        """When new frames keep landing, /api/frame keeps returning
        the newest -- not the first one published."""
        monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)
        buf = StreamBuffer()
        buf.publish(b"first")
        buf.publish(b"second")
        buf.publish(b"third")
        client = _make_client(stream_buffer=buf)
        r = client.get("/api/frame", headers=HEADERS)
        assert r.content == b"third"


# ---- /api/stream -----------------------------------------------------------


def _read_first_part(client: TestClient, url: str, *, headers=None) -> bytes:
    """Open the MJPEG stream, read until we have one full part.

    TestClient.stream() drives the response synchronously via httpx.
    We read in small chunks until the boundary appears twice -- once
    opening the part, once closing it / starting the next part.
    """
    chunks: list[bytes] = []
    with client.stream("GET", url, headers=headers) as r:
        assert r.status_code == 200
        for chunk in r.iter_bytes():
            chunks.append(chunk)
            joined = b"".join(chunks)
            if joined.count(b"--avis-frame-boundary") >= 2:
                break
    return b"".join(chunks)


class TestStreamHappyPath:
    def test_stream_content_type_advertises_boundary(self, warm_client):
        with warm_client.stream("GET", "/api/stream", headers=HEADERS) as r:
            assert r.status_code == 200
            ct = r.headers["content-type"]
            assert ct.startswith("multipart/x-mixed-replace")
            assert "boundary=avis-frame-boundary" in ct

    def test_stream_first_part_is_initial_frame(self, warm_client):
        body = _read_first_part(warm_client, "/api/stream", headers=HEADERS)
        assert b"--avis-frame-boundary" in body
        assert b"Content-Type: image/jpeg" in body
        assert JPEG_A in body

    def test_stream_subscriber_released_after_response(self, warm_client, warm_buffer):
        before = warm_buffer.subscriber_count
        with warm_client.stream("GET", "/api/stream", headers=HEADERS) as r:
            assert r.status_code == 200
            for chunk in r.iter_bytes():
                if chunk:
                    break
        # Allow StreamingResponse cleanup to run.
        for _ in range(20):
            if warm_buffer.subscriber_count == before:
                break
            time.sleep(0.05)
        assert warm_buffer.subscriber_count == before


class TestStreamErrors:
    def test_stream_503_when_no_buffer(self, no_buffer_client):
        r = no_buffer_client.get("/api/stream", headers=HEADERS)
        assert r.status_code == 503

    def test_stream_503_when_subscriber_cap_reached(self, monkeypatch):
        """When the buffer's only subscriber slot is taken, the
        route must 503 with Retry-After: 10 rather than degrade the
        existing stream.

        Tested at the route-mapping level rather than via two
        parallel streams through the TestClient -- the sync test
        client serializes requests, so concurrent streaming via
        that path is an anyio deadlock waiting to happen. The
        buffer's own subscriber-cap behavior is exercised in
        tests/web/test_stream_buffer.py.
        """
        monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)
        buf = StreamBuffer(capacity=2, max_subscribers=1)
        buf.publish(JPEG_A)
        held = buf.subscribe()
        try:
            client = _make_client(stream_buffer=buf)
            r = client.get("/api/stream", headers=HEADERS)
            assert r.status_code == 503
            assert r.headers.get("retry-after") == "10"
        finally:
            held.close()


# ---- Wire format helpers ---------------------------------------------------


class TestWireFormat:
    def test_frame_chunk_format(self):
        """Spot-check the helper's output -- the route's correctness
        depends on it producing valid multipart parts."""
        from src.web.routes.stream import _frame_chunk

        chunk = _frame_chunk(b"abc")
        assert chunk.startswith(b"--avis-frame-boundary\r\n")
        assert b"Content-Type: image/jpeg\r\n" in chunk
        assert b"Content-Length: 3\r\n" in chunk
        assert chunk.endswith(b"\r\nabc\r\n")
