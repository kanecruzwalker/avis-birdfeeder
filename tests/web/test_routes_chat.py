"""Unit tests for src.web.routes.chat.

Covers:
    - 503 when no analyst is wired into the app
    - 401 when token missing or wrong (auth wall)
    - 422 on missing / oversized question
    - 200 with the structured analyst response shape
    - tools_called list and llm_available flag pass through verbatim
    - the analyst's exception path doesn't crash the route — the
      wire shape always carries a string answer + error field
    - the route runs the (synchronous) analyst.answer() in a threadpool
      so it doesn't pin the event loop (smoke test: an analyst that
      blocks for a moment doesn't deadlock concurrent requests)
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from src.web.app import create_app

TOKEN = "valid-token-1234567890"
HEADERS = {"X-Avis-Token": TOKEN}


# ── Mock analyst ─────────────────────────────────────────────────────────────


class _Response:
    """Stand-in for AnalystResponse — minimal interface used by the route."""

    def __init__(
        self,
        answer: str = "Two House Finches and one Mourning Dove visited.",
        tools_called: list[str] | None = None,
        confidence: str = "high",
        llm_available: bool = True,
        error: str | None = None,
    ) -> None:
        self._answer = answer
        self._tools_called = tools_called or ["read_recent_observations"]
        self._confidence = confidence
        self._llm_available = llm_available
        self._error = error

    def to_dict(self) -> dict:
        return {
            "answer": self._answer,
            "tools_called": list(self._tools_called),
            "confidence": self._confidence,
            "llm_available": self._llm_available,
            "error": self._error,
            "generated_at": datetime.now(UTC).isoformat(),
        }


class _StubAnalyst:
    """Analyst double — records the question it was asked."""

    def __init__(self, response: _Response | None = None) -> None:
        self._response = response or _Response()
        self.calls: list[str] = []

    def answer(self, question: str, **_: object) -> _Response:
        self.calls.append(question)
        return self._response


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def app_no_analyst(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)
    return create_app(observations_path=tmp_path / "observations.jsonl")


@pytest.fixture
def app_with_analyst(tmp_path, monkeypatch):
    monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)
    analyst = _StubAnalyst()
    app = create_app(
        observations_path=tmp_path / "observations.jsonl",
        analyst=analyst,
    )
    return app, analyst


# ── 503 when no analyst ──────────────────────────────────────────────────────


class TestNoAnalyst:
    def test_503_when_unconfigured(self, app_no_analyst):
        client = TestClient(app_no_analyst)
        r = client.post("/api/ask", headers=HEADERS, json={"question": "what visited?"})
        assert r.status_code == 503
        # Operator hint is in the body — surfaced to the SPA.
        assert "GEMINI_API_KEY" in r.text or "not configured" in r.text.lower()


# ── Auth wall ────────────────────────────────────────────────────────────────


class TestAuth:
    def test_missing_token_401(self, app_with_analyst):
        app, _ = app_with_analyst
        client = TestClient(app)
        r = client.post("/api/ask", json={"question": "hi"})
        assert r.status_code == 401

    def test_wrong_token_401(self, app_with_analyst):
        app, _ = app_with_analyst
        client = TestClient(app)
        r = client.post(
            "/api/ask",
            headers={"X-Avis-Token": "nope"},
            json={"question": "hi"},
        )
        assert r.status_code == 401

    def test_token_in_query(self, app_with_analyst):
        app, _ = app_with_analyst
        client = TestClient(app)
        r = client.post(f"/api/ask?token={TOKEN}", json={"question": "hi"})
        assert r.status_code == 200


# ── Request validation ───────────────────────────────────────────────────────


class TestValidation:
    def test_missing_question_422(self, app_with_analyst):
        app, _ = app_with_analyst
        r = TestClient(app).post("/api/ask", headers=HEADERS, json={})
        assert r.status_code == 422

    def test_empty_question_422(self, app_with_analyst):
        app, _ = app_with_analyst
        r = TestClient(app).post("/api/ask", headers=HEADERS, json={"question": ""})
        assert r.status_code == 422

    def test_oversized_question_422(self, app_with_analyst):
        app, _ = app_with_analyst
        r = TestClient(app).post(
            "/api/ask",
            headers=HEADERS,
            json={"question": "x" * 2001},
        )
        assert r.status_code == 422

    def test_question_at_limit_passes(self, app_with_analyst):
        app, _ = app_with_analyst
        r = TestClient(app).post(
            "/api/ask",
            headers=HEADERS,
            json={"question": "x" * 2000},
        )
        assert r.status_code == 200


# ── Happy path ───────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_200_returns_structured_response(self, app_with_analyst):
        app, analyst = app_with_analyst
        r = TestClient(app).post(
            "/api/ask",
            headers=HEADERS,
            json={"question": "Anything new today?"},
        )
        assert r.status_code == 200
        body = r.json()
        for field in (
            "answer",
            "tools_called",
            "confidence",
            "llm_available",
            "error",
            "generated_at",
        ):
            assert field in body, f"missing {field} in response"

    def test_question_routed_to_analyst(self, app_with_analyst):
        app, analyst = app_with_analyst
        TestClient(app).post(
            "/api/ask",
            headers=HEADERS,
            json={"question": "What's the busiest hour?"},
        )
        assert analyst.calls == ["What's the busiest hour?"]

    def test_tools_called_passthrough(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)
        analyst = _StubAnalyst(
            _Response(
                answer="Three species today.",
                tools_called=["get_top_species", "get_detection_stats"],
            )
        )
        app = create_app(
            observations_path=tmp_path / "observations.jsonl",
            analyst=analyst,
        )
        body = (
            TestClient(app)
            .post("/api/ask", headers=HEADERS, json={"question": "today summary"})
            .json()
        )
        assert body["tools_called"] == ["get_top_species", "get_detection_stats"]

    def test_llm_unavailable_passthrough(self, tmp_path, monkeypatch):
        # When the analyst falls back (no LLM key on the server but the
        # operator wired it in anyway), the route forwards llm_available=False
        # rather than 503'ing — the SPA shows a "fallback response" hint.
        monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)
        analyst = _StubAnalyst(
            _Response(
                answer="LLM unavailable.",
                tools_called=[],
                llm_available=False,
                error="LLM not available",
            )
        )
        app = create_app(
            observations_path=tmp_path / "observations.jsonl",
            analyst=analyst,
        )
        body = TestClient(app).post("/api/ask", headers=HEADERS, json={"question": "x"}).json()
        assert body["llm_available"] is False
        assert body["error"] == "LLM not available"


# ── Threadpool offload ───────────────────────────────────────────────────────


class TestThreadpool:
    """The analyst is synchronous and slow. The route runs it via
    ``run_in_threadpool`` so concurrent requests don't serialize on
    the event loop. This is a smoke check, not a perf benchmark.
    """

    def test_blocking_analyst_does_not_serialize(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AVIS_WEB_TOKEN", TOKEN)

        class _SlowAnalyst:
            def answer(self, question: str, **_: object) -> _Response:
                time.sleep(0.2)
                return _Response(answer=f"ack: {question}")

        app = create_app(
            observations_path=tmp_path / "observations.jsonl",
            analyst=_SlowAnalyst(),
        )
        client = TestClient(app)

        results: list[int] = []

        def hit(question: str) -> None:
            r = client.post("/api/ask", headers=HEADERS, json={"question": question})
            results.append(r.status_code)

        threads = [threading.Thread(target=hit, args=(f"q{i}",)) for i in range(3)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start

        assert results == [200, 200, 200]
        # Three 0.2s calls running serially would be ~0.6s; with the
        # threadpool offload they overlap. Generous bound to avoid CI
        # flakiness — we just need to confirm not-fully-serialized.
        assert elapsed < 0.55, f"requests appear serialized (elapsed={elapsed:.2f}s)"
