"""Chat proxy to :class:`BirdAnalystAgent`.

``POST /api/ask`` takes a natural language question, hands it to the
analyst's ``answer()`` method, and returns the structured response the
SPA renders as a chat bubble.

The route is gated at startup, not per-request: the agent is wired in
through ``create_app(analyst=...)`` and stashed on ``app.state``. When
no analyst is configured (the common case during tests, or when
``GEMINI_API_KEY`` is unset on the operator's box), every call returns
503. Tests inject a mock analyst the same way production does.

``BirdAnalystAgent.answer()`` is synchronous — it makes a blocking
LangChain call out to Gemini that can take seconds — so we run it in a
threadpool to avoid pinning uvicorn's event loop.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..auth import RequireToken

router = APIRouter(
    prefix="/api",
    tags=["chat"],
    dependencies=[RequireToken],
)


# ── Wire shapes ─────────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    """User question routed to the analyst.

    Length cap is generous — long-form questions are fine — but bounded
    so a runaway client can't spam multi-MB payloads through the LLM.
    """

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural language question about the feeder.",
    )


class AskResponse(BaseModel):
    """Structured analyst answer.

    Mirrors :meth:`AnalystResponse.to_dict` so the SPA can render the
    same shape regardless of whether the LLM was reached or fell back.
    """

    answer: str
    tools_called: list[str]
    confidence: str
    llm_available: bool
    error: str | None = None
    generated_at: str


# ── Route ───────────────────────────────────────────────────────────────────


def _analyst(request: Request) -> Any:
    """Return the configured analyst or 503 if none was wired in."""
    analyst = getattr(request.app.state, "analyst", None)
    if analyst is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Analyst not configured. Set GEMINI_API_KEY in the "
                "dashboard's environment and restart, or run the "
                "agent + dashboard in the same process."
            ),
        )
    return analyst


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Forward the question to the analyst and return its answer."""
    analyst = _analyst(request)
    response = await run_in_threadpool(analyst.answer, payload.question)
    return AskResponse(**response.to_dict())
