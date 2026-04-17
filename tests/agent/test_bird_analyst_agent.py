"""
tests/agent/test_bird_analyst_agent.py

Unit tests for BirdAnalystAgent.

All LLM calls are mocked — tests never hit the Gemini API.
Tests verify the agent's structure, fallback behavior, tool execution
routing, and response parsing.

What is NOT tested here:
    - Actual LLM reasoning quality (can't unit test)
    - Real Gemini API responses (integration test, needs key)
    - Tool implementations (tested in tests/agent/tools/)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.bird_analyst_agent import (
    AnalystDecision,
    AnalystResponse,
    BirdAnalystAgent,
    _ToolExecutor,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_agent(
    tmp_path: Path,
    enabled: bool = True,
    llm_available: bool = False,  # default False — no real API in tests
) -> BirdAnalystAgent:
    obs_path = tmp_path / "observations.jsonl"
    obs_path.write_text("")
    dec_path = tmp_path / "analyst_decisions.jsonl"
    agent = BirdAnalystAgent(
        observations_path=str(obs_path),
        decisions_log_path=str(dec_path),
        daily_summaries_dir=str(tmp_path / "summaries"),
        provider="gemini",
        model="gemini-2.0-flash",
        enabled=enabled,
    )
    # Override _client to control llm_available without real API
    if not llm_available:
        agent._client = None
    return agent


def _make_mock_client(response_text: str = "No action needed.", tools: list[str] | None = None):
    """Return a mock LLM client that returns a fixed response."""
    client = MagicMock()
    client.available = True
    client.run_with_tools = MagicMock(return_value=(response_text, tools or []))
    return client


# ── AnalystDecision ───────────────────────────────────────────────────────────


class TestAnalystDecision:
    def test_default_values(self) -> None:
        d = AnalystDecision()
        assert d.reasoning == ""
        assert d.switch_mode is None
        assert d.push_message is None
        assert d.generate_report is False
        assert d.feeder_alert is None
        assert d.tools_called == []
        assert d.llm_available is True

    def test_custom_values(self) -> None:
        d = AnalystDecision(
            reasoning="Mode confidence dropped.",
            switch_mode="fixed_crop",
            tools_called=["get_detection_stats", "switch_detection_mode"],
        )
        assert d.switch_mode == "fixed_crop"
        assert len(d.tools_called) == 2


# ── AnalystResponse ───────────────────────────────────────────────────────────


class TestAnalystResponse:
    def test_to_dict_is_json_serialisable(self) -> None:
        r = AnalystResponse(
            answer="7 House Finches visited today.",
            data={"species": [{"code": "HOFI", "count": 7}]},
            tools_called=["get_top_species"],
            confidence="high",
        )
        d = r.to_dict()
        # Must serialise without error
        json_str = json.dumps(d)
        assert "House Finches" in json_str

    def test_to_dict_includes_generated_at(self) -> None:
        r = AnalystResponse(answer="test")
        d = r.to_dict()
        assert "generated_at" in d

    def test_error_field_preserved(self) -> None:
        r = AnalystResponse(answer="", error="API timeout")
        d = r.to_dict()
        assert d["error"] == "API timeout"

    def test_llm_available_false_in_dict(self) -> None:
        r = AnalystResponse(answer="fallback", llm_available=False)
        assert r.to_dict()["llm_available"] is False


# ── BirdAnalystAgent construction ─────────────────────────────────────────────


class TestBirdAnalystAgentInit:
    def test_disabled_agent_llm_not_available(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, enabled=False)
        assert agent.llm_available is False

    def test_no_api_key_gracefully_unavailable(self, tmp_path: Path) -> None:
        # No GEMINI_API_KEY in test environment — should not raise
        agent = _make_agent(tmp_path, enabled=True)
        assert agent.llm_available is False  # no key = not available

    def test_paths_stored(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        assert "observations.jsonl" in agent.observations_path
        assert "analyst_decisions.jsonl" in agent.decisions_log_path


# ── advise() fallback ─────────────────────────────────────────────────────────


class TestAdvise:
    def test_returns_none_when_llm_unavailable(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, llm_available=False)
        result = agent.advise()
        assert result is None

    def test_returns_decision_when_llm_available(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent._client = _make_mock_client("No action needed.", tools=["log_analyst_decision"])

        result = agent.advise(current_mode="fixed_crop")

        assert result is not None
        assert isinstance(result, AnalystDecision)
        assert result.llm_available is True

    def test_switch_mode_inferred_from_tool_call(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        # Simulate client returning switch_detection_mode in tools_called
        agent._client = _make_mock_client(
            "Switching to yolo — confidence is higher.",
            tools=["get_detection_stats", "switch_detection_mode", "log_analyst_decision"],
        )
        vc = MagicMock()
        vc.detection_mode = "yolo"  # tool already changed it

        result = agent.advise(vision_capture=vc, current_mode="fixed_crop")

        assert result is not None
        assert result.switch_mode == "yolo"

    def test_generate_report_inferred_from_tool_call(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent._client = _make_mock_client(
            "End of day — generating report.",
            tools=["generate_daily_report", "log_analyst_decision"],
        )
        result = agent.advise()
        assert result is not None
        assert result.generate_report is True

    def test_feeder_alert_flagged_when_health_checked(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent._client = _make_mock_client(
            "Feeder health checked.",
            tools=["get_feeder_health", "log_analyst_decision"],
        )
        result = agent.advise()
        assert result is not None
        assert result.feeder_alert is not None

    def test_advise_survives_client_exception(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent._client = MagicMock()
        agent._client.available = True
        agent._client.run_with_tools = MagicMock(side_effect=Exception("network error"))

        result = agent.advise()
        assert result is None  # exception → returns None → fallback


# ── answer() ─────────────────────────────────────────────────────────────────


class TestAnswer:
    def test_fallback_when_llm_unavailable(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, llm_available=False)
        resp = agent.answer("What birds visited today?")

        assert isinstance(resp, AnalystResponse)
        assert resp.llm_available is False
        assert resp.error is not None
        assert len(resp.answer) > 0  # fallback message present

    def test_returns_answer_string(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent._client = _make_mock_client(
            "3 House Finches visited in the last hour.",
            tools=["get_top_species"],
        )
        resp = agent.answer("What birds visited today?")

        assert "House Finch" in resp.answer
        assert "get_top_species" in resp.tools_called
        assert resp.llm_available is True

    def test_to_dict_is_json_serialisable(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path, llm_available=False)
        resp = agent.answer("Any unusual sightings?")
        d = resp.to_dict()
        json.dumps(d)  # must not raise

    def test_confidence_high_when_tools_called(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent._client = _make_mock_client("Seen 3 species today.", tools=["get_top_species"])
        resp = agent.answer("How many species today?")
        assert resp.confidence == "high"

    def test_confidence_low_when_no_tools(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent._client = _make_mock_client("I don't have enough data.", tools=[])
        resp = agent.answer("What is the meaning of life?")
        assert resp.confidence == "low"

    def test_answer_survives_client_exception(self, tmp_path: Path) -> None:
        agent = _make_agent(tmp_path)
        agent._client = MagicMock()
        agent._client.available = True
        agent._client.run_with_tools = MagicMock(side_effect=RuntimeError("timeout"))
        resp = agent.answer("test")
        assert resp.error is not None
        assert resp.llm_available is True  # LLM was available, just errored


# ── _ToolExecutor ─────────────────────────────────────────────────────────────


class TestToolExecutor:
    def _make_executor(self, tmp_path: Path) -> _ToolExecutor:
        obs_path = tmp_path / "observations.jsonl"
        obs_path.write_text("")
        return _ToolExecutor(
            observations_path=str(obs_path),
            decisions_log_path=str(tmp_path / "decisions.jsonl"),
            daily_summaries_dir=str(tmp_path / "summaries"),
            vision_capture=None,
            notifier=None,
            current_mode="fixed_crop",
        )

    def test_unknown_tool_returns_error(self, tmp_path: Path) -> None:
        ex = self._make_executor(tmp_path)
        result = ex.execute("nonexistent_tool", {})
        assert "error" in result

    def test_read_recent_observations_returns_dict(self, tmp_path: Path) -> None:
        ex = self._make_executor(tmp_path)
        result = ex.execute("read_recent_observations", {"hours": 1.0})
        assert "total_detections" in result
        assert result["total_detections"] == 0  # empty log

    def test_get_top_species_returns_dict(self, tmp_path: Path) -> None:
        ex = self._make_executor(tmp_path)
        result = ex.execute("get_top_species", {"n": 5, "hours": 24.0})
        assert "species" in result

    def test_get_time_context_returns_dict(self, tmp_path: Path) -> None:
        ex = self._make_executor(tmp_path)
        result = ex.execute("get_time_context", {})
        assert "activity_period" in result

    def test_get_feeder_health_returns_status(self, tmp_path: Path) -> None:
        ex = self._make_executor(tmp_path)
        result = ex.execute("get_feeder_health", {})
        assert "status" in result

    def test_switch_mode_without_capture_returns_failure(self, tmp_path: Path) -> None:
        ex = self._make_executor(tmp_path)
        result = ex.execute("switch_detection_mode", {"new_mode": "yolo", "reason": "test"})
        assert result["success"] is False  # no vision_capture

    def test_switch_mode_with_capture_succeeds(self, tmp_path: Path) -> None:
        ex = self._make_executor(tmp_path)
        vc = MagicMock()
        vc.detection_mode = "fixed_crop"
        ex.vision_capture = vc
        result = ex.execute("switch_detection_mode", {"new_mode": "yolo", "reason": "test"})
        assert result["success"] is True
        assert vc.detection_mode == "yolo"

    def test_log_analyst_decision_writes_file(self, tmp_path: Path) -> None:
        ex = self._make_executor(tmp_path)
        result = ex.execute("log_analyst_decision", {
            "reasoning": "Nothing notable.",
            "actions_taken": [],
            "observations_summary": "3 detections in last hour.",
        })
        assert result["success"] is True
        dec_path = tmp_path / "decisions.jsonl"
        assert dec_path.exists()
        entry = json.loads(dec_path.read_text().strip())
        assert entry["reasoning"] == "Nothing notable."

    def test_push_notification_without_notifier_returns_failure(self, tmp_path: Path) -> None:
        ex = self._make_executor(tmp_path)
        result = ex.execute("push_notification", {"message": "test"})
        assert result["success"] is False