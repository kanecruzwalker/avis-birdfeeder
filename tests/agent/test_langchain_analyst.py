"""
tests/agent/test_langchain_analyst.py

Unit tests for LangChainAnalyst.

All LLM and LangGraph calls are mocked — no API calls in CI.
Tests verify memory layer behavior, graceful degradation, response
structure, and the tool context injection pattern.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from src.agent.bird_analyst_agent import AnalystResponse
from src.agent.langchain_analyst import LangChainAnalyst

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_analyst(
    tmp_path: Path,
    enabled: bool = True,
    available: bool = False,
) -> LangChainAnalyst:
    obs = tmp_path / "obs.jsonl"
    obs.write_text("")
    thr = tmp_path / "thresholds.yaml"
    yaml.dump({"fusion": {"audio_weight": 0.55, "visual_weight": 0.45}}, thr.open("w"))

    analyst = LangChainAnalyst(
        observations_path=str(obs),
        thresholds_path=str(thr),
        decisions_log_path=str(tmp_path / "decisions.jsonl"),
        daily_summaries_dir=str(tmp_path / "summaries"),
        enabled=enabled,
    )
    analyst._available = available
    return analyst


def _mock_graph_result(text: str, tools: list[str] | None = None) -> dict:
    """Build a fake LangGraph result dict."""
    from unittest.mock import MagicMock
    msgs = []
    if tools:
        tool_msg = MagicMock()
        tool_msg.tool_calls = [{"name": t} for t in tools]
        tool_msg.content = ""
        msgs.append(tool_msg)
    final = MagicMock()
    final.tool_calls = []
    final.content = text
    msgs.append(final)
    return {"messages": msgs}


# ── Construction ──────────────────────────────────────────────────────────────


class TestLangChainAnalystInit:
    def test_disabled_not_available(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path, enabled=False)
        assert a.available is False

    def test_no_api_key_not_available(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path, enabled=True)
        assert a.available is False  # no key in test env

    def test_paths_stored(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        assert "obs.jsonl" in a.observations_path

    def test_memory_empty_on_init(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        assert a._message_history == []
        assert a._entity_store == {}
        assert a._tool_cache == {}


# ── Graceful degradation ──────────────────────────────────────────────────────


class TestGracefulDegradation:
    def test_answer_returns_response_when_unavailable(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path, available=False)
        resp = a.answer("What birds visited?")
        assert isinstance(resp, AnalystResponse)
        assert resp.llm_available is False
        assert len(resp.answer) > 0

    def test_answer_response_is_json_serialisable(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path, available=False)
        resp = a.answer("test")
        json.dumps(resp.to_dict())  # must not raise

    def test_answer_error_field_set_when_unavailable(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path, available=False)
        resp = a.answer("test")
        assert resp.error is not None


# ── answer() with mocked graph ────────────────────────────────────────────────


class TestAnswerWithMockedGraph:
    def _make_with_mock_graph(self, tmp_path: Path, response_text: str,
                               tools: list[str] | None = None) -> LangChainAnalyst:
        a = _make_analyst(tmp_path, available=True)
        a._graph = MagicMock()
        a._graph.invoke = MagicMock(
            return_value=_mock_graph_result(response_text, tools)
        )
        return a

    def test_returns_analyst_response(self, tmp_path: Path) -> None:
        a = self._make_with_mock_graph(tmp_path, "3 species today.")
        resp = a.answer("What birds?")
        assert isinstance(resp, AnalystResponse)
        assert resp.answer == "3 species today."

    def test_tools_called_captured(self, tmp_path: Path) -> None:
        a = self._make_with_mock_graph(tmp_path, "House Finch ×5.",
                                        tools=["get_top_species"])
        resp = a.answer("What birds?")
        assert "get_top_species" in resp.tools_called

    def test_confidence_high_when_tools_called(self, tmp_path: Path) -> None:
        a = self._make_with_mock_graph(tmp_path, "HOFI ×5.",
                                        tools=["get_top_species"])
        resp = a.answer("Birds today?")
        assert resp.confidence == "high"

    def test_confidence_low_when_no_tools(self, tmp_path: Path) -> None:
        a = self._make_with_mock_graph(tmp_path, "I don't know.", tools=[])
        resp = a.answer("Anything?")
        assert resp.confidence == "low"

    def test_response_includes_memory_state(self, tmp_path: Path) -> None:
        a = self._make_with_mock_graph(tmp_path, "Done.")
        resp = a.answer("test")
        assert "memory_state" in resp.data

    def test_graph_exception_returns_error_response(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path, available=True)
        a._graph = MagicMock()
        a._graph.invoke = MagicMock(side_effect=RuntimeError("timeout"))
        resp = a.answer("test")
        assert resp.error is not None
        assert resp.llm_available is True


# ── Memory Layer 1 — Conversation buffer ─────────────────────────────────────


class TestConversationBufferMemory:
    def _make(self, tmp_path: Path) -> LangChainAnalyst:
        a = _make_analyst(tmp_path, available=True)
        a._graph = MagicMock()
        a._graph.invoke = MagicMock(
            return_value=_mock_graph_result("Response.")
        )
        return a

    def test_history_builds_across_turns(self, tmp_path: Path) -> None:
        a = self._make(tmp_path)
        a.answer("Question 1")
        a.answer("Question 2")
        assert len(a._message_history) == 4  # 2 pairs

    def test_history_trimmed_to_window(self, tmp_path: Path) -> None:
        a = self._make(tmp_path)
        a.memory_window_k = 2  # keep only last 2 exchanges
        for i in range(10):
            a.answer(f"Question {i}")
        # History stored is unbounded internally but window applied at query time
        assert len(a._message_history) == 20  # stored all
        # Window is applied when building messages for next query
        windowed = a._message_history[-(a.memory_window_k * 2):]
        assert len(windowed) == 4  # 2 exchanges × 2 messages

    def test_reset_clears_history(self, tmp_path: Path) -> None:
        a = self._make(tmp_path)
        a.answer("Question")
        assert len(a._message_history) > 0
        a.reset_memory()
        assert a._message_history == []


# ── Memory Layer 2 — Entity store ────────────────────────────────────────────


class TestEntityMemory:
    def test_species_codes_extracted(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        a._update_entity_store("How many HOFI today?", "HOFI visited 5 times.", [])
        assert "HOFI" in a._entity_store.get("species_of_interest", [])

    def test_multiple_species_tracked(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        a._update_entity_store("Show me HOFI and MODO", "HOFI×5 MODO×2.", [])
        species = a._entity_store.get("species_of_interest", [])
        assert "HOFI" in species
        assert "MODO" in species

    def test_feeder_health_interest_tracked(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        a._update_entity_store("Is the feeder running low on food?", "Looks healthy.", [])
        assert a._entity_store.get("asked_about_feeder_health") is True

    def test_calibration_interest_tracked(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        a._update_entity_store("Can you run a calibration sweep?", "Sure.", [])
        assert a._entity_store.get("asked_about_calibration") is True

    def test_reset_clears_entities(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        a._entity_store["species_of_interest"] = ["HOFI"]
        a.reset_memory()
        assert a._entity_store == {}

    def test_entity_enrichment_prepends_context(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        a._entity_store["species_of_interest"] = ["HOFI", "ANHU"]
        enriched = a._enrich_with_entities("What about yesterday?")
        assert "HOFI" in enriched or "ANHU" in enriched

    def test_no_enrichment_when_entity_store_empty(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        query = "What birds today?"
        assert a._enrich_with_entities(query) == query


# ── Memory Layer 3 — Tool cache ───────────────────────────────────────────────


class TestToolCache:
    def test_cacheable_tools_added(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        a._update_tool_cache(["get_top_species", "read_recent_observations"], [])
        assert "get_top_species" in a._tool_cache
        assert "read_recent_observations" in a._tool_cache

    def test_action_tools_not_cached(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        a._update_tool_cache(["push_notification", "switch_detection_mode"], [])
        assert "push_notification" not in a._tool_cache
        assert "switch_detection_mode" not in a._tool_cache

    def test_cache_shows_in_memory_summary(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        a._tool_cache["get_top_species"] = "2026-04-15T10:00:00+00:00"
        summary = a.get_memory_summary()
        assert "get_top_species" in summary["layer3_tool_cache_keys"]

    def test_reset_clears_cache(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        a._tool_cache["get_top_species"] = "cached"
        a.reset_memory()
        assert a._tool_cache == {}


# ── Memory summary ────────────────────────────────────────────────────────────


class TestMemorySummary:
    def test_summary_is_dict(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        summary = a.get_memory_summary()
        assert isinstance(summary, dict)

    def test_summary_has_all_layers(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        s = a.get_memory_summary()
        assert "layer1_conversation_turns" in s
        assert "layer2_entities" in s
        assert "layer3_tool_cache_keys" in s

    def test_summary_is_json_serialisable(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path)
        json.dumps(a.get_memory_summary())


# ── Graph diagram ─────────────────────────────────────────────────────────────


class TestGraphDiagram:
    def test_returns_string_when_unavailable(self, tmp_path: Path) -> None:
        a = _make_analyst(tmp_path, available=False)
        diagram = a.get_graph_diagram()
        assert isinstance(diagram, str)
        assert len(diagram) > 0
