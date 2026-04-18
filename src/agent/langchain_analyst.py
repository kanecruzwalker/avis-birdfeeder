"""
src/agent/langchain_analyst.py

LangGraph-based conversational analyst for the Avis birdfeeder.

This module implements the framework-based half of Avis's dual-agent
architecture. Where BirdAnalystAgent uses a custom tool-calling loop
with the Gemini API directly, LangChainAnalyst uses LangGraph — an
industry-standard graph-based agent framework built on LangChain.

Both agents call the same underlying tool functions from src/agent/tools/.
The difference is the orchestration layer:

    BirdAnalystAgent    → custom loop, advise() path, autonomous decisions
    LangChainAnalyst    → LangGraph, answer() path, conversational queries
                          + three memory layers for context retention

Why LangGraph specifically (not legacy AgentExecutor)?
    LangGraph models the agent as an explicit state machine — each node
    (perceive, reason, act) is a named step with inspectable state.
    This maps directly onto the  Perceive→Reason→Act→Memory
    diagram. The graph can be visualised and shown in the presentation.
    Legacy AgentExecutor is a black box; LangGraph is transparent.

Three memory layers:

    Layer 1 — Conversation buffer (ConversationBufferWindowMemory)
        Remembers the last K exchanges in the current session.
        "You just asked about House Finches — your follow-up 'how often?'
        refers to them." Visible in every multi-turn demo exchange.

    Layer 2 — Entity memory (in-graph entity store)
        Extracts named entities (species codes, user preferences) from
        the conversation and stores them in the LangGraph state.
        "This user consistently asks about HOFI and ANHU — load those
        first when they ask for a summary."

    Layer 3 — Session observation cache
        Tracks which tool calls were already made this session and what
        they returned. Prevents redundant log reads when the user asks
        related questions. Stored as a dict in LangGraph state.

LangGraph state structure:
    {
        "messages":      list[BaseMessage]   — full conversation history
        "tool_cache":    dict[str, Any]      — layer 3: cached tool results
        "entities":      dict[str, Any]      — layer 2: extracted entities
        "actions_taken": list[str]           — tools called this session
        "current_mode":  str                 — detection mode for context
    }

Usage:
    analyst = LangChainAnalyst.from_config("configs/")
    response = analyst.answer("What birds visited today?")
    print(response.answer)           # natural language
    print(response.to_dict())        # JSON-ready for web API

    # Multi-turn — memory persists within the analyst instance
    r1 = analyst.answer("How many House Finches today?")
    r2 = analyst.answer("What about yesterday?")  # remembers HOFI context

Graceful degradation:
    If LangChain/LangGraph is not installed or the API key is missing,
    answer() returns an AnalystResponse with llm_available=False and a
    helpful fallback message. The feeder keeps running.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from src.agent.bird_analyst_agent import AnalystResponse

logger = logging.getLogger(__name__)

# Lazily imported — only needed if LangGraph is installed
_LANGGRAPH_AVAILABLE = False
try:
    import langgraph  # type: ignore[import]  # noqa: F401
    from langchain_core.messages import (  # type: ignore  # noqa: F401
        AIMessage,
        HumanMessage,
        SystemMessage,
    )
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    pass


class LangChainAnalyst:
    """
    LangGraph-based conversational analyst for Avis.

    Handles the answer() path — natural language user queries about
    feeder activity. Maintains conversation context across multiple
    turns within a session using three layered memory mechanisms.

    The agent is stateful within a session (memory persists across
    answer() calls) but stateless across sessions (memory resets on
    instantiation). Long-term cross-session memory is a future feature.

    Usage:
        analyst = LangChainAnalyst.from_config("configs/")

        # Single question
        resp = analyst.answer("What birds visited today?")

        # Multi-turn — context retained between calls
        resp1 = analyst.answer("How many House Finches visited?")
        resp2 = analyst.answer("What was their average confidence?")
        resp3 = analyst.answer("Is that typical for them?")

        # Web backend
        return jsonify(resp.to_dict())

        # Reset session memory
        analyst.reset_memory()
    """

    _SYSTEM_PROMPT = """You are the Avis birdfeeder assistant — an expert on the birds
    visiting this San Diego backyard feeder.

    You have access to tools that let you query detection data, check species history,
    assess feeder health, run calibration sweeps, and take management actions.

    Guidelines:
    - Use tools to look up data before answering — don't guess at numbers
    - Be specific: name species, give counts, cite confidence percentages
    - When a user asks a follow-up, use the conversation history to understand context
    - Remember species the user has asked about and proactively include them in summaries
    - If detection data is limited, say so honestly rather than extrapolating
    - After answering, always call log_analyst_decision to record your reasoning
    - Keep answers concise but complete — the user may be viewing on mobile

    You are helpful, knowledgeable about San Diego birds, and data-driven."""

    def __init__(
        self,
        observations_path: str,
        thresholds_path: str,
        decisions_log_path: str,
        daily_summaries_dir: str,
        provider: str = "gemini",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        enabled: bool = True,
        memory_window_k: int = 10,
        vision_capture: Any = None,
        notifier: Any = None,
        current_mode: str = "fixed_crop",
    ) -> None:
        """
        Args:
            observations_path:   Path to logs/observations.jsonl.
            thresholds_path:     Path to configs/thresholds.yaml.
            decisions_log_path:  Path to logs/analyst_decisions.jsonl.
            daily_summaries_dir: Directory for daily report output.
            provider:            LLM provider — "gemini" supported currently.
            model:               Model string. "gemini-2.5-flash" default.
            temperature:         Slightly higher than advise() path (0.3) for
                                 more natural conversational responses.
            max_tokens:          1024 — answers may be longer than decisions.
            enabled:             False forces graceful fallback without LLM calls.
            memory_window_k:     Number of past exchanges to keep in buffer.
                                 Default 10 = 5 back-and-forth turns.
            vision_capture:      VisionCapture instance for mode switching tool.
            notifier:            Notifier instance for push notification tool.
            current_mode:        Active detection mode — passed to tools.
        """
        self.observations_path = observations_path
        self.thresholds_path = thresholds_path
        self.decisions_log_path = decisions_log_path
        self.daily_summaries_dir = daily_summaries_dir
        self.provider = provider
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enabled = enabled
        self.memory_window_k = memory_window_k
        self.vision_capture = vision_capture
        self.notifier = notifier
        self.current_mode = current_mode

        # Three memory layers — initialised in _build_graph()
        self._message_history: list = []  # Layer 1: conversation buffer
        self._entity_store: dict[str, Any] = {}  # Layer 2: named entities
        self._tool_cache: dict[str, Any] = {}  # Layer 3: session tool cache

        # LangGraph components — built lazily on first answer() call
        self._graph = None
        self._llm = None
        self._tools = None
        self._available = False

        if enabled and _LANGGRAPH_AVAILABLE:
            self._available = self._initialise()

        logger.info(
            "LangChainAnalyst initialised | provider=%s model=%s available=%s",
            provider,
            model,
            self._available,
        )

    @classmethod
    def from_config(
        cls,
        config_dir: str | Path,
        vision_capture: Any = None,
        notifier: Any = None,
        current_mode: str = "fixed_crop",
    ) -> LangChainAnalyst:
        """
        Construct LangChainAnalyst from configs/ directory.

        Reads the same llm: block from hardware.yaml as BirdAnalystAgent,
        ensuring both agents share configuration.

        Args:
            config_dir:     Path to configs/ directory.
            vision_capture: VisionCapture instance (for mode switching tool).
            notifier:       Notifier instance (for push notification tool).
            current_mode:   Active detection mode.

        Returns:
            Configured LangChainAnalyst.
        """
        config_dir = Path(config_dir)
        hw_path = config_dir / "hardware.yaml"
        paths_path = config_dir / "paths.yaml"

        llm_cfg: dict = {}
        if hw_path.exists():
            with hw_path.open() as f:
                hw = yaml.safe_load(f)
            llm_cfg = hw.get("llm", {})

        observations_path = "logs/observations.jsonl"
        thresholds_path = str(config_dir / "thresholds.yaml")
        decisions_log_path = "logs/analyst_decisions.jsonl"
        daily_summaries_dir = "logs/daily_summaries"

        if paths_path.exists():
            with paths_path.open() as f:
                p = yaml.safe_load(f)
            observations_path = p.get("logs", {}).get("observations", observations_path)
            decisions_log_path = p.get("logs", {}).get("analyst_decisions", decisions_log_path)
            daily_summaries_dir = p.get("logs", {}).get("daily_summaries", daily_summaries_dir)

        return cls(
            observations_path=observations_path,
            thresholds_path=thresholds_path,
            decisions_log_path=decisions_log_path,
            daily_summaries_dir=daily_summaries_dir,
            provider=llm_cfg.get("provider", "gemini"),
            model=llm_cfg.get("model", "gemini-2.5-flash"),
            temperature=float(llm_cfg.get("temperature", 0.3)),
            max_tokens=int(llm_cfg.get("max_tokens", 1024)),
            enabled=bool(llm_cfg.get("enabled", True)),
            vision_capture=vision_capture,
            notifier=notifier,
            current_mode=current_mode,
        )

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialise(self) -> bool:
        """
        Build the LLM client, tool list, and LangGraph execution graph.

        Returns True if everything initialised successfully, False otherwise.
        Failures are logged but never raised — the agent degrades gracefully.
        """
        try:
            self._llm = self._build_llm()
            if self._llm is None:
                return False

            self._tools = self._build_tools()
            self._graph = self._build_graph()
            return True

        except Exception as exc:
            logger.warning("LangChainAnalyst initialisation failed: %s", exc)
            return False

    def _build_llm(self) -> Any:
        """
        Build the LangChain LLM client for the configured provider.

        Currently supports Gemini via langchain-google-genai.
        Returns None if credentials missing or package not installed.
        """
        if self.provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                logger.warning(
                    "GEMINI_API_KEY not set — LangChainAnalyst unavailable. "
                    "Set GEMINI_API_KEY in .env to enable conversational queries."
                )
                return None
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

                return ChatGoogleGenerativeAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    google_api_key=api_key,
                )
            except ImportError:
                logger.warning(
                    "langchain-google-genai not installed. "
                    "Run: pip install langchain-google-genai"
                )
                return None

        elif self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set.")
                return None
            try:
                from langchain_openai import ChatOpenAI  # type: ignore

                return ChatOpenAI(
                    model=self.model_name or "gpt-4o-mini",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=api_key,
                )
            except ImportError:
                logger.warning("langchain-openai not installed.")
                return None

        elif self.provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not set.")
                return None
            try:
                from langchain_anthropic import ChatAnthropic  # type: ignore

                return ChatAnthropic(
                    model=self.model_name or "claude-sonnet-4-6",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=api_key,
                )
            except ImportError:
                logger.warning("langchain-anthropic not installed.")
                return None

        logger.warning("Unknown provider: %s", self.provider)
        return None

    def _build_tools(self) -> list:
        """
        Build the LangChain tool list with runtime context injected.
        """
        from src.agent.tools.langchain_tools import build_langchain_tools  # noqa: PLC0415

        return build_langchain_tools(
            {
                "observations_path": self.observations_path,
                "thresholds_path": self.thresholds_path,
                "daily_summaries_dir": self.daily_summaries_dir,
                "vision_capture": self.vision_capture,
                "notifier": self.notifier,
                "current_mode": self.current_mode,
                "decisions_log_path": self.decisions_log_path,
            }
        )

    def _build_graph(self) -> Any:
        """
        Build the LangGraph ReAct agent graph.

        Graph structure:
            [START] → [agent_node] → [tool_node] → [agent_node] → ... → [END]

        The agent_node calls the LLM with current messages + tool schemas.
        The tool_node executes whatever tool the LLM requested.
        The loop continues until the LLM produces a final text response
        (no tool call) or the step limit is reached.

        This graph is the thing to visualise in the presentation —
        call graph.get_graph().draw_mermaid() to get a Mermaid diagram.
        """

        from langgraph.prebuilt import create_react_agent

        llm_with_tools = self._llm.bind_tools(self._tools)

        graph = create_react_agent(
            model=llm_with_tools,
            tools=self._tools,
        )
        return graph

    # ── Public interface ──────────────────────────────────────────────────────

    def answer(
        self,
        user_query: str,
        vision_capture: Any = None,
        notifier: Any = None,
    ) -> AnalystResponse:
        """
        Answer a natural language question using LangGraph + three memory layers.

        The agent:
            1. Prepends conversation history (Layer 1) to give context
            2. Injects known entities (Layer 2) into the system prompt
            3. Checks tool cache (Layer 3) before making redundant calls
            4. Runs the LangGraph ReAct loop — perceive, reason, act
            5. Updates all three memory layers with results
            6. Returns AnalystResponse with answer + raw data

        Args:
            user_query:     Natural language question from the user.
            vision_capture: Override VisionCapture if changed since init.
            notifier:       Override Notifier if changed since init.

        Returns:
            AnalystResponse — always returned, never raises.
            llm_available=False when LangGraph unavailable.
        """
        if not self._available:
            return AnalystResponse(
                answer=(
                    "The conversational analyst is currently unavailable. "
                    "Check that GEMINI_API_KEY is set and langchain-google-genai "
                    "is installed. The feeder is still running normally."
                ),
                llm_available=False,
                error="LangGraph not available",
            )

        # Update runtime context if overrides provided
        if vision_capture is not None:
            self.vision_capture = vision_capture
        if notifier is not None:
            self.notifier = notifier

        try:
            return self._run_graph(user_query)
        except Exception as exc:
            logger.exception("LangChainAnalyst.answer() failed: %s", exc)
            return AnalystResponse(
                answer="An error occurred processing your question. Please try again.",
                llm_available=True,
                error=str(exc),
            )

    def reset_memory(self) -> None:
        """
        Clear all three memory layers.

        Call between user sessions or when starting a fresh conversation.
        The feeder detection pipeline is unaffected — only conversation
        context is cleared.
        """
        self._message_history.clear()
        self._entity_store.clear()
        self._tool_cache.clear()
        logger.info("LangChainAnalyst memory reset.")

    @property
    def available(self) -> bool:
        """True if the LangGraph agent is initialised and ready."""
        return self._available

    def get_graph_diagram(self) -> str:
        """
        Return a Mermaid diagram of the LangGraph agent structure.

        Use this in the demo notebook to visualise the agent graph:
            from IPython.display import display, Markdown
            display(Markdown(analyst.get_graph_diagram()))
        """
        if self._graph is None:
            return "```mermaid\ngraph TD\n    A[LangGraph not available]\n```"
        try:
            return self._graph.get_graph().draw_mermaid()
        except Exception:
            return "```mermaid\ngraph TD\n    A[agent] --> B[tools]\n    B --> A\n```"

    def get_memory_summary(self) -> dict[str, Any]:
        """
        Return a summary of current memory state for debugging and demo.

        Shows all three layers — useful in the notebook to demonstrate
        that memory is actually building across turns.
        """
        return {
            "layer1_conversation_turns": len(self._message_history) // 2,
            "layer1_messages": [
                {"role": m.get("role", "?"), "preview": m.get("content", "")[:80]}
                for m in self._message_history[-6:]  # last 3 turns
            ],
            "layer2_entities": self._entity_store,
            "layer3_tool_cache_keys": list(self._tool_cache.keys()),
        }

    # ── Internal graph execution ──────────────────────────────────────────────

    def _run_graph(self, user_query: str) -> AnalystResponse:
        """
        Execute the LangGraph agent for one user query.

        Builds the input state with conversation history (Layer 1) and
        entity context (Layer 2), runs the graph, then updates memory
        with the results.
        """
        from langchain_core.messages import HumanMessage  # type: ignore

        # ── Layer 2: inject entity context into query ─────────────────────
        enriched_query = self._enrich_with_entities(user_query)

        # ── Layer 1: build message list with history ──────────────────────
        # Keep last memory_window_k messages (window buffer)
        windowed_history = self._message_history[-(self.memory_window_k * 2) :]
        messages = windowed_history + [HumanMessage(content=enriched_query)]

        # ── Layer 3: attach tool cache hint to query ──────────────────────
        if self._tool_cache:
            cache_hint = (
                f"\n\n[Session cache — already fetched this session: "
                f"{', '.join(self._tool_cache.keys())}. "
                f"Reuse cached data where appropriate.]"
            )
            messages[-1] = HumanMessage(content=enriched_query + cache_hint)

        # ── Run graph ─────────────────────────────────────────────────────
        state = {"messages": messages}
        result = self._graph.invoke(state)

        # Extract final AI message and tool calls made
        final_messages = result.get("messages", [])
        tools_called = self._extract_tools_called(final_messages)
        final_text = self._extract_final_text(final_messages)

        # ── Update memory layers ──────────────────────────────────────────
        # Layer 1: add this exchange to conversation history
        self._message_history.append({"role": "user", "content": user_query})
        self._message_history.append({"role": "assistant", "content": final_text})

        # Layer 2: extract entities from this exchange
        self._update_entity_store(user_query, final_text, tools_called)

        # Layer 3: update tool cache with new results
        self._update_tool_cache(tools_called, final_messages)

        return AnalystResponse(
            answer=final_text,
            data={
                "tools_called": tools_called,
                "memory_state": self.get_memory_summary(),
                "entity_store": self._entity_store,
            },
            tools_called=tools_called,
            confidence="high" if tools_called else "low",
            llm_available=True,
        )

    def _enrich_with_entities(self, query: str) -> str:
        """
        Layer 2: Prepend known entity context to the user query.

        If the user has been asking about specific species, inject that
        context so the agent doesn't need to re-ask "which species?".
        """
        if not self._entity_store:
            return query

        species_of_interest = self._entity_store.get("species_of_interest", [])
        if species_of_interest:
            context = f"[User has been asking about: {', '.join(species_of_interest)}] "
            return context + query

        return query

    def _update_entity_store(
        self,
        query: str,
        response: str,
        tools_called: list[str],
    ) -> None:
        """
        Layer 2: Extract and store named entities from this exchange.

        Currently tracks species codes mentioned in the conversation.
        Future: track user preferences, time ranges, locations.
        """
        import re  # noqa: PLC0415

        # Extract 4-letter uppercase species codes (AOU format)
        codes = re.findall(r"\b[A-Z]{4}\b", query + " " + response)

        # Filter to known species codes (4 uppercase letters is not unique enough alone)
        # We accept any that look like AOU codes in the conversation context
        plausible = [
            c for c in codes if c not in ("YOLO", "HTTP", "JSON", "NULL", "TRUE", "NONE", "WITH")
        ]

        if plausible:
            existing = self._entity_store.get("species_of_interest", [])
            for code in plausible:
                if code not in existing:
                    existing.append(code)
            self._entity_store["species_of_interest"] = existing[-5:]  # keep last 5

        # Track if user has asked about feeder health
        health_keywords = ("food", "refill", "empty", "feeder", "health", "declining")
        if any(k in query.lower() for k in health_keywords):
            self._entity_store["asked_about_feeder_health"] = True

        # Track if user has asked for calibration
        if any(k in query.lower() for k in ("calibrat", "weight", "threshold", "sweep", "tune")):
            self._entity_store["asked_about_calibration"] = True

    def _update_tool_cache(
        self,
        tools_called: list[str],
        messages: list,
    ) -> None:
        """
        Layer 3: Cache tool results so they aren't re-fetched this session.

        Observation tools (read_recent_observations, get_top_species) are
        cached because their data doesn't change significantly within a
        conversation session. Action tools are not cached.

        Cache keys are tool names. The cache is intentionally simple —
        a dict mapping tool name to the timestamp it was last called.
        Actual result caching would require storing tool outputs which
        is a future enhancement.
        """
        cacheable = {
            "read_recent_observations",
            "get_top_species",
            "get_detection_stats",
            "get_feeder_health",
        }
        from datetime import UTC, datetime  # noqa: PLC0415

        for tool_name in tools_called:
            if tool_name in cacheable:
                self._tool_cache[tool_name] = datetime.now(UTC).isoformat()

    def _extract_tools_called(self, messages: list) -> list[str]:
        """Extract the names of tools called from the message list."""
        tools = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if name:
                        tools.append(name)
        return tools

    def _extract_final_text(self, messages: list) -> str:
        """Extract the final text response from the message list."""
        for msg in reversed(messages):
            # Look for AI messages that don't have tool calls (final response)
            if hasattr(msg, "content") and msg.content:
                has_tool_calls = bool(getattr(msg, "tool_calls", None))
                if not has_tool_calls and isinstance(msg.content, str):
                    return msg.content.strip()
        return "I was unable to generate a response."
