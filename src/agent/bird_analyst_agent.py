"""
src/agent/bird_analyst_agent.py

LLM-based agentic analyst for the Avis birdfeeder system.

BirdAnalystAgent wraps a large language model (Gemini by default) and
gives it a set of tools it can call to perceive the feeder state and
take actions.
"Agentic AI" requirement:

    Perceive → Reason → Act → Memory

    Perceive: tools read observations.jsonl and system state
    Reason:   LLM decides what to do based on what it perceives
    Act:      LLM calls tools to switch modes, push notifications, write reports
    Memory:   every reasoning cycle is logged to analyst_decisions.jsonl

Two modes of operation:

    advise(context) — proactive mode, called by ExperimentOrchestrator
        The agent is given current system state and recent observations,
        reasons about what to do, and returns an AnalystDecision.
        The orchestrator executes the decision. This replaces the fixed
        30-minute timer logic with LLM reasoning.

    answer(user_query) — reactive mode, called by web backend (future)
        The agent receives a natural language question, calls appropriate
        observation tools, and returns a structured AnalystResponse with
        a natural language answer and the raw data used.

Provider abstraction:
    The LLM provider is configured via hardware.yaml (llm.provider).
    Currently supports "gemini" with a clean interface for adding
    "openai" or "anthropic" later. The provider abstraction is
    intentionally thin — all three APIs support function/tool calling
    with similar patterns.

Graceful degradation:
    If the LLM is unavailable (no API key, network down, quota exceeded),
    advise() returns None. ExperimentOrchestrator falls back to its fixed
    schedule. The feeder keeps running regardless of LLM availability.

Config keys consumed from hardware.yaml (under llm:):
    provider:              "gemini" (default) | "openai" | "anthropic"
    model:                 Model string, e.g. "gemini-2.0-flash"
    temperature:           0.2 recommended — consistent decisions over creativity
    max_tokens:            512 — enough for reasoning + tool calls
    enabled:               true/false — set false to force fallback
    fallback_to_schedule:  true — orchestrator falls back if advise() returns None

Environment variables:
    GEMINI_API_KEY     — required when provider is "gemini"
    OPENAI_API_KEY     — required when provider is "openai"
    ANTHROPIC_API_KEY  — required when provider is "anthropic"
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.agent.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


# ── Response data structures ──────────────────────────────────────────────────


@dataclass
class AnalystDecision:
    """
    Structured output from BirdAnalystAgent.advise().

    Returned to ExperimentOrchestrator which executes the decisions.
    All fields have safe defaults so partial responses don't crash the loop.

    Attributes:
        reasoning:          The LLM's chain-of-thought (logged to decisions log).
        switch_mode:        New detection mode to apply, or None if no switch.
        push_message:       Text to push to Pushover, or None if no push.
        generate_report:    True if the agent decided to generate a daily report.
        feeder_alert:       Non-None if the agent flagged a feeder health concern.
        tools_called:       Names of tools the LLM called this cycle.
        llm_available:      False if this decision came from fallback logic.
    """

    reasoning: str = ""
    switch_mode: str | None = None
    push_message: str | None = None
    generate_report: bool = False
    feeder_alert: str | None = None
    tools_called: list[str] = field(default_factory=list)
    llm_available: bool = True


@dataclass
class AnalystResponse:
    """
    Structured output from BirdAnalystAgent.answer().

    Returned to the web backend for user-facing queries.
    Designed to be JSON-serialisable — the web API can return it directly.

    Attributes:
        answer:         Natural language answer to the user's question.
        data:           Raw structured data from the tools called.
        tools_called:   Which tools were used to answer.
        confidence:     "high" | "medium" | "low" — how confident the agent is.
        llm_available:  False if the answer was generated without LLM reasoning.
        error:          Non-None if something went wrong.
    """

    answer: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    tools_called: list[str] = field(default_factory=list)
    confidence: str = "medium"
    llm_available: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict for web API responses."""
        return {
            "answer": self.answer,
            "data": self.data,
            "tools_called": self.tools_called,
            "confidence": self.confidence,
            "llm_available": self.llm_available,
            "error": self.error,
            "generated_at": datetime.now(UTC).isoformat(),
        }


# ── LLM provider clients ──────────────────────────────────────────────────────


class _GeminiClient:
    """
    Thin wrapper around langchain-google-genai for tool-calling conversations.

    Uses ChatGoogleGenerativeAI instead of google-generativeai SDK directly —
    avoids a protobuf version conflict with tensorflow-cpu in the dev environment.
    The external interface (available, run_with_tools) is unchanged.
    """

    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import]

            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set in environment.")

            self._llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=api_key,
            )
            self._available = True
            logger.info("Gemini client initialised via langchain-google-genai | model=%s", model)
        except Exception as exc:
            logger.warning("Gemini client failed to initialise: %s", exc)
            self._llm = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def run_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tool_executor: _ToolExecutor,
        max_rounds: int = 6,
    ) -> tuple[str, list[str]]:
        """
        Run a multi-turn tool-calling conversation via LangChain invoke.
        """
        if not self._available:
            return "", []

        try:
            from langchain_core.messages import (  # type: ignore[import]
                HumanMessage,
                SystemMessage,
                ToolMessage,
            )

            tools = self._build_langchain_tools(tool_executor)
            llm_with_tools = self._llm.bind_tools(tools)

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]

            tools_called = []
            rounds = 0

            while rounds < max_rounds:
                response = llm_with_tools.invoke(messages)
                messages.append(response)

                if not response.tool_calls:
                    return response.content or "", tools_called

                for tc in response.tool_calls:
                    name = tc["name"]
                    args = tc["args"]
                    tool_id = tc.get("id", name)
                    tools_called.append(name)

                    logger.debug("LLM calling tool: %s(%s)", name, args)
                    result = tool_executor.execute(name, args)

                    messages.append(
                        ToolMessage(
                            content=json.dumps(result, default=str),
                            tool_call_id=tool_id,
                        )
                    )

                rounds += 1

            return getattr(response, "content", "") or "", tools_called

        except Exception as exc:
            logger.warning("Gemini run_with_tools failed: %s", exc)
            return "", []

    def _build_langchain_tools(self, tool_executor: _ToolExecutor) -> list:
        """Build LangChain tools with runtime context injected."""
        from src.agent.tools.langchain_tools import build_langchain_tools  # noqa: PLC0415

        return build_langchain_tools(
            {
                "observations_path": tool_executor.observations_path,
                "thresholds_path": "configs/thresholds.yaml",
                "daily_summaries_dir": tool_executor.daily_summaries_dir,
                "vision_capture": tool_executor.vision_capture,
                "notifier": tool_executor.notifier,
                "current_mode": tool_executor.current_mode,
                "decisions_log_path": tool_executor.decisions_log_path,
            }
        )


class _ToolExecutor:
    """
    Executes tool calls on behalf of the LLM.

    Bridges the gap between what the LLM requests (JSON args) and what
    the tool functions need (some args injected at runtime by the agent).

    Runtime-injected args (vision_capture, notifier, paths) are set by
    BirdAnalystAgent before each reasoning cycle. The LLM never sees or
    supplies these — it only provides the args listed in the tool schema.
    """

    def __init__(
        self,
        observations_path: str,
        decisions_log_path: str,
        daily_summaries_dir: str,
        vision_capture: Any = None,
        notifier: Any = None,
        current_mode: str = "fixed_crop",
    ) -> None:
        self.observations_path = observations_path
        self.decisions_log_path = decisions_log_path
        self.daily_summaries_dir = daily_summaries_dir
        self.vision_capture = vision_capture
        self.notifier = notifier
        self.current_mode = current_mode

    def execute(self, tool_name: str, args: dict[str, Any]) -> Any:
        """
        Execute a tool by name with the given LLM-supplied arguments.

        Injects runtime context (paths, vision_capture, notifier) that
        the LLM doesn't know about. Returns a JSON-serialisable result.
        """
        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            logger.warning("Unknown tool requested by LLM: %s", tool_name)
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            # Inject runtime args that aren't in the LLM schema
            if tool_name in (
                "read_recent_observations",
                "get_detection_stats",
                "query_species_history",
                "get_top_species",
                "get_feeder_health",
            ):
                return fn(observations_path=self.observations_path, **args)

            elif tool_name == "switch_detection_mode":
                return fn(vision_capture=self.vision_capture, **args)

            elif tool_name == "generate_daily_report":
                return fn(
                    observations_path=self.observations_path,
                    daily_summaries_dir=self.daily_summaries_dir,
                    **args,
                )

            elif tool_name == "push_notification":
                return fn(notifier=self.notifier, **args)

            elif tool_name == "log_analyst_decision":
                return fn(
                    decisions_log_path=self.decisions_log_path,
                    mode_at_decision=self.current_mode,
                    **args,
                )

            elif tool_name == "get_time_context":
                from src.agent.tools.system_tools import get_time_context  # noqa: PLC0415

                return get_time_context()

            elif tool_name == "get_current_system_status":
                from src.agent.tools.system_tools import get_current_system_status  # noqa: PLC0415

                return get_current_system_status(**args)

            else:
                return fn(**args)

        except Exception as exc:
            logger.exception("Tool %s raised exception: %s", tool_name, exc)
            return {"error": str(exc)}


# ── Main agent class ──────────────────────────────────────────────────────────


class BirdAnalystAgent:
    """
    LLM-based agentic analyst for the Avis birdfeeder.

    Wraps a language model and gives it tools to perceive feeder state
    and take actions. Called by ExperimentOrchestrator every N minutes
    to advise on detection mode, feeder health, and notifications.

    Also provides answer() for reactive user queries — this will be called
    by the web backend when it's built.

    Usage (orchestrator integration):
        agent = BirdAnalystAgent.from_config("configs/")
        decision = agent.advise(vision_capture=vc, notifier=n, ...)
        if decision and decision.switch_mode:
            orchestrator._apply_detection_mode(decision.switch_mode)

    Usage (web backend, future):
        response = agent.answer("What birds visited today?")
        return jsonify(response.to_dict())
    """

    # System prompt sent to the LLM at the start of every reasoning cycle.
    # Tuned for consistent, conservative decisions — we don't want the agent
    # switching modes every cycle or spamming notifications.
    _ADVISE_SYSTEM_PROMPT = """You are the Avis birdfeeder analyst AI.
    Your job is to monitor a backyard bird feeder in San Diego, CA and make
    decisions that improve detection quality and keep the feeder owner informed.

    You have access to tools that let you read detection data, check system status,
    switch detection modes, generate reports, send notifications, and log your reasoning.

    Guidelines for good decisions:
    - Read recent observations and check detection stats BEFORE making any decision
    - Only switch detection modes if the data clearly shows one mode outperforms the other
    (mean confidence difference > 0.05 or detection rate difference > 30%)
    - Don't switch modes more than once per hour — give each mode time to produce data
    - Only push notifications for genuinely notable events (unusual species, feeder alert,
    daily summary) — not for every detection
    - Check feeder health periodically (every few hours during active periods)
    - ALWAYS call log_analyst_decision as your final action to record your reasoning

    Be analytical and conservative. Explain your reasoning clearly."""

    _ANSWER_SYSTEM_PROMPT = """You are the Avis birdfeeder assistant.
    Answer questions about bird activity at the feeder clearly and helpfully.
    Use the available tools to look up data before answering.
    Be specific with numbers and species names.
    If you don't have enough data to answer confidently, say so."""

    def __init__(
        self,
        observations_path: str,
        decisions_log_path: str,
        daily_summaries_dir: str,
        provider: str = "gemini",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 512,
        enabled: bool = True,
    ) -> None:
        self.observations_path = observations_path
        self.decisions_log_path = decisions_log_path
        self.daily_summaries_dir = daily_summaries_dir
        self.enabled = enabled
        self._provider = provider

        self._client = None
        if enabled:
            self._client = self._init_client(provider, model, temperature, max_tokens)

        status = "enabled" if (self._client and self._client.available) else "disabled/fallback"
        logger.info(
            "BirdAnalystAgent initialised | provider=%s model=%s status=%s",
            provider,
            model,
            status,
        )

    def _init_client(self, provider: str, model: str, temperature: float, max_tokens: int) -> Any:
        """
        Initialise the LLM client for the configured provider.

        Returns None if the provider is unknown or initialisation fails.
        The agent degrades gracefully — callers check client.available.
        """
        if provider == "gemini":
            return _GeminiClient(model=model, temperature=temperature, max_tokens=max_tokens)
        elif provider == "openai":
            # Placeholder for OpenAI integration — same interface, different SDK
            logger.warning(
                "OpenAI provider not yet implemented. "
                "Set provider to 'gemini' or contribute an _OpenAIClient class."
            )
            return None
        elif provider == "anthropic":
            logger.warning(
                "Anthropic provider not yet implemented. "
                "Set provider to 'gemini' or contribute an _AnthropicClient class."
            )
            return None
        else:
            logger.warning("Unknown LLM provider: %s", provider)
            return None

    @classmethod
    def from_config(cls, config_dir: str | Path) -> BirdAnalystAgent:
        """
        Construct BirdAnalystAgent from configs/ directory.

        Reads hardware.yaml for llm: block and paths.yaml for log paths.
        All llm: keys are optional — the agent has safe defaults for all.

        Args:
            config_dir: Path to the configs/ directory.

        Returns:
            Configured BirdAnalystAgent.
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
        decisions_log_path = "logs/analyst_decisions.jsonl"
        daily_summaries_dir = "logs/daily_summaries"

        if paths_path.exists():
            with paths_path.open() as f:
                paths_cfg = yaml.safe_load(f)
            observations_path = paths_cfg.get("logs", {}).get("observations", observations_path)
            decisions_log_path = paths_cfg.get("logs", {}).get(
                "analyst_decisions", decisions_log_path
            )
            daily_summaries_dir = paths_cfg.get("logs", {}).get(
                "daily_summaries", daily_summaries_dir
            )

        return cls(
            observations_path=observations_path,
            decisions_log_path=decisions_log_path,
            daily_summaries_dir=daily_summaries_dir,
            provider=llm_cfg.get("provider", "gemini"),
            model=llm_cfg.get("model", "gemini-2.5-flash"),
            temperature=float(llm_cfg.get("temperature", 0.2)),
            max_tokens=int(llm_cfg.get("max_tokens", 512)),
            enabled=bool(llm_cfg.get("enabled", True)),
        )

    @property
    def llm_available(self) -> bool:
        """True if the LLM client is initialised and available."""
        return (
            self.enabled and self._client is not None and getattr(self._client, "available", False)
        )

    # ── Proactive advise() ────────────────────────────────────────────────────

    def advise(
        self,
        vision_capture: Any = None,
        notifier: Any = None,
        current_mode: str = "fixed_crop",
        uptime_seconds: float = 0.0,
        window_elapsed_minutes: float = 0.0,
        window_total_minutes: float = 30.0,
        last_detection: dict[str, Any] | None = None,
    ) -> AnalystDecision | None:
        """
        Run one LLM reasoning cycle and return a structured decision.

        Called by ExperimentOrchestrator every N minutes. The orchestrator
        executes whatever the agent decides — it does not interpret the
        LLM's raw output, only the structured AnalystDecision fields.

        If the LLM is unavailable, returns None. The orchestrator falls
        back to its fixed schedule when this happens.

        Args:
            vision_capture:         VisionCapture instance (for mode switching).
            notifier:               Notifier instance (for push notifications).
            current_mode:           Currently active detection mode.
            uptime_seconds:         Seconds since orchestrator started.
            window_elapsed_minutes: Minutes elapsed in current A/B window.
            window_total_minutes:   Configured A/B window length.
            last_detection:         Dict with species/confidence/minutes_ago, or None.

        Returns:
            AnalystDecision if LLM reasoning completed, None if unavailable.
        """
        if not self.llm_available:
            logger.debug("LLM not available — advise() returning None (fallback to schedule).")
            return None

        last = last_detection or {}
        user_message = (
            f"Current detection mode: {current_mode}\n"
            f"Uptime: {uptime_seconds / 3600:.1f} hours\n"
            f"Current A/B window: {window_elapsed_minutes:.0f}/{window_total_minutes:.0f} min\n"
            f"Last detection: {last.get('species', 'none')} "
            f"{last.get('minutes_ago', 'N/A')} min ago "
            f"(conf={last.get('confidence', 'N/A')})\n\n"
            f"Please assess the current situation and decide what actions, if any, to take. "
            f"Remember to call log_analyst_decision at the end."
        )

        executor = _ToolExecutor(
            observations_path=self.observations_path,
            decisions_log_path=self.decisions_log_path,
            daily_summaries_dir=self.daily_summaries_dir,
            vision_capture=vision_capture,
            notifier=notifier,
            current_mode=current_mode,
        )

        try:
            text, tools_called = self._client.run_with_tools(
                system_prompt=self._ADVISE_SYSTEM_PROMPT,
                user_message=user_message,
                tool_executor=executor,
            )
        except Exception as exc:
            logger.warning("advise() LLM call failed: %s", exc)
            return None

        return self._parse_advise_response(text, tools_called, executor)

    def _parse_advise_response(
        self,
        text: str,
        tools_called: list[str],
        executor: _ToolExecutor,
    ) -> AnalystDecision:
        """
        Build an AnalystDecision from the LLM's tool call history.

        Rather than parsing free text (fragile), we infer decisions from
        which tools were actually called and what args were passed.
        The LLM's text response becomes the reasoning field.
        """
        decision = AnalystDecision(
            reasoning=text,
            tools_called=tools_called,
            llm_available=True,
        )

        # Infer switch_mode from whether switch_detection_mode was called
        # The executor captures the last call's new_mode via vision_capture
        if "switch_detection_mode" in tools_called and executor.vision_capture:
            decision.switch_mode = getattr(executor.vision_capture, "detection_mode", None)

        # Infer push from push_notification being called
        if "push_notification" in tools_called:
            decision.push_message = "[pushed by agent]"

        # Infer report generation
        if "generate_daily_report" in tools_called:
            decision.generate_report = True

        # Infer feeder alert from get_feeder_health result
        if "get_feeder_health" in tools_called:
            # The alert text is in the decisions log — flag it for orchestrator
            decision.feeder_alert = "feeder_health_checked"

        logger.info(
            "Analyst decision | tools=%s switch=%s report=%s push=%s",
            tools_called,
            decision.switch_mode,
            decision.generate_report,
            decision.push_message is not None,
        )

        return decision

    # ── Reactive answer() ─────────────────────────────────────────────────────

    def answer(
        self,
        user_query: str,
        vision_capture: Any = None,
        notifier: Any = None,
        current_mode: str = "fixed_crop",
    ) -> AnalystResponse:
        """
        Answer a natural language question about the feeder using tool calls.

        Called by the web backend for user-facing queries. The agent calls
        appropriate observation tools, synthesises the data, and returns
        a structured response with both the natural language answer and
        the raw data used to generate it.

        Args:
            user_query:     The user's question in natural language.
            vision_capture: VisionCapture instance (may be needed for status).
            notifier:       Notifier instance (not used for read-only queries).
            current_mode:   Active detection mode for context.

        Returns:
            AnalystResponse — always returned even on LLM failure.
            On failure: answer is a generic fallback, llm_available=False.
        """
        if not self.llm_available:
            return AnalystResponse(
                answer=(
                    "The AI analyst is currently unavailable. "
                    "The feeder is still running and logging detections. "
                    "Check logs/observations.jsonl for raw data."
                ),
                llm_available=False,
                error="LLM not available",
            )

        executor = _ToolExecutor(
            observations_path=self.observations_path,
            decisions_log_path=self.decisions_log_path,
            daily_summaries_dir=self.daily_summaries_dir,
            vision_capture=vision_capture,
            notifier=notifier,
            current_mode=current_mode,
        )

        try:
            text, tools_called = self._client.run_with_tools(
                system_prompt=self._ANSWER_SYSTEM_PROMPT,
                user_message=user_query,
                tool_executor=executor,
            )

            if not text:
                text = "I was unable to generate a response. Please try again."

            return AnalystResponse(
                answer=text,
                data={},
                tools_called=tools_called,
                confidence="high" if tools_called else "low",
                llm_available=True,
            )

        except Exception as exc:
            logger.warning("answer() failed: %s", exc)
            return AnalystResponse(
                answer="An error occurred while processing your question.",
                llm_available=True,
                error=str(exc),
            )
