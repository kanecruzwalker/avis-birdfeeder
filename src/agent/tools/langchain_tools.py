"""
src/agent/tools/langchain_tools.py

LangChain @tool adapters for BirdAnalystAgent tools.

This module is the adapter layer between our custom tool functions and
the LangChain/LangGraph framework. Each function here is a thin wrapper
that:
    1. Applies the @tool decorator so LangChain can introspect and call it
    2. Injects the runtime context (observations_path, thresholds_path, etc.)
       that the raw tool functions need but LangChain agents don't supply
    3. Returns a string — LangChain tools must return strings, not dicts

Why keep the raw tools separate from the LangChain wrappers?
    The raw tools (observation_tools.py, calibration_tools.py, etc.) are
    framework-agnostic. They are called by our custom BirdAnalystAgent via
    Gemini function calling, and by LangChainAnalyst via these wrappers.
    The logic lives once; the framework interface is just a thin skin.

    This means:
    - Adding a new framework in future = add one wrapper file, zero logic changes
    - Testing the tools = test the raw functions, not the LangChain wrappers
    - The course report can honestly say "we implemented tools in a framework-
      agnostic way and integrated them with both a custom agent and LangChain"

Runtime context injection:
    LangChain tools are plain functions — they cannot receive runtime objects
    (observations_path, vision_capture, notifier) as call arguments from the LLM.
    We solve this with a factory pattern: build_langchain_tools(context) returns
    a list of @tool-decorated functions that close over the context dict.
    LangChainAnalyst calls build_langchain_tools() once at init time.

Usage:
    tools = build_langchain_tools({
        "observations_path": "logs/observations.jsonl",
        "thresholds_path":   "configs/thresholds.yaml",
        "daily_summaries_dir": "logs/daily_summaries",
        "vision_capture":    vc,   # may be None
        "notifier":          n,    # may be None
        "current_mode":      "fixed_crop",
    })
    agent = create_react_agent(llm, tools, prompt)
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_langchain_tools(context: dict[str, Any]) -> list:
    """
    Build LangChain @tool-decorated functions with runtime context injected.

    Args:
        context: Dict containing runtime values the tools need:
            observations_path    (str)
            thresholds_path      (str)
            daily_summaries_dir  (str)
            vision_capture       (VisionCapture | None)
            notifier             (Notifier | None)
            current_mode         (str)
            decisions_log_path   (str)

    Returns:
        List of LangChain Tool objects ready to pass to an agent.
    """
    try:
        from langchain_core.tools import tool  # type: ignore[import]
    except ImportError:
        logger.error(
            "langchain-core not installed. "
            "Run: pip install langchain-core langchain-google-genai langgraph"
        )
        return []

    obs_path = context.get("observations_path", "logs/observations.jsonl")
    thr_path = context.get("thresholds_path", "configs/thresholds.yaml")
    summaries_dir = context.get("daily_summaries_dir", "logs/daily_summaries")
    vision_capture = context.get("vision_capture")
    notifier = context.get("notifier")
    current_mode = context.get("current_mode", "fixed_crop")
    decisions_log = context.get("decisions_log_path", "logs/analyst_decisions.jsonl")

    # ── Observation tools ─────────────────────────────────────────────────────

    @tool
    def read_recent_observations(hours: float = 1.0) -> str:
        """
        Read recent bird detections from the feeder observation log.
        Returns a summary of what birds have been detected recently.
        Use this first to understand current feeder activity.
        """
        from src.agent.tools.observation_tools import (  # noqa: PLC0415
            read_recent_observations as _fn,
        )

        result = _fn(observations_path=obs_path, hours=hours)
        return json.dumps(result)

    @tool
    def get_detection_stats(hours: float = 2.0) -> str:
        """
        Compare fixed_crop vs yolo detection mode performance.
        Returns mean confidence and detection rates for each mode.
        Use when deciding whether to switch detection strategies.
        """
        from src.agent.tools.observation_tools import (  # noqa: PLC0415
            get_detection_stats as _fn,
        )

        result = _fn(observations_path=obs_path, hours=hours)
        return json.dumps(result)

    @tool
    def query_species_history(species_code: str, days: float = 7.0) -> str:
        """
        Get detection history for a specific bird species.
        species_code should be a 4-letter AOU code like 'HOFI' for House Finch.
        Use this to answer questions about a particular species over time.
        """
        from src.agent.tools.observation_tools import (  # noqa: PLC0415
            query_species_history as _fn,
        )

        result = _fn(observations_path=obs_path, species_code=species_code, days=days)
        return json.dumps(result)

    @tool
    def get_top_species(n: int = 5, hours: float = 24.0) -> str:
        """
        Get the most frequently detected bird species in a time window.
        Use for questions like 'what birds visited today?'
        Returns species ranked by detection count with mean confidence.
        """
        from src.agent.tools.observation_tools import get_top_species as _fn  # noqa: PLC0415

        result = _fn(observations_path=obs_path, n=n, hours=hours)
        return json.dumps(result)

    @tool
    def get_feeder_health(comparison_days: int = 3) -> str:
        """
        Assess feeder food level from detection activity trends.
        Declining detections over days suggest the feeder needs refilling.
        Returns status: healthy, declining, or low — with reasoning.
        """
        from src.agent.tools.observation_tools import get_feeder_health as _fn  # noqa: PLC0415

        result = _fn(observations_path=obs_path, comparison_days=comparison_days)
        return json.dumps(result)

    # ── System tools ──────────────────────────────────────────────────────────

    @tool
    def get_time_context() -> str:
        """
        Get current time and expected bird activity level.
        Dawn and dusk are peak activity periods — use this to contextualise
        low detection counts before raising feeder health concerns.
        """
        from src.agent.tools.system_tools import get_time_context as _fn  # noqa: PLC0415

        return json.dumps(_fn())

    # ── Calibration tools ─────────────────────────────────────────────────────

    @tool
    def run_fusion_weight_sweep(hours: float = 6.0) -> str:
        """
        Find the optimal audio/visual fusion weight combination.
        Sweeps weight combinations on recent live data and returns the best split.
        Use this periodically to keep fusion weights tuned to current conditions.
        Returns recommended weights and the improvement over current settings.
        """
        from src.agent.tools.calibration_tools import (  # noqa: PLC0415
            run_fusion_weight_sweep as _fn,
        )

        result = _fn(observations_path=obs_path, hours=hours)
        return json.dumps(result)

    @tool
    def evaluate_detection_threshold(hours: float = 6.0) -> str:
        """
        Find the optimal confidence threshold for triggering notifications.
        Sweeps threshold values and shows the sensitivity/precision tradeoff.
        Returns the recommended threshold based on recent field conditions.
        """
        from src.agent.tools.calibration_tools import (  # noqa: PLC0415
            evaluate_detection_threshold as _fn,
        )

        result = _fn(observations_path=obs_path, hours=hours)
        return json.dumps(result)

    @tool
    def compare_model_backends(hours: float = 24.0) -> str:
        """
        Compare Hailo NPU vs CPU inference quality from the observation log.
        Shows whether the NPU is producing higher confidence classifications.
        Use this to verify Hailo is providing real quality improvement, not
        just speed improvement.
        """
        from src.agent.tools.calibration_tools import (  # noqa: PLC0415
            compare_model_backends as _fn,
        )

        result = _fn(observations_path=obs_path, hours=hours)
        return json.dumps(result)

    @tool
    def apply_fusion_weights(audio_weight: float, visual_weight: float) -> str:
        """
        Apply new fusion weights to the system configuration.
        Only call this after run_fusion_weight_sweep confirms improvement > 0.02
        and the sweep used at least 10 dual-modality observations.
        audio_weight + visual_weight must equal 1.0.
        This immediately affects all subsequent detections.
        """
        from src.agent.tools.calibration_tools import apply_fusion_weights as _fn  # noqa: PLC0415

        result = _fn(
            audio_weight=audio_weight,
            visual_weight=visual_weight,
            thresholds_path=thr_path,
        )
        return json.dumps(result)

    # ── Action tools ──────────────────────────────────────────────────────────

    @tool
    def switch_detection_mode(new_mode: str, reason: str) -> str:
        """
        Switch the active detection mode between fixed_crop and yolo.
        new_mode must be 'fixed_crop' or 'yolo'.
        Always provide a reason explaining why the switch is warranted.
        Only switch when detection stats clearly favour the new mode.
        """
        from src.agent.tools.action_tools import switch_detection_mode as _fn  # noqa: PLC0415

        result = _fn(
            new_mode=new_mode,
            vision_capture=vision_capture,
            reason=reason,
        )
        return json.dumps(result)

    @tool
    def generate_daily_report(for_date: str = "") -> str:
        """
        Build and save the daily summary report for today or a specific date.
        for_date should be 'YYYY-MM-DD' format, or leave empty for today.
        Writes .md and .json files to the daily summaries directory.
        Returns the push-ready summary message.
        """
        from src.agent.tools.action_tools import generate_daily_report as _fn  # noqa: PLC0415

        result = _fn(
            observations_path=obs_path,
            daily_summaries_dir=summaries_dir,
            for_date=for_date if for_date else None,
        )
        return json.dumps(result)

    @tool
    def push_notification(message: str) -> str:
        """
        Send a push notification to the feeder owner via Pushover.
        Use sparingly — only for notable events like unusual species,
        feeder health alerts, daily summaries, or calibration updates.
        Keep messages under 200 characters for clean mobile display.
        """
        from src.agent.tools.action_tools import push_notification as _fn  # noqa: PLC0415

        result = _fn(message=message, notifier=notifier)
        return json.dumps(result)

    @tool
    def log_analyst_decision(
        reasoning: str,
        actions_taken: str,
        observations_summary: str,
    ) -> str:
        """
        Log your reasoning and actions to the decisions audit log.
        Always call this as your final action every reasoning cycle,
        even when taking no other actions — this is your chain-of-thought record.
        actions_taken should be a comma-separated list of tool names called,
        or 'none' if no actions were taken.
        """
        from src.agent.tools.action_tools import log_analyst_decision as _fn  # noqa: PLC0415

        actions_list = (
            [a.strip() for a in actions_taken.split(",") if a.strip()]
            if actions_taken and actions_taken.lower() != "none"
            else []
        )
        result = _fn(
            decisions_log_path=decisions_log,
            reasoning=reasoning,
            actions_taken=actions_list,
            observations_summary=observations_summary,
            mode_at_decision=current_mode,
        )
        return json.dumps(result)

    return [
        read_recent_observations,
        get_detection_stats,
        query_species_history,
        get_top_species,
        get_feeder_health,
        get_time_context,
        run_fusion_weight_sweep,
        evaluate_detection_threshold,
        compare_model_backends,
        apply_fusion_weights,
        switch_detection_mode,
        generate_daily_report,
        push_notification,
        log_analyst_decision,
    ]
