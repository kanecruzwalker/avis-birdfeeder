"""
src/agent/tools/system_tools.py

System state tools for BirdAnalystAgent.

These tools let the LLM perceive the current hardware and agent state
before making decisions. They are read-only — no side effects.

Tools in this module:
    get_current_system_status  — current mode, uptime, recent activity summary
    get_time_context           — time of day, season hints for reasoning
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def get_current_system_status(
    current_mode: str,
    uptime_seconds: float,
    window_elapsed_minutes: float,
    window_total_minutes: float,
    last_detection_species: str | None,
    last_detection_confidence: float | None,
    last_detection_minutes_ago: float | None,
) -> dict[str, Any]:
    """
    Return a snapshot of the current system state.

    This tool is called at the start of every LLM reasoning cycle so
    the agent knows its current context before deciding what to do.

    Args:
        current_mode:                  Active detection mode ("fixed_crop"|"yolo").
        uptime_seconds:                Seconds since ExperimentOrchestrator started.
        window_elapsed_minutes:        Minutes elapsed in the current A/B window.
        window_total_minutes:          Configured A/B window duration.
        last_detection_species:        Common name of most recent detection, or None.
        last_detection_confidence:     Confidence of most recent detection, or None.
        last_detection_minutes_ago:    Minutes since last detection, or None.

    Returns:
        Flat dict describing current state — all values JSON serialisable.
    """
    uptime_hours = uptime_seconds / 3600
    window_pct = (window_elapsed_minutes / window_total_minutes * 100
                  if window_total_minutes > 0 else 0)

    return {
        "current_detection_mode": current_mode,
        "uptime_hours": round(uptime_hours, 2),
        "current_window": {
            "elapsed_minutes": round(window_elapsed_minutes, 1),
            "total_minutes": window_total_minutes,
            "percent_complete": round(window_pct, 1),
        },
        "last_detection": {
            "species": last_detection_species,
            "confidence": last_detection_confidence,
            "minutes_ago": round(last_detection_minutes_ago, 1)
            if last_detection_minutes_ago is not None else None,
        },
        "status": "running",
    }


def get_time_context() -> dict[str, Any]:
    """
    Return time-of-day context to help the LLM reason about bird activity.

    Birds are most active at dawn and dusk. The agent can use this to
    decide whether low detection counts are expected (midday) or
    concerning (dawn/dusk with no activity = possible empty feeder).

    Returns:
        {
            "utc_now": str,
            "hour_utc": int,
            "activity_period": "dawn" | "morning" | "midday" |
                               "afternoon" | "dusk" | "night",
            "expected_activity": "high" | "medium" | "low",
            "note": str   # plain-English context for LLM
        }
    """
    now = datetime.now(UTC)
    hour = now.hour

    # Rough activity bands based on San Diego sunrise/sunset patterns
    # Dawn: 5-7, Morning: 7-10, Midday: 10-14, Afternoon: 14-17, Dusk: 17-19, Night: 19-5
    if 5 <= hour < 7:
        period, expected = "dawn", "high"
        note = "Peak activity window — birds most active at dawn."
    elif 7 <= hour < 10:
        period, expected = "morning", "high"
        note = "High activity window — morning feeding period."
    elif 10 <= hour < 14:
        period, expected = "midday", "medium"
        note = "Moderate activity — midday lull is normal."
    elif 14 <= hour < 17:
        period, expected = "afternoon", "medium"
        note = "Moderate activity — afternoon feeding resumes."
    elif 17 <= hour < 19:
        period, expected = "dusk", "high"
        note = "Peak activity window — birds active before roosting."
    else:
        period, expected = "night", "low"
        note = "Low activity expected — most birds roosting."

    return {
        "utc_now": now.strftime("%Y-%m-%d %H:%M UTC"),
        "hour_utc": hour,
        "activity_period": period,
        "expected_activity": expected,
        "note": note,
    }
