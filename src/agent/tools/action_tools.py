"""
src/agent/tools/action_tools.py

Action tools for BirdAnalystAgent — tools that cause side effects.

Unlike observation and system tools (read-only), these tools change state:
they switch hardware modes, write files, and send notifications.

The agent calls these only after reasoning that the action is warranted.
Each action is logged so the decision trail is auditable.

Tools in this module:
    switch_detection_mode   — change VisionCapture.detection_mode at runtime
    generate_daily_report   — build + write daily summary report
    push_notification       — send a text push via Pushover
    log_analyst_decision    — write the LLM's reasoning to decisions log

Why is log_analyst_decision a tool?
    The LLM calls it explicitly as the last step of every reasoning cycle,
    embedding its own chain-of-thought into the log. This makes the agent's
    decisions auditable and provides the "qualitative examples" content
    required by the course rubric (Phase 4: show where the system succeeds
    and fails with explanation).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def switch_detection_mode(
    new_mode: str,
    vision_capture: Any,
    reason: str = "",
) -> dict[str, Any]:
    """
    Switch the active detection mode on VisionCapture at runtime.

    Changes are in-memory only — hardware.yaml committed defaults are
    not touched. The Pi always boots into the config-file default
    (fixed_crop) and the agent switches from there.

    Args:
        new_mode:       "fixed_crop" or "yolo". Validated before applying.
        vision_capture: VisionCapture instance from the running agent.
                        Passed by BirdAnalystAgent at call time — not
                        serialised into the tool schema.
        reason:         Plain-English reason for the switch (logged).

    Returns:
        {"success": bool, "previous_mode": str, "new_mode": str, "reason": str}
    """
    valid_modes = ("fixed_crop", "yolo")
    if new_mode not in valid_modes:
        return {
            "success": False,
            "previous_mode": "unknown",
            "new_mode": new_mode,
            "reason": f"Invalid mode '{new_mode}'. Must be one of {valid_modes}.",
        }

    if vision_capture is None:
        return {
            "success": False,
            "previous_mode": "unknown",
            "new_mode": new_mode,
            "reason": "VisionCapture not available — cannot switch mode.",
        }

    previous = getattr(vision_capture, "detection_mode", "unknown")
    vision_capture.detection_mode = new_mode

    logger.info(
        "Agent switched detection mode: %s → %s | reason: %s",
        previous, new_mode, reason,
    )

    return {
        "success": True,
        "previous_mode": previous,
        "new_mode": new_mode,
        "reason": reason,
    }


def generate_daily_report(
    observations_path: str,
    daily_summaries_dir: str,
    for_date: str | None = None,
) -> dict[str, Any]:
    """
    Build the daily summary report and write it to disk.

    Calls ReportBuilder internally — the agent does not need to know
    about the report format, only that it can trigger generation.

    Args:
        observations_path:   Path to logs/observations.jsonl.
        daily_summaries_dir: Directory for .md/.json output.
        for_date:            ISO date string "YYYY-MM-DD", or None for today.

    Returns:
        {
            "success": bool,
            "report_date": str,
            "md_path": str,
            "json_path": str,
            "total_detections": int,
            "unique_species": int,
            "push_message": str,   # the summary text ready to push
        }
    """
    # Import here to avoid circular imports — tools are loaded by the agent
    from src.notify.report_builder import ReportBuilder  # noqa: PLC0415
    from datetime import date as _date  # noqa: PLC0415

    try:
        target_date = (
            _date.fromisoformat(for_date) if for_date
            else datetime.now(UTC).date()
        )
        builder = ReportBuilder(observations_path=observations_path)
        report = builder.build_daily_summary(for_date=target_date)
        md_path, json_path = builder.write_daily_summary(
            report=report,
            output_dir=daily_summaries_dir,
        )
        return {
            "success": True,
            "report_date": target_date.isoformat(),
            "md_path": str(md_path),
            "json_path": str(json_path),
            "total_detections": report.total_detections,
            "unique_species": report.unique_species,
            "push_message": report.to_push_message(),
        }
    except Exception as exc:
        logger.exception("generate_daily_report failed: %s", exc)
        return {
            "success": False,
            "report_date": for_date or "unknown",
            "md_path": "",
            "json_path": "",
            "total_detections": 0,
            "unique_species": 0,
            "push_message": "",
        }


def push_notification(
    message: str,
    notifier: Any,
) -> dict[str, Any]:
    """
    Send a plain-text push notification via the configured Notifier.

    The agent calls this when it decides a human should be alerted —
    for example after detecting an unusual species, inferring the
    feeder is low, or completing an experiment window comparison.

    Args:
        message:  The notification body. Keep under 500 chars for clean
                  mobile display. The agent is responsible for brevity.
        notifier: Notifier instance from the running agent.

    Returns:
        {"success": bool, "message_sent": str}
    """
    if notifier is None:
        return {"success": False, "message_sent": ""}

    try:
        notifier._push_text(message)
        logger.info("Agent pushed notification: %s", message[:80])
        return {"success": True, "message_sent": message}
    except Exception as exc:
        logger.warning("Agent push_notification failed: %s", exc)
        return {"success": False, "message_sent": ""}


def log_analyst_decision(
    decisions_log_path: str,
    reasoning: str,
    actions_taken: list[str],
    observations_summary: str,
    mode_at_decision: str,
) -> dict[str, Any]:
    """
    Write the LLM's reasoning and actions to the analyst decisions log.

    The agent calls this as the final step of every reasoning cycle,
    regardless of whether it took any actions. This creates an auditable
    trail of what the LLM observed, what it concluded, and what it did.

    The log file (logs/analyst_decisions.jsonl) is the primary evidence
    for the course rubric requirement to show agent reasoning.

    Args:
        decisions_log_path:   Path to logs/analyst_decisions.jsonl.
        reasoning:            The LLM's chain-of-thought explanation.
        actions_taken:        List of action tool names called this cycle.
        observations_summary: Brief description of what was observed.
        mode_at_decision:     Detection mode active when decision was made.

    Returns:
        {"success": bool, "logged_at": str}
    """
    path = Path(decisions_log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "mode": mode_at_decision,
        "observations_summary": observations_summary,
        "reasoning": reasoning,
        "actions_taken": actions_taken,
    }

    try:
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return {"success": True, "logged_at": entry["timestamp"]}
    except Exception as exc:
        logger.warning("log_analyst_decision failed: %s", exc)
        return {"success": False, "logged_at": ""}