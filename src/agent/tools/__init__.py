"""
src/agent/tools/__init__.py

Tool registry for BirdAnalystAgent.

Exports all tool functions and their Gemini function-calling schemas.
The schema is what gets sent to the LLM — it tells the model what each
tool does, what arguments it takes, and what it returns.

Why define schemas here rather than auto-generating from docstrings?
    Gemini's function-calling API requires a specific JSON schema format.
    Hand-writing the schemas means we control exactly what the LLM sees —
    descriptions are tuned for the LLM to make good decisions, not just
    to document the code. The two concerns (code docs vs LLM prompting)
    are kept separate.

Adding a new tool:
    1. Implement the function in the appropriate module
    2. Import it here
    3. Add its schema to TOOL_SCHEMAS
    4. Add it to TOOL_REGISTRY with its name as the key
    The agent picks it up automatically — no other changes needed.
"""

from src.agent.tools.action_tools import (
    generate_daily_report,
    log_analyst_decision,
    push_notification,
    switch_detection_mode,
)
from src.agent.tools.observation_tools import (
    get_detection_stats,
    get_feeder_health,
    get_top_species,
    query_species_history,
    read_recent_observations,
)
from src.agent.tools.system_tools import get_current_system_status, get_time_context

__all__ = [
    "read_recent_observations",
    "get_detection_stats",
    "query_species_history",
    "get_top_species",
    "get_feeder_health",
    "get_current_system_status",
    "get_time_context",
    "switch_detection_mode",
    "generate_daily_report",
    "push_notification",
    "log_analyst_decision",
    "TOOL_SCHEMAS",
    "TOOL_REGISTRY",
]

TOOL_SCHEMAS = [
    {
        "name": "read_recent_observations",
        "description": (
            "Read recent bird detections from the observation log. "
            "Use this first to understand what has been happening at the feeder."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hours": {"type": "number", "description": "How many hours back to look. Default 1.0."},
                "max_results": {"type": "integer", "description": "Max detections to return. Default 50."},
            },
            "required": [],
        },
    },
    {
        "name": "get_detection_stats",
        "description": (
            "Compare fixed_crop vs yolo detection performance over a time window. "
            "Use this when deciding whether to switch detection modes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hours": {"type": "number", "description": "Time window in hours. Default 2.0."},
            },
            "required": [],
        },
    },
    {
        "name": "query_species_history",
        "description": (
            "Get detection history for a specific bird species over recent days. "
            "Use this to answer user questions about a particular species."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "species_code": {"type": "string", "description": "4-letter AOU code, e.g. 'HOFI'."},
                "days": {"type": "number", "description": "Days back to search. Default 7."},
            },
            "required": ["species_code"],
        },
    },
    {
        "name": "get_top_species",
        "description": (
            "Get the most frequently detected bird species in a recent time window. "
            "Use for 'what birds visited today?' type questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "description": "How many top species to return. Default 5."},
                "hours": {"type": "number", "description": "Time window in hours. Default 24.0."},
            },
            "required": [],
        },
    },
    {
        "name": "get_feeder_health",
        "description": (
            "Assess feeder food level by analysing detection activity trends. "
            "Declining detections over days may indicate the feeder needs refilling. "
            "Returns status: healthy / declining / low."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "comparison_days": {"type": "integer", "description": "Days to compare for trend. Default 3."},
            },
            "required": [],
        },
    },
    {
        "name": "get_time_context",
        "description": (
            "Get current time and expected bird activity level for this time of day. "
            "Dawn and dusk are peak activity. Use to contextualise low detection counts."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "switch_detection_mode",
        "description": (
            "Switch the active detection mode between 'fixed_crop' and 'yolo'. "
            "Only switch if the stats justify it. Always provide a clear reason."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "new_mode": {"type": "string", "enum": ["fixed_crop", "yolo"], "description": "Mode to switch to."},
                "reason": {"type": "string", "description": "Why you are switching modes."},
            },
            "required": ["new_mode", "reason"],
        },
    },
    {
        "name": "generate_daily_report",
        "description": (
            "Build the daily summary report and write it to disk as .md and .json. "
            "Returns the push-ready summary message. Call at end of day or on user request."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "for_date": {"type": "string", "description": "ISO date 'YYYY-MM-DD'. Omit for today."},
            },
            "required": [],
        },
    },
    {
        "name": "push_notification",
        "description": (
            "Send a push notification to the feeder owner. "
            "Use for notable events only: unusual sighting, feeder alert, daily summary. "
            "Keep under 200 characters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Notification text. Be concise."},
            },
            "required": ["message"],
        },
    },
    {
        "name": "log_analyst_decision",
        "description": (
            "Log your reasoning and actions to the decisions log. "
            "Always call this as the LAST step of every reasoning cycle, "
            "even if you took no actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "What you observed, concluded, and why."},
                "actions_taken": {"type": "array", "items": {"type": "string"}, "description": "Action tools called this cycle."},
                "observations_summary": {"type": "string", "description": "One-sentence summary of what was observed."},
            },
            "required": ["reasoning", "actions_taken", "observations_summary"],
        },
    },
]

TOOL_REGISTRY = {
    "read_recent_observations": read_recent_observations,
    "get_detection_stats": get_detection_stats,
    "query_species_history": query_species_history,
    "get_top_species": get_top_species,
    "get_feeder_health": get_feeder_health,
    "get_current_system_status": get_current_system_status,
    "get_time_context": get_time_context,
    "switch_detection_mode": switch_detection_mode,
    "generate_daily_report": generate_daily_report,
    "push_notification": push_notification,
    "log_analyst_decision": log_analyst_decision,
}
