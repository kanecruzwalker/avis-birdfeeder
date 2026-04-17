"""
src/agent/tools/observation_tools.py

Observation query tools for BirdAnalystAgent.

These tools give the LLM read access to the observations log and split
data. Each function is a plain Python callable — the agent registers them
in its tool registry and the LLM decides when to call them.

Why plain functions rather than methods?
    Tools need to be serialisable as JSON schema for the Gemini function-
    calling API. Plain functions with type annotations and docstrings map
    directly to the schema format. No class state needed — each tool
    receives everything it needs as arguments.

Tools in this module:
    read_recent_observations  — raw observation list for a time window
    get_detection_stats       — aggregated stats per detection mode
    query_species_history     — history for a specific species
    get_top_species           — ranked species by detection count
    get_feeder_health         — infers feeder activity trend (food level proxy)

The feeder health tool is the most interesting one from a research angle:
    declining detections + lower confidence + shorter inter-detection gaps
    correlate with less food in the tray. This lets the agent reason about
    physical feeder state purely from classification signal — no hardware sensor.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean, stdev
from typing import Any

logger = logging.getLogger(__name__)


# ── Tool: read_recent_observations ────────────────────────────────────────────


def read_recent_observations(
    observations_path: str,
    hours: float = 1.0,
    max_results: int = 50,
) -> dict[str, Any]:
    """
    Read the most recent bird observations from the log.

    Returns a summary dict the LLM can reason about — not raw JSONL.
    Raw JSONL would use too many tokens; this distils what matters.

    Args:
        observations_path: Path to logs/observations.jsonl.
        hours:             How many hours back to look. Default 1.0.
        max_results:       Cap on observations returned in the list.

    Returns:
        {
            "window_hours": float,
            "total_detections": int,
            "unique_species": int,
            "detections": [{"species", "confidence", "mode", "timestamp"}, ...],
            "mean_confidence": float,
            "mode_counts": {"fixed_crop": N, "yolo": M},
        }
    """
    path = Path(observations_path)
    now = datetime.now(UTC)
    since = now - timedelta(hours=hours)

    detections = []
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obs = json.loads(line)
                    ts = datetime.fromisoformat(obs.get("timestamp", ""))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    if ts >= since:
                        detections.append({
                            "species": obs.get("common_name", obs.get("species_code", "?")),
                            "species_code": obs.get("species_code", "?"),
                            "confidence": round(float(obs.get("fused_confidence", 0)), 3),
                            "mode": obs.get("detection_mode", "fixed_crop"),
                            "timestamp": ts.strftime("%H:%M:%S"),
                            "has_audio": obs.get("audio_result") is not None,
                            "has_visual": obs.get("visual_result") is not None,
                        })
                except Exception:
                    continue

    confidences = [d["confidence"] for d in detections]
    mode_counts = Counter(d["mode"] for d in detections)
    species_set = {d["species_code"] for d in detections}

    return {
        "window_hours": hours,
        "total_detections": len(detections),
        "unique_species": len(species_set),
        "mean_confidence": round(mean(confidences), 3) if confidences else 0.0,
        "mode_counts": dict(mode_counts),
        "detections": detections[-max_results:],  # most recent first after slice
    }


# ── Tool: get_detection_stats ─────────────────────────────────────────────────


def get_detection_stats(
    observations_path: str,
    hours: float = 2.0,
) -> dict[str, Any]:
    """
    Compare detection performance between fixed_crop and yolo modes.

    Useful for the agent to decide whether to switch modes — if one mode
    has significantly higher mean confidence or detection rate, the agent
    can reason about switching.

    Args:
        observations_path: Path to logs/observations.jsonl.
        hours:             Window to analyse. Default 2.0 (covers one full
                           A/B cycle at 30-minute windows).

    Returns:
        {
            "window_hours": float,
            "by_mode": {
                "fixed_crop": {"detections": N, "mean_confidence": F,
                               "std_confidence": F, "unique_species": N,
                               "detections_per_hour": F},
                "yolo": {...}
            },
            "recommendation": str   # plain-English summary for LLM context
        }
    """
    path = Path(observations_path)
    now = datetime.now(UTC)
    since = now - timedelta(hours=hours)

    by_mode: dict[str, dict[str, list]] = defaultdict(lambda: {
        "confidences": [], "species": set()
    })

    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obs = json.loads(line)
                    ts = datetime.fromisoformat(obs.get("timestamp", ""))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    if ts < since:
                        continue
                    mode = obs.get("detection_mode", "fixed_crop")
                    conf = float(obs.get("fused_confidence", 0))
                    code = obs.get("species_code", "?")
                    by_mode[mode]["confidences"].append(conf)
                    by_mode[mode]["species"].add(code)
                except Exception:
                    continue

    result: dict[str, Any] = {"window_hours": hours, "by_mode": {}}
    for mode, data in by_mode.items():
        confs = data["confidences"]
        n = len(confs)
        result["by_mode"][mode] = {
            "detections": n,
            "mean_confidence": round(mean(confs), 3) if confs else 0.0,
            "std_confidence": round(stdev(confs), 3) if len(confs) > 1 else 0.0,
            "unique_species": len(data["species"]),
            "detections_per_hour": round(n / hours, 2),
        }

    # Plain-English recommendation for LLM context
    modes = result["by_mode"]
    if len(modes) == 2 and "fixed_crop" in modes and "yolo" in modes:
        fc_conf = modes["fixed_crop"]["mean_confidence"]
        yo_conf = modes["yolo"]["mean_confidence"]
        delta = yo_conf - fc_conf
        if abs(delta) < 0.02:
            rec = "Both modes performing similarly. No strong reason to switch."
        elif delta > 0:
            rec = f"yolo showing {delta:.2f} higher mean confidence than fixed_crop."
        else:
            rec = f"fixed_crop showing {abs(delta):.2f} higher mean confidence than yolo."
    elif len(modes) == 1:
        mode_name = list(modes.keys())[0]
        rec = f"Only {mode_name} data available in this window — no comparison yet."
    else:
        rec = "Insufficient data for comparison."

    result["recommendation"] = rec
    return result


# ── Tool: query_species_history ───────────────────────────────────────────────


def query_species_history(
    observations_path: str,
    species_code: str,
    days: float = 7.0,
) -> dict[str, Any]:
    """
    Return detection history for a specific species over the last N days.

    Useful for user queries like "how often have House Finches visited?"
    or for the agent to reason about species patterns over time.

    Args:
        observations_path: Path to logs/observations.jsonl.
        species_code:      4-letter AOU code, e.g. "HOFI". Case-insensitive.
        days:              How many days back to search. Default 7.

    Returns:
        {
            "species_code": str,
            "days_searched": float,
            "total_detections": int,
            "detections_per_day": float,
            "mean_confidence": float,
            "first_seen": str | None,     # ISO timestamp of first detection
            "last_seen": str | None,      # ISO timestamp of most recent
            "daily_counts": {             # date string → count
                "2026-04-15": 5,
                ...
            },
            "peak_hour": int | None,      # UTC hour with most detections
        }
    """
    path = Path(observations_path)
    code = species_code.upper()
    now = datetime.now(UTC)
    since = now - timedelta(days=days)

    detections: list[datetime] = []
    confidences: list[float] = []
    hour_counts: Counter = Counter()

    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obs = json.loads(line)
                    if obs.get("species_code", "").upper() != code:
                        continue
                    ts = datetime.fromisoformat(obs.get("timestamp", ""))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    if ts < since:
                        continue
                    detections.append(ts)
                    confidences.append(float(obs.get("fused_confidence", 0)))
                    hour_counts[ts.hour] += 1
                except Exception:
                    continue

    daily: Counter = Counter()
    for ts in detections:
        daily[ts.date().isoformat()] += 1

    return {
        "species_code": code,
        "days_searched": days,
        "total_detections": len(detections),
        "detections_per_day": round(len(detections) / days, 2),
        "mean_confidence": round(mean(confidences), 3) if confidences else 0.0,
        "first_seen": detections[0].isoformat() if detections else None,
        "last_seen": detections[-1].isoformat() if detections else None,
        "daily_counts": dict(sorted(daily.items())),
        "peak_hour": hour_counts.most_common(1)[0][0] if hour_counts else None,
    }


# ── Tool: get_top_species ─────────────────────────────────────────────────────


def get_top_species(
    observations_path: str,
    n: int = 5,
    hours: float = 24.0,
) -> dict[str, Any]:
    """
    Return the top N most frequently detected species in the last N hours.

    Useful for user queries like "what birds visited today?" and for the
    agent's daily summary reasoning.

    Args:
        observations_path: Path to logs/observations.jsonl.
        n:                 How many species to return. Default 5.
        hours:             Time window. Default 24.0 (today).

    Returns:
        {
            "window_hours": float,
            "total_detections": int,
            "species": [
                {"rank": 1, "code": "HOFI", "name": "House Finch",
                 "count": 12, "mean_confidence": 0.87},
                ...
            ]
        }
    """
    path = Path(observations_path)
    now = datetime.now(UTC)
    since = now - timedelta(hours=hours)

    counts: Counter = Counter()
    names: dict[str, str] = {}
    confs: dict[str, list[float]] = defaultdict(list)
    total = 0

    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obs = json.loads(line)
                    ts = datetime.fromisoformat(obs.get("timestamp", ""))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    if ts < since:
                        continue
                    code = obs.get("species_code", "?")
                    counts[code] += 1
                    names[code] = obs.get("common_name", code)
                    confs[code].append(float(obs.get("fused_confidence", 0)))
                    total += 1
                except Exception:
                    continue

    ranked = [
        {
            "rank": i + 1,
            "code": code,
            "name": names.get(code, code),
            "count": count,
            "mean_confidence": round(mean(confs[code]), 3),
        }
        for i, (code, count) in enumerate(counts.most_common(n))
    ]

    return {
        "window_hours": hours,
        "total_detections": total,
        "species": ranked,
    }


# ── Tool: get_feeder_health ───────────────────────────────────────────────────


def get_feeder_health(
    observations_path: str,
    comparison_days: int = 3,
) -> dict[str, Any]:
    """
    Infer feeder food level from detection activity trends.

    Low food in the tray causes fewer bird visits, lower confidence
    (birds leave quickly, partial crops), and declining activity across
    consecutive days. This tool computes a trend signal the agent can
    use to push a "feeder may need refilling" notification.

    No physical sensor required — the signal comes entirely from the
    classification pipeline.

    Args:
        observations_path: Path to logs/observations.jsonl.
        comparison_days:   Number of recent days to compare. Default 3.
                           Day N-1 vs Day N trend across this window.

    Returns:
        {
            "status": "healthy" | "declining" | "low" | "unknown",
            "daily_detections": [{"date": str, "count": int, "mean_conf": float}, ...],
            "trend_direction": "up" | "down" | "flat" | "unknown",
            "trend_pct_change": float,   # % change from first to last day
            "alert": str | None,         # human-readable alert if low activity
            "reasoning": str,            # plain-English explanation for LLM
        }
    """
    path = Path(observations_path)
    now = datetime.now(UTC)

    # Bucket by day for the last comparison_days days
    daily_counts: dict[str, list[float]] = defaultdict(list)
    for d in range(comparison_days):
        day = (now - timedelta(days=d)).date().isoformat()
        daily_counts.setdefault(day, [])

    if path.exists():
        cutoff = now - timedelta(days=comparison_days)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obs = json.loads(line)
                    ts = datetime.fromisoformat(obs.get("timestamp", ""))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    if ts < cutoff:
                        continue
                    day = ts.date().isoformat()
                    daily_counts[day].append(float(obs.get("fused_confidence", 0)))
                except Exception:
                    continue

    daily_summary = sorted([
        {
            "date": day,
            "count": len(confs),
            "mean_conf": round(mean(confs), 3) if confs else 0.0,
        }
        for day, confs in daily_counts.items()
    ], key=lambda x: x["date"])

    counts = [d["count"] for d in daily_summary]

    if len(counts) < 2 or all(c == 0 for c in counts):
        return {
            "status": "unknown",
            "daily_detections": daily_summary,
            "trend_direction": "unknown",
            "trend_pct_change": 0.0,
            "alert": None,
            "reasoning": "Not enough data to assess feeder health.",
        }

    first = counts[0] if counts[0] > 0 else 1
    last = counts[-1]
    pct_change = round((last - first) / first * 100, 1)

    if pct_change < -50:
        status = "low"
        direction = "down"
        alert = (
            f"⚠️ Feeder activity dropped {abs(pct_change):.0f}% over {comparison_days} days. "
            f"Feeder may need refilling."
        )
        reasoning = (
            f"Detection count fell from {first} to {last} over {comparison_days} days "
            f"({pct_change:.0f}%). This level of decline typically indicates low food supply."
        )
    elif pct_change < -20:
        status = "declining"
        direction = "down"
        alert = None
        reasoning = (
            f"Detection count declining ({pct_change:.0f}% over {comparison_days} days). "
            f"Worth monitoring — may indicate food is getting low."
        )
    elif pct_change > 10:
        status = "healthy"
        direction = "up"
        alert = None
        reasoning = f"Detection activity increasing ({pct_change:.0f}%). Feeder appears well-stocked."
    else:
        status = "healthy"
        direction = "flat"
        alert = None
        reasoning = f"Detection activity stable ({pct_change:.0f}% change). Feeder appears healthy."

    return {
        "status": status,
        "daily_detections": daily_summary,
        "trend_direction": direction,
        "trend_pct_change": pct_change,
        "alert": alert,
        "reasoning": reasoning,
    }