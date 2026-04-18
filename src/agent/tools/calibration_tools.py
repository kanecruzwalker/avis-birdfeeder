"""
src/agent/tools/calibration_tools.py

Self-calibration tools for BirdAnalystAgent and LangChainAnalyst.

These tools allow the agent to autonomously tune the classification
pipeline based on real field data — closing the loop between observation
and configuration without human intervention.

Why self-calibration matters:
    The fusion weights (audio=0.55, visual=0.45) and confidence threshold
    (0.70) were chosen empirically in the lab. Field conditions differ:
    ambient noise, lighting, feeder distance, and seasonal species mix all
    affect which modality is more reliable at any given time. An agent that
    can re-tune these parameters from live data is more accurate than one
    locked to lab-tuned constants.

How it works:
    All calibration tools read from observations.jsonl — data that is
    already being collected by the running pipeline. No new sensors,
    no new data collection required. The agent is tuning itself purely
    from the signal it is already producing.

The calibration loop (agent perspective):
    1. run_fusion_weight_sweep()  → find optimal audio/visual split
    2. evaluate_detection_threshold() → find optimal confidence gate
    3. compare_model_backends()   → verify Hailo vs CPU quality
    4. apply_fusion_weights()     → write winning config to thresholds.yaml
    The agent calls these tools, reasons about the results, and decides
    whether to apply changes. It logs every decision to analyst_decisions.jsonl.

Tools in this module:
    run_fusion_weight_sweep       — grid search over audio/visual weights
    evaluate_detection_threshold  — precision/recall sweep over threshold
    compare_model_backends        — Hailo vs CPU confidence comparison
    apply_fusion_weights          — write new weights to thresholds.yaml
                                    (side effect — agent calls after sweep confirms)

Academic note:
    This directly addressing "parameter sensitivity"
     and "optimization" AI function category.
    The sweep results populate the fusion_weight_sensitivity plot in
    notebooks/phase7_evaluation.ipynb automatically.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ── Tool: run_fusion_weight_sweep ─────────────────────────────────────────────


def run_fusion_weight_sweep(
    observations_path: str,
    hours: float = 6.0,
    weight_steps: int = 5,
) -> dict[str, Any]:
    """
    Grid search over audio/visual fusion weights on recent live data.

    Evaluates how many observations would have crossed the current
    confidence threshold at each weight combination, and computes the
    mean fused confidence for each. Returns the weight pair that maximises
    confident detections — the agent uses this to decide whether to
    recalibrate.

    Why live data rather than the held-out test set?
        The test set measures generalisation. This tool measures what
        works *right now at this feeder* — different lighting, different
        mic position, different local species mix. Both are useful but
        they answer different questions.

    Args:
        observations_path: Path to logs/observations.jsonl.
        hours:             How many hours of recent data to use.
                           Default 6.0 — covers several A/B windows.
                           Increase to 24.0 for a full-day sweep.
        weight_steps:      Number of weight values to try between 0.0 and 1.0.
                           Default 5 gives [0.2, 0.4, 0.5, 0.6, 0.8].
                           Increase for finer resolution (costs more compute).

    Returns:
        {
            "hours_analysed": float,
            "n_observations": int,
            "n_with_both_modalities": int,
            "sweep_results": [
                {"audio_weight": float, "visual_weight": float,
                 "mean_fused_confidence": float,
                 "detections_above_threshold": int,
                 "coverage_pct": float},
                ...  sorted best to worst by mean_fused_confidence
            ],
            "recommended": {
                "audio_weight": float,
                "visual_weight": float,
                "mean_fused_confidence": float,
                "improvement_over_current": float,  # delta vs 0.55/0.45
            },
            "current_weights": {"audio": 0.55, "visual": 0.45},
            "reasoning": str,   # plain-English explanation for agent
        }
    """
    obs = _load_observations(observations_path, hours=hours)

    # Only sweep over observations where both modalities fired
    # Single-modality observations aren't affected by fusion weights
    both_modal = [
        o for o in obs if o.get("audio_result") is not None and o.get("visual_result") is not None
    ]

    if len(both_modal) < 5:
        return {
            "hours_analysed": hours,
            "n_observations": len(obs),
            "n_with_both_modalities": len(both_modal),
            "sweep_results": [],
            "recommended": None,
            "current_weights": {"audio": 0.55, "visual": 0.45},
            "reasoning": (
                f"Only {len(both_modal)} dual-modality observations in last {hours}h. "
                f"Need at least 5 to run a meaningful sweep. "
                f"Keep current weights and check again later."
            ),
        }

    # Extract per-observation audio and visual confidences
    # We re-compute fused confidence from components rather than using
    # the stored fused_confidence — that was computed with old weights
    samples = []
    for o in both_modal:
        audio_conf = float(
            o.get("audio_result", {}).get("confidence", 0.0)
            if isinstance(o.get("audio_result"), dict)
            else 0.0
        )
        visual_conf = float(
            o.get("visual_result", {}).get("confidence", 0.0)
            if isinstance(o.get("visual_result"), dict)
            else 0.0
        )
        if audio_conf > 0 and visual_conf > 0:
            samples.append((audio_conf, visual_conf))

    if not samples:
        return {
            "hours_analysed": hours,
            "n_observations": len(obs),
            "n_with_both_modalities": len(both_modal),
            "sweep_results": [],
            "recommended": None,
            "current_weights": {"audio": 0.55, "visual": 0.45},
            "reasoning": "Could not extract per-modality confidences from observations.",
        }

    # Generate weight grid
    step = 1.0 / (weight_steps + 1)
    weights = [round(step * i, 2) for i in range(1, weight_steps + 1)]

    threshold = 0.70  # production threshold from thresholds.yaml
    current_audio_w = 0.55

    sweep_results = []
    for aw in weights:
        vw = round(1.0 - aw, 2)
        fused = [aw * a + vw * v for a, v in samples]
        above = sum(1 for f in fused if f >= threshold)
        sweep_results.append(
            {
                "audio_weight": aw,
                "visual_weight": vw,
                "mean_fused_confidence": round(mean(fused), 4),
                "detections_above_threshold": above,
                "coverage_pct": round(above / len(fused) * 100, 1),
            }
        )

    sweep_results.sort(key=lambda x: x["mean_fused_confidence"], reverse=True)
    best = sweep_results[0]

    # Compute current performance for comparison
    current_fused = [current_audio_w * a + (1 - current_audio_w) * v for a, v in samples]
    current_mean_conf = mean(current_fused)
    improvement = round(best["mean_fused_confidence"] - current_mean_conf, 4)

    # Plain-English reasoning for the agent
    if abs(improvement) < 0.01:
        reasoning = (
            f"Sweep across {len(weights)} weight combinations on {len(samples)} "
            f"dual-modality observations. Best weights "
            f"(audio={best['audio_weight']}, visual={best['visual_weight']}) "
            f"give only {improvement:+.3f} mean confidence vs current. "
            f"Difference is negligible — recommend keeping current weights."
        )
    elif improvement > 0:
        reasoning = (
            f"Sweep found improvement: audio={best['audio_weight']}, "
            f"visual={best['visual_weight']} gives "
            f"{improvement:+.3f} higher mean confidence "
            f"({best['mean_fused_confidence']:.3f} vs {current_mean_conf:.3f}) "
            f"on {len(samples)} recent observations. "
            f"Recommend applying new weights."
        )
    else:
        reasoning = (
            f"Current weights are already near-optimal on recent data. "
            f"Best alternative gives {improvement:.3f} change. Keep current."
        )

    return {
        "hours_analysed": hours,
        "n_observations": len(obs),
        "n_with_both_modalities": len(both_modal),
        "n_samples_swept": len(samples),
        "sweep_results": sweep_results,
        "recommended": {
            "audio_weight": best["audio_weight"],
            "visual_weight": best["visual_weight"],
            "mean_fused_confidence": best["mean_fused_confidence"],
            "improvement_over_current": improvement,
        },
        "current_weights": {"audio": current_audio_w, "visual": 1 - current_audio_w},
        "reasoning": reasoning,
    }


# ── Tool: evaluate_detection_threshold ───────────────────────────────────────


def evaluate_detection_threshold(
    observations_path: str,
    hours: float = 6.0,
    threshold_range: tuple[float, float] = (0.60, 0.85),
    steps: int = 6,
) -> dict[str, Any]:
    """
    Sweep confidence thresholds to find the optimal sensitivity/precision balance.

    Higher threshold = fewer false positives but misses low-confidence
    real detections. Lower threshold = catches more birds but risks
    background noise triggering notifications.

    This tool estimates the tradeoff from live data — the agent can then
    decide whether to tighten or loosen the gate based on current conditions.

    Args:
        observations_path: Path to logs/observations.jsonl.
        hours:             Time window for analysis. Default 6.0.
        threshold_range:   (min, max) thresholds to sweep. Default (0.60, 0.85).
        steps:             Number of threshold values. Default 6.

    Returns:
        {
            "hours_analysed": float,
            "n_observations": int,
            "current_threshold": float,
            "sweep_results": [
                {"threshold": float, "detections_passed": int,
                 "detections_suppressed": int, "pass_rate_pct": float,
                 "mean_confidence_passed": float},
                ...
            ],
            "recommended_threshold": float,
            "reasoning": str,
        }
    """
    obs = _load_observations(observations_path, hours=hours)
    confidences = [float(o.get("fused_confidence", 0)) for o in obs]

    if len(confidences) < 5:
        return {
            "hours_analysed": hours,
            "n_observations": len(obs),
            "current_threshold": 0.70,
            "sweep_results": [],
            "recommended_threshold": 0.70,
            "reasoning": f"Only {len(obs)} observations in last {hours}h. Need 5+ for sweep.",
        }

    low, high = threshold_range
    step_size = (high - low) / (steps - 1) if steps > 1 else 0
    thresholds = [round(low + step_size * i, 3) for i in range(steps)]

    current_threshold = 0.70
    sweep_results = []

    for t in thresholds:
        passed = [c for c in confidences if c >= t]
        suppressed = len(confidences) - len(passed)
        sweep_results.append(
            {
                "threshold": t,
                "detections_passed": len(passed),
                "detections_suppressed": suppressed,
                "pass_rate_pct": round(len(passed) / len(confidences) * 100, 1),
                "mean_confidence_passed": round(mean(passed), 3) if passed else 0.0,
            }
        )

    # Recommend the threshold that passes ≥70% of detections with ≥0.75 mean conf
    # This is a practical heuristic — the agent can override with its own reasoning
    recommended = current_threshold
    for r in sweep_results:
        if r["pass_rate_pct"] >= 70 and r["mean_confidence_passed"] >= 0.75:
            recommended = r["threshold"]
            break

    current_result = next(
        (r for r in sweep_results if abs(r["threshold"] - current_threshold) < 0.01),
        sweep_results[len(sweep_results) // 2],
    )

    reasoning = (
        f"Swept {len(thresholds)} thresholds on {len(confidences)} observations. "
        f"Current threshold {current_threshold} passes "
        f"{current_result['pass_rate_pct']:.0f}% of detections. "
        f"Recommended threshold {recommended} balances coverage and precision."
    )

    return {
        "hours_analysed": hours,
        "n_observations": len(obs),
        "current_threshold": current_threshold,
        "sweep_results": sweep_results,
        "recommended_threshold": recommended,
        "reasoning": reasoning,
    }


# ── Tool: compare_model_backends ──────────────────────────────────────────────


def compare_model_backends(
    observations_path: str,
    hours: float = 24.0,
) -> dict[str, Any]:
    """
    Compare Hailo NPU vs CPU inference quality from the observation log.

    Observations tagged with detection_mode="yolo" used the Hailo pipeline.
    Observations tagged "fixed_crop" may have used either backend depending
    on hailo.enabled. This tool compares confidence distributions between
    recent windows to help the agent decide whether NPU inference is
    producing better results than CPU fallback.

    Note: This is a signal-level comparison, not a latency benchmark.
    Latency data lives in notebooks/hailo_benchmark.ipynb.
    This tool asks: does the NPU produce higher-confidence classifications?

    Args:
        observations_path: Path to logs/observations.jsonl.
        hours:             Time window. Default 24.0 for a full day.

    Returns:
        {
            "hours_analysed": float,
            "by_detection_mode": {
                "fixed_crop": {"n": int, "mean_conf": float, "std_conf": float},
                "yolo":       {"n": int, "mean_conf": float, "std_conf": float},
            },
            "hailo_delta": float,   # yolo_mean - fixed_crop_mean
            "recommendation": str,
            "reasoning": str,
        }
    """
    obs = _load_observations(observations_path, hours=hours)

    by_mode: dict[str, list[float]] = defaultdict(list)
    for o in obs:
        mode = o.get("detection_mode", "fixed_crop")
        conf = float(o.get("fused_confidence", 0))
        by_mode[mode].append(conf)

    result: dict[str, Any] = {
        "hours_analysed": hours,
        "by_detection_mode": {},
    }

    for mode, confs in by_mode.items():
        result["by_detection_mode"][mode] = {
            "n": len(confs),
            "mean_conf": round(mean(confs), 4) if confs else 0.0,
            "std_conf": round(stdev(confs), 4) if len(confs) > 1 else 0.0,
        }

    fc = result["by_detection_mode"].get("fixed_crop", {})
    yo = result["by_detection_mode"].get("yolo", {})

    if not fc or not yo:
        result["hailo_delta"] = 0.0
        result["recommendation"] = "insufficient_data"
        result["reasoning"] = (
            "Need observations from both detection modes to compare. "
            "Run A/B experiment for at least one full cycle first."
        )
        return result

    delta = round(yo["mean_conf"] - fc["mean_conf"], 4)
    result["hailo_delta"] = delta

    if delta > 0.03:
        result["recommendation"] = "prefer_yolo"
        result["reasoning"] = (
            f"YOLO/Hailo mode shows {delta:+.3f} higher mean confidence "
            f"({yo['mean_conf']:.3f} vs {fc['mean_conf']:.3f}) "
            f"over {hours}h. Hailo NPU producing better crops for classifier."
        )
    elif delta < -0.03:
        result["recommendation"] = "prefer_fixed_crop"
        result["reasoning"] = (
            f"Fixed crop shows {abs(delta):.3f} higher mean confidence "
            f"({fc['mean_conf']:.3f} vs {yo['mean_conf']:.3f}). "
            f"YOLO crops may be misaligned — check feeder mounting."
        )
    else:
        result["recommendation"] = "no_preference"
        result["reasoning"] = (
            f"Both modes performing similarly (delta={delta:+.3f}). "
            f"A/B switching is not degrading quality. Continue experiment."
        )

    return result


# ── Tool: apply_fusion_weights ────────────────────────────────────────────────


def apply_fusion_weights(
    audio_weight: float,
    visual_weight: float,
    thresholds_path: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Write new fusion weights to configs/thresholds.yaml.

    This is the only calibration tool with a side effect — it modifies
    a config file. The agent should only call this after:
        1. run_fusion_weight_sweep() confirms improvement > 0.02
        2. The improvement is based on at least 10 dual-modal observations

    The write preserves all other thresholds.yaml content — only
    fusion.audio_weight and fusion.visual_weight are updated.

    Args:
        audio_weight:     New audio weight. Must be in (0, 1).
        visual_weight:    New visual weight. Must equal 1 - audio_weight.
        thresholds_path:  Path to configs/thresholds.yaml.
        dry_run:          If True, validate and return what would be written
                          without actually writing. Use for testing.

    Returns:
        {
            "success": bool,
            "dry_run": bool,
            "previous": {"audio": float, "visual": float},
            "applied": {"audio": float, "visual": float},
            "path": str,
            "message": str,
        }
    """
    # Validate weights
    audio_weight = round(float(audio_weight), 3)
    visual_weight = round(float(visual_weight), 3)

    if not (0 < audio_weight < 1):
        return {
            "success": False,
            "dry_run": dry_run,
            "previous": {},
            "applied": {},
            "path": thresholds_path,
            "message": f"Invalid audio_weight={audio_weight}. Must be in (0, 1).",
        }

    weight_sum = round(audio_weight + visual_weight, 3)
    if abs(weight_sum - 1.0) > 0.01:
        return {
            "success": False,
            "dry_run": dry_run,
            "previous": {},
            "applied": {},
            "path": thresholds_path,
            "message": (
                f"Weights must sum to 1.0. Got {weight_sum}. "
                f"Set visual_weight = {round(1.0 - audio_weight, 3)}."
            ),
        }

    path = Path(thresholds_path)
    if not path.exists():
        return {
            "success": False,
            "dry_run": dry_run,
            "previous": {},
            "applied": {},
            "path": thresholds_path,
            "message": f"thresholds.yaml not found at {thresholds_path}.",
        }

    with path.open() as f:
        config = yaml.safe_load(f)

    previous = {
        "audio": float(config.get("fusion", {}).get("audio_weight", 0.55)),
        "visual": float(config.get("fusion", {}).get("visual_weight", 0.45)),
    }

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "previous": previous,
            "applied": {"audio": audio_weight, "visual": visual_weight},
            "path": thresholds_path,
            "message": (
                f"Dry run — would update fusion weights: "
                f"audio {previous['audio']} → {audio_weight}, "
                f"visual {previous['visual']} → {visual_weight}."
            ),
        }

    # Apply update — preserve all other config keys
    if "fusion" not in config:
        config["fusion"] = {}
    config["fusion"]["audio_weight"] = audio_weight
    config["fusion"]["visual_weight"] = visual_weight

    with path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(
        "Fusion weights updated: audio %s→%s visual %s→%s",
        previous["audio"],
        audio_weight,
        previous["visual"],
        visual_weight,
    )

    return {
        "success": True,
        "dry_run": False,
        "previous": previous,
        "applied": {"audio": audio_weight, "visual": visual_weight},
        "path": thresholds_path,
        "message": (
            f"Fusion weights updated: "
            f"audio {previous['audio']} → {audio_weight}, "
            f"visual {previous['visual']} → {visual_weight}."
        ),
    }


# ── Shared helper ─────────────────────────────────────────────────────────────


def _load_observations(
    observations_path: str,
    hours: float,
) -> list[dict[str, Any]]:
    """
    Load observations from the last N hours. Shared by all calibration tools.
    Malformed lines are skipped silently — robustness over strictness.
    """
    path = Path(observations_path)
    if not path.exists():
        return []

    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    results = []

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
                if ts >= cutoff:
                    results.append(obs)
            except Exception:
                continue

    return results
