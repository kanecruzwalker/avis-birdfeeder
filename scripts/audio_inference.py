"""
scripts/audio_inference.py

Standalone BirdNET inference script for Python 3.11 subprocess bridge.

This script is intentionally separate from the main src/ package because it
must run under Python 3.11 (tflite_runtime requirement) while the main agent
runs under Python 3.13 (picamera2 requirement). The agent calls this script
as a subprocess and reads the JSON result from stdout.

Usage:
    python scripts/audio_inference.py <wav_path> <species_yaml_path> [min_conf]

Output (stdout):
    On success:  JSON object with keys:
                     species_code, common_name, scientific_name,
                     confidence, error (null)
    On no bird:  JSON object with error="NO_BIRD_DETECTED"
    On failure:  JSON object with error=<message>

Exit codes:
    0 — inference completed (check 'error' field for logical errors)
    1 — unexpected exception before any output
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: audio_inference.py <wav> <species_yaml> [min_conf]"}))
        sys.exit(0)

    wav_path       = Path(sys.argv[1])
    species_yaml   = Path(sys.argv[2])
    min_conf       = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10

    # Validate inputs
    if not wav_path.exists():
        print(json.dumps({"error": f"WAV not found: {wav_path}"}))
        sys.exit(0)

    if not species_yaml.exists():
        print(json.dumps({"error": f"Species YAML not found: {species_yaml}"}))
        sys.exit(0)

    # Load species lookup
    try:
        import yaml
        with species_yaml.open() as f:
            species_cfg = yaml.safe_load(f)
        sci_to_code: dict[str, str] = {}
        code_to_meta: dict[str, dict] = {}
        for s in species_cfg["species"]:
            code = s["code"]
            sci  = s["scientific_name"]
            sci_to_code[sci] = code
            code_to_meta[code] = {
                "common_name":     s["common_name"],
                "scientific_name": sci,
            }
    except Exception as exc:
        print(json.dumps({"error": f"Species YAML load failed: {exc}"}))
        sys.exit(0)

    # Run BirdNET inference
    try:
        try:
            import tflite_runtime.interpreter as tflite  # noqa: F401
        except ImportError:
            from tensorflow import lite as tflite  # noqa: F401

        from birdnetlib import Recording
        from birdnetlib.analyzer import Analyzer

        analyzer = Analyzer()
        recording = Recording(analyzer, str(wav_path), min_conf=min_conf)
        recording.analyze()
        detections = recording.detections

    except Exception as exc:
        print(json.dumps({"error": f"BirdNET inference failed: {exc}"}))
        sys.exit(0)

    if not detections:
        print(json.dumps({"error": "NO_BIRD_DETECTED"}))
        sys.exit(0)

    # Filter to SD species
    sd_detections = []
    for d in detections:
        sci  = d.get("scientific_name", "")
        code = sci_to_code.get(sci)
        if code is not None:
            sd_detections.append((code, float(d["confidence"]), sci))

    if not sd_detections:
        print(json.dumps({"error": "NO_BIRD_DETECTED"}))
        sys.exit(0)

    best_code, best_conf, best_sci = max(sd_detections, key=lambda x: x[1])
    meta = code_to_meta.get(best_code, {})

    print(json.dumps({
        "species_code":    best_code,
        "common_name":     meta.get("common_name", best_code),
        "scientific_name": meta.get("scientific_name", best_sci),
        "confidence":      best_conf,
        "error":           None,
    }))


if __name__ == "__main__":
    main()
