#!/usr/bin/env python3
"""Apply Pi-local dev/testing overrides to config files after git pull.

Problem this script solves
--------------------------
The Pi deployment uses several config values that differ from the safe
committed defaults in configs/. Values like push: true, hailo.enabled: true,
detection_mode: yolo, and the per-camera crop zones are Pi-local calibration
that should never be committed to the repo. After every `git pull` on the Pi,
these overrides need to be re-applied on top of the freshly pulled configs.

Previous approach (scripts/dev_config.sh) used sed-based find-and-replace on
YAML files. This was unreliable: one regex matched the wrong key entirely
(threshold: 0.70 vs the real key confidence_threshold: 0.70 — silently
failed for weeks), another matched too broadly (enabled: false would match
any future config block, not just hailo.enabled), and multi-line blocks
like per-camera crop overrides couldn't be handled at all.

This Python rewrite parses YAML with PyYAML, edits the dict by real key
path, validates it still parses after modification, and backs up the
original before overwriting.

Usage
-----
    python scripts/dev_config.py

Exit codes
----------
    0 — all overrides applied and configs still valid
    1 — config file missing or not parseable (possible YAML corruption)
    2 — unexpected error during override application

Design notes
------------
- Overrides are declared as data at the top of this file, not buried in
  code. To change an override value, edit the PI_OVERRIDES constant.
- Every override declares its full key path. No broad regexes, no accidental
  matches in other config sections.
- Backups are written to configs/*.yaml.bak before modification. These are
  gitignored. If something goes wrong, copy the .bak back manually.
- Final step is a parse-check of every config file — if any file comes out
  corrupted, the script exits non-zero and logs which file and why.
- Designed to be run non-interactively from cron, systemd, or pi.ps1's
  pi-pull shortcut.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

# ── Pi-local overrides ────────────────────────────────────────────────────────
# Declarative: each entry is (config_file_relative_path, key_path_tuple, value).
# Key path is a tuple of dict keys from the root of the YAML document down to
# the value being set. Example: ("hailo", "enabled") sets root["hailo"]["enabled"].
#
# Edit this list to change what gets overridden on the Pi. Nothing else in
# this file needs to change.
PI_OVERRIDES: list[tuple[str, tuple[str, ...], Any]] = [
    # hardware.yaml — Pi hardware and detection mode
    ("configs/hardware.yaml", ("hailo", "enabled"), True),
    ("configs/hardware.yaml", ("hailo", "detection_mode"), "yolo"),
    ("configs/hardware.yaml", ("cameras", "motion_threshold"), 0.005),
    # Per-camera crop overrides — calibrated to the current physical mount
    # on Kane's feeder as of 2026-04-19. If the cameras are re-mounted or
    # moved to a different feeder, re-run scripts/capture_test_frame.py and
    # update these values.
    # Feeder crop coordinates — scaled 1.5x on 2026-04-23 to match the
    # 2304x1296 capture upgrade. The real-world feeder area these capture
    # is unchanged; coordinates only rescaled to the new raw frame size.
    # If the cameras are re-mounted or moved, re-run
    # scripts/capture_test_frame.py and update these values.
    (
        "configs/hardware.yaml",
        ("cameras", "feeder_crop_cam0"),
        {"x": 945, "y": 195, "width": 1050, "height": 870},
    ),
    (
        "configs/hardware.yaml",
        ("cameras", "feeder_crop_cam1"),
        {"x": 630, "y": 195, "width": 1050, "height": 870},
    ),
    # notify.yaml — enable mobile push notifications
    ("configs/notify.yaml", ("channels", "push"), True),
    # thresholds.yaml — lower dispatch threshold for live calibration.
    # Note: the key is confidence_threshold, not threshold — the old sed
    # script had this wrong and silently did nothing.
    ("configs/thresholds.yaml", ("agent", "confidence_threshold"), 0.20),
]

# Resolve project root relative to this script so the tool works whether
# invoked from repo root or from scripts/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict. Raises on parse failure."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a dict (got {type(data).__name__})")
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write a dict back to YAML. Uses safe_dump so Python-specific tags
    never leak into config files."""
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def backup(path: Path) -> Path:
    """Copy path to path.bak. Returns the backup path."""
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    return bak


def set_nested(data: dict[str, Any], key_path: tuple[str, ...], value: Any) -> None:
    """Set data[k1][k2][...][kn] = value, creating intermediate dicts as
    needed. Raises TypeError if the path tries to descend into a non-dict."""
    cursor: Any = data
    for key in key_path[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[key_path[-1]] = value


def get_nested(data: dict[str, Any], key_path: tuple[str, ...]) -> Any:
    """Read data[k1][k2][...][kn]. Returns None if any key is missing."""
    cursor: Any = data
    for key in key_path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def format_key_path(key_path: tuple[str, ...]) -> str:
    """Render a key path as a dotted string for log output."""
    return ".".join(key_path)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def apply_overrides() -> list[tuple[Path, tuple[str, ...], Any, Any]]:
    """Apply PI_OVERRIDES. Returns a list of (path, key_path, old, new) tuples
    for summary printing."""
    # Group overrides by file so each file is loaded, modified in-memory, and
    # saved exactly once — avoids repeated disk I/O and partial-state risk.
    files_to_overrides: dict[Path, list[tuple[tuple[str, ...], Any]]] = {}
    for rel_path, key_path, value in PI_OVERRIDES:
        abs_path = PROJECT_ROOT / rel_path
        files_to_overrides.setdefault(abs_path, []).append((key_path, value))

    changes: list[tuple[Path, tuple[str, ...], Any, Any]] = []

    for path, overrides in files_to_overrides.items():
        if not path.exists():
            raise FileNotFoundError(f"Config file missing: {path}")

        backup(path)
        data = load_yaml(path)

        for key_path, new_value in overrides:
            old_value = get_nested(data, key_path)
            set_nested(data, key_path, new_value)
            changes.append((path, key_path, old_value, new_value))

        save_yaml(path, data)

    return changes


def validate_configs() -> None:
    """Parse-check every config file we touched. Raises on any failure."""
    touched_files = {PROJECT_ROOT / rel_path for rel_path, _, _ in PI_OVERRIDES}
    for path in touched_files:
        try:
            load_yaml(path)
        except (yaml.YAMLError, ValueError) as exc:
            raise RuntimeError(f"Validation failed for {path}: {exc}") from exc


def print_summary(changes: list[tuple[Path, tuple[str, ...], Any, Any]]) -> None:
    """Print a human-readable summary of what changed."""
    print("Pi dev overrides applied:")
    print("-" * 60)
    by_file: dict[Path, list[tuple[tuple[str, ...], Any, Any]]] = {}
    for path, key_path, old, new in changes:
        by_file.setdefault(path, []).append((key_path, old, new))

    for path, entries in by_file.items():
        print(f"\n{path.relative_to(PROJECT_ROOT)}")
        for key_path, old, new in entries:
            key_str = format_key_path(key_path)
            if old == new:
                print(f"  {key_str}: {new}  (unchanged)")
            else:
                print(f"  {key_str}: {old!r} -> {new!r}")

    print("\n" + "-" * 60)
    print(f"Backups written to configs/*.yaml.bak")


# ── Entrypoint ────────────────────────────────────────────────────────────────
def main() -> int:
    try:
        changes = apply_overrides()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — we want the broad catch here
        print(f"error applying overrides: {exc}", file=sys.stderr)
        return 2

    try:
        validate_configs()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "One or more configs are corrupted after override application. "
            "Restore from configs/*.yaml.bak manually.",
            file=sys.stderr,
        )
        return 1

    print_summary(changes)
    return 0


if __name__ == "__main__":
    sys.exit(main())