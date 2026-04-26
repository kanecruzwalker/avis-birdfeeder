"""
Create chronological 70/15/15 train/val/test splits from verified_labels.jsonl.

Run once before Track 3 training. Outputs to:
    data/splits/deployment_train.csv
    data/splits/deployment_val.csv
    data/splits/deployment_test.csv

Each CSV has columns:
    file_path        — absolute path to the capture image
    species_code     — verified label (one of 23 classes including OTHER+CALT)
    other_species_code — present and non-null only when species_code == OTHER
    capture_timestamp — ISO 8601 UTC, what we sort by
    cam_index        — extracted from filename for camera-stratification check
    agreed_with_pre_label — kept for reference, not used by training

Stratification check:
    Warns if any class has < 3 examples in val or test (likely to give
    unstable per-class metrics). Does NOT modify the split — only flags.

Usage:
    python -m tools.labeler.ui.make_deployment_splits

    # Or with custom paths:
    python -m tools.labeler.ui.make_deployment_splits \\
        --verified data/labels/verified_labels.jsonl \\
        --output-dir data/splits
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

CAM_PATTERN = re.compile(r"_cam(\d+)\.")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: {path} not found.", file=sys.stderr)
        sys.exit(1)
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _extract_cam(filename: str) -> int:
    """Pull cam0/cam1 from filenames like 20260424_133057_292152_cam1.png."""
    m = CAM_PATTERN.search(filename)
    return int(m.group(1)) if m else -1


def _resolve_path(record: dict, images_dir: Path) -> str:
    """Map verified record to its image file path on disk."""
    return str((images_dir / record["image_filename"]).resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified", type=Path,
                        default=Path("data/labels/verified_labels.jsonl"))
    parser.add_argument("--images-dir", type=Path,
                        default=Path("data/captures/images"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/splits"))
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--min-class-examples", type=int, default=3,
                        help="Warn if any class has < N records in val or test.")
    args = parser.parse_args(argv)

    print(f"Reading {args.verified}...")
    raw_records = _read_jsonl(args.verified)
    print(f"  Loaded {len(raw_records)} verified records.\n")

    # Build flat DataFrame with everything we need
    rows = []
    skipped = 0
    for r in raw_records:
        capture_ts = r.get("pre_label", {}).get("capture_timestamp")
        if not capture_ts:
            # Fall back to verified_at if pre_label is missing capture_timestamp
            capture_ts = r.get("verified_at")
        if not capture_ts:
            skipped += 1
            continue
        rows.append({
            "file_path": _resolve_path(r, args.images_dir),
            "species_code": r["species_code"],
            "other_species_code": r.get("other_species_code"),
            "capture_timestamp": capture_ts,
            "cam_index": _extract_cam(r["image_filename"]),
            "agreed_with_pre_label": r.get("agreed_with_pre_label"),
            "image_filename": r["image_filename"],
        })

    if skipped:
        print(f"WARNING: skipped {skipped} records with no usable timestamp.\n")

    df = pd.DataFrame(rows)

    # Effective species code — what the training pipeline will use as the label
    # OTHER+CALT becomes "CALT" (a new class). Other sentinels stay as-is.
    df["effective_class"] = df.apply(
        lambda row: row["other_species_code"]
        if row["species_code"] == "OTHER"
        else row["species_code"],
        axis=1,
    )

    # Sort chronologically — earliest first
    df = df.sort_values("capture_timestamp").reset_index(drop=True)

    # Compute split boundaries
    n = len(df)
    train_end = int(n * args.train_fraction)
    val_end = int(n * (args.train_fraction + args.val_fraction))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    print(f"Total verified records:    {n}")
    print(f"Train  ({train_end:>5}, {args.train_fraction*100:.0f}%): "
          f"{train['capture_timestamp'].min()} → {train['capture_timestamp'].max()}")
    print(f"Val    ({len(val):>5}, {args.val_fraction*100:.0f}%): "
          f"{val['capture_timestamp'].min()} → {val['capture_timestamp'].max()}")
    print(f"Test   ({len(test):>5}, {(1-args.train_fraction-args.val_fraction)*100:.0f}%): "
          f"{test['capture_timestamp'].min()} → {test['capture_timestamp'].max()}")
    print()

    # Per-split class distribution
    print("─" * 60)
    print("Class distribution per split (effective_class)")
    print("─" * 60)
    print(f"{'class':12} {'train':>7} {'val':>5} {'test':>5}")
    print("─" * 36)
    classes = sorted(df["effective_class"].unique())
    warnings = []
    for code in classes:
        t = (train["effective_class"] == code).sum()
        v = (val["effective_class"] == code).sum()
        s = (test["effective_class"] == code).sum()
        marker = ""
        if v < args.min_class_examples or s < args.min_class_examples:
            marker = " ⚠"
            warnings.append((code, t, v, s))
        print(f"{code:12} {t:>7} {v:>5} {s:>5}{marker}")

    if warnings:
        print()
        print(f"⚠ {len(warnings)} class(es) have < {args.min_class_examples} records "
              "in val or test:")
        for code, t, v, s in warnings:
            print(f"   {code}: train={t}, val={v}, test={s}")
        print("  Per-class metrics for these will be unstable — interpret with care.")

    # Camera stratification check
    print()
    print("─" * 60)
    print("Camera distribution per split")
    print("─" * 60)
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        cam_counts = Counter(split_df["cam_index"])
        total = len(split_df)
        cam0_pct = cam_counts.get(0, 0) / total * 100 if total else 0
        cam1_pct = cam_counts.get(1, 0) / total * 100 if total else 0
        print(f"  {split_name:5}: cam0={cam_counts.get(0, 0):>4} ({cam0_pct:.1f}%)  "
              f"cam1={cam_counts.get(1, 0):>4} ({cam1_pct:.1f}%)")

    # Save splits
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "deployment_train.csv"
    val_path = args.output_dir / "deployment_val.csv"
    test_path = args.output_dir / "deployment_test.csv"

    cols = ["file_path", "species_code", "other_species_code", "effective_class",
            "capture_timestamp", "cam_index", "agreed_with_pre_label", "image_filename"]
    train[cols].to_csv(train_path, index=False)
    val[cols].to_csv(val_path, index=False)
    test[cols].to_csv(test_path, index=False)

    print()
    print("─" * 60)
    print("Wrote:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print("─" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())