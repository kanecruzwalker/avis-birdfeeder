"""Diagnostic script: inspect verified_labels.jsonl.

Read-only. Validates each record against the VerifiedLabel schema, prints
per-species distribution, agreement rate with the pre-labeler, OTHER
breakdown, and flags any anomalies (duplicates, malformed records,
orphan filenames not in pre_labels.jsonl).

Usage:
    python -m tools.labeler.ui.inspect

    # or with custom paths:
    python -m tools.labeler.ui.inspect \\
        --verified data/labels/verified_labels.jsonl \\
        --pre-labels data/labels/pre_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from tools.labeler.schema import (
    KNOWN_SPECIES_CODES,
    SENTINELS,
    VerifiedLabel,
)


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file, returning list of parsed-JSON dicts. Skips blank lines."""
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(f"  ⚠ {path.name}:{lineno} → JSON parse error: {exc}")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified", type=Path, default=Path("data/labels/verified_labels.jsonl"))
    parser.add_argument("--pre-labels", type=Path, default=Path("data/labels/pre_labels.jsonl"))
    args = parser.parse_args(argv)

    print(f"\nInspecting: {args.verified}")
    print(f"   against: {args.pre_labels}\n")

    verified_raw = _read_jsonl(args.verified)
    pre_raw = _read_jsonl(args.pre_labels)

    if not verified_raw:
        print(f"No verified records found at {args.verified}.")
        if pre_raw:
            print(f"({len(pre_raw)} pre-labels exist — go verify some!)")
        return 0

    # ── 1. Schema validation ──────────────────────────────────────────
    print("─" * 60)
    print("Schema validation")
    print("─" * 60)
    valid: list[VerifiedLabel] = []
    schema_errors = 0
    for i, raw in enumerate(verified_raw):
        try:
            valid.append(VerifiedLabel.model_validate(raw))
        except ValidationError as exc:
            schema_errors += 1
            print(f"  ⚠ record {i}: {exc.error_count()} validation error(s)")
            for err in exc.errors()[:3]:
                print(f"      - {' → '.join(str(x) for x in err['loc'])}: {err['msg']}")
    if schema_errors == 0:
        print(f"  ✓ All {len(valid)} records pass schema validation.")
    else:
        print(f"  ⚠ {schema_errors}/{len(verified_raw)} records have schema errors.")

    if not valid:
        return 1

    # ── 2. Per-species distribution ───────────────────────────────────
    print("\n" + "─" * 60)
    print("Distribution by verified species_code")
    print("─" * 60)
    by_species = Counter(v.species_code for v in valid)
    for code, count in by_species.most_common():
        bar = "█" * min(count, 30)
        marker = ""
        if code in SENTINELS:
            marker = "  ← sentinel"
        elif code not in KNOWN_SPECIES_CODES:
            marker = "  ← UNEXPECTED (not in vocab + not sentinel)"
        print(f"  {code:8} {count:4}  {bar}{marker}")

    # ── 3. OTHER breakdown ─────────────────────────────────────────────
    others = [v for v in valid if v.species_code == "OTHER"]
    if others:
        print("\n" + "─" * 60)
        print("OTHER → other_species_code breakdown")
        print("─" * 60)
        other_counts = Counter(v.other_species_code for v in others)
        for code, count in other_counts.most_common():
            note = ""
            if code in KNOWN_SPECIES_CODES:
                note = "  ← in known vocab? (might be a confusion)"
            print(f"  {code:6} {count:4}{note}")
        print(f"\n  Total OTHER: {len(others)} ({len(others) / len(valid) * 100:.1f}% of verified)")

    # ── 4. Agreement-with-pre-label rate ──────────────────────────────
    print("\n" + "─" * 60)
    print("Agreement with pre-labeler")
    print("─" * 60)
    agree_count = sum(1 for v in valid if v.agreed_with_pre_label is True)
    disagree_count = sum(1 for v in valid if v.agreed_with_pre_label is False)
    null_count = sum(1 for v in valid if v.agreed_with_pre_label is None)
    print(f"  Agreed:    {agree_count:4} ({agree_count/len(valid)*100:5.1f}%)")
    print(f"  Corrected: {disagree_count:4} ({disagree_count/len(valid)*100:5.1f}%)")
    if null_count:
        print(f"  Unset:     {null_count:4} ({null_count/len(valid)*100:5.1f}%)")

    # Per-species agreement rate (only for species with ≥3 records)
    by_pre_species = defaultdict(lambda: [0, 0])  # [agree, total]
    for v in valid:
        pre_code = v.pre_label.llm_response.species_code if v.pre_label else "?"
        by_pre_species[pre_code][1] += 1
        if v.agreed_with_pre_label is True:
            by_pre_species[pre_code][0] += 1
    rows = [(p, a, t) for p, (a, t) in by_pre_species.items() if t >= 3]
    if rows:
        print("\n  Per-pre-label-species (≥3 reviewed):")
        rows.sort(key=lambda x: (-x[2], x[0]))
        for pre_code, agree, total in rows:
            rate = agree / total
            print(f"    {pre_code:8} {agree:3}/{total:<3}  ({rate*100:5.1f}%)")

    # ── 5. Duplicate detection ────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Duplicate / orphan checks")
    print("─" * 60)
    filenames = [v.image_filename for v in valid]
    fname_counts = Counter(filenames)
    dupes = [(f, c) for f, c in fname_counts.items() if c > 1]
    if dupes:
        print(f"  ⚠ {len(dupes)} filename(s) appear multiple times:")
        for f, c in dupes[:10]:
            print(f"      {f}  ({c}×)")
        print("    (the store always atomic-rewrites on correction, so this means")
        print("    something wrote duplicates — investigate before retraining.)")
    else:
        print("  ✓ No duplicate filenames.")

    # Orphans: verified records pointing to filenames NOT in pre_labels.jsonl
    pre_filenames = {p.get("image_filename") for p in pre_raw}
    orphans = [v.image_filename for v in valid if v.image_filename not in pre_filenames]
    if orphans:
        print(f"  ⚠ {len(orphans)} verified record(s) reference unknown image_filenames:")
        for f in orphans[:5]:
            print(f"      {f}")
    else:
        print("  ✓ All verified records map to a pre-label.")

    # ── 6. Reviewer notes ──────────────────────────────────────────────
    notes = [v for v in valid if v.reviewer_notes]
    if notes:
        print("\n" + "─" * 60)
        print(f"Records with reviewer notes ({len(notes)})")
        print("─" * 60)
        for v in notes[-5:]:  # last 5
            print(f"  [{v.species_code}{('·' + v.other_species_code) if v.other_species_code else ''}] "
                  f"{v.image_filename}")
            print(f"    “{v.reviewer_notes}”")

    # ── 7. Time span ──────────────────────────────────────────────────
    if len(valid) >= 2:
        timestamps = sorted(v.verified_at for v in valid)
        first, last = timestamps[0], timestamps[-1]
        span = last - first
        print(f"\n  Verification span: {first.isoformat()} → {last.isoformat()}")
        print(f"  Duration: {span}")
        if span.total_seconds() > 60:
            rate = len(valid) / (span.total_seconds() / 60)
            print(f"  Rate: ~{rate:.1f} records/min")

    # ── 8. Coverage ───────────────────────────────────────────────────
    if pre_raw:
        coverage = len(valid) / len(pre_raw) * 100
        remaining = len(pre_raw) - len(valid)
        print("\n" + "─" * 60)
        print(f"Overall coverage: {len(valid)} / {len(pre_raw)}  ({coverage:.1f}%)")
        print(f"Remaining: {remaining} pre-labels to review")
        print("─" * 60)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
