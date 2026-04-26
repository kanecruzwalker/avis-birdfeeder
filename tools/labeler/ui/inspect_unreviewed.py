"""Diagnose what's in the unreviewed pre-label backlog.

Useful when most of the dataset has been verified but a chunk remains
untouched. Tells you the species distribution of the unreviewed records
and whether spot-checking is worth the time, or whether the remainder
can be auto-confirmed for the dominant species.

Read-only — touches no files.

Usage:
    python -m tools.labeler.ui.inspect_unreviewed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified", type=Path, default=Path("data/labels/verified_labels.jsonl"))
    parser.add_argument("--pre-labels", type=Path, default=Path("data/labels/pre_labels.jsonl"))
    args = parser.parse_args(argv)

    pre = _read_jsonl(args.pre_labels)
    verified = _read_jsonl(args.verified)
    verified_filenames = {v["image_filename"] for v in verified}

    unreviewed = [p for p in pre if p["image_filename"] not in verified_filenames]

    print(f"\nTotal pre-labels:    {len(pre)}")
    print(f"Verified:            {len(verified)}")
    print(f"Unreviewed backlog:  {len(unreviewed)}\n")

    if not unreviewed:
        print("No backlog — all pre-labels have been reviewed.")
        return 0

    # ── Distribution by Gemini's pre-label ──
    print("─" * 60)
    print("Unreviewed backlog — Gemini's pre-labels")
    print("─" * 60)
    by_species = Counter(p["llm_response"]["species_code"] for p in unreviewed)
    for code, count in by_species.most_common():
        pct = count / len(unreviewed) * 100
        print(f"  {code:8} {count:5}  ({pct:5.1f}%)")

    # ── Confidence distribution for the dominant pre-label ──
    dominant_code = by_species.most_common(1)[0][0]
    dominant_records = [p for p in unreviewed if p["llm_response"]["species_code"] == dominant_code]
    if len(dominant_records) >= 10:
        print(f"\n─ Confidence histogram for unreviewed {dominant_code} ({len(dominant_records)} records) ─")
        bins = [0] * 10  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
        for r in dominant_records:
            c = r["llm_response"]["confidence"]
            idx = min(int(c * 10), 9)
            bins[idx] += 1
        for i, count in enumerate(bins):
            lo, hi = i / 10, (i + 1) / 10
            bar = "█" * min(count // max(len(dominant_records) // 60, 1), 60)
            print(f"  {lo:.1f}–{hi:.1f}  {count:5}  {bar}")

    # ── Cross-reference: verified disagreement rate for the dominant code ──
    verified_for_dominant = [
        v for v in verified
        if v.get("pre_label", {}).get("llm_response", {}).get("species_code") == dominant_code
    ]
    if verified_for_dominant:
        agreed = sum(1 for v in verified_for_dominant if v.get("agreed_with_pre_label") is True)
        rate = agreed / len(verified_for_dominant) * 100
        print("\n─ Sanity check ─")
        print(f"  Of {len(verified_for_dominant)} reviewed {dominant_code} records, "
              f"{agreed} agreed with pre-label ({rate:.1f}%).")
        if rate >= 99.0 and len(verified_for_dominant) >= 100:
            print(f"\n  → At {rate:.1f}% agreement on {len(verified_for_dominant)} reviewed examples,")
            print(f"    the remaining {len(dominant_records)} unreviewed {dominant_code} records")
            print(f"    are statistically very likely also {dominant_code}.")
            print("    You could safely bulk-confirm them, OR spot-check a random sample.")
            print("    See README §'Bulk-confirming a high-confidence bucket' for both options.")
        elif rate >= 90.0:
            print(f"\n  → {rate:.1f}% agreement is high but not unanimous. Worth at least")
            print(f"    spot-checking a random 50 of the unreviewed {dominant_code} records")
            print("    before bulk action.")
        else:
            print(f"\n  → {rate:.1f}% agreement is mixed. Continue manual review of the backlog;")
            print("    bulk-confirming would inject errors.")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())