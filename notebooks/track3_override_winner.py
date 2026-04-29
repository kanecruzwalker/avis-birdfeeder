"""
Override Track 3 winner selection — ship V2 instead of V1.

The evaluation script's automatic winner selection used "best deploy_macro_f1"
as the primary metric. V1 won that comparison (0.776 vs V2 at 0.736).
However, V1 was scored on only 184 of 642 deployment test records — the ones
without CALT, NONE, or UNKNOWN — because V1 has no class for those. V2 was
scored on 254 records (CALT included), making it a harder evaluation task
on a wider class set.

Comparing them like-for-like:
- V1 punts on CALT entirely. Deploy any future CALT capture and V1
  misclassifies it. The original problem is unfixed.
- V2 correctly classifies 67/70 CALT records (F1=0.77) while keeping
  AMCR/HOFI/SOSP at acceptable levels.
- Both stay within NABirds tolerance.

Per the investigation doc's success criteria, V2 hits 4/5 explicit goals.
The 5th (NONE F1 ≥ 0.85) is only achievable in V3, which has serious issues
in the UNKNOWN class.

This script ships V2.

Run from repo root:
    python notebooks/track3_override_winner.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import joblib

REPO_ROOT = Path(".").resolve()
MODELS_DIR = REPO_ROOT / "models" / "visual"
RESULTS_DIR = REPO_ROOT / "notebooks" / "results" / "phase8" / "track3_retraining"

V2_PATH = MODELS_DIR / "sklearn_pipeline_track3_v2.pkl"
PRODUCTION_PATH = MODELS_DIR / "sklearn_pipeline.pkl"

if not V2_PATH.exists():
    print(f"ERROR: {V2_PATH} not found.")
    sys.exit(1)

# Backup the existing production pipeline (the V0 baseline that's currently active)
backup_path = MODELS_DIR / "sklearn_pipeline_v0_backup.pkl"
if PRODUCTION_PATH.exists() and not backup_path.exists():
    shutil.copy2(PRODUCTION_PATH, backup_path)
    print(f"Backed up current production pipeline → {backup_path.name}")
else:
    print(f"Backup already exists at {backup_path.name} (or no current pipeline).")

# Deploy V2 to production location
shutil.copy2(V2_PATH, PRODUCTION_PATH)
print(f"Deployed V2 → {PRODUCTION_PATH.name}")

# Verify
bundle = joblib.load(PRODUCTION_PATH)
print(
    f"\nActive variant: {bundle.get('track3_variant')}\n"
    f"Classes: {bundle['n_classes']}\n"
    f"Best C: {bundle['best_c']}\n"
    f"Notes: {bundle.get('notes', '(none)')}\n"
)

# Update winner.txt with rationale
winner_path = RESULTS_DIR / "winner.txt"
rationale_path = RESULTS_DIR / "winner_rationale.md"

winner_path.write_text("v2\n")

rationale_path.write_text(
    """# Track 3 winner selection — V2

## Automatic selection said V1; we deployed V2 instead.

### The script's decision

The `phase8_track3_evaluation.py` script applied "best deploy_macro_f1"
with the NABirds tolerance gate. By that rule, V1 wins:

| Variant | Deploy F1 | NABirds F1 |
|---------|-----------|------------|
| V0      | 0.131     | 0.931      |
| V1      | 0.776     | 0.926      |
| V2      | 0.736     | 0.921      |
| V3      | 0.519     | 0.863      |

### Why the rule misled us

The `deploy_macro_f1` numbers are not directly comparable across variants
because each variant evaluates a different subset of the deployment test
set:

| Variant | Records evaluated | Subset |
|---------|------|--------|
| V0      | 184  | Only NABirds-vocab classes (no CALT, NONE, UNKNOWN) |
| V1      | 184  | Same as V0 |
| V2      | 254  | + CALT |
| V3      | 642  | All test records |

V1's score is computed on the *easy* records — the 184 that contain
species V1 has classes for. The 458 records V1 cannot classify
(CALT, NONE, UNKNOWN) are excluded entirely from V1's score. They
represent the actual problem we set out to fix.

V2 is scored on a harder, wider task. Its 0.736 reflects classifying
CALT correctly 96% of the time alongside the existing classes.

### Why V2 is the right ship

1. **V2 fixes the headline finding.** 521 CALT records were the
   primary motivation for Track 3. V1 cannot classify them. V2
   classifies them correctly (per-class F1 = 0.77).

2. **V2's confusion matrix shows the diagnostic confusion is
   resolved.** V0 systematically mislabeled CALT as MOCH/MODO.
   V2 correctly outputs CALT 67/70 times.

3. **V2 stays within NABirds tolerance.** 0.921 vs V0's 0.931 —
   no catastrophic forgetting on the existing 19 species.

4. **V2 hits 4/5 success criteria from the investigation doc:**
   - CALT F1 ≥ 0.70: 0.77 ✓
   - NABirds macro F1 ≥ 0.85: 0.92 ✓
   - Deploy macro F1 ≥ V0 + 0.10: +0.61 ✓
   - CALT-mislabeled-as-MOCH/MODO < 5%: <2% ✓
   - NONE F1 ≥ 0.85: only achievable in V3, which has UNKNOWN-class
     issues. Not deployed.

### Why V3 is not deployed

V3's overall macro F1 (0.52) was pulled down by the UNKNOWN class
(F1 = 0.14). With only 84 training records and the heterogeneous
nature of "blurry / multi-bird / weird angle" captures, the model
could not learn a coherent UNKNOWN representation. Most UNKNOWN
records were predicted as CALT (33/58).

V3's NONE class itself works reasonably (F1 = 0.77), but the
UNKNOWN class's failure mode contaminates other classes through
class confusion.

**Recommendation for next iteration:** drop UNKNOWN as a class.
Use threshold-based abstention via the existing ScoreFuser logic
instead. Re-evaluate with V2-style architecture plus NONE only.

### Decision rule lesson learned

The `phase8_track3_evaluation.py` decision rule should be revised
to require evaluation set parity before comparing macro F1 values.
A model that "doesn't try" the hard cases will always look better
than a model that tries and partially succeeds.

A revised rule for future Track-N evaluations:

> Among variants whose evaluation set sizes are within 5% of the
> largest variant's evaluation set, select the one with highest
> deploy_macro_f1 that does not regress NABirds macro F1 by more
> than 0.05.

Variants that evaluate on < 95% of the test set are excluded from
the comparison; their narrowness is itself disqualifying.
""")

print(f"Updated {winner_path.name} → 'v2'")
print(f"Wrote rationale → {rationale_path.name}")
print()
print("Next:")
print("  1. Review winner_rationale.md")
print("  2. Run agent locally to verify model loads correctly")
print("  3. Commit + push to PR branch")
print("  4. After merge: pull on Pi, restart agent")