# Track 3 winner selection — V2

## Automatic selection said V1; we deployed V2 instead.

### The script's decision

The phase8_track3_evaluation.py script applied "best deploy_macro_f1"
with the NABirds tolerance gate. By that rule, V1 wins:

| Variant | Deploy F1 | NABirds F1 |
|---------|-----------|------------|
| V0      | 0.131     | 0.931      |
| V1      | 0.776     | 0.926      |
| V2      | 0.736     | 0.921      |
| V3      | 0.519     | 0.863      |

### Why the rule misled us

The deploy_macro_f1 numbers are not directly comparable across variants
because each variant evaluates a different subset of the deployment test
set:

| Variant | Records evaluated | Subset |
|---------|------|--------|
| V0      | 184  | Only NABirds-vocab classes (no CALT, NONE, UNKNOWN) |
| V1      | 184  | Same as V0 |
| V2      | 254  | + CALT |
| V3      | 642  | All test records |

V1 is scored on the easy records — the 184 that contain species V1 has
classes for. The 458 records V1 cannot classify (CALT, NONE, UNKNOWN)
are excluded entirely from V1 score. They represent the actual problem
we set out to fix.

V2 is scored on a harder, wider task. Its 0.736 reflects classifying
CALT correctly 96% of the time alongside the existing classes.

### Why V2 is the right ship

1. V2 fixes the headline finding. 521 CALT records were the primary
   motivation for Track 3. V1 cannot classify them. V2 classifies them
   correctly (per-class F1 = 0.77).

2. V2 confusion matrix shows the diagnostic confusion is resolved.
   V0 systematically mislabeled CALT as MOCH/MODO. V2 correctly outputs
   CALT 67/70 times.

3. V2 stays within NABirds tolerance. 0.921 vs V0 0.931 — no
   catastrophic forgetting on the existing 19 species.

4. V2 hits 4/5 success criteria from the investigation doc:
   - CALT F1 at least 0.70: actual 0.77 PASS
   - NABirds macro F1 at least 0.85: actual 0.92 PASS
   - Deploy macro F1 at least V0 plus 0.10: actual +0.61 PASS
   - CALT-mislabeled-as-MOCH/MODO under 5%: actual under 2% PASS
   - NONE F1 at least 0.85: only achievable in V3, which has UNKNOWN
     class issues. Not deployed.

### Why V3 is not deployed

V3 overall macro F1 (0.52) was pulled down by the UNKNOWN class
(F1 = 0.14). With only 84 training records and the heterogeneous
nature of "blurry / multi-bird / weird angle" captures, the model
could not learn a coherent UNKNOWN representation. Most UNKNOWN
records were predicted as CALT (33/58).

V3 NONE class itself works reasonably (F1 = 0.77), but the
UNKNOWN class failure mode contaminates other classes through
class confusion.

Recommendation for next iteration: drop UNKNOWN as a class. Use
threshold-based abstention via the existing ScoreFuser logic
instead. Re-evaluate with V2-style architecture plus NONE only.

### Decision rule lesson learned

The phase8_track3_evaluation.py decision rule should be revised
to require evaluation set parity before comparing macro F1 values.
A model that does not try the hard cases will always look better
than a model that tries and partially succeeds.

A revised rule for future Track-N evaluations:

  Among variants whose evaluation set sizes are within 5% of the
  largest variant evaluation set, select the one with highest
  deploy_macro_f1 that does not regress NABirds macro F1 by more
  than 0.05.

Variants that evaluate on under 95% of the test set are excluded
from the comparison; their narrowness is itself disqualifying.
