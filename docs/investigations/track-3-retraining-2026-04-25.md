# Track 3 — Visual Classifier Retraining

**Date:** 2026-04-25
**Author:** Kane Cruz-Walker (with assistance)
**Status:** Active — implementation begins immediately after doc review
**Branch:** `feature/track-3-retraining` (created from main after Layer 2 merge)
**Predecessor:** Layer 2 (`feature/layer-2-review-ui`, merged 2026-04-26)
**Related artifacts:**
- `docs/investigations/labeling-assistant-ui-2026-04-25.md`
- `notebooks/visual_efficientnet.ipynb` (Phase 4 — current model)
- `notebooks/phase7_evaluation.ipynb` (Phase 7 — baseline metrics)
- `notebooks/classifier_retrain_experiment_2026-04-23.py` (preliminary retrain experiments)

---

## TL;DR

The visual classifier is currently a frozen EfficientNet-B0 + sklearn
LogisticRegression head trained on 19 SD species from NABirds. It achieves
macro F1 ≈ 0.93 on NABirds held-out but performs poorly on real deployment
captures, because the feeder's actual visual distribution is different from
the NABirds training distribution. The dominant out-of-vocabulary species
(California Towhee, CALT) accounts for 12.2% of verified deployment data
and is being mislabeled as MOCH or MODO.

Track 3 retrains the LogReg head — keeping the frozen extractor — on a
mixed dataset of NABirds + 4280 verified deployment captures, with three
new explicit classes (CALT, NONE, UNIDENTIFIED) for a total of 23 classes.
Four model variants are trained and compared against both the original
NABirds test set (preserves benchmark) and a held-out deployment test set
(measures real-world performance). The best variant by deployment macro F1
becomes the new production model.

---

## Hypothesis

The visual classifier's poor deployment performance is primarily caused
by **distribution mismatch between training data (NABirds — curated bird
photos) and deployment data (feeder camera captures at variable distance,
lighting, and angle)**, compounded by **incomplete vocabulary** (CALT
absent) and **lack of explicit no-bird / uncertain handling**.

The retraining will succeed if:
1. Adding CALT to the vocabulary brings CALT-bucket F1 from 0.0
   (current — every CALT misclassified) to > 0.7.
2. Adding NONE as an explicit class reduces false-positive bird
   notifications when the feeder is empty.
3. The model retains > 0.85 macro F1 on the original NABirds test set
   (no catastrophic forgetting on already-known classes).
4. Deployment macro F1 improves measurably over the V0 baseline on
   the held-out deployment test set.

---

## Motivation

### What we know from Layer 2

After 4280 manual reviews from the first deployment data batch:

| Pre-label species | Verified records | Agreement rate |
|---|---|---|
| NONE | 2100 | 99.9% |
| SOSP | 1039 | 96.4% |
| HOFI | 382 | 69.9% |
| MODO | 225 | **0.0%** |
| AMCR | 198 | 100.0% |
| MOCH | 156 | **0.0%** |
| WREN | 66 | 0.0% |
| HOSP | 50 | 0.0% |
| (and 7 more low-agreement species) | 535 total | 0.0% across all |

The agreement column tells two stories. High agreement on NONE/SOSP/AMCR
means the pre-labeler is reliable on those. Zero agreement on MODO/MOCH/etc
means those are visual hallucinations — the pre-labeler proposed those
species but they are not actually visiting.

The OTHER + CALT bucket (521 records, 12.2% of verified) is the
out-of-vocabulary signal. Reviewers identified California Towhee
confidently and consistently, but the model has no class for it, so the
pre-labeler substituted whatever in-vocab class looked closest (mockingbird,
dove, sparrow).

**The diagnostic conclusion:** the deployment data has a much narrower
species set than the training data anticipated (4 visiting species vs
20 trained), AND one of those four is missing from the vocabulary.
Track 3 corrects both issues with a targeted, additive retraining.

### Why retrain now

Three forcing functions:

1. **Dataset size is sufficient.** 4280 verified records covering 4 visual
   species is enough to retrain a LogReg head with statistical confidence.
   The CALT cohort (521) is well above the rule-of-thumb minimum (~50)
   for a new class.
2. **Production model performance is the bottleneck.** Per the Pushover
   notification logs, the agent is sending high-confidence MOCH and MODO
   notifications that are actually CALT. Every false notification
   degrades reviewer trust. Track 3 directly addresses this.
3. **Architecture is favorable to fast iteration.** The LogReg head trains
   in seconds on cached features, so we can run multiple variants
   overnight and pick the best. There is no expensive training cycle to
   worry about.

### Why not retrain the EfficientNet backbone

The phase 4 evaluation showed fine-tuning the full EfficientNet on the
NABirds SD subset produces macro F1 = 0.097 — *worse than baseline SVM*.
Frozen features + LogReg head produced macro F1 = 0.931. This is a
documented finding in the project: **the limited training data is
insufficient to fine-tune a 5M-parameter network without overfitting**.

Adding 4280 deployment records does not change this calculus. The new
records are well below the threshold needed for full fine-tuning (which
typically wants 10K+ examples per class). The frozen-feature architecture
remains correct. Track 3 changes only the LogReg head.

---

## Methodology

### Split strategy

**Date-based chronological splits, applied to the verified deployment
data only. NABirds splits remain unchanged.**

Procedure:
1. Sort all 4280 verified records by `pre_label.capture_timestamp`.
2. Earliest 70% (~2996) → `deployment_train`
3. Next 15% (~642) → `deployment_val`
4. Latest 15% (~642) → `deployment_test` (held out, never seen during
   training or hyperparameter selection)

This is more honest than random splitting because:
- A capture from 2026-04-22 14:31:05 cam0 and another from 2026-04-22
  14:31:42 cam1 are highly correlated (same bird, seconds apart, slightly
  different angle). Random splitting puts them on different sides of the
  boundary; chronological splitting respects this temporal locality.
- The test set becomes "captures from days the model has never seen"
  which is the deployment task on day N+1 in production.

**Stratification check:** After splitting, verify each split contains a
reasonable fraction of every class. If a class is concentrated in
specific dates (CALT visits only happen at certain hours, for example),
the chronological split could starve a class on one side. If this
happens, we apply **stratified-within-bucket** splitting as fallback —
keep the chronological intent, but ensure ≥3 examples of every class in
val and test.

**Why not random:** Random within-day splits inflate scores by ~0.05–0.10
F1 in our experience, because spatially or temporally adjacent frames
leak information. This makes the model look better than it is.

**Why not by camera (e.g., cam0 = train, cam1 = test):** Cam0 and cam1
have slightly different angles but the visual distributions are nearly
identical (verified by the Layer 2 inspector — 51% of captures from each).
Camera-based splits don't test temporal generalization, which is the
real production failure mode.

### NABirds split — unchanged

The existing splits at `data/splits/visual_train.csv` / `visual_val.csv` /
`visual_test.csv` are preserved as-is. These contain 19 species from the
SD subset of NABirds. Track 3 trains on the union of these and the new
deployment splits.

### Class set — 23 total classes

| Source | Classes | Records (verified) |
|---|---|---|
| NABirds existing | 19 species | ~2700 train, ~340 val, ~672 test |
| CALT (new, OOV) | 1 species | 521 deployment (chronologically split) |
| NONE (new, sentinel) | 1 class | 2104 deployment (chronologically split) |
| UNIDENTIFIED (new, sentinel) | 1 class | 149 deployment (chronologically split) |
| AMCR, HOFI, SOSP overlap | 3 species | 1496 deployment added to existing |
| **Total** | **23** | NABirds + 4280 deployment |

Some species (DOWO, BLPH, WBNU, CAVI, WCSP, etc.) have no deployment
data — they didn't visit the feeder during this period. They retain
their NABirds training data and are evaluated only on NABirds test set.
**This is a known feature of Track 3, not a bug.** The model preserves
broad bird-species knowledge for fusion with audio (BirdNET detects
many species the visual side has never seen visually) while specializing
on what's actually visiting.

### Class balancing

NONE has 2104 records — 4× larger than CALT (521) and 14× larger than
UNIDENTIFIED (149). Without rebalancing, LogReg will bias toward
predicting NONE for everything because that maximizes accuracy.

**All Track 3 variants use `class_weight="balanced"` in LogisticRegression**
to apply inverse-frequency weighting at training time. This was a tested
hyperparameter in `classifier_retrain_experiment_2026-04-23.py`.

We do not undersample NONE — keeping all 2104 records preserves variety
in negative examples (different lighting, angles, backgrounds), which is
valuable for the model's ability to distinguish "no bird" from
"backlit bird" or "leaf shadow that looks bird-shaped."

### Hyperparameter tuning

Each variant's LogReg head tunes `C ∈ {0.01, 0.1, 1.0, 10.0, 100.0}` on
the **deployment val set** specifically. The `C` value that maximizes
deployment val macro F1 is selected, then evaluated on both test sets.

This procedure mirrors the existing `visual_efficientnet.ipynb`
Section 11 pattern but with deployment val instead of NABirds val for
selection — because deployment performance is the metric we care about.

`max_iter=5000` to ensure convergence. `solver="lbfgs"` (default).
`random_state=42` for reproducibility.

### Variants — 4 models

| Variant | Training data | Classes | C selection | Tests |
|---|---|---|---|---|
| **V0 — Production baseline** | NABirds train only | 19 (SD species) | Existing C=1.0 (from cell 25) | "Where are we now? Lower bound." |
| **V1 — NABirds + deployment, no new classes** | NABirds train + deployment_train (filtered to known species: HOFI, SOSP, AMCR) | 19 | Tuned on deployment val | "Does just adding deployment-domain examples help?" |
| **V2 — Vocabulary expansion (CALT)** | NABirds train + deployment_train (HOFI, SOSP, AMCR, CALT) | 20 | Tuned on deployment val | "Does adding CALT to the vocabulary fix the headline confusion?" |
| **V3 — Full retrain (CALT + NONE + UNIDENTIFIED)** | NABirds train + deployment_train (all 4 species + sentinels) | 23 | Tuned on deployment val | "Do explicit NONE/UNIDENTIFIED classes improve overall behavior?" |

Each variant produces a `models/visual/sklearn_pipeline_track3_VN.pkl`
and a `models/visual/track3_VN/` results directory with confusion
matrices, per-class F1 plots, and metrics CSV.

### Optional V4 (time permitting)

**V4 — Prototype network head.** Same training data as V3 but classifies
by k-NN on the frozen features instead of LogReg. This is an architecture
ablation, not a data ablation. Tests whether the LogReg head choice
matters or whether any reasonable classifier on top of frozen features
performs similarly. Skip V4 if implementation runs long.

### Evaluation framework

Both test sets are evaluated for every variant:

**NABirds test (672 images, 19 species)** — preserved benchmark
- Macro F1, weighted F1, accuracy
- Per-class F1
- Confusion matrix
- Cohen's kappa
- Balanced accuracy

For variants with classes not in NABirds (CALT, NONE, UNIDENTIFIED), the
NABirds test evaluation excludes those classes' columns from the
confusion matrix — they would never appear in NABirds, by definition.

**Deployment test (642 records, all 23 classes)** — real-world performance
- Macro F1, weighted F1, accuracy (on the 23-class label set)
- Per-class F1 (especially CALT, NONE, UNIDENTIFIED)
- Confusion matrix focused on actual deployment classes
- Per-class precision and recall (separately, not just F1, because
  recall on UNIDENTIFIED matters less than precision on CALT)
- Calibration: when the model says 90% confidence, is it right 90% of the
  time? (Reliability diagram)

**Existing phase 7 framework reuse:** `phase7_evaluation.ipynb` already
implements most of these metrics (ROC-AUC, Cohen's kappa, balanced
accuracy, per-class precision/recall, confusion matrices, qualitative
galleries, dataset-size ablation, fusion-weight sensitivity). Track 3's
evaluation notebook (`phase8_track3_evaluation.ipynb`) imports these
functions rather than reimplementing them.

### Decision rule for production

The variant with the highest **deployment test macro F1** that does not
regress NABirds test macro F1 by more than 0.05 is selected for
production. This rule prevents specializing so hard on the feeder that
we lose the broader species knowledge needed for audio/visual fusion.

If V3 (the maximalist variant) wins, it ships. If V2 wins because
explicit NONE/UNIDENTIFIED classes hurt overall behavior, V2 ships.
The data tells us which.

---

## Implementation plan

### Files to create

```
notebooks/
├── phase8_track3_training.ipynb     ← V0/V1/V2/V3 training and artifact saving
├── phase8_track3_evaluation.ipynb   ← All 4 variants on both test sets
└── results/
    └── phase8/
        └── track3_retraining/       ← Confusion matrices, plots, metrics CSV

models/
└── visual/
    ├── sklearn_pipeline_track3_v0.pkl   ← copy of current sklearn_pipeline.pkl, renamed
    ├── sklearn_pipeline_track3_v1.pkl
    ├── sklearn_pipeline_track3_v2.pkl
    ├── sklearn_pipeline_track3_v3.pkl
    └── frozen_extractor.pt              ← unchanged

data/
└── splits/
    ├── visual_train.csv                  ← unchanged (NABirds)
    ├── visual_val.csv                    ← unchanged
    ├── visual_test.csv                   ← unchanged
    ├── deployment_train.csv              ← NEW — 70% of verified, chronological
    ├── deployment_val.csv                ← NEW — 15% of verified, chronological
    └── deployment_test.csv               ← NEW — 15% of verified, chronological

src/
└── vision/
    └── classify.py                       ← Update MODEL_VERSION constant + label map handling

configs/
├── species.yaml                          ← Add CALT entry (with code, common_name, scientific_name)
└── thresholds.yaml                       ← Possibly update visual confidence threshold for new classes

docs/
└── investigations/
    └── track-3-retraining-2026-04-25.md  ← This document

CHANGELOG.md                              ← New entry for Track 3
```

### Files to update

- `tools/labeler/schema.py` — already supports OTHER+code; no schema changes needed.
- `src/vision/classify.py` — `MODEL_VERSION` constant updates from
  `frozen-efficientnet-b0-logreg-sdbirds-v1` → `track3-v{0,1,2,3}` based on
  which variant is deployed. Label map handling already works for any
  number of classes.
- `configs/species.yaml` — add CALT entry. NONE and UNIDENTIFIED are
  sentinels and already understood by the schema layer; they don't need
  species.yaml entries unless we want common_name display in the UI
  (probably yes for completeness).
- `CHANGELOG.md` — new entry under [Unreleased].

### Implementation sequence (ordered, blocking)

1. **Create deployment splits** — chronological 70/15/15 by capture_timestamp.
   Output to `data/splits/deployment_*.csv`. Stratification check.
2. **Extract features for deployment data** — run frozen extractor over
   all 4280 verified images, cache to disk so subsequent variant training
   doesn't re-extract. Same procedure as `visual_efficientnet.ipynb`
   Section 10 but for new images.
3. **Train V0** — copy existing `sklearn_pipeline.pkl` to
   `sklearn_pipeline_track3_v0.pkl` and tag with track3-v0 version.
   No retraining needed; this is the baseline.
4. **Train V1** — NABirds + deployment HOFI/SOSP/AMCR. Tune C on deployment val. Save.
5. **Train V2** — V1 + CALT. Tune C. Save.
6. **Train V3** — V2 + NONE + UNIDENTIFIED. Tune C. Save.
7. **Evaluate all 4 variants** on both NABirds test and deployment test.
   Generate confusion matrices, per-class plots, metrics CSV.
8. **Select winner** by decision rule above.
9. **Deploy winner** by copying its `.pkl` to `sklearn_pipeline.pkl`
   (the file `VisualClassifier` loads at runtime). MODEL_VERSION constant updated.
10. **Restart agent on Pi** to pick up new model.
11. **Document findings** in this investigation doc's appendix and in CHANGELOG.

Steps 1–8 happen tonight on the laptop. Steps 9–11 happen tomorrow morning
after a final review.

---

## Success criteria

A successful Track 3 retraining means:

| Criterion | Target | Measurement |
|---|---|---|
| CALT class exists with usable performance | F1 ≥ 0.70 on deployment test | Per-class F1 on deployment_test |
| NONE class reduces false-positive notifications | F1 ≥ 0.85 on deployment test | Per-class F1 on deployment_test |
| Existing classes preserved | NABirds macro F1 ≥ 0.85 | Macro F1 on NABirds test |
| Real-world improvement | Deployment macro F1 ≥ V0 + 0.10 | Macro F1 on deployment_test |
| Headline confusion fixed | CALT-mislabeled-as-MOCH/MODO drops to <5% | Confusion matrix off-diagonal counts |

If 4 of 5 criteria hit, ship V-winner. If <4 of 5, investigate before
shipping — most likely cause is data leakage, label noise in deployment_train,
or class imbalance the rebalancing didn't address.

### Headline metric for the writeup

The CS 450 writeup will use this single number as the headline finding:

> **Macro F1 on held-out deployment test set, V-winner vs V0**:
> X.XX vs Y.YY (Δ = +Z.ZZ)

This is one number that tells the entire story: the targeted retraining
moved deployment performance from Y.YY to X.XX. The component-level
analysis (per-class, by-class confusion) provides the explanation.

---

## Risks and mitigations

### Risk 1: Distribution shift not the real problem

**Hypothesis check:** what if performance is bad because the captures
themselves are low-quality (cropping issues, motion blur, etc.) and even
adding training data doesn't help?

**Mitigation:** V1 isolates this. If V1 (deployment data, no new classes)
doesn't improve over V0, the issue is image quality, not vocabulary or
training distribution. We'd then look at the preprocessing pipeline, not
the classifier.

### Risk 2: CALT examples are systematically different from CALT-in-the-wild

**Hypothesis check:** the 521 CALT records are all from one feeder, one
camera setup, one time-of-day distribution. A model trained on these
might overfit to the specific visual characteristics of these captures
(e.g., always seen from above-right at this specific feeder height).

**Mitigation:** the chronological test split catches this. If V2's CALT
F1 is 0.95 on deployment_train but 0.40 on deployment_test, we know the
model learned the camera setup, not the species. Cross-camera (cam0 vs
cam1) disagreement on CALT predictions in deployment is also a flag.

### Risk 3: NONE class is too easy and dominates

**Hypothesis check:** with 2104 NONE examples, the model could learn
"if the image is mostly green/brown, predict NONE" and ignore everything
else.

**Mitigation:** class_weight="balanced" addresses this directly.
Sensitivity check: train V3 with class_weight=None and compare. If
balanced beats unbalanced by < 0.02 macro F1, we'd want to investigate
whether the rebalancing is doing what we think.

### Risk 4: Catastrophic forgetting on rare NABirds species

**Hypothesis check:** species in NABirds with few examples (DOWO, BLPH,
WBNU, etc.) might have their decision boundaries shifted by adding
4280 mostly-feeder examples. The new signal could swamp the old.

**Mitigation:** NABirds test set evaluation catches this. If a rare-class
F1 drops > 0.1, we revisit. Most likely fix is upweighting NABirds-only
examples when training V2/V3. We have not built this in to the initial
plan because the simpler `class_weight="balanced"` should handle it.

### Risk 5: V4 prototype network outperforms V3 LogReg

**Hypothesis check:** if V4 wins clearly, it suggests the LogReg head was
the bottleneck and we should rethink the architecture more broadly.

**Mitigation:** V4 is optional and time-permitting. If we run it and it
wins, ship V3 anyway for this PR (the simpler, more conventional
choice), and write up V4 as a "future work" finding. Don't change
architecture under pressure.

### Risk 6: Deployment test set has labeling errors

**Hypothesis check:** Layer 2 inspection showed 99.9% pre-label agreement
on NONE, but human verification was done by one reviewer (Kane). Personal
biases in edge cases could subtly favor or disfavor certain labels.

**Mitigation:** Daniel can spot-check 50 random records from
deployment_test. If disagreement rate is < 5%, accept the test set as
ground truth. If higher, we need a clearer labeling protocol before
shipping anything.

### Risk 7: Two reviewers disagree on the agent's notification policy

**Hypothesis check:** if V3 ships with explicit NONE prediction, the
agent could use that to suppress notifications when feeder is empty.
But the existing fusion logic uses confidence threshold, not class
prediction. We'd need to update `agent.py` to handle "predicted class
== NONE" specially, which is out of Track 3's scope.

**Mitigation:** Track 3 ships the model only. Agent integration is a
follow-up task. The new model's NONE predictions are logged but ignored
by the agent until a separate PR updates the notification rules.

---

## Rollback plan

If V-winner deployed to Pi performs worse than V0 in production (notification
quality drops, or new failure modes emerge):

1. **Immediate rollback:** rename `sklearn_pipeline.pkl` to
   `sklearn_pipeline_track3_failed.pkl`. Copy the original V0 file
   (preserved as `sklearn_pipeline_track3_v0.pkl`) back to
   `sklearn_pipeline.pkl`. Restart agent. < 5 minutes.
2. **Investigate:** logs from the live deployment show which captures
   failed and which classes were predicted. Compare to deployment_test
   expectations.
3. **Decide:** if a single class is problematic, V2 (without that class)
   may be a fallback ship. If multiple, retrain with revised class set
   or reweighting.

The frozen extractor never changes, so rollback only requires swapping
the LogReg head — no model architecture changes, no Hailo recompilation.

---

## Open questions (to resolve during implementation)

1. **Should NONE be modeled as a 23rd class or as an abstention threshold?**
   Initial plan: 23rd class (V3). If V3's NONE confusion is high
   (predicting NONE when there's a faint bird), we may try V3-alt with
   NONE as threshold-based abstention instead.

2. **How does the agent handle UNIDENTIFIED predictions?** Initial plan:
   pass through to fusion as a low-confidence visual modality. Audio
   has the chance to dominate. This is consistent with current fusion
   behavior; no agent changes needed.

3. **Should we add capture_timestamp to the verified records' metadata
   for future retraining?** Already present via `pre_label.capture_timestamp`.
   No change needed.

4. **Do we re-evaluate phase 7's fusion weight (currently audio=0.55,
   visual=0.45) after Track 3 ships?** Probably yes, but as a separate
   PR. Track 3 changes the visual model's calibration which might shift
   the optimal fusion weight. Out of scope for this PR.

---

## Architecture diagram

```
                    ┌──────────────────────────────────────┐
                    │   NABirds SD subset (existing)       │
                    │   ~2700 train  ~340 val  ~672 test   │
                    │   19 species                         │
                    └──────────────┬───────────────────────┘
                                   │
                                   │ via existing data/splits/visual_*.csv
                                   ▼
                    ┌──────────────────────────────────────┐
                    │   Frozen EfficientNet-B0 extractor   │ ← unchanged from Phase 4
                    │   (timm pretrained, no gradients)    │
                    │   Output: 1280-dim features          │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────┐
                    │      Cached features (laptop disk)   │ ← reused across V0–V3
                    └──────────────┬───────────────────────┘
                                   │
                          ┌────────┼────────┐
                          ▼        ▼        ▼
                       ┌────┐  ┌────┐  ┌────┐  ┌────┐
                       │ V0 │  │ V1 │  │ V2 │  │ V3 │   ← LogReg heads
                       └────┘  └────┘  └────┘  └────┘   trained per variant
                          │      │      │      │
                          └──────┴──────┴──────┘
                                   │
                                   ▼
                    ┌──────────────────────────────────────┐
                    │   Evaluation on both test sets       │
                    │   - NABirds test (preserve benchmark)│
                    │   - Deployment test (real world)     │
                    └──────────────┬───────────────────────┘
                                   │
                                   ▼
                            select V-winner
                                   │
                                   ▼
                    ┌──────────────────────────────────────┐
                    │  models/visual/sklearn_pipeline.pkl  │ ← deploy to Pi
                    └──────────────────────────────────────┘


                    ┌──────────────────────────────────────┐
                    │  Layer 2 verified labels (4280)      │
                    │  Chronologically split:              │
                    │   ~2996 train  ~642 val  ~642 test   │
                    │  Classes used per variant:           │
                    │   V0: none                           │
                    │   V1: HOFI/SOSP/AMCR                 │
                    │   V2: + CALT                         │
                    │   V3: + NONE + UNIDENTIFIED          │
                    └──────────────────────────────────────┘
```

---

## Appendix: notes for the writeup

**The narrative arc:**

1. **Problem identification (Phase 7).** Held-out NABirds eval showed
   macro F1 = 0.93. Production deployment showed catastrophic
   misclassifications.

2. **Diagnosis (Layer 1 + Layer 2).** Built tooling to label deployment
   data. Found:
   - 4 species visited (AMCR, HOFI, SOSP, CALT)
   - CALT (12.2% of captures) was out-of-vocabulary
   - Pre-labeler's hallucinations on OOV captures revealed that LLM
     vision confidence is unreliable on small, low-resolution feeder
     images.

3. **Targeted intervention (Track 3, this PR).** Retrained the LogReg
   head on a mixed dataset. Added CALT, NONE, UNIDENTIFIED. Compared
   four variants. Selected best by decision rule.

4. **Result (to be filled in after implementation).** Deployment
   macro F1 went from X.XX to Y.YY. CALT confusion (mislabeled as
   MOCH/MODO) dropped from N% to M%.

**This is a paper-shaped story.** Clean problem, principled methodology,
defensible evaluation, measured outcome. Even if Track 3 produces
modest gains (rather than transformative ones), the methodology
section is publishable.

**Anticipated questions from a grader:**

- *"Why didn't you use CLIP/DINO/foundation model X?"* Answered above —
  preserves established baseline, allows clean attribution of
  improvement to data, not architecture.
- *"Is your test set really held-out?"* Yes — chronological split, latest
  15% of verified data, never seen during training or hyperparameter
  selection. Methodology section explains.
- *"What about cross-validation?"* The frozen-feature architecture makes
  CV optional. We have a clear val/test split, and the cost of training
  variants is < 1 minute each. CV would add complexity without changing
  conclusions.
- *"Did you check for label noise?"* Layer 2 inspector reported 99.9%
  pre-label agreement on NONE, no duplicates, no orphans. Daniel can
  spot-check deployment_test for additional confidence.
- *"How do you know you didn't overfit hyperparameters to deployment val?"*
  We tune on deployment val and report on deployment test, which is
  the standard separation. Both are chronologically separated.

---

## Sign-off

**Investigation reviewed:** Pending
**Implementation begins:** Immediately after sign-off
**Branch:** `feature/track-3-retraining`
**Estimated implementation time:** 4–6 hours
**Estimated evaluation time:** 1–2 hours
**Target deployment:** Tomorrow morning (Pi restart with new model)

---