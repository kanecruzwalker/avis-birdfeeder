# Labeling Assistant — Investigation

**Date:** 2026-04-24
**Branch:** `feat/labeling-assistant`
**Author:** Kane Cruz-Walker

## Hypothesis

An agent-assisted labeling tool will enable rapid curation of training data from deployment-captured images, reducing manual labeling effort from ~1 minute per image to ~10 seconds per image (a 6x speedup). This enables retraining the visual classifier on feeder-distribution data within the course timeline.

## Motivation

The visual classifier (frozen EfficientNet-B0 + LogReg) achieves macro F1 = 0.931 on the NABirds held-out test set but fails on deployment:

- Visual confidence on real feeder birds: mean 0.137, stdev 0.008 (April 24, 1.5 hours, n=138)
- Classifier outputs essentially its prior regardless of visual input
- Same wrong predictions at 2x capture resolution (stdev 0.008) as at 1x (stdev 0.009)

Two prior branches ruled out input-level fixes:
- `feat/adaptive-yolo-crop` (PR #56): tighter crop strategies produced no improvement
- `feat/camera-quality-2x` (PR #58): 2.25x more pixels per bird produced no improvement

This establishes the classifier's failure is feature-level out-of-distribution, not input-level. NABirds training distribution (side-view birds at medium distance, diverse natural backgrounds) does not match deployment (overhead view at ~30cm onto a wooden feeder with seed and orange slices). The remaining lever is retraining on deployment-distribution data.

## Why a labeling assistant (not manual labeling)

Manual labeling 300-500 frames at ~1 minute per frame = 5-8 hours of focused work. Realistic for a single-person effort but tedious, error-prone, and scales poorly.

Agent-assisted pre-labeling reduces this to review-and-confirm:
- Agent examines each image plus any available audio hint
- Proposes species code with reasoning
- Human verifies or corrects
- Effective labeling rate: 300-500 frames in 1-2 hours of focused review

The agent doesn't replace human judgment. It provides a starting point so the reviewer operates in "is this right?" mode instead of "what is this?" mode.

## Architecture

### Layer 1: Pre-labeling (automated batch) — THIS PR

Input:
- Images in `data/captures/images/`
- Observation records in `logs/observations.jsonl`

Output:
- `data/labels/pre_labels.jsonl` (one record per image)

Process per image:
1. Find matching observation record by filename timestamp
2. Extract audio hint from `observation.audio_result` (if any)
3. Query Gemini 2.5 Flash with image + audio hint + species reference
4. Parse structured response: species code, confidence, reasoning, bird_visible
5. Write pre-label record to JSONL

### Layer 2: Human review UI (interactive) — separate PR

Input: `data/labels/pre_labels.jsonl` + images
Output: `data/labels/verified_labels.jsonl`

FastAPI + simple HTML. Per image: show image, pre-label, audio context. User confirms, corrects, or marks skip. Keyboard shortcuts for speed.

### Layer 3: Training data export (automated) — separate PR

Input: `data/labels/verified_labels.jsonl`
Output: `data/training/{train,val,test}/{species}/*.png`

Temporal train/val/test split (60/20/20) to avoid near-duplicate leakage across splits.

### Layer 4: Retraining pipeline (automated) — separate PR

Input: `data/training/`
Output: `models/visual/sklearn_pipeline_v2.pkl`

Extract features with existing frozen EfficientNet extractor. Train new LogReg head. Evaluate on held-out split before Pi deployment.

## This PR scope

Layer 1 only. Layers 2-4 are separate PRs, each independently reviewable and testable.

## Design decisions

### Why Gemini 2.5 Flash (not 2.5 Pro or 3.1)?

- Vision capability confirmed
- Pricing: ~$0.0004 per image (1000 images = ~$0.40)
- Already used by `BirdAnalystAgent` via `langchain-google-genai` — no new dependency
- Sufficient for constrained 20-class closed-set classification with visual references
- Pro / 3.1 add cost without meaningful quality gain for this task

### Why `langchain-google-genai` (not `google-generativeai`)?

- Already in codebase (Phase 6 agentic layer)
- Avoids protobuf conflict with tensorflow-cpu 2.21.0 (documented in PR #42)
- Consistent with `BirdAnalystAgent` and `LangChainAnalyst` patterns
- Supports `with_structured_output(PydanticSchema)` for reliable parsing
- Multimodal via `HumanMessage` with `image_url` content blocks

### Why structured output via Pydantic schema?

- Parsing free-text LLM responses is fragile
- `with_structured_output(PreLabel)` handles parsing and validation
- Failed parses surface as exceptions we can log and skip, not silent corruption
- Schema becomes documentation of what we expect from the model

### Why temporal train/val/test split (future, Layer 3)?

- Frames captured seconds apart are near-duplicates (same bird pose, similar lighting)
- Random split leaks near-duplicates across train/val/test, inflating measured performance
- Temporal split: train = oldest 60%, val = middle 20%, test = newest 20%
- Approximates real deployment scenario: "train on past, predict on future"

## Agent prompting strategy

The Gemini prompt provides:
- Camera context: overhead-mounted, ~30cm above wooden feeder tray with seed and orange slices
- Full 20-species reference with identifying visual characteristics
- Audio hint as a separate signal, explicitly marked as potentially misleading
- Empty feeder handling: `species_code="NONE"`
- Uncertainty expression: `uncertain_between` field for alternatives

Prompt engineering notes:
- Zero-shot with rich species reference instead of few-shot examples
- Structured JSON output, not prose
- Low temperature (0.1) for consistent classification
- Audio hint provided but explicitly not required to match — prevents hallucination
- Model instructed to override audio hint when visual evidence contradicts

## Success criteria

**Functional:**
- Processes 100 images without crashes
- Outputs valid `PreLabel` records for each successful image
- Handles Gemini API errors gracefully (retry once, then skip)
- Runs in under 10 seconds per image (network-bounded)

**Quality:**
- Pre-label agreement with human review >=70% on a 50-image spot check
- "NONE" (empty feeder) correctly identified in >=90% of clearly empty frames
- Cost per image under $0.001

**Operational:**
- Resumable: re-running skips already-labeled images
- Observable: progress bar, error count, per-image timing
- Auditable: prompt version logged with each record for debugging

## Risks

- **Gemini hallucinates a species:** structured output + retry + skip on parse failure. Human review catches substantive errors.
- **API rate limits:** configurable delay between requests, backoff on 429.
- **Cost overrun:** hard cap on image count per run, confirm before scaling.
- **Pre-label bias feeds into training:** if Gemini consistently errs one way, retraining on those labels would propagate bias. Mitigation: mandatory human review, pre-labels are suggestions not ground truth.
- **Audio hint overfitting:** if Gemini copies audio_hint instead of looking at image, visual signal is lost. Mitigation: explicit prompt instruction, empty-audio test cases during development.

## Rollback

Pre-labeling is an offline batch process. No runtime deployment impact. To roll back: delete `data/labels/pre_labels.jsonl` and re-run if needed. Nothing in `src/` changes. The tool lives in `tools/labeler/` specifically to signal it is dev-time only.

## Follow-up work (separate PRs)

- `feat/labeling-assistant-ui` — FastAPI review interface (Layer 2)
- `feat/training-data-export` — JSONL to folder-structured training set (Layer 3)
- `feat/retrain-classifier-v2` — retraining pipeline + deployment (Layer 4)
- `docs/classifier-v2-evaluation` — measurement of improvement

## Open questions

1. **Which subset to pre-label first?** Starting with most recent 1000 frames (post-PR #51 color fix, April 21+). Empty frames get labeled as "NONE" for free, providing negative class data.
2. **Retain all 20 species in first retrain, or subset?** TBD based on review. Rare species may be dropped or augmented with NABirds data (domain-gap risk acknowledged).
3. **NONE as a class or as a gate?** TBD. Can train 21-class classifier (20 species + NONE), or keep NONE as gate output. Experiment both.
4. **How to handle near-duplicate frames?** Raw capture rate produces near-duplicates. Either deduplicate by perceptual hash before labeling, or label all and rely on temporal split to separate them.


## Smoke test results (2026-04-24)

Ran the pre-labeler on 21 curated images with known ground truth to
validate the end-to-end pipeline.

**Setup:**
- 21 images across 6 categories (HOFI, AMCR, CALT, SOSP, UNKNOWN, NONE)
- No audio hints (--no-observations) — pure visual test
- Gemini 2.5 Flash, temperature 0.1, PROMPT_VERSION v1.0
- Cost: $0.02, wall clock: 106s (avg 5s per image)

**Quantitative results:**
- Exact matches: 16/21 (76%)
- All in-vocabulary species (HOFI, AMCR, SOSP): 11/11 correct (100%)
- Empty feeder detection: 5/5 correct (100%)
- Out-of-vocabulary (CALT): 0/3 correct (0%)
- Low-information images (blurry, edge-cropped): 0/2 correct (0%)

**Qualitative findings:**

1. Gemini correctly identifies all in-vocabulary species with well-calibrated
   confidence (0.85-0.98 range). Reasoning includes specific visual features,
   not generic platitudes.

2. Empty feeder detection is perfect and high-confidence (1.00). The tool
   can reliably separate bird-present from bird-absent frames without
   human review — a significant labor saving.

3. Gemini does NOT use UNKNOWN on out-of-vocabulary or low-information
   images. Instead it picks the closest-matching species from the allowed
   list with confidence 0.85+. The prompt instruction to use UNKNOWN when
   uncertain was insufficient to overcome the pull of "must pick from list."

4. Reasoning text accurately describes observed features even when the
   species label is wrong. For California Towhee images labeled as MOCH,
   Gemini's reasoning described "plain grey-brown plumage with long tail"
   — accurate towhee features, incorrectly mapped to Mockingbird.

**Implications for downstream stages:**

- Human review (Layer 2) will catch the ~24% of mis-predictions. At 76%
  pre-label accuracy, reviewer spends ~10s confirming correct labels and
  ~30s correcting wrong ones — still a 3-4x speedup over manual labeling.
- Empty feeder labels (the largest single category) are essentially
  free — just spot-check a few per review session.
- Gemini's over-confidence on uncertain cases is a prompt-engineering
  limitation, not a blocker. Ship v1.0, iterate later.

**Follow-up work surfaced:**

1. `feat/species-discovery` — Aggregate "wrong" predictions with similar
   reasoning patterns to identify candidate species missing from
   configs/species.yaml. California Towhee is the first known candidate.

2. `prompt/v1.1` — Re-iterate system prompt to more aggressively use
   UNKNOWN on out-of-vocabulary and low-information inputs. Consider
   few-shot examples of when UNKNOWN is the correct answer.

3. `feat/add-calt-species` — Add California Towhee (CALT) to species.yaml,
   audio_label_map.json, and visual_label_map.json. Requires retraining
   BirdNET species filter and adding training data for the new class.