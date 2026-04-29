# Avis Labeling Assistant

Agent-assisted labeling tool that pre-labels Avis deployment images using
Gemini 2.5 Flash vision, enabling rapid human review to produce training
data for classifier retraining.

## What this is

The visual classifier (frozen EfficientNet-B0 + LogReg, trained on
NABirds) achieves macro F1 = 0.931 on the held-out test set but fails
on deployment. Visual confidence on real feeder birds averages ~0.14 —
essentially the classifier's prior. The cause is feature-level
out-of-distribution: NABirds side-view side-angle photos do not match
our overhead-view feeder captures.

The fix is retraining on deployment-distribution data. This tool
addresses the data-collection problem: labeling ~500 captured images
is tedious manual work. Gemini 2.5 Flash pre-labels each image with a
suggested species + reasoning, and a human reviewer confirms or
corrects in a separate review UI (upcoming PR).

## Pipeline layers

This PR implements Layer 1. Layers 2-4 are separate follow-up PRs.

1. **Pre-labeling** (this PR): `python -m tools.labeler` →
   `data/labels/pre_labels.jsonl`
2. **Human review UI** (future PR): `python -m tools.labeler.server` →
   `data/labels/verified_labels.jsonl`
3. **Training data export** (future PR): JSONL → folder-structured
   `data/training/{train,val,test}/{species}/*.png`
4. **Retraining** (future PR): new `sklearn_pipeline_v2.pkl`

## Quick start

```bash
# Smoke test on 10 recent images (costs ~$0.01)
python -m tools.labeler --limit 10 --verbose

# Real batch: latest 500 images, post-PR #51 color fix only
python -m tools.labeler --limit 500 --post-pr51

# Resume a previous run (already-labeled images are skipped automatically)
python -m tools.labeler --limit 500 --post-pr51

# Primary camera only
python -m tools.labeler --limit 500 --post-pr51 --camera cam0
```

## Requirements

- `GEMINI_API_KEY` or `GOOGLE_API_KEY` in `.env` (LangChain falls back
  from `GOOGLE_API_KEY` to `GEMINI_API_KEY` automatically)
- `langchain-google-genai` (already in `requirements.txt`)
- Capture images at `data/captures/images/` (default path, configurable
  with `--image-dir`)
- Optional: `logs/observations.jsonl` for audio hints (default path,
  `--no-observations` to skip)

## Output format

Pre-labels are appended to `data/labels/pre_labels.jsonl`, one
`PreLabel` record per line:

```json
{
  "image_path": "/mnt/data/avis-birdfeeder/data/captures/images/...",
  "image_filename": "20260424_141605_420369_cam0.png",
  "capture_timestamp": "2026-04-24T14:16:05.420369+00:00",
  "observation_timestamp": "2026-04-24T14:16:06.123456+00:00",
  "audio_hint": "HOFI",
  "audio_confidence": 0.85,
  "llm_response": {
    "bird_visible": true,
    "species_code": "HOFI",
    "confidence": 0.92,
    "reasoning": "Small streaky brown bird with red head on chest.",
    "uncertain_between": null
  },
  "model_name": "gemini-2.5-flash",
  "prompt_version": "v1.0",
  "labeled_at": "2026-04-24T18:55:35.000000+00:00",
  "elapsed_seconds": 3.33
}
```

## Cost and timing

- **Model**: Gemini 2.5 Flash ($0.30 / 1M input, $2.50 / 1M output tokens)
- **Per image**: ~$0.0004 (mostly input tokens for the image)
- **Per batch of 500**: ~$0.20, ~45 minutes wall clock
- **Per batch of 1000**: ~$0.40, ~90 minutes wall clock

## Design decisions

See `docs/investigations/labeling-assistant-2026-04-24.md` for the full
investigation. Key choices:

- **Gemini 2.5 Flash over Pro**: sufficient quality for constrained
  20-class classification, 10x cheaper than Pro.
- **`langchain-google-genai` over `google-generativeai`**: avoids
  protobuf conflict with tensorflow-cpu (documented in PR #42).
- **Pydantic `with_structured_output`**: reliable parsing via schema
  validation rather than free-text regex.
- **Audio hint is weak context only**: prompt instructs Gemini to
  prefer visual evidence over audio match. Audio hints below 0.30
  confidence are dropped to avoid noise.

## Known limitations (v1.0)

Found via 21-image smoke test on 2026-04-24 (76% accuracy):

1. **Out-of-vocabulary species**: Gemini picks closest-matching species
   instead of returning `UNKNOWN`. California Towhee (not in
   `configs/species.yaml`) was labeled as Northern Mockingbird.
2. **Over-confidence**: blurry or ambiguous images get 0.85+ confidence
   instead of the 0.4-0.7 range the prompt requested.
3. **No `NONE` vs bird disagreement handling**: if Gemini says NONE but
   audio detected a bird, we log both but don't flag the disagreement.

All three are addressable via prompt iteration (bump `PROMPT_VERSION`
in `tools/labeler/prompts.py` and re-run). They are NOT blockers — the
human review step catches misclassifications.

## File layout

```
tools/labeler/
├── __init__.py              # package docstring
├── __main__.py              # CLI entry (python -m tools.labeler)
├── schema.py                # Pydantic models: PreLabel, PreLabelResponse, VerifiedLabel
├── prompts.py               # Species reference + system/user prompt builders
├── pre_labeler.py           # Agent: Gemini call, retries, batch loop
└── README.md                # this file
```