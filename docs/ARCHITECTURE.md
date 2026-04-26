# Architecture

## System Overview

Avis is an agentic AI system. The central `BirdAgent` runs a continuous perception-decision-action loop on the Raspberry Pi. It does not wait for human input — it monitors its environment autonomously and takes action when it detects a bird.

```
┌─────────────────────────── BirdAgent Loop ─────────────────────────────┐
│                                                                         │
│  PERCEIVE         DETECT            CLASSIFY            ACT             │
│  ────────         ──────            ────────            ───             │
│  Mic input   →   Presence      →   Audio species   ─┐                  │
│  Camera feed →   Detection     →   Visual species  ─┴→ Fuse → Notify  │
│                                                          ↓              │
│                                                         Log             │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `src/agent/`
The orchestration layer. `bird_agent.py` owns the main loop. It calls into the audio and visual pipelines, passes results to fusion, and dispatches notifications and logs. **Nothing else should import from agent** — agent imports from everything else, not the reverse.

### `src/audio/`
Three-stage pipeline:
1. `capture.py` — Grabs frames from the Pi Camera(s) on motion trigger.
   Optional bird-presence gate (Hailo YOLO or CPU YOLO) filters empty frames
   before classification (Phase 8 Branch 2).
2. `preprocess.py` — Resizes, normalizes, and optionally augments images.
   Output is HWC float32 (224, 224, 3), ImageNet-normalized.
3. `classify.py` — Wraps frozen EfficientNet-B0 + sklearn LogReg inference,
   returns a `ClassificationResult`. Currently 20 classes (Track 3 V2 model:
   19 NABirds species + CALT). Either the CPU PyTorch path or the Hailo
   NPU path is selected at runtime based on `hardware.yaml`. Both paths
   produce identical 1280-dim feature vectors for the LogReg head.

### `src/vision/`
Three-stage pipeline:
1. `capture.py` — Grabs frames from the Pi Camera(s) on motion trigger
2. `preprocess.py` — Resizes, normalizes, and optionally augments images
3. `classify.py` — Wraps EfficientNet/MobileNet inference, returns a `ClassificationResult`

### `src/fusion/`
`combiner.py` takes one `ClassificationResult` from audio and one from visual and produces a single fused species prediction with a combined confidence score. The weighting strategy (configurable via YAML) can be equal weighting, confidence-weighted, or learned.

### `src/notify/`
`notifier.py` receives a confirmed `BirdObservation` and dispatches it via configured channels (push notification, email, local log file). Channels are toggled in `.env`.

### `src/data/`
- `schema.py` — Pydantic models. **The single source of truth for data shapes across all modules.**
- `downloader.py` — Utilities for fetching and caching datasets.

## Data Flow

```
Raw audio (WAV)     →  [preprocess]  →  Mel spectrogram  →  [BirdNET]   →  ClassificationResult
(subprocess)
Raw image (frame)   →  [motion gate] →  [bird gate*]      →  [classify] →  ClassificationResult
(frozen EN+LR)
↓
[ScoreFuser] → BirdObservation
↓
[Notifier] → Push + Log
(observations.jsonl)

bird-presence gate optional, configurable via hardware.yaml detection_mode

The visual classifier produces predictions over 20 species (Track 3 V2):
the original 19 NABirds-trained species (AMCR, AMRO, ANHU, BLPH, DOWO,
EUST, HOFI, HOORI, HOSP, LEGO, MOCH, MODO, OCWA, SOSP, SPTO, WBNU,
WCSP, WREN, YRUM) plus CALT (California Towhee) added from verified
deployment data.
```

## Configuration Philosophy

**No magic numbers in source code.** All tunable values live in `configs/`:

| File | Contents |
|------|----------|
| `configs/species.yaml` | List of San Diego region species. Drives dataset filtering, model output labels, and UI display names. Currently 20 species after Track 3 added CALT. |

| `configs/thresholds.yaml` | Detection confidence cutoffs, audio window duration |
| `configs/paths.yaml` | Dataset paths, model weight paths, log output path |
| `configs/notify.yaml` | Notification channel config (which channels are active) |

## Hardware Integration Notes

- The Hailo AI HAT+ connects via PCIe. Inference calls go through the `hailort` Python bindings.
- Both cameras connect via CSI ribbon. `picamera2` library handles capture for both.
- The Fifine USB mic appears as a standard ALSA audio device. `sounddevice` handles capture.
- All heavy training happens on laptop/Colab — the Pi is **inference-only**.


## Visual Classifier Retraining Workflow (Track 3)

The frozen EfficientNet-B0 backbone and the LogReg head are decoupled.
The backbone weights never change — only the head is retrained when new
labeled deployment data is available.

┌────────────────────────────────────┐
                       │  models/visual/frozen_extractor.pt │  ← never retrained
                       │  (timm EfficientNet-B0 ImageNet)   │
                       └─────────────┬──────────────────────┘
                                     │ 1280-dim features
                                     ▼
                       ┌────────────────────────────────────┐
                       │  models/visual/sklearn_pipeline.pkl│  ← retrained per Track-N
                       │  (StandardScaler + LogReg head)    │
                       └────────────────────────────────────┘

      A new retraining iteration follows this pattern:

1. **Layer 1** (`tools/labeler/`) generates pre-labels via Gemini vision
   over recent captures.
2. **Layer 2** (`tools/labeler/ui/`) presents the pre-labels for human
   verification, producing `data/labels/verified_labels.jsonl`.
3. **Track-N** (`notebooks/phase8_track{N}_*.py`) creates chronological
   splits, trains LogReg variants on cached frozen features, evaluates
   on both NABirds test (preserved benchmark) and deployment test
   (real-world performance), and selects a winner.
4. The winner's `.pkl` is copied to `models/visual/sklearn_pipeline.pkl`
   for production. The previous baseline is preserved at
   `sklearn_pipeline_v0_backup.pkl` for rollback.

This decoupling is a deliberate architectural choice: it keeps the
expensive feature extractor stable (no risk of catastrophic forgetting,
no Hailo recompilation needed when the head changes) while making
vocabulary expansion and class adjustments cheap (LogReg head trains in
seconds on cached features).                 
## Dependency Boundaries

```
agent  →  audio, vision, fusion, notify, data
audio  →  data
vision →  data
fusion →  data
notify →  data
data   →  (no internal deps)
```

This means `data/schema.py` can always be imported without pulling in any ML dependencies. Tests for schema run in milliseconds.
