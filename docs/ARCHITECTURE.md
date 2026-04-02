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
1. `capture.py` — Records audio from the USB mic in configurable window sizes
2. `preprocess.py` — Converts raw WAV to mel spectrograms (the input format for BirdNET and our fine-tuned model)
3. `classify.py` — Wraps BirdNET inference via Python 3.11 subprocess bridge
   (scripts/audio_inference.py) on Pi. Falls back to direct birdnetlib call
   on dev machines. Returns a `ClassificationResult`

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
Raw audio (WAV)     →  [preprocess] → Spectrogram → [classify] → ClassificationResult
Raw image (frame)   →  [preprocess] → Tensor      → [classify] → ClassificationResult
                                                                          ↓
                                                               [fusion] → BirdObservation
                                                                          ↓
                                                              [notify] → User + Log
```

## Configuration Philosophy

**No magic numbers in source code.** All tunable values live in `configs/`:

| File | Contents |
|------|----------|
| `configs/species.yaml` | List of San Diego region species (drives all dataset filtering) |
| `configs/thresholds.yaml` | Detection confidence cutoffs, audio window duration |
| `configs/paths.yaml` | Dataset paths, model weight paths, log output path |
| `configs/notify.yaml` | Notification channel config (which channels are active) |

## Hardware Integration Notes

- The Hailo AI HAT+ connects via PCIe. Inference calls go through the `hailort` Python bindings.
- Both cameras connect via CSI ribbon. `picamera2` library handles capture for both.
- The Fifine USB mic appears as a standard ALSA audio device. `sounddevice` handles capture.
- All heavy training happens on laptop/Colab — the Pi is **inference-only**.

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
