# Avis — Project Roadmap

> Full development plan from scaffold to deployed product.
> Update phase status as work progresses.
> Detailed implementation lives in `docs/ARCHITECTURE.md`.

---

## Status Legend
- ✅ Complete
- 🔄 In progress
- ⏳ Planned
- 🔮 Future / post-course

---

## ✅ Phase 1 — Repository Scaffold
**Goal:** Professional, tested foundation before a single ML line is written.

- [x] Full `src/` module structure with stubs and docstrings
- [x] Pydantic schemas — `ClassificationResult`, `BirdObservation`
- [x] 28 passing unit tests, 0 warnings
- [x] GitHub Actions CI — lint + format + tests on every PR
- [x] 4 YAML configs — species, thresholds, paths, notify
- [x] `docs/` — ARCHITECTURE, SETUP, CONTRIBUTING, ONBOARDING
- [x] Custom non-commercial license
- [x] PR template and branch conventions established

---

## ✅ Phase 2 — Preprocessing Pipelines
**Goal:** Both audio and visual data pipelines fully implemented and tested
on laptop with synthetic data — no hardware required.

### Audio
- [x] `src/audio/preprocess.py` — WAV → mono → normalize → mel spectrogram
- [x] `tests/audio/test_preprocess.py` — synthetic WAV input tests (PR #6)
- [x] `scripts/download_datasets.py` — Xeno-canto SD subset via API (PR #11)

### Visual
- [x] `src/vision/preprocess.py` — resize → normalize → augment (PR #8)
- [x] `tests/vision/test_preprocess.py` — synthetic image input tests (PR #8)

### Data
- [x] `configs/species.yaml` — expanded to 20 SD region species (PR #11)
- [x] `src/data/downloader.py` — dataset fetch and cache utilities (PR #11)
- [x] `src/data/splitter.py` — train/val/test split generation 60/20/20 (PR #12)
- [x] `docs/DATASETS.md` — dataset sources and preprocessing decisions
- [x] Verified live: ANHU audio download, NABirds visual splits (3342 images)

### Known issues carried into Phase 3
- CAVI has no NABirds visual data (pre-AOU-split dataset)
- Full Xeno-canto download (all 20 species) not yet run — run before Phase 3

---

## ✅ Phase 3 — Baseline Models
**Goal:** KNN and SVM baselines trained and evaluated. Establishes the
performance floor required by the course before introducing pretrained models.

### Audio baseline
- [x] KNN classifier on MFCC features (librosa)
- [X] Evaluate: accuracy, precision, recall, F1, confusion matrix
- [X] `notebooks/audio_baseline.ipynb`

### Visual baseline
- [X] SVM classifier on flattened/HOG image features
- [X] Evaluate: same metrics as audio
- [X] `notebooks/visual_baseline.ipynb`

### Fusion + Notify (implement stubs)
- [X] `src/fusion/combiner.py` — weighted confidence score fusion
- [X] `src/notify/notifier.py` — log and print channels
- [x] Tests for both modules (no hardware required)

### Experiment tracking
- [x] `notebooks/results/experiments.csv` — running log, 2 rows (KNN + SVM)
- [x] `models/baselines/*.pkl` — frozen baseline artifacts for Phase 4 comparison

### Known issues carried into Phase 4
- CAVI missing from both audio and visual splits (pre-2016 AOU split)
- AMCR too few audio files (2) — skipped from audio training
- Thin-data audio species (BLPH, DOWO, MODO, YRUM) — F1=0.00, data limited
- BirdNET and EfficientNet must beat: audio macro F1 > 0.191, visual macro F1 > 0.121

---

## ✅ Phase 4 — Pretrained Model Integration
**Goal:** BirdNET and EfficientNet integrated, fine-tuned on SD species,
evaluated against baselines. All training on laptop/Colab.

### Audio
- [x] BirdNET pretrained weights downloaded and integrated
- [x] Fine-tune on Xeno-canto SD subset
- [x] `src/audio/classify.py` — AudioClassifier implemented
- [x] Evaluate vs KNN baseline — convergence curves generated
- [x] `notebooks/audio_birdnet.ipynb`

### Visual
- [x] EfficientNet-B0 pretrained weights via timm
- [x] Fine-tune on NABirds SD subset
- [x] `src/vision/classify.py` — VisualClassifier implemented
- [x] Evaluate vs SVM baseline — convergence curves generated
- [x] `notebooks/visual_efficientnet.ipynb`

### Agent (wire classifiers in)
- [x] `src/agent/bird_agent.py` — `_cycle()` implemented with real classifiers
- [x] `src/agent/bird_agent.py` — `from_config()` reads all YAML configs
- [ ] Integration test: full pipeline on recorded test clips (no Pi yet)

### Known results
- Audio CNN from scratch: accuracy=0.116, macro F1=0.089 — below KNN baseline
- BirdNET pretrained: accuracy=0.744, macro F1=0.776 — beats KNN 4x
- EfficientNet fine-tuned: accuracy=0.103, macro F1=0.097 — below SVM baseline
- Frozen EfficientNet + LogReg: accuracy=0.930, macro F1=0.931 — beats SVM 8x
- Key finding: pretrained features transfer dramatically better than
  training from scratch on limited SD species data

---

## 🔄 Phase 5 — Hardware Deployment
**Goal:** System running end-to-end on Raspberry Pi 5 with live capture and push notifications.

### Software (complete)
- [x] `src/audio/capture.py` — Fifine USB mic capture (sounddevice) (PR #26)
- [x] `src/vision/capture.py` — dual Pi Camera Module 3 capture (picamera2) (PR #26)
- [x] `src/vision/stereo.py` — StereoEstimator Phase 6 stub (PR #26)
- [x] `configs/hardware.yaml` — Pi hardware constants and feeder crop config (PR #26)
- [x] `src/audio/classify.py` — updated to BirdNET (F1=0.776) (PR #26)
- [x] `src/vision/classify.py` — updated to frozen EfficientNet + LogReg (F1=0.931) (PR #26)
- [x] `src/agent/bird_agent.py` — live capture loop, cooldown suppression (PR #26)
- [x] `src/notify/notifier.py` — Pushover push notifications, webhook stub (PR #26)
- [x] `src/data/schema.py` — dual-camera and Phase 6 stereo fields added (PR #26)
- [x] `src/fusion/combiner.py` — dual camera visual_result_2 support (PR #26)
- [x] 331 tests passing, CI green
- [x] Push notification confirmed working on device


### Hardware (complete)
- [x] Pi 5 hardware confirmed: dual IMX708 cameras, Fifine mic, SSD, Hailo HAT+
- [x] Two-venv subprocess bridge: Python 3.13 (picamera2) + Python 3.11 (tflite)
- [x] `scripts/audio_inference.py` — BirdNET subprocess bridge (PR #30)
- [x] `scripts/capture_test_frame.py` — feeder crop tuning utility
- [x] Copy source + configs to Pi, set `push: true` in Pi's `notify.yaml`
- [x] Run `python -m src.agent.bird_agent` — live detection and push notification confirmed
- [x] Full audio+visual fusion confirmed live on device
- [x] Push notifications confirmed end to end
- [ ] Mount cameras at feeder — 8cm horizontal baseline, rigid parallel mount
- [ ] Run scripts/capture_test_frame.py, tune feeder_crop in hardware.yaml

### Model compilation (deferred to Phase 6)
- [ ] Export audio model to ONNX → compile to Hailo .hef
- [x] Export visual model to ONNX → compile to Hailo .hef
- [x] Verify inference speed meets real-time requirements
- [x] Hailo HAILO8L benchmark — EfficientNet-B0 HEF compiled and validated
- [x] `notebooks/hailo_benchmark.ipynb` — benchmark results with live Pi data
- [x] `src/vision/hailo_extractor.py` — HailoVisualExtractor class
---

## ⏳ Phase 6 — Evaluation + Report
**Goal:** Complete course deliverables with full quantitative evaluation.

- [ ] Final evaluation on held-out test set (both modalities)
- [ ] Confusion matrices for audio and visual classifiers
- [ ] Comparison table: baseline vs fine-tuned vs fused
- [ ] Demo recording — agent running on Pi, detecting real SD birds
- [ ] Course report written and submitted
- [ ] `CHANGELOG.md` brought fully up to date

- [x] `src/vision/hailo_detector.py` — HailoDetector, YOLOv8s on HAILO8L
- [x] YOLO detect-then-classify pipeline wired into VisionCapture
- [x] Shared VDevice — YOLO + EfficientNet both on NPU simultaneously

---

## 🔮 Post-Course / Future Directions

These are not course requirements but represent the long-term vision:

- **Weatherproof enclosure** — solar-powered, field-deployable hardware
- **Mobile app** — iOS/Android push notifications with species photos
- **Species expansion** — beyond San Diego to North American dataset
- **Learned fusion** — replace weighted average with a small trained combiner
- **Community logging** — multi-feeder network contributing to eBird-style database
- **Commercial product** — hardware + software kit for ornithologists and hobbyists
  (see LICENSE for commercial use terms)

---

## Development Timeline

| Phase | Description | Key Targets | Status |
|-------|-------------|-------------|--------|
| 1 | Repository scaffold | Module structure, schemas, CI, configs | ✅ |
| 2 | Preprocessing pipelines | WAV → spectrogram, image normalize, dataset download | ✅ |
| 3 | Baseline models | KNN audio, SVM visual, fusion + notify implementation | ✅ |
| 4 | Pretrained model integration | BirdNET + EfficientNet fine-tuned on SD species | ✅ |
| 5 | Hardware deployment | Hailo inference, Pi cameras + mic, push notifications | ✅ |
| 6 | Evaluation and report | Metrics, confusion matrices, demo recording, submission | ⏳ |
---