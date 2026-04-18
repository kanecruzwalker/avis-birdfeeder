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
- [x] BirdAnalystAgent — custom LLM tool-calling agent (Gemini)
- [x] LangChainAnalyst — LangGraph agent with 3 memory layers
- [x] ExperimentOrchestrator — autonomous Pi controller, systemd boot
- [x] 14 shared calibration + observation tools
- [x] agent_demo.ipynb — validated with live Gemini API calls on Pi
- [ ] Mount cameras at feeder — tune feeder_crop


## ✅ Phase 6 — Hardware + Agentic Layer (Complete)
- [x] Hailo EfficientNet-B0 HEF compiled and deployed (6.3× CPU speedup)
- [x] HailoVisualExtractor wired into VisualClassifier
- [x] HailoDetector — YOLOv8s on HAILO8L
- [x] Shared VDevice — YOLO + EfficientNet both on NPU simultaneously
- [x] Push notifications with image attachments
- [x] BirdAnalystAgent — custom Gemini tool-calling agent
- [x] LangChainAnalyst — LangGraph ReAct agent, 3 memory layers
- [x] ExperimentOrchestrator — autonomous Pi controller, systemd boot
- [x] 14 shared calibration + observation tools
- [x] ReportBuilder — daily + window reports (.md + .json)
- [x] agent_demo.ipynb — validated with live Gemini API calls on Pi
- [x] phase7_evaluation.ipynb scaffold

## ✅ Phase 7 — Held-out Evaluation + Autonomous Deployment (Complete)
- [x] Held-out test set evaluation (audio n=86, visual n=672)
- [x] Confusion matrices for all models
- [x] Comparison table: baseline vs pretrained vs fused
- [x] Dataset size ablation curve
- [x] Fusion weight sensitivity analysis
- [x] Pi systemd boot autostart installed and verified
- [x] pi.ps1 laptop management shortcuts
- [x] BaselineOptimizer stub (OpenClaw-targeted)

## ⏳ Phase 8 — Extended Metrics, Stereo Vision, Web Dashboard
**Goal:** Complete all remaining course deliverables and build the
user-facing product layer.

### 8A — Extended Metrics (feat/extended-metrics)
- [ ] ROC-AUC per class added to phase7_evaluation.ipynb
- [ ] Per-class precision and recall table
- [ ] Professor rubric metrics verified 
- [ ] LangGraph Mermaid agent diagram rendered and saved
- [ ] Fusion weight retuning to optimal (audio=0.05, F1=0.974)
- [ ] All metrics feeding into report section

### 8B — Stereo Vision (feat/stereo-vision)
- [ ] scripts/calibrate_stereo.py — checkerboard calibration on Pi
- [ ] StereoEstimator._rectify() — undistort + stereo rectification
- [ ] StereoEstimator._compute_disparity() — SGBM disparity map
- [ ] StereoEstimator._disparity_to_depth() — depth from baseline + focal length
- [ ] BirdObservation.estimated_depth_cm populated on live detections
- [ ] Agent tool: get_last_depth_estimate() — LLM can reason about bird distance
- [ ] Tests for stereo pipeline (synthetic calibration data)

### 8C — Web Dashboard + Live Feed (feat/web-dashboard)
- [ ] src/web/app.py — FastAPI server
      GET /            — HTML dashboard (single page)
      GET /stream      — MJPEG live video (640×360 @ 10fps, background thread)
      GET /frame       — latest annotated detection frame (JPEG, YOLO box overlay)
      GET /latest      — latest BirdObservation as JSON
      GET /observations — last N detections from observations.jsonl
      GET /status      — agent uptime, detection count, current mode
      POST /ask        — conversational query → BirdAnalystAgent.answer()
      GET /health      — unauthenticated health check
- [ ] src/web/frame_annotator.py — PIL: draw YOLO box + species + confidence
- [ ] src/web/stream_buffer.py — thread-safe ring buffer for MJPEG frames
- [ ] src/web/templates/index.html — responsive single-page dashboard
      Live MJPEG stream | Latest annotated detection | Recent observations
      Agent chat interface | Status bar (mode, uptime, detection count)
- [ ] src/agent/cli.py — terminal Q&A interface (dev path)
      Interactive loop: reads stdin, calls analyst.answer(), prints response
- [ ] Token auth middleware (AVIS_WEB_TOKEN in .env)
      All endpoints except /health require token query param or header
      Protects Gemini API credits and observation data
- [ ] Tailscale installation and configuration
      scripts/install_tailscale.sh — one-command setup
      Persistent private IP, end-to-end encrypted, no public exposure
      Access revocable per-device from Tailscale admin panel
- [ ] ngrok support for ephemeral demo sharing
      scripts/start_ngrok.sh — starts tunnel, prints public URL
      Use only for time-limited demos, never left running unattended
- [ ] CORS headers for cross-origin access
- [ ] avis.service update — start web server alongside orchestrator

### 8D — Report + Slides (last, after 8A-8C merged)
- [ ] IEEE report — three chapters:
      Ch1: Baseline problem (classical features + limited data = near-random)
      Ch2: Transfer learning solution (BirdNET + EfficientNet = dramatic improvement)
      Ch3: Agentic system (autonomous monitoring + LLM reasoning + live dashboard)
- [ ] Slide deck using Phase 7 charts, confusion matrices, live Pi screenshots
- [ ] Demo video — Pi booting autonomously, detection firing, dashboard showing
      live feed, conversational query answered by agent

### 8E — Stubs (architecture documented, implementation post-course)
- [ ] BaselineOptimizer implementation (OpenClaw framework)
      Already stubbed in src/agent/baseline_optimizer.py
- [ ] Dataset expansion (additional Xeno-canto species + iNaturalist visual)
      Already stubbed in scripts/expand_dataset.py (to be created)
- [ ] Webhook push to web dashboard (notifier.py _webhook() stub already exists)

## 🔮 Post-Course / Future
- Commercial weatherproof enclosure + solar power
- Mobile app (iOS/Android) with species photo push notifications
- Species expansion beyond 20 SD species to full North American dataset
- Multi-feeder network contributing to eBird-style community database
- OpenClaw BaselineOptimizer — autonomous hyperparameter search
- Stereo-based size estimation for species disambiguation







- [ ] Phase 7 evaluation — held-out test set, confusion matrices, comparison table
- [ ] Course report and slide deck

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