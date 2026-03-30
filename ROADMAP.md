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

## ⏳ Phase 2 — Preprocessing Pipelines
**Goal:** Both audio and visual data pipelines fully implemented and tested
on laptop with synthetic data — no hardware required.

### Audio
- [ ] `scripts/download_datasets.py` — Xeno-canto San Diego subset
- [ ] `src/audio/preprocess.py` — WAV → mono → normalize → mel spectrogram
- [ ] `tests/audio/test_preprocess.py` — synthetic WAV input tests
- [ ] `docs/DATASETS.md` — document all dataset sources and preprocessing decisions

### Visual
- [ ] `scripts/download_nabirds.py` — NABirds San Diego species subset
- [ ] `src/vision/preprocess.py` — resize → normalize → augment (PIL/torchvision)
- [ ] `tests/vision/test_preprocess.py` — synthetic image input tests

### Data
- [ ] Expand `configs/species.yaml` from 15 → full SD region species list
- [ ] `src/data/downloader.py` — implement dataset fetch and cache utilities
- [ ] Train/val/test split generation (60/20/20)

---

## ⏳ Phase 3 — Baseline Models
**Goal:** KNN and SVM baselines trained and evaluated. Establishes the
performance floor required by the course before introducing pretrained models.

### Audio baseline
- [ ] KNN classifier on MFCC features (librosa)
- [ ] Evaluate: accuracy, precision, recall, F1, confusion matrix
- [ ] `notebooks/audio_baseline.ipynb`

### Visual baseline
- [ ] SVM classifier on flattened/HOG image features
- [ ] Evaluate: same metrics as audio
- [ ] `notebooks/visual_baseline.ipynb`

### Fusion + Notify (implement stubs)
- [ ] `src/fusion/combiner.py` — weighted confidence score fusion
- [ ] `src/notify/notifier.py` — log and print channels
- [ ] Tests for both modules (no hardware required)

---

## ⏳ Phase 4 — Pretrained Model Integration
**Goal:** BirdNET and EfficientNet integrated, fine-tuned on SD species,
evaluated against baselines. All training on laptop/Colab.

### Audio
- [ ] BirdNET pretrained weights downloaded and integrated
- [ ] Fine-tune on Xeno-canto SD subset
- [ ] `src/audio/classify.py` — AudioClassifier implemented
- [ ] Evaluate vs KNN baseline — convergence curves generated
- [ ] `notebooks/audio_finetuning.ipynb`

### Visual
- [ ] EfficientNet-B0 pretrained weights via timm
- [ ] Fine-tune on NABirds SD subset
- [ ] `src/vision/classify.py` — VisualClassifier implemented
- [ ] Evaluate vs SVM baseline — convergence curves generated
- [ ] `notebooks/visual_finetuning.ipynb`

### Agent (wire classifiers in)
- [ ] `src/agent/bird_agent.py` — `_cycle()` implemented with real classifiers
- [ ] `src/agent/bird_agent.py` — `from_config()` reads all YAML configs
- [ ] Integration test: full pipeline on recorded test clips (no Pi yet)

---

## ⏳ Phase 5 — Hardware Deployment
**Goal:** System running end-to-end on Raspberry Pi 5 with Hailo inference.

### Model compilation
- [ ] Export audio model to ONNX → compile to Hailo .hef
- [ ] Export visual model to ONNX → compile to Hailo .hef
- [ ] Verify inference speed meets real-time requirements

### Pi deployment
- [ ] `src/audio/capture.py` — Fifine USB mic capture (sounddevice)
- [ ] `src/vision/capture.py` — Pi Camera Module 3 capture (picamera2)
- [ ] Hardware tests marked `@pytest.mark.hardware` — run on Pi only
- [ ] End-to-end test: real bird detection, classification, notification

### Notification
- [ ] `src/notify/notifier.py` — push notification channel (Pushover)
- [ ] Live stream via RTSP or Flask (secondary camera)

---

## ⏳ Phase 6 — Evaluation + Report
**Goal:** Complete course deliverables with full quantitative evaluation.

- [ ] Final evaluation on held-out test set (both modalities)
- [ ] Confusion matrices for audio and visual classifiers
- [ ] Comparison table: baseline vs fine-tuned vs fused
- [ ] Demo recording — agent running on Pi, detecting real SD birds
- [ ] Course report written and submitted
- [ ] `CHANGELOG.md` brought fully up to date

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
| 2 | Preprocessing pipelines | WAV → spectrogram, image normalize, dataset download | ⏳ |
| 3 | Baseline models | KNN audio, SVM visual, fusion + notify implementation | ⏳ |
| 4 | Pretrained model integration | BirdNET + EfficientNet fine-tuned on SD species | ⏳ |
| 5 | Hardware deployment | Hailo inference, Pi cameras + mic, push notifications | ⏳ |
| 6 | Evaluation and report | Metrics, confusion matrices, demo recording, submission | ⏳ |
---