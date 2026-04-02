# Changelog

All notable changes to Avis are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

### Hardware deployment (Phase 5 complete)
- Pi 5 running Debian Trixie (Python 3.13.5) with Hailo AI HAT+ confirmed
- Dual IMX708 cameras (indices 0 and 1, 1536×864) confirmed via rpicam-hello
- Fifine USB mic (sounddevice index 1, 48kHz) confirmed
- Two-venv architecture: Python 3.13 venv for picamera2/visual pipeline,
  pyenv Python 3.11 for BirdNET/tflite_runtime subprocess bridge
- `scripts/audio_inference.py` — standalone BirdNET inference script for
  Python 3.11 subprocess bridge (PR #30)
- System validated live — fused detections confirmed:
  Black Phoebe 100%, Mourning Dove 90%, House Finch 92%, House Sparrow 79%
- Pending: camera physical mounting, feeder crop tuning, Hailo compilation

---

## [0.5.0] — Phase 5 Hardware Deployment (Software Complete)

### Added
- `src/audio/capture.py` — `AudioCapture`: Fifine USB mic (device index 1,
  48kHz native, no resampling), 3-second windows, RMS energy gate discards
  silent frames before BirdNET inference (PR #26)
- `src/vision/capture.py` — `VisionCapture`: dual Pi Camera Module 3 via
  picamera2, simultaneous capture, rolling background motion gate, feeder
  crop ROI applied before 224×224 downsampling, saves raw frames to
  `data/captures/images/` (PR #26)
- `src/vision/stereo.py` — `StereoEstimator` Phase 6 stub: full interface
  defined (calibrate, estimate, _rectify, _compute_disparity,
  _disparity_to_depth), all methods raise `NotImplementedError` until
  Phase 6 stereo calibration (PR #26)
- `configs/hardware.yaml` — Pi hardware constants: mic device index, sample
  rate, camera indices, capture resolution, feeder crop zone (x, y, w, h),
  stereo baseline, Hailo device address (PR #26)
- `src/notify/notifier.py` — `_push()` implemented via Pushover API (urllib,
  no SDK); credentials from `.env`; graceful degradation when credentials
  missing; `_webhook()` Phase 6 stub for future web app backend (PR #26)
- `src/notify/notifier.py` — `enable_webhook`, `webhook_url`,
  `webhook_timeout_seconds`, `webhook_auth_header` parameters and
  `notify.yaml` webhook config block (PR #26)
- Push notification confirmed working on device — House Finch, 87%
  confidence (PR #26)

### Changed
- `src/audio/classify.py` — `AudioClassifier` updated to BirdNET inference
  via birdnetlib (F1=0.776 vs F1=0.089 CNN from scratch). `predict()` now
  takes a WAV file path. Added `NoBirdDetectedError` for graceful degradation
  when no SD species detected (PR #26)
- `src/vision/classify.py` — `VisualClassifier` updated to frozen
  EfficientNet-B0 backbone + sklearn StandardScaler + LogisticRegression
  pipeline (F1=0.931 vs F1=0.097 fine-tuned). Added `camera_index` param
  passed through to `ClassificationResult` (PR #26)
- `src/agent/bird_agent.py` — `_cycle()` wired to live `AudioCapture` and
  `VisionCapture`. Both cameras classify independently. Cooldown suppression
  via `_is_on_cooldown()` prevents notification spam for repeat detections.
  `NoBirdDetectedError` handled as soft audio failure (PR #26)
- `src/data/schema.py` — `ClassificationResult` gains `camera_index` field.
  `BirdObservation` gains `visual_result_2`, `image_path_2`, `detection_box`,
  `estimated_depth_cm`, `estimated_size_cm`, `stereo_calibrated` — all
  optional with `None` defaults, fully backward compatible (PR #26)
- `src/fusion/combiner.py` — `fuse()` accepts optional `visual_result_2`.
  `_select_best_visual()` picks higher confidence or averages when both
  cameras agree on species (PR #26)
- `configs/notify.yaml` — added push/webhook channel toggles and webhook
  config block. `push: false` committed as safe default — set `true` on
  Pi deployment (PR #26)
- `configs/paths.yaml` — added Phase 5 model paths: `visual_frozen_extractor`,
  `visual_sklearn`, `stereo_calibration`, Hailo `.hef` paths (PR #26)
- `.gitignore` — expanded `models/**/*.pt` to cover subdirectory weights,
  removed accidentally tracked binary files (PR #26)
- `notebooks/visual_efficientnet.ipynb` — added cell 28: saves
  `frozen_extractor.pt` + `sklearn_pipeline.pkl` with verification round-trip
  (PR #26)
- `.env.example` — added `PUSHOVER_USER_KEY`, `PUSHOVER_APP_TOKEN`,
  `WEBHOOK_AUTH_TOKEN` with setup instructions (PR #27)
- `docs/SETUP.md` — added Phase 5 model artifact generation section (PR #26)
- `docs/DATASETS.md` — added model artifacts table (PR #26)

### Test count
- 331 passing, 0 failing, CI green

---

## [0.4.3] — Phase 4 Frozen EfficientNet + Linear Classifier

### Added
- `notebooks/visual_efficientnet.ipynb` Section 11 — frozen EfficientNet-B0
  feature extractor + LogisticRegression; test accuracy=0.930, macro F1=0.931,
  weighted F1=0.930 (19 species, n=672); 8x improvement over SVM baseline (PR #X)
- `notebooks/results/visual_linear_confusion_matrix.png`,
  `visual_linear_per_class_f1.png` — evaluation plots for frozen+linear approach

### Known results
- Best C=0.1 on val set; all 19 species beat SVM baseline individually
- DOWO and SOSP: F1=1.00; HOSP lowest at F1=0.85

---

## [0.4.2] — Phase 4 BirdNET Pretrained Audio

### Added
- `notebooks/audio_birdnet.ipynb` Section 10-13 — BirdNET pretrained inference
  via birdnetlib 0.9.0; test accuracy=0.744, macro F1=0.776, weighted F1=0.823
  (18 species, n=86); 4x improvement over KNN baseline (PR #X)
- `notebooks/results/audio_birdnet_confusion_matrix.png`,
  `audio_birdnet_per_class_f1.png` — BirdNET evaluation plots
- `resampy==0.4.3` — required for birdnetlib MP3 decoding via librosa

### Known results
- 66/86 test files got a known-species detection (coverage 77%)
- BLPH, DOWO, HOSP, MODO: F1=1.00; ANHU lowest at F1=0.67

---

## [0.4.1] — Phase 4 Classifier Modules + Agent Wiring

### Added
- `src/audio/classify.py` — AudioClassifier wrapping `_build_audio_cnn`;
  lazy loading, from_config() reads paths.yaml, predict() on mel spectrograms
- `src/vision/classify.py` — VisualClassifier wrapping EfficientNet-B0 via
  timm; lazy loading, HWC->CHW transpose at inference
- `src/agent/bird_agent.py` — from_config() and _cycle() implemented;
  graceful degradation per modality, threshold gate before dispatch
- `scripts/generate_label_map.py` — derives label maps from split CSVs;
  writes models/label_map.json, audio_label_map.json, visual_label_map.json
- `tests/audio/test_classify.py` — 18 tests
- `tests/vision/test_classify.py` — 19 tests
- `tests/agent/test_bird_agent.py` — 20 tests
- `tests/scripts/test_generate_label_map.py` — 10 tests

---

## [0.4.0] — Phase 4 Baseline CNN Models

### Added
- `notebooks/audio_birdnet.ipynb` — CNN from scratch on mel spectrograms;
  test accuracy=0.116, macro F1=0.089 (18 species, n=86); underperforms
  KNN baseline — insufficient data for CNN from scratch
- `notebooks/visual_efficientnet.ipynb` — EfficientNet-B0 fine-tuned;
  test accuracy=0.103, macro F1=0.097 (19 species, n=672); overfits on
  limited data
- `notebooks/results/experiments.csv` — cleaned 6-row canonical log with
  deduplication guard in all Phase 4 notebooks
- `notebooks/results/phase3/`, `phase4/` — result PNGs organized by phase

### Changed
- `requirements.txt` — added birdnetlib==0.9.0, tensorflow-cpu==2.21.0,
  resampy==0.4.3 under Phase 4 BirdNET inference section

---

## [0.3.2] — Phase 3 Fusion + Notify

### Added
- `src/fusion/combiner.py` — `ScoreFuser` fully implemented: equal, weighted,
  and max confidence fusion strategies; winner-takes-all species disagreement
  handling; graceful single-modality fallback; `from_config()` reads
  `configs/thresholds.yaml` (PR #18)
- `src/notify/notifier.py` — `Notifier` fully implemented: JSONL log channel
  appends to `logs/observations.jsonl`; print channel formats via
  `message_template`; `from_config()` reads `notify.yaml` + `paths.yaml`;
  push/email deferred to Phase 5 (PR #18)
- `tests/fusion/test_combiner.py` — expanded from 7 to 44 tests covering all
  strategies, disagreement resolution, single-modality fallback, and
  `from_config()` (PR #18)
- `tests/notify/test_notifier.py` — 30 new tests: init, log, print, dispatch,
  and `from_config()` (PR #18)

---

## [0.3.1] — Phase 3 Visual Baseline

### Added
- `notebooks/visual_baseline.ipynb` — SVM classifier on HOG + color histogram
  features (26340-dim vector); C selection on val set (best C=10.0);
  test accuracy=0.213, macro F1=0.121 (19 species, n=672) (PR #17)
- `notebooks/results/visual_baseline_*.png` — frozen C-selection, confusion
  matrix, and per-class F1 plots
- `notebooks/results/experiments.csv` — second row appended (SVM visual)
- `models/baselines/visual_svm_baseline.pkl` — trained SVM + scaler +
  label encoder saved for Phase 4 comparison (gitignored)

### Changed
- `requirements.txt` — added scikit-image==0.24.0 for HOG feature extraction

### Known results
- Top performer: DOWO F1=0.81 (distinctive black/white pattern)
- YRUM: high recall (0.78) but low precision — class imbalance artifact
- 12 species scored F1=0.00 — expected with HOG+color on limited data

---

## [0.3.0] — Phase 3 Audio Baseline

### Added
- `notebooks/audio_baseline.ipynb` — KNN classifier on MFCC features
  (80-dim mean+std vector, n_mfcc=40); k selection on val set (best k=3);
  test accuracy=0.302, macro F1=0.191 (18 species, n=86) (PR #16)
- `notebooks/results/audio_baseline_*.png` — frozen k-selection, confusion
  matrix, and per-class F1 plots
- `notebooks/results/experiments.csv` — running experiment log, first row
  appended (KNN audio)
- `models/baselines/audio_knn_baseline.pkl` — trained KNN + scaler +
  label encoder saved for Phase 4 comparison (gitignored)
- `notebooks/` directory — established with `results/` subdirectory for
  all notebook output artifacts

### Changed
- `requirements.txt` — added scikit-learn==1.8.0, matplotlib==3.10.8,
  pandas==3.0.2 under new Phase 3 section
- `.gitignore` — added `models/**/*.pkl` to exclude trained baseline artifacts

### Known results
- Top performers: OCWA F1=0.80, WCSP F1=0.63, HOFI F1=0.55
- Thin-data species (ANHU, BLPH, DOWO, MODO, YRUM): F1=0.00 — data
  limitation, not model failure

---

## [0.2.4] — Phase 2 Split Generation

### Added
- `src/data/splitter.py` — stratified 60/20/20 train/val/test split generator
  for both audio and visual modalities, `NABIRDS_CLASS_MAP` covering all 20
  SD species including plumage variants, deterministic via fixed seed
- `scripts/generate_splits.py` — CLI with `--audio-only`, `--visual-only`,
  `--train-ratio`, `--val-ratio` flags, reads all config from YAML
- `tests/data/test_splitter.py` — 32 synthetic tests, no real dataset files
  required (PR #12)

### Fixed
- `configs/thresholds.yaml` — added `splits` section (`train_ratio`,
  `val_ratio`, `random_seed`), fixed `audio_weight`/`visual_weight`
  indentation (were at root level, now correctly nested under `fusion`)

### Known issues
- CAVI (California Scrub-Jay) has no NABirds visual data — NABirds predates
  the 2016 AOU split of Western Scrub-Jay. Audio training unaffected.
  To be addressed in Phase 3.

---

## [0.2.3] — Phase 2 Species Expansion

### Changed
- `configs/species.yaml` — expanded from 15 to 20 SD region species (PR #11)
  - Removed 5 non-SD species: BCCH, NOCA, WTSP, CEDW, YWAR
  - Added 10 genuine SD backyard/feeder species: AMCR, SPTO, BLPH, HOSP,
    EUST, WCSP, HOORI, WBNU, OCWA, YRUM
  - Entries grouped by resident vs seasonal
  - Source: eBird SD frequency data + San Diego Field Ornithologists checklist

---

## [0.2.2] — Phase 2 Data Pipeline

### Added
- `src/data/downloader.py` — Xeno-canto API v3 pagination, quality filtering
  (A/B), idempotent download loop with metadata sidecar, NABirds structural
  verification utilities (PR #11)
- `scripts/download_datasets.py` — CLI with `--dry-run`, `--species`,
  `--max-per-species` flags, API key from `.env` (PR #11)
- `tests/data/test_downloader.py` — 37 synthetic tests, all network calls
  mocked, no internet required (PR #11)
- `docs/DATASETS.md` — dataset sources, licenses, manual NABirds setup
  steps, split schema
- `.env.example` — added `XENO_CANTO_API_KEY` (PR #10)

---

## [0.2.1] — Phase 2 Vision Preprocessing

### Added
- `src/vision/preprocess.py` — full image preprocessing pipeline:
  `load_image`, `resize`, `normalize`, `augment`, `preprocess_frame`,
  `preprocess_file` (PR #8)
- `tests/vision/test_preprocess.py` — 40 unit tests, fully synthetic,
  no hardware or real image files required (PR #8)
- Output is HWC float32 (224, 224, 3), ImageNet-normalized — CHW
  transpose deferred to classify.py in Phase 4

---

## [0.2.0] - Phase 2 Audio Preprocessing

### Added
- `src/audio/preprocess.py` — full WAV → mel spectrogram pipeline:
  `load_wav`, `normalize`, `to_mel_spectrogram`, `preprocess_file`,
  `preprocess_array` (PR #6)
- `tests/audio/test_preprocess.py` — 31 unit tests, fully synthetic,
  no hardware or real audio files required (PR #6)

### Fixed
- CI workflow now installs from `requirements.txt` so librosa and all
  runtime dependencies are available during test runs (PR #6)
- Removed unused `librosa.display` import from `preprocess.py` (PR #6)


---

## [0.1.1] - CI and Docs Cleanup

### Added
- `ROADMAP.md` — full 6-phase development plan with status tracking (PR #4)
- `docs/ONBOARDING.md` — contributor setup guide for new team members (PR #1)
- GitHub Actions CI workflow — lint + format + tests on every PR (PR #2)
- `.github/pull_request_template.md` — structured PR checklist (PR #2)

### Fixed
- Pydantic `model_` namespace warning in `ClassificationResult` — added
  `model_config = {"protected_namespaces": ()}` (PR #1)
- CI badge URL corrected in `README.md` (PR #3)
- README team table updated, Daniel handle placeholder noted (PR #3)

---

## [0.1.0] — Phase 1 Scaffold

### Added
- Initial repository scaffold (PR #1)
  - Full `src/` module structure with stubs and docstrings
  - `src/data/schema.py` — Pydantic models: `ClassificationResult`,
    `BirdObservation`, `Modality`
  - `configs/` YAML system — species, thresholds, paths, notify
  - `tests/` mirroring `src/` — 28 passing tests, 0 warnings
  - `docs/ARCHITECTURE.md` — system design and dependency rules
  - `docs/SETUP.md` — clone, install, run instructions
  - `docs/CONTRIBUTING.md` — branch naming, commit format, PR rules
  - `requirements.txt` and `requirements-dev.txt` — all versions pinned
  - `.env.example` — documented environment variables
  - `.gitignore` — excludes venv, datasets, model weights, logs
  - `pyproject.toml` — ruff and pytest configuration
  - Custom non-commercial license

---

<!--
TEMPLATE for new entries — add above [Unreleased] when starting a phase:

## [0.X.0] — Phase N Description

### Added
- New features or files

### Changed
- Changes to existing functionality

### Fixed
- Bug fixes

### Removed
- Removed features
-->