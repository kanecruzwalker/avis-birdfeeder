# Changelog

All notable changes to Avis are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

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