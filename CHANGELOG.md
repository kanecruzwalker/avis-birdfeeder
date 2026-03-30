# Changelog

All notable changes to Avis are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

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