# Avis — Contributor Quickstart Guide

> Get from zero to running the agent and making your first contribution.
> Estimated time: 20-30 minutes.

---

## What You're Setting Up

Avis is an agentic AI bird identification system running on a Raspberry Pi 5.
This guide gets your **laptop dev environment** running so you can write code,
run tests, and contribute via pull requests. Pi hardware is not required for
development — all tests run on synthetic data.

---

## Step 1 — Prerequisites

Install these before anything else:

| Tool | Version | Install |
|------|---------|---------|
| Python | **3.11.9** | `winget install Python.Python.3.11` (Windows) / `brew install python@3.11` (Mac) |
| Git | Any recent | https://git-scm.com/downloads |
| VS Code | Any recent | https://code.visualstudio.com |

Verify Python:
```bash
python --version   # must show 3.11.x
# Windows:
py -3.11 --version
```

> ⚠️ Use Python 3.11 on your dev machine. The Pi uses 3.13 with a special
> two-venv setup — that's only relevant for Pi deployment, not development.

---

## Step 2 — Clone and Install
```bash
# Clone the repo
git clone https://github.com/kanecruzwalker/avis-birdfeeder.git
cd avis-birdfeeder

# Create virtual environment — use Python 3.11 explicitly
py -3.11 -m venv .venv          # Windows
python3.11 -m venv .venv        # Mac/Linux

# Activate
.venv\Scripts\activate          # Windows PowerShell
source .venv/bin/activate       # Mac/Linux

# You should see (.venv) in your prompt
python --version                # confirm 3.11.x

# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## Step 3 — Verify Everything Works
```bash
# Run all tests — should show 331 passed, 0 warnings
python -m pytest tests/ -v

# Lint check — should show "All checks passed!"
ruff check src/ tests/

# Format check
ruff format --check src/ tests/
```

If all three pass you're fully set up.

---

## Step 4 — Environment File
```bash
cp .env.example .env
```

Leave values blank for now — credentials are only needed for Pi deployment
and dataset downloads. Never commit `.env` — it's in `.gitignore`.

---

## Step 5 — Understand the Project Structure
```
avis-birdfeeder/
├── src/
│   ├── agent/      # BirdAgent — the main agentic loop
│   ├── audio/      # Mic capture → WAV → BirdNET classification
│   ├── vision/     # Camera capture → EfficientNet classification
│   ├── fusion/     # Combines audio + visual confidence scores
│   ├── notify/     # Push notifications, logging
│   └── data/       # Shared Pydantic schemas — read this first
├── configs/        # All tunable values (thresholds, paths, species)
├── scripts/        # One-off utilities (dataset download, inference)
├── notebooks/      # Training experiments and evaluation
├── tests/          # Mirrors src/ — one test file per source file
└── docs/           # Architecture, setup, contributing guides
```

**Start here:** Read `src/data/schema.py` — it defines `ClassificationResult`
and `BirdObservation`, the data shapes that flow through the entire system.

**System flow:**
```
Mic → AudioCapture → AudioClassifier (BirdNET) ──┐
                                                   ├→ ScoreFuser → Notifier
Cameras → VisionCapture → VisualClassifier ────────┘
```

---

## Step 6 — Key Commands at a Glance

| What | Command |
|------|---------|
| Run all tests | `python -m pytest tests/ -v` |
| Run one module's tests | `python -m pytest tests/audio/ -v` |
| Run tests by name | `python -m pytest tests/ -v -k "schema"` |
| Lint check | `ruff check src/ tests/` |
| Auto-fix lint | `ruff check src/ tests/ --fix` |
| Format code | `ruff format src/ tests/` |
| Run the agent (Pi only) | `python -m src.agent.bird_agent` |
| Download audio datasets | `python scripts/download_datasets.py` |
| Generate train/val/test splits | `python scripts/generate_splits.py` |
| Generate label maps | `python scripts/generate_label_map.py` |
| Launch Jupyter | `jupyter notebook` |

---

## Step 7 — Git Workflow (Every Contribution)
```bash
# 1. Always start from an up-to-date main
git checkout main
git pull origin main

# 2. Create a branch for your work
git checkout -b feature/your-feature-name

# 3. Write code in src/, write tests in tests/
# 4. Run tests frequently as you go
python -m pytest tests/ -v

# 5. Before committing — format and lint
ruff format src/ tests/
ruff check src/ tests/

# 6. Commit with a meaningful message
git add src/ tests/
git commit -m "feat(audio): add mel spectrogram caching"

# 7. Push and open a PR on GitHub
git push origin feature/your-feature-name
```

**Never push directly to `main`.** Every change goes through a PR.

---

## Step 8 — Branch Naming
```
feature/short-description    # new functionality
fix/short-description        # bug fix
docs/short-description       # documentation only
refactor/short-description   # cleanup, no behavior change
test/short-description       # adding or fixing tests
```

---

## Step 9 — Commit Message Format
```
type(scope): short description

Types:  feat, fix, docs, test, refactor, chore
Scopes: agent, audio, vision, fusion, notify, data, ci, config
```

Examples:
```
feat(audio): add confidence score normalization
fix(vision): handle missing frame gracefully in capture loop
test(fusion): add tests for single-modality fallback
docs(setup): add Pi deployment instructions
chore(deps): pin scipy to 1.11.0
```

---

## Step 10 — Pull Request Rules

1. PR must pass CI (lint + format + tests) before merging
2. Fill out the PR template — describe what changed and why
3. Self-review your diff before requesting review
4. One logical change per PR — don't bundle unrelated changes
5. Every new function in `src/` needs at least one test in `tests/`

---

## Step 11 — Rules That Matter

- **No magic numbers in src/** — every threshold and path lives in `configs/`
- **Tests that need the Pi** get marked `@pytest.mark.hardware` — CI skips them
- **Tests that need model weights** get marked `@pytest.mark.requires_model`
- **Run ruff before every commit** — CI enforces both format and lint
- **Never commit `.env`** — credentials stay local

---

## Step 12 — Good First Contributions

If you're looking for somewhere to start:

- **Run an experiment** — add a row to `notebooks/results/experiments.csv`
  by trying a different fusion weight (audio=0.6, visual=0.4) and measuring
  F1 on the validation set. Document results in the notebook.
- **Add a species** — follow the instructions in `docs/DATASETS.md` to add
  a new San Diego species to `configs/species.yaml`, download recordings,
  regenerate splits, and retrain.
- **Tune a threshold** — the agent fires at fused confidence >= 0.70.
  Plot precision/recall curves on the validation set for thresholds 0.60–0.85
  and find the optimal value. Document in a notebook.
- **Improve a test** — find a function in `src/` with only one test and add
  edge case coverage. Check coverage with `python -m pytest --cov=src tests/`.
- **Write capture_test_frame tests** — `scripts/capture_test_frame.py` has
  no tests yet. Add `tests/scripts/test_capture_test_frame.py`.

---

## If Something Is Broken

1. Make sure venv is active — check for `(.venv)` in your prompt
2. Run `pip install -r requirements-dev.txt` — a dep may have been added
3. Run `git pull origin main` — you may be behind
4. Check the Actions tab on GitHub — if CI is red, main may have a known issue

---

## Further Reading

| Doc | What it covers |
|-----|---------------|
| `docs/ARCHITECTURE.md` | Full system design and dependency rules |
| `docs/SETUP.md` | Pi deployment, dataset download, notebook setup |
| `docs/CONTRIBUTING.md` | Detailed PR and commit conventions |
| `docs/DATASETS.md` | Dataset sources, licenses, split schema |
| `CHANGELOG.md` | Full history of what was built in each phase |
| `ROADMAP.md` | Phase plan, what's done, what's next |