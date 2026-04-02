# Setup Guide

## Prerequisites

- Python 3.11 or higher
- Git
- (On Pi) Raspberry Pi OS Bookworm 64-bit, Hailo drivers installed

## Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/avis-birdfeeder.git
cd avis-birdfeeder

# Create virtual environment
python -m venv .venv

# Activate (Mac/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install runtime dependencies
pip install -r requirements.txt

# Install dev dependencies (linting, testing)
pip install -r requirements-dev.txt
```

## Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in values. See `.env.example` for all required variables with descriptions.

## Running the Agent

```bash
python -m src.agent.bird_agent
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Running the Linter

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```

## Raspberry Pi — Hailo Setup

Before running inference on the Pi, ensure the Hailo runtime is installed:

```bash
# Follow the official Hailo RPi5 setup guide:
# https://github.com/hailo-ai/hailo-rpi5-examples
pip install hailort  # version must match your HailoRT runtime
```

Verify the Hailo HAT+ is detected:

```bash
hailortcli fw-control identify
```

## Dataset Download

Datasets are not committed to the repo. Use the download script after setup:

```bash
python scripts/download_datasets.py
```

See `docs/DATASETS.md` for full dataset documentation.



## Jupyter Notebook Setup

Notebooks require the project venv to be registered as a Jupyter kernel.
Run this once after completing the install steps above:

```bash
# Activate venv first, then:
pip install ipykernel
python -m ipykernel install --user --name=avis-venv --display-name "Avis (venv)"
```

Launch Jupyter from the project root:

```bash
jupyter notebook
```

Open any notebook from the browser UI. Before running, confirm the kernel shows
**Avis (venv)** in the top right. If it shows a different kernel, go to
**Kernel → Change Kernel → Avis (venv)** — otherwise imports will fail.

Run a full notebook cleanly with **Kernel → Restart & Run All**.

See `notebooks/README.md` for full notebook documentation.

## Generate Splits

After downloading datasets, generate the train/val/test split manifests:

```bash
python scripts/generate_splits.py
```

This writes CSV files to `data/splits/`. Re-run if the species list changes.
Do not change `random_seed` in `configs/thresholds.yaml` after splits are
first generated — it would invalidate all trained models.


## Generate Label Maps

After generating splits, generate the label map JSON files used by the classifiers:
```bash
python scripts/generate_label_map.py
```

This writes `models/label_map.json`, `models/audio_label_map.json`, and
`models/visual_label_map.json`. These files are committed to the repo —
you only need to run this if you change the species list or retrain splits.
```

**requirements.txt** — small typo, missing space before `#HOG`:
```
scikit-image==0.24.0       # HOG (Histogram of Oriented Gradients)...


## Generate Phase 5 Model Artifacts

After downloading datasets and generating splits, train the Phase 5 inference models:

**Audio (BirdNET):**
BirdNET loads pretrained weights from the `birdnetlib` package automatically.
No additional download or training step required.

**Visual (frozen EfficientNet + LogReg):**
Run `notebooks/visual_efficientnet.ipynb` (all cells, including cell 28 — Save artifacts).
This generates:
- `models/visual/frozen_extractor.pt` — EfficientNet-B0 feature extractor backbone
- `models/visual/sklearn_pipeline.pkl` — StandardScaler + LogisticRegression pipeline

These files are gitignored — run the notebook after cloning on any new machine.