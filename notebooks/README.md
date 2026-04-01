# Avis — Notebooks

This directory contains Jupyter notebooks for baseline training, pretrained model
fine-tuning, and evaluation. Notebooks are the exploratory and training environment —
all production code lives in `src/`.

---

## Structure

```
notebooks/
├── audio_baseline.ipynb          # Phase 3 — KNN on MFCC features
├── visual_baseline.ipynb         # Phase 3 — SVM on HOG + color histogram
├── audio_birdnet.ipynb           # Phase 4 — BirdNET fine-tuning (planned)
├── visual_efficientnet.ipynb     # Phase 4 — EfficientNet-B0 fine-tuning (planned)
└── results/
    ├── experiments.csv           # Running experiment log — one row per training run
    ├── audio_baseline_*.png      # Frozen Phase 3 audio baseline plots
    └── visual_baseline_*.png     # Frozen Phase 3 visual baseline plots
```

---

## Setup — Jupyter Kernel

Notebooks must run with the project's virtual environment, not the system Python.
Run this once after setting up your venv:

```bash
# Activate venv first
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac/Linux

# Register the venv as a Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=avis-venv --display-name "Avis (venv)"

# Launch Jupyter
jupyter notebook
```

In the Jupyter UI: **Kernel → Change Kernel → Avis (venv)** before running any notebook.
If the kernel is wrong, imports will fail with `ModuleNotFoundError`.

---

## Running a Notebook

1. Activate venv and launch Jupyter: `jupyter notebook`
2. Open the notebook from the browser UI
3. Confirm kernel is **Avis (venv)** (top right of notebook)
4. **Kernel → Restart & Run All** to run from clean state

Each notebook is self-contained — it reads configs from `configs/`, loads split
CSVs from `data/splits/`, and saves outputs to `notebooks/results/`.

---

## Experiment Log — `results/experiments.csv`

Every notebook appends one row to `notebooks/results/experiments.csv` when it
completes. This file is the single source of truth for all training runs across
all phases — do not edit it manually.

### Schema

| Column | Description | Example |
|--------|-------------|---------|
| `phase` | Phase number | `3` |
| `notebook` | Notebook filename | `audio_baseline.ipynb` |
| `modality` | `audio` or `visual` | `audio` |
| `model` | Model name | `KNN` |
| `features` | Feature extraction description | `MFCC mean+std (n_mfcc=40, sr=48000)` |
| `best_params` | Best hyperparameters from val set | `k=3, metric=euclidean` |
| `test_accuracy` | Overall test accuracy | `0.3023` |
| `macro_f1` | Macro-averaged F1 (unweighted) | `0.1913` |
| `weighted_f1` | Weighted F1 (by class size) | `0.2925` |
| `n_train` | Training samples used | `334` |
| `n_test` | Test samples evaluated | `86` |
| `species_count` | Number of species in test set | `18` |
| `timestamp` | When the run completed | `2026-03-31 05:19` |
| `notes` | Free-text notes | `Thin-data species: [...]` |

### Current runs

| Phase | Notebook | Model | Test Acc | Macro F1 |
|-------|----------|-------|----------|----------|
| 3 | audio_baseline.ipynb | KNN | 0.3023 | 0.1913 |
| 3 | visual_baseline.ipynb | SVM | 0.2128 | 0.1208 |

Phase 4 targets: audio macro F1 > 0.1913, visual macro F1 > 0.1208.

---

## Frozen Outputs

Each notebook saves plots to `notebooks/results/` when it runs. These are committed
to the repo as a frozen visual record of that training run. Do not delete or overwrite
existing PNGs — they are the audit trail.

| File pattern | Contents |
|-------------|----------|
| `audio_baseline_k_selection.png` | Val accuracy vs k for KNN |
| `audio_baseline_confusion_matrix.png` | Per-class confusion matrix |
| `audio_baseline_f1_per_class.png` | Per-class F1 bar chart |
| `visual_baseline_c_selection.png` | Val accuracy vs C for SVM |
| `visual_baseline_confusion_matrix.png` | Per-class confusion matrix |
| `visual_baseline_f1_per_class.png` | Per-class F1 bar chart |

---

## Saved Models

Each notebook saves a `.pkl` artifact to `models/baselines/`:

| File | Contents |
|------|----------|
| `models/baselines/audio_knn_baseline.pkl` | Trained KNN + scaler + label encoder + metrics |
| `models/baselines/visual_svm_baseline.pkl` | Trained SVM + scaler + label encoder + metrics |

These files are gitignored — regenerate by re-running the notebook.
Load in Phase 4 for direct comparison:

```python
import pickle
with open("models/baselines/audio_knn_baseline.pkl", "rb") as f:
    baseline = pickle.load(f)
print(f"KNN baseline macro F1: {baseline['macro_f1']:.4f}")
```

---

## Adding a New Notebook

When adding a Phase 4 or later notebook:

1. Follow the naming convention: `{modality}_{model}.ipynb`
   e.g. `audio_birdnet.ipynb`, `visual_efficientnet.ipynb`
2. Read `SAMPLE_RATE` and other params from `configs/thresholds.yaml` — never hardcode
3. Read split paths from `configs/paths.yaml`
4. Save all plots to `RESULTS_DIR = PROJECT_ROOT / "notebooks" / "results"`
5. Append a row to `experiments.csv` in the final cell (copy the pattern from
   `audio_baseline.ipynb` cell 12)
6. Run **Kernel → Restart & Run All** before committing — outputs must be current
