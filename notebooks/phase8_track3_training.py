"""
Track 3 training script — produces V0/V1/V2/V3 LogReg pipeline bundles.

Run from repo root:
    python notebooks/phase8_track3_training.py

Outputs (per investigation doc):
    models/visual/sklearn_pipeline_track3_v0.pkl  (copy of existing baseline)
    models/visual/sklearn_pipeline_track3_v1.pkl  (NABirds + deployment, 19 classes)
    models/visual/sklearn_pipeline_track3_v2.pkl  (V1 + CALT)
    models/visual/sklearn_pipeline_track3_v3.pkl  (V2 + NONE + UNIDENTIFIED)
    notebooks/results/phase8/track3_retraining/track3_features_cache.npz

The features cache is reused across V0/V1/V2/V3 — feature extraction
runs once even though we train four heads.

This script can be copied verbatim into a Jupyter notebook by splitting
on the `# ── CELL N ──` markers.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Make src/ importable from notebooks/
sys.path.insert(0, str(Path(".").resolve()))

from src.vision.preprocess import preprocess_file as visual_preprocess_file


# ── CELL 1 — Setup ──────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

REPO_ROOT = Path(".").resolve()
SPLITS_DIR = REPO_ROOT / "data" / "splits"
MODELS_DIR = REPO_ROOT / "models" / "visual"
RESULTS_DIR = REPO_ROOT / "notebooks" / "results" / "phase8" / "track3_retraining"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = RESULTS_DIR / "track3_features_cache.npz"

# Load existing NABirds label map for V0 baseline copy
with open(REPO_ROOT / "models" / "visual_label_map.json") as f:
    nabirds_label_raw = json.load(f)
nabirds_label_map = {int(k): v for k, v in nabirds_label_raw.items()}
print(f"NABirds label map: {len(nabirds_label_map)} species")


# ── CELL 2 — Load splits ────────────────────────────────────────────────

print("\n─ Loading splits ─")
nabirds_train = pd.read_csv(SPLITS_DIR / "visual_train.csv")
nabirds_val = pd.read_csv(SPLITS_DIR / "visual_val.csv")
nabirds_test = pd.read_csv(SPLITS_DIR / "visual_test.csv")

deployment_train = pd.read_csv(SPLITS_DIR / "deployment_train.csv")
deployment_val = pd.read_csv(SPLITS_DIR / "deployment_val.csv")
deployment_test = pd.read_csv(SPLITS_DIR / "deployment_test.csv")

print(f"NABirds:    train={len(nabirds_train)}, val={len(nabirds_val)}, test={len(nabirds_test)}")
print(f"Deployment: train={len(deployment_train)}, val={len(deployment_val)}, test={len(deployment_test)}")

# Effective classes
deployment_train_classes = sorted(deployment_train["effective_class"].unique())
print(f"Deployment classes: {deployment_train_classes}")


# ── CELL 3 — Build frozen extractor ─────────────────────────────────────

print("\n─ Building frozen EfficientNet-B0 ─")
extractor = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=0,
    global_pool="avg",
)
extractor.eval()
extractor = extractor.to(DEVICE)
for p in extractor.parameters():
    p.requires_grad = False
print("Extractor ready (1280-dim output).")


# ── CELL 4 — Extract features (cached) ──────────────────────────────────

def extract_features(df: pd.DataFrame, desc: str) -> tuple[np.ndarray, np.ndarray]:
    """Run frozen extractor over all images in df, return (features, label_strings)."""
    features = []
    label_strs = []
    failed = 0
    n = len(df)

    for i, row in enumerate(df.itertuples(index=False)):
        try:
            frame = visual_preprocess_file(str(row.file_path))
            tensor = (
                torch.from_numpy(frame.astype(np.float32))
                .permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            )
            with torch.no_grad():
                feat = extractor(tensor).cpu().numpy().squeeze()
        except Exception:
            feat = np.zeros(1280, dtype=np.float32)
            failed += 1

        features.append(feat)
        # For NABirds rows, label is in `species_code`. For deployment rows, in `effective_class`.
        label_strs.append(getattr(row, "effective_class", row.species_code))

        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"  {desc}: {i+1}/{n} ({failed} failed)")

    return np.vstack(features), np.array(label_strs)


if CACHE_PATH.exists():
    print(f"\nLoading cached features from {CACHE_PATH}...")
    cache = np.load(CACHE_PATH, allow_pickle=True)
    X_nabirds_train = cache["X_nabirds_train"]
    y_nabirds_train = cache["y_nabirds_train"]
    X_nabirds_val = cache["X_nabirds_val"]
    y_nabirds_val = cache["y_nabirds_val"]
    X_nabirds_test = cache["X_nabirds_test"]
    y_nabirds_test = cache["y_nabirds_test"]
    X_deploy_train = cache["X_deploy_train"]
    y_deploy_train = cache["y_deploy_train"]
    X_deploy_val = cache["X_deploy_val"]
    y_deploy_val = cache["y_deploy_val"]
    X_deploy_test = cache["X_deploy_test"]
    y_deploy_test = cache["y_deploy_test"]
    print(f"  Loaded shapes: nabirds_train={X_nabirds_train.shape}, deploy_train={X_deploy_train.shape}")
else:
    print("\n─ Extracting features (will take 30-60 minutes total) ─")
    print("\nNABirds:")
    X_nabirds_train, y_nabirds_train = extract_features(nabirds_train, "nabirds_train")
    X_nabirds_val, y_nabirds_val = extract_features(nabirds_val, "nabirds_val")
    X_nabirds_test, y_nabirds_test = extract_features(nabirds_test, "nabirds_test")
    print("\nDeployment:")
    X_deploy_train, y_deploy_train = extract_features(deployment_train, "deploy_train")
    X_deploy_val, y_deploy_val = extract_features(deployment_val, "deploy_val")
    X_deploy_test, y_deploy_test = extract_features(deployment_test, "deploy_test")

    print(f"\nCaching features to {CACHE_PATH}...")
    np.savez_compressed(
        CACHE_PATH,
        X_nabirds_train=X_nabirds_train, y_nabirds_train=y_nabirds_train,
        X_nabirds_val=X_nabirds_val, y_nabirds_val=y_nabirds_val,
        X_nabirds_test=X_nabirds_test, y_nabirds_test=y_nabirds_test,
        X_deploy_train=X_deploy_train, y_deploy_train=y_deploy_train,
        X_deploy_val=X_deploy_val, y_deploy_val=y_deploy_val,
        X_deploy_test=X_deploy_test, y_deploy_test=y_deploy_test,
    )
    print("Cached.")


# ── CELL 5 — Train all 4 variants ───────────────────────────────────────

# Helper to fit + tune C on validation set
def train_and_tune(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    label_to_idx: dict, scaler: StandardScaler,
    use_balanced: bool = True,
) -> tuple[LogisticRegression, float]:
    """
    Train LogReg for each C, return (best_clf, best_C).
    Tunes on macro F1 on the val set.
    """
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    # Map string labels to int indices
    y_train_idx = np.array([label_to_idx[s] for s in y_train])
    y_val_idx = np.array([label_to_idx[s] for s in y_val])

    best_C = 1.0
    best_f1 = -1.0
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(
            C=C,
            class_weight="balanced" if use_balanced else None,
            solver="lbfgs",
            max_iter=5000,
            random_state=42,
        )
        clf.fit(X_train_s, y_train_idx)
        val_preds = clf.predict(X_val_s)
        val_f1 = f1_score(y_val_idx, val_preds, average="macro", zero_division=0)
        print(f"    C={C:7.2f}  val macro F1={val_f1:.4f}{' ★' if val_f1 > best_f1 else ''}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_C = C

    # Refit at best_C
    clf = LogisticRegression(
        C=best_C, class_weight="balanced" if use_balanced else None,
        solver="lbfgs", max_iter=5000, random_state=42,
    )
    clf.fit(X_train_s, y_train_idx)
    return clf, best_C, best_f1


def save_bundle(name: str, scaler: StandardScaler, clf: LogisticRegression,
                label_map: dict, n_classes: int, best_C: float, notes: str) -> Path:
    """Save sklearn pipeline bundle."""
    bundle = {
        "scaler": scaler,
        "clf": clf,
        "label_map": label_map,
        "n_classes": n_classes,
        "best_c": best_C,
        "track3_variant": name,
        "notes": notes,
    }
    out_path = MODELS_DIR / f"sklearn_pipeline_track3_{name}.pkl"
    joblib.dump(bundle, out_path)
    return out_path


print("\n" + "=" * 64)
print("Training all 4 variants")
print("=" * 64)


# ── V0: Baseline — copy existing pipeline ─────────────────
print("\n─ V0: Production baseline (copy existing) ─")
existing_pipeline = MODELS_DIR / "sklearn_pipeline.pkl"
v0_path = MODELS_DIR / "sklearn_pipeline_track3_v0.pkl"
if existing_pipeline.exists():
    shutil.copy2(existing_pipeline, v0_path)
    # Tag the copy with track3 metadata
    bundle = joblib.load(v0_path)
    bundle["track3_variant"] = "v0"
    bundle["notes"] = "Production baseline (NABirds-only LogReg, copied unchanged)."
    joblib.dump(bundle, v0_path)
    print(f"  Saved {v0_path}")
else:
    print(f"  WARNING: {existing_pipeline} not found. V0 must be built from NABirds first.")


# ── V1: NABirds + deployment (existing 19 species only) ─────────────────
print("\n─ V1: NABirds + deployment (no new classes) ─")
# Keep only deployment rows whose effective_class is in NABirds vocabulary
nabirds_vocab = set(nabirds_label_map.values())
mask_train = np.isin(y_deploy_train, list(nabirds_vocab))
mask_val = np.isin(y_deploy_val, list(nabirds_vocab))
print(f"  Deployment train: {mask_train.sum()}/{len(y_deploy_train)} rows in NABirds vocab")
print(f"  Deployment val:   {mask_val.sum()}/{len(y_deploy_val)} rows in NABirds vocab")

X_v1_train = np.vstack([X_nabirds_train, X_deploy_train[mask_train]])
y_v1_train = np.concatenate([y_nabirds_train, y_deploy_train[mask_train]])
X_v1_val = np.vstack([X_nabirds_val, X_deploy_val[mask_val]])
y_v1_val = np.concatenate([y_nabirds_val, y_deploy_val[mask_val]])

v1_classes = sorted(set(y_v1_train))
v1_label_to_idx = {code: i for i, code in enumerate(v1_classes)}
v1_label_map = {i: code for code, i in v1_label_to_idx.items()}

scaler_v1 = StandardScaler().fit(X_v1_train)
clf_v1, c_v1, f1_v1 = train_and_tune(X_v1_train, y_v1_train, X_v1_val, y_v1_val,
                                     v1_label_to_idx, scaler_v1)
save_bundle("v1", scaler_v1, clf_v1, v1_label_map, len(v1_classes), c_v1,
            f"NABirds+deployment, 19 classes, balanced, best C={c_v1}, val F1={f1_v1:.3f}")
print(f"  Saved with C={c_v1}, val F1={f1_v1:.3f}")


# ── V2: V1 + CALT ─────────────────
print("\n─ V2: V1 + CALT ─")
v2_vocab = nabirds_vocab | {"CALT"}
mask_train = np.isin(y_deploy_train, list(v2_vocab))
mask_val = np.isin(y_deploy_val, list(v2_vocab))
print(f"  Deployment train: {mask_train.sum()}/{len(y_deploy_train)} rows")
print(f"  Deployment val:   {mask_val.sum()}/{len(y_deploy_val)} rows")

X_v2_train = np.vstack([X_nabirds_train, X_deploy_train[mask_train]])
y_v2_train = np.concatenate([y_nabirds_train, y_deploy_train[mask_train]])
X_v2_val = np.vstack([X_nabirds_val, X_deploy_val[mask_val]])
y_v2_val = np.concatenate([y_nabirds_val, y_deploy_val[mask_val]])

v2_classes = sorted(set(y_v2_train))
v2_label_to_idx = {code: i for i, code in enumerate(v2_classes)}
v2_label_map = {i: code for code, i in v2_label_to_idx.items()}

scaler_v2 = StandardScaler().fit(X_v2_train)
clf_v2, c_v2, f1_v2 = train_and_tune(X_v2_train, y_v2_train, X_v2_val, y_v2_val,
                                     v2_label_to_idx, scaler_v2)
save_bundle("v2", scaler_v2, clf_v2, v2_label_map, len(v2_classes), c_v2,
            f"V1+CALT, {len(v2_classes)} classes, balanced, best C={c_v2}, val F1={f1_v2:.3f}")
print(f"  Saved with C={c_v2}, val F1={f1_v2:.3f}")


# ── V3: V2 + NONE + UNIDENTIFIED ─────────────────
print("\n─ V3: V2 + NONE + UNIDENTIFIED ─")
v3_vocab = nabirds_vocab | {"CALT", "NONE", "UNKNOWN"}
mask_train = np.isin(y_deploy_train, list(v3_vocab))
mask_val = np.isin(y_deploy_val, list(v3_vocab))
print(f"  Deployment train: {mask_train.sum()}/{len(y_deploy_train)} rows")
print(f"  Deployment val:   {mask_val.sum()}/{len(y_deploy_val)} rows")

X_v3_train = np.vstack([X_nabirds_train, X_deploy_train[mask_train]])
y_v3_train = np.concatenate([y_nabirds_train, y_deploy_train[mask_train]])
X_v3_val = np.vstack([X_nabirds_val, X_deploy_val[mask_val]])
y_v3_val = np.concatenate([y_nabirds_val, y_deploy_val[mask_val]])

v3_classes = sorted(set(y_v3_train))
v3_label_to_idx = {code: i for i, code in enumerate(v3_classes)}
v3_label_map = {i: code for code, i in v3_label_to_idx.items()}

scaler_v3 = StandardScaler().fit(X_v3_train)
clf_v3, c_v3, f1_v3 = train_and_tune(X_v3_train, y_v3_train, X_v3_val, y_v3_val,
                                     v3_label_to_idx, scaler_v3)
save_bundle("v3", scaler_v3, clf_v3, v3_label_map, len(v3_classes), c_v3,
            f"V2+NONE+UNKNOWN, {len(v3_classes)} classes, balanced, best C={c_v3}, val F1={f1_v3:.3f}")
print(f"  Saved with C={c_v3}, val F1={f1_v3:.3f}")


# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("Training complete — all four variants saved")
print("=" * 64)
print(f"  V0: {MODELS_DIR / 'sklearn_pipeline_track3_v0.pkl'}")
print(f"  V1: {MODELS_DIR / 'sklearn_pipeline_track3_v1.pkl'}  C={c_v1}  val_f1={f1_v1:.3f}")
print(f"  V2: {MODELS_DIR / 'sklearn_pipeline_track3_v2.pkl'}  C={c_v2}  val_f1={f1_v2:.3f}")
print(f"  V3: {MODELS_DIR / 'sklearn_pipeline_track3_v3.pkl'}  C={c_v3}  val_f1={f1_v3:.3f}")
print("\nNext: run phase8_track3_evaluation.py to evaluate on both test sets.")