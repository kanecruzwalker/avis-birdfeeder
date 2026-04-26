"""
Track 3 evaluation script — runs all 4 variants on both test sets.

Run from repo root:
    python notebooks/phase8_track3_evaluation.py

Inputs:
    models/visual/sklearn_pipeline_track3_v0.pkl
    models/visual/sklearn_pipeline_track3_v1.pkl
    models/visual/sklearn_pipeline_track3_v2.pkl
    models/visual/sklearn_pipeline_track3_v3.pkl
    notebooks/results/phase8/track3_retraining/track3_features_cache.npz

Outputs:
    notebooks/results/phase8/track3_retraining/
        confusion_v{0,1,2,3}_nabirds.png
        confusion_v{0,1,2,3}_deployment.png
        per_class_f1_v{0,1,2,3}.png
        comparison_table.csv
        winner.txt    ← which variant should ship to production
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

# Make src/ importable
sys.path.insert(0, str(Path(".").resolve()))


# ── Setup ───────────────────────────────────────────────────────────────

REPO_ROOT = Path(".").resolve()
MODELS_DIR = REPO_ROOT / "models" / "visual"
RESULTS_DIR = REPO_ROOT / "notebooks" / "results" / "phase8" / "track3_retraining"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = RESULTS_DIR / "track3_features_cache.npz"

# Decision rule: deployment macro F1 wins, but NABirds macro F1 must not
# drop by more than NABIRDS_TOLERANCE.
NABIRDS_TOLERANCE = 0.05


# ── Load cached features ────────────────────────────────────────────────

print("Loading cached features...")
if not CACHE_PATH.exists():
    print(f"ERROR: feature cache not found at {CACHE_PATH}")
    print("Run phase8_track3_training.py first.")
    sys.exit(1)

cache = np.load(CACHE_PATH, allow_pickle=True)
X_nabirds_test = cache["X_nabirds_test"]
y_nabirds_test = cache["y_nabirds_test"]
X_deploy_test = cache["X_deploy_test"]
y_deploy_test = cache["y_deploy_test"]
print(f"  NABirds test: {X_nabirds_test.shape}, classes: {sorted(set(y_nabirds_test))}")
print(f"  Deployment test: {X_deploy_test.shape}, classes: {sorted(set(y_deploy_test))}")


# ── Evaluate one variant ────────────────────────────────────────────────

def evaluate_variant(variant_name: str) -> dict:
    """Load a variant and evaluate it on both test sets. Return metrics dict."""
    print("\n" + "=" * 64)
    print(f"Evaluating variant {variant_name.upper()}")
    print("=" * 64)

    bundle_path = MODELS_DIR / f"sklearn_pipeline_track3_{variant_name}.pkl"
    if not bundle_path.exists():
        print(f"  WARNING: {bundle_path} not found. Skipping.")
        return None
    bundle = joblib.load(bundle_path)
    scaler = bundle["scaler"]
    clf = bundle["clf"]
    label_map = bundle["label_map"]
    label_to_idx = {v: k for k, v in label_map.items()}
    classes_in_model = sorted(label_map.values())
    print(f"  Loaded: {len(classes_in_model)} classes, C={bundle['best_c']}")

    metrics = {"variant": variant_name, "n_classes": len(classes_in_model)}

    # ── NABirds test eval ──
    # Filter to classes the model knows about (in case test has classes the model wasn't trained on)
    nb_mask = np.isin(y_nabirds_test, classes_in_model)
    if nb_mask.sum() > 0:
        X_nb = scaler.transform(X_nabirds_test[nb_mask])
        y_nb_true_idx = np.array([label_to_idx[s] for s in y_nabirds_test[nb_mask]])
        y_nb_pred_idx = clf.predict(X_nb)

        nb_macro_f1 = f1_score(y_nb_true_idx, y_nb_pred_idx, average="macro", zero_division=0)
        nb_accuracy = (y_nb_pred_idx == y_nb_true_idx).mean()
        nb_kappa = cohen_kappa_score(y_nb_true_idx, y_nb_pred_idx)
        nb_balanced = balanced_accuracy_score(y_nb_true_idx, y_nb_pred_idx)

        metrics["nabirds_macro_f1"] = nb_macro_f1
        metrics["nabirds_accuracy"] = nb_accuracy
        metrics["nabirds_kappa"] = nb_kappa
        metrics["nabirds_balanced_acc"] = nb_balanced
        metrics["nabirds_n_eval"] = int(nb_mask.sum())

        print(f"\n  NABirds test ({nb_mask.sum()}/{len(y_nabirds_test)} rows in model vocab):")
        print(f"    macro F1   = {nb_macro_f1:.4f}")
        print(f"    accuracy   = {nb_accuracy:.4f}")
        print(f"    kappa      = {nb_kappa:.4f}")
        print(f"    balanced   = {nb_balanced:.4f}")

        # Confusion matrix on NABirds test
        nb_classes_present = sorted(set(y_nb_true_idx))
        nb_class_names = [label_map[i] for i in nb_classes_present]
        cm = confusion_matrix(y_nb_true_idx, y_nb_pred_idx, labels=nb_classes_present)
        fig, ax = plt.subplots(figsize=(12, 10))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb_class_names).plot(
            ax=ax, colorbar=True, xticks_rotation=45)
        ax.set_title(f"Track 3 {variant_name.upper()} — NABirds Test Confusion Matrix")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"confusion_{variant_name}_nabirds.png", dpi=150)
        plt.close(fig)
    else:
        print("  No NABirds test rows in model vocabulary. Skipping NABirds eval.")
        metrics["nabirds_macro_f1"] = float("nan")

    # ── Deployment test eval ──
    dep_mask = np.isin(y_deploy_test, classes_in_model)
    if dep_mask.sum() > 0:
        X_dep = scaler.transform(X_deploy_test[dep_mask])
        y_dep_true_idx = np.array([label_to_idx[s] for s in y_deploy_test[dep_mask]])
        y_dep_pred_idx = clf.predict(X_dep)

        dep_macro_f1 = f1_score(y_dep_true_idx, y_dep_pred_idx, average="macro", zero_division=0)
        dep_accuracy = (y_dep_pred_idx == y_dep_true_idx).mean()
        dep_kappa = cohen_kappa_score(y_dep_true_idx, y_dep_pred_idx)
        dep_balanced = balanced_accuracy_score(y_dep_true_idx, y_dep_pred_idx)

        metrics["deploy_macro_f1"] = dep_macro_f1
        metrics["deploy_accuracy"] = dep_accuracy
        metrics["deploy_kappa"] = dep_kappa
        metrics["deploy_balanced_acc"] = dep_balanced
        metrics["deploy_n_eval"] = int(dep_mask.sum())

        print(f"\n  Deployment test ({dep_mask.sum()}/{len(y_deploy_test)} rows in model vocab):")
        print(f"    macro F1   = {dep_macro_f1:.4f}")
        print(f"    accuracy   = {dep_accuracy:.4f}")
        print(f"    kappa      = {dep_kappa:.4f}")
        print(f"    balanced   = {dep_balanced:.4f}")

        # Per-class F1 on deployment test
        dep_classes_present = sorted(set(y_dep_true_idx))
        dep_class_names = [label_map[i] for i in dep_classes_present]
        per_class_f1 = f1_score(y_dep_true_idx, y_dep_pred_idx, average=None,
                                 labels=dep_classes_present, zero_division=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(dep_class_names, per_class_f1)
        ax.axhline(dep_macro_f1, color="red", linestyle="--",
                  label=f"Macro F1 = {dep_macro_f1:.3f}")
        ax.set_xlabel("Species")
        ax.set_ylabel("F1 score")
        ax.set_title(f"Track 3 {variant_name.upper()} — Deployment Per-class F1")
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"per_class_f1_{variant_name}.png", dpi=150)
        plt.close(fig)

        # Confusion matrix on deployment test
        cm = confusion_matrix(y_dep_true_idx, y_dep_pred_idx, labels=dep_classes_present)
        fig, ax = plt.subplots(figsize=(12, 10))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dep_class_names).plot(
            ax=ax, colorbar=True, xticks_rotation=45)
        ax.set_title(f"Track 3 {variant_name.upper()} — Deployment Test Confusion Matrix")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"confusion_{variant_name}_deployment.png", dpi=150)
        plt.close(fig)

        # Detailed per-class report (saved as text)
        report = classification_report(
            y_dep_true_idx, y_dep_pred_idx,
            labels=dep_classes_present,
            target_names=dep_class_names,
            zero_division=0,
        )
        (RESULTS_DIR / f"classification_report_{variant_name}.txt").write_text(report)

        # Save the per-class F1 dictionary too
        per_class_dict = dict(zip(dep_class_names, per_class_f1))
        for cls, f1 in per_class_dict.items():
            metrics[f"deploy_f1_{cls}"] = f1
    else:
        print("  No deployment test rows in model vocabulary. Skipping deployment eval.")
        metrics["deploy_macro_f1"] = float("nan")

    return metrics


# ── Evaluate all variants ───────────────────────────────────────────────

all_metrics = []
for variant in ["v0", "v1", "v2", "v3"]:
    m = evaluate_variant(variant)
    if m:
        all_metrics.append(m)

# ── Comparison table ───────────────────────────────────────────────────

print("\n" + "=" * 64)
print("COMPARISON TABLE")
print("=" * 64)
df = pd.DataFrame(all_metrics)
df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

# Print key metrics
key_cols = ["variant", "n_classes",
            "nabirds_macro_f1", "nabirds_accuracy",
            "deploy_macro_f1", "deploy_accuracy"]
print(df[key_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# ── Decision rule ───────────────────────────────────────────────────────

print("\n" + "=" * 64)
print("DECISION RULE: best deploy_macro_f1 with nabirds_macro_f1 ≥ V0 - 0.05")
print("=" * 64)

v0_row = df[df["variant"] == "v0"].iloc[0] if (df["variant"] == "v0").any() else None
if v0_row is not None:
    v0_nabirds = v0_row["nabirds_macro_f1"]
    print(f"V0 NABirds macro F1: {v0_nabirds:.4f}")
    print(f"Tolerance: {NABIRDS_TOLERANCE}")
    print(f"Threshold: NABirds macro F1 ≥ {v0_nabirds - NABIRDS_TOLERANCE:.4f}")
else:
    v0_nabirds = None

# Filter to variants meeting NABirds constraint
qualified = df.copy()
if v0_nabirds is not None and not pd.isna(v0_nabirds):
    qualified = qualified[
        qualified["nabirds_macro_f1"] >= (v0_nabirds - NABIRDS_TOLERANCE)
    ]

if len(qualified) == 0:
    print("\nNO variants meet the NABirds tolerance threshold.")
    print("Manual review required — check whether the deployment data caused")
    print("catastrophic forgetting, or whether NABirds tolerance is too tight.")
    winner = None
else:
    # Best deployment macro F1 among qualified
    qualified = qualified.sort_values("deploy_macro_f1", ascending=False)
    winner_row = qualified.iloc[0]
    winner = winner_row["variant"]
    print(f"\nWINNER: {winner.upper()}")
    print(f"  NABirds macro F1: {winner_row['nabirds_macro_f1']:.4f}")
    print(f"  Deploy macro F1:  {winner_row['deploy_macro_f1']:.4f}")
    delta = winner_row["deploy_macro_f1"] - v0_row["deploy_macro_f1"] if v0_row is not None else float("nan")
    print(f"  Δ from V0:        {delta:+.4f}")

    # Write winner.txt
    (RESULTS_DIR / "winner.txt").write_text(f"{winner}\n")
    print(f"\n  Wrote {RESULTS_DIR / 'winner.txt'}")

print("\n" + "=" * 64)
print("Evaluation complete.")
print(f"Results in: {RESULTS_DIR}")
print("=" * 64)

if winner:
    print(f"\nNext step: copy sklearn_pipeline_track3_{winner}.pkl to "
          "sklearn_pipeline.pkl to deploy.")
    print("Then restart agent on Pi.")