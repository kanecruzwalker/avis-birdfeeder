"""
Classifier retraining experiment — do different hyperparameters fix the
deployment collapse, or is domain shift the root cause?

Strategy:
1. Extract EfficientNet features from NABirds training data (cached if possible)
2. Train 4 LogReg variants with different (C, class_weight) combinations
3. For each variant, measure:
   a. Held-out test set macro F1 (sanity check)
   b. Coefficient norm spread (class balance indicator)
4. Save all 4 models for subsequent deployment-data evaluation

Run from: avis-birdfeeder/ (repo root)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
import torch
import timm
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from src.vision.preprocess import preprocess_frame

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Load splits
splits_dir = Path("data/splits")
visual_train = pd.read_csv(splits_dir / "visual_train.csv")
visual_val = pd.read_csv(splits_dir / "visual_val.csv")
visual_test = pd.read_csv(splits_dir / "visual_test.csv")
print(f"Train: {len(visual_train)}, Val: {len(visual_val)}, Test: {len(visual_test)}")

# Build label map
all_codes = sorted(set(visual_train["species_code"]) | set(visual_val["species_code"]) | set(visual_test["species_code"]))
code_to_idx = {code: i for i, code in enumerate(all_codes)}
idx_to_code = {i: code for code, i in code_to_idx.items()}
n_classes = len(all_codes)
print(f"Classes: {n_classes} → {all_codes}")

# Load frozen extractor
print("Building frozen EfficientNet-B0...")
extractor = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg")
extractor.eval().to(DEVICE)
for p in extractor.parameters():
    p.requires_grad = False

def extract_features(df, desc):
    features, labels = [], []
    n = len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        try:
            # PIL load, preprocess_frame handles sizing
            from PIL import Image
            img = np.array(Image.open(row.file_path).convert("RGB"))
            frame = preprocess_frame(img, width=224, height=224)
            tensor = torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = extractor(tensor).cpu().numpy().squeeze()
        except Exception as e:
            feat = np.zeros(1280, dtype=np.float32)
        features.append(feat)
        labels.append(code_to_idx[row.species_code])
        if (i+1) % 200 == 0 or (i+1) == n:
            print(f"  {desc}: {i+1}/{n}")
    return np.array(features), np.array(labels)

print("\nExtracting features (this will take a while)...")
X_train, y_train = extract_features(visual_train, "train")
X_val, y_val = extract_features(visual_val, "val")
X_test, y_test = extract_features(visual_test, "test")

# Fit scaler
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Train 4 variants
variants = {
    "baseline_C0.1_None":      dict(C=0.1, class_weight=None),
    "A_C10_balanced":          dict(C=10.0, class_weight="balanced"),
    "B_C1_balanced":           dict(C=1.0, class_weight="balanced"),
    "C_C10_None":              dict(C=10.0, class_weight=None),
}

results = {}
for name, params in variants.items():
    print(f"\n{'='*60}\nTraining variant: {name}")
    print(f"Params: {params}")
    clf = LogisticRegression(
        C=params["C"],
        class_weight=params["class_weight"],
        solver="lbfgs",
        max_iter=5000,
        random_state=42,
    )
    clf.fit(X_train_s, y_train)
    
    # Metrics
    train_preds = clf.predict(X_train_s)
    val_preds = clf.predict(X_val_s)
    test_preds = clf.predict(X_test_s)
    train_f1 = f1_score(y_train, train_preds, average="macro", zero_division=0)
    val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    
    # Coefficient balance
    norms = np.linalg.norm(clf.coef_, axis=1)
    
    print(f"  Train macro F1: {train_f1:.3f}")
    print(f"  Val macro F1:   {val_f1:.3f}")
    print(f"  Test macro F1:  {test_f1:.3f}")
    print(f"  Coef norms: min={norms.min():.3f} max={norms.max():.3f} ratio={norms.max()/norms.min():.2f}x")
    print(f"  Converged: {clf.n_iter_}")
    
    # Save as bundle
    bundle = {
        "scaler": scaler,
        "clf": clf,
        "label_map": idx_to_code,
        "n_classes": n_classes,
        "best_c": params["C"],
        "hyperparams_experiment": name,
    }
    out_path = Path(f"models/visual/sklearn_pipeline_{name}.pkl")
    joblib.dump(bundle, out_path)
    print(f"  Saved: {out_path}")
    
    results[name] = {
        "train_f1": train_f1, "val_f1": val_f1, "test_f1": test_f1,
        "coef_ratio": norms.max() / norms.min(),
    }

# Summary
print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
for name, r in results.items():
    print(f"{name:30s} val_f1={r['val_f1']:.3f} test_f1={r['test_f1']:.3f} coef_ratio={r['coef_ratio']:.2f}x")