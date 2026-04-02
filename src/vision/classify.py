"""
src/vision/classify.py

Wraps the frozen EfficientNet-B0 feature extractor + sklearn LogisticRegression
pipeline to produce a ClassificationResult.

Architecture (Phase 5):
    - Feature extractor: EfficientNet-B0 pretrained on ImageNet (via timm),
      num_classes=0, global_pool="avg" — outputs 1280-dim feature vectors.
      Backbone weights are frozen — no gradient computation at inference.
    - Classifier head: StandardScaler + LogisticRegression trained on
      extracted features from our 19 SD species NABirds subset.
      Trained in notebooks/visual_efficientnet.ipynb (Section 10-11).

Why this architecture instead of end-to-end fine-tuning?
    Phase 4 evaluation showed frozen EfficientNet + LogReg achieves
    macro F1=0.931 vs fine-tuned EfficientNet macro F1=0.097 on our
    19-species SD dataset. The pretrained ImageNet features transfer
    directly to bird species — fine-tuning on our limited data overfits.

Artifacts loaded by _load() (saved by notebooks/visual_efficientnet.ipynb cell 28):
    models/visual/frozen_extractor.pt   — EfficientNet backbone state dict
    models/visual/sklearn_pipeline.pkl  — {"scaler", "clf", "label_map", "n_classes"}

Input:  preprocessed frame — float32 array of shape (224, 224, 3), HWC,
        ImageNet-normalized, as returned by src.vision.preprocess.preprocess_frame().
Output: ClassificationResult (src.data.schema)

Design notes:
    - Both artifacts loaded lazily on first predict() call — __init__ stays fast.
    - HWC → CHW transpose applied here (not in preprocess).
    - camera_index is passed through to ClassificationResult so the agent and
      fuser know which camera produced this result.
    - _build_efficientnet() kept as the single source of truth for the extractor
      architecture — matches the notebook exactly.

Phase 6 note:
    When the Hailo .hef compilation succeeds, a HailoVisualClassifier subclass
    will override predict() to run the compiled model via hailort bindings.
    The interface (input shape, output ClassificationResult) stays identical.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import timm
import torch
import torch.nn as nn
import yaml

from src.data.schema import ClassificationResult, Modality

logger = logging.getLogger(__name__)


class VisualClassifier:
    """
    Wrapper around frozen EfficientNet-B0 + LogisticRegression for SD bird
    species classification.

    The 1280-dim feature extractor is frozen at ImageNet pretrained weights.
    A StandardScaler + LogisticRegression head trained on NABirds SD species
    images provides the final classification.

    Usage:
        classifier = VisualClassifier.from_config("configs/paths.yaml")
        frame = preprocess_frame(raw_frame)           # (224, 224, 3) float32
        result = classifier.predict(frame, camera_index=0)
    """

    MODEL_VERSION = "frozen-efficientnet-b0-logreg-sdbirds-v1"

    def __init__(
        self,
        extractor_path: str,
        sklearn_path: str,
        species_list_path: str,
        device: str | None = None,
    ) -> None:
        """
        Args:
            extractor_path:    Path to frozen EfficientNet backbone checkpoint (.pt).
                               Saved by notebooks/visual_efficientnet.ipynb cell 28.
            sklearn_path:      Path to sklearn pipeline bundle (.pkl).
                               Contains scaler, clf, label_map, n_classes.
            species_list_path: Path to configs/species.yaml (code → names).
            device:            Torch device string ('cpu', 'cuda'). Auto-detected if None.
                               Note: feature extraction only — sklearn inference is CPU-only.
        """
        self.extractor_path = Path(extractor_path)
        self.sklearn_path = Path(sklearn_path)
        self.species_list_path = Path(species_list_path)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Lazy-loaded on first predict() call
        self._extractor: nn.Module | None = None
        self._scaler = None  # sklearn StandardScaler
        self._clf = None  # sklearn LogisticRegression
        self._label_map: dict[int, str] = {}  # int index → species_code
        self._species_meta: dict[str, dict] = {}  # code → {common_name, scientific_name}

        logger.info(
            "VisualClassifier initialized | extractor=%s sklearn=%s device=%s",
            self.extractor_path,
            self.sklearn_path,
            self.device,
        )

    @classmethod
    def from_config(cls, config_path: str) -> VisualClassifier:
        """
        Construct a VisualClassifier from configs/paths.yaml.

        Reads:
            models.visual_frozen_extractor → extractor_path
            models.visual_sklearn          → sklearn_path

        Args:
            config_path: Path to configs/paths.yaml.

        Returns:
            Configured VisualClassifier instance (artifacts not yet loaded).
        """
        config_path = Path(config_path)
        with config_path.open() as f:
            cfg = yaml.safe_load(f)

        extractor_path = cfg["models"]["visual_frozen_extractor"]
        sklearn_path = cfg["models"]["visual_sklearn"]
        species_list_path = config_path.parent / "species.yaml"

        logger.info("VisualClassifier.from_config | paths.yaml=%s", config_path)
        return cls(
            extractor_path=extractor_path,
            sklearn_path=str(sklearn_path),
            species_list_path=str(species_list_path),
        )

    def _load(self) -> None:
        """
        Lazily load both artifacts and species metadata.

        Called automatically on first predict(). Separated from __init__ so
        that constructing a VisualClassifier in tests does not require artifacts
        on disk — only predict() requires them.

        Raises:
            RuntimeError: If either artifact file is missing.
        """
        if not self.extractor_path.exists():
            raise RuntimeError(
                f"Frozen extractor not found at {self.extractor_path}. "
                "Run notebooks/visual_efficientnet.ipynb cell 28 to save artifacts."
            )
        if not self.sklearn_path.exists():
            raise RuntimeError(
                f"Sklearn pipeline not found at {self.sklearn_path}. "
                "Run notebooks/visual_efficientnet.ipynb cell 28 to save artifacts."
            )

        # ── Species metadata ──────────────────────────────────────────────────
        with self.species_list_path.open() as f:
            species_cfg = yaml.safe_load(f)
        self._species_meta = {
            s["code"]: {
                "common_name": s["common_name"],
                "scientific_name": s["scientific_name"],
            }
            for s in species_cfg["species"]
        }

        # ── Sklearn pipeline bundle ───────────────────────────────────────────
        bundle = joblib.load(self.sklearn_path)
        self._scaler = bundle["scaler"]
        self._clf = bundle["clf"]
        self._label_map = {int(k): v for k, v in bundle["label_map"].items()}
        n_classes = bundle["n_classes"]

        # ── EfficientNet feature extractor ────────────────────────────────────
        checkpoint = torch.load(self.extractor_path, map_location=self.device)
        self._extractor = _build_efficientnet()
        self._extractor.load_state_dict(checkpoint["model_state_dict"])
        self._extractor.to(self.device)
        self._extractor.eval()

        # Freeze all parameters — no gradients needed at inference
        for param in self._extractor.parameters():
            param.requires_grad = False

        logger.info(
            "VisualClassifier loaded | classes=%d device=%s extractor=%s",
            n_classes,
            self.device,
            self.extractor_path,
        )

    def predict(
        self,
        frame: np.ndarray,
        camera_index: int | None = None,
    ) -> ClassificationResult:
        """
        Run inference on a preprocessed frame and return the top species prediction.

        Pipeline:
            frame (224, 224, 3) HWC float32
                → CHW transpose + batch dim → EfficientNet → 1280-dim feature
                → StandardScaler → LogisticRegression.predict_proba()
                → top species + confidence → ClassificationResult

        Args:
            frame:        Float32 array of shape (224, 224, 3), HWC,
                          ImageNet-normalized, as returned by preprocess_frame().
            camera_index: Which camera captured this frame (0=primary, 1=secondary).
                          Passed through to ClassificationResult for fusion tracking.

        Returns:
            ClassificationResult for the highest-confidence species prediction.

        Raises:
            ValueError:  If frame has unexpected shape.
            RuntimeError: If artifacts have not been saved yet.
        """
        if self._extractor is None:
            self._load()

        if frame.ndim != 3 or frame.shape != (224, 224, 3):
            raise ValueError(
                f"Expected frame of shape (224, 224, 3), got {frame.shape}. "
                "Ensure preprocess_frame() was called before predict()."
            )

        # ── Feature extraction ────────────────────────────────────────────────
        # HWC (224, 224, 3) → CHW (3, 224, 224) → batch (1, 3, 224, 224)
        tensor = (
            torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        with torch.no_grad():
            # extractor outputs (1, 1280) — global average pooled features
            features = self._extractor(tensor).cpu().numpy()  # shape (1, 1280)

        # ── Sklearn classification ────────────────────────────────────────────
        features_scaled = self._scaler.transform(features)  # (1, 1280)
        probas = self._clf.predict_proba(features_scaled)[0]  # (n_classes,)
        pred_idx = int(probas.argmax())
        confidence = float(probas[pred_idx])
        species_code = self._label_map[pred_idx]
        meta = self._species_meta.get(species_code, {})

        logger.debug(
            "Visual predict: %s (%.3f) camera=%s",
            species_code,
            confidence,
            camera_index,
        )

        return ClassificationResult(
            species_code=species_code,
            common_name=meta.get("common_name", species_code),
            scientific_name=meta.get("scientific_name", ""),
            confidence=confidence,
            modality=Modality.VISUAL,
            model_version=self.MODEL_VERSION,
            camera_index=camera_index,
        )


def _build_efficientnet() -> nn.Module:
    """
    Build the frozen EfficientNet-B0 feature extractor.

    Configuration:
        num_classes=0      — removes the classification head, outputs features
        global_pool="avg"  — global average pooling → 1280-dim output vector
        pretrained=False   — weights loaded from checkpoint, not downloaded

    This is the single source of truth for the extractor architecture.
    Must match the extractor built in notebooks/visual_efficientnet.ipynb
    cell 23 exactly — mismatches cause load_state_dict() to fail with
    unexpected key errors.

    Returns:
        EfficientNet-B0 feature extractor. Caller loads weights via
        load_state_dict() and calls .eval() before inference.
    """
    return timm.create_model(
        "efficientnet_b0",
        pretrained=False,  # weights loaded from checkpoint
        num_classes=0,  # no classification head — outputs 1280-dim features
        global_pool="avg",  # global average pool before output
    )
