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

Phase 6 Hailo integration:
    When hardware.yaml has hailo.enabled: true and the compiled .hef exists at
    hailo.models.visual_hef, HailoVisualExtractor is used for feature extraction
    instead of the CPU PyTorch path. The sklearn LogReg head always runs on CPU.
    Falls back to CPU silently if hailo_platform is unavailable or HEF is missing.
    The interface (input shape, output ClassificationResult) is identical for both
    paths — BirdAgent and ScoreFuser are unaware of which backend is active.
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

    The 1280-dim feature extractor runs on either CPU (PyTorch) or the Hailo
    HAILO8L NPU, depending on hardware.yaml configuration. The sklearn LogReg
    head always runs on CPU regardless of backend.

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
        hailo_hef_path: str | None = None,
        hailo_enabled: bool = False,
        device: str | None = None,
        shared_vdevice: object | None = None,
    ) -> None:
        """
        Args:
            extractor_path:    Path to frozen EfficientNet backbone checkpoint (.pt).
                               Saved by notebooks/visual_efficientnet.ipynb cell 28.
            sklearn_path:      Path to sklearn pipeline bundle (.pkl).
                               Contains scaler, clf, label_map, n_classes.
            species_list_path: Path to configs/species.yaml (code → names).
            hailo_hef_path:    Path to compiled .hef file for Hailo inference.
                               Read from hardware.yaml hailo.models.visual_hef.
                               If None or file absent, CPU path is used.
            hailo_enabled:     Whether to attempt Hailo inference. Controlled by
                               hardware.yaml hailo.enabled. Falls back to CPU if
                               hailo_platform unavailable or HEF missing.
            device:            Torch device string ('cpu', 'cuda'). Auto-detected if None.
                               Only used for the CPU inference path.
        """
        self.extractor_path = Path(extractor_path)
        self.sklearn_path = Path(sklearn_path)
        self.species_list_path = Path(species_list_path)
        self.hailo_hef_path = Path(hailo_hef_path) if hailo_hef_path else None
        self.hailo_enabled = hailo_enabled
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Lazy-loaded on first predict() call
        self._extractor: nn.Module | None = None
        self._scaler = None  # sklearn StandardScaler
        self._clf = None  # sklearn LogisticRegression
        self._label_map: dict[int, str] = {}  # int index → species_code
        self._species_meta: dict[str, dict] = {}  # code → {common_name, scientific_name}
        self._hailo_extractor = None  # HailoVisualExtractor, lazy-loaded
        self._shared_vdevice = shared_vdevice

        logger.info(
            "VisualClassifier initialized | extractor=%s sklearn=%s device=%s hailo=%s",
            self.extractor_path,
            self.sklearn_path,
            self.device,
            "enabled" if hailo_enabled else "disabled",
        )

    @classmethod
    def from_config(
        cls, config_path: str, shared_vdevice: object | None = None
    ) -> VisualClassifier:
        """
        Construct a VisualClassifier from configs/paths.yaml.

        Reads:
            models.visual_frozen_extractor → extractor_path
            models.visual_sklearn          → sklearn_path

        Also reads configs/hardware.yaml if present:
            hailo.enabled                  → hailo_enabled
            hailo.models.visual_hef        → hailo_hef_path

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

        # Read Hailo config from hardware.yaml if available
        hailo_enabled = False
        hailo_hef_path = None
        hw_path = config_path.parent / "hardware.yaml"
        if hw_path.exists():
            with hw_path.open() as f:
                hw = yaml.safe_load(f)
            hailo_cfg = hw.get("hailo", {})
            hailo_enabled = hailo_cfg.get("enabled", False)
            hailo_hef_path = hailo_cfg.get("models", {}).get("visual_hef")

        logger.info("VisualClassifier.from_config | paths.yaml=%s", config_path)
        return cls(
            extractor_path=extractor_path,
            sklearn_path=str(sklearn_path),
            species_list_path=str(species_list_path),
            hailo_hef_path=hailo_hef_path,
            hailo_enabled=hailo_enabled,
            shared_vdevice=shared_vdevice,
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

        # ── EfficientNet feature extractor (CPU path) ─────────────────────────
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

    def _load_hailo(self) -> bool:
        """
        Attempt to load HailoVisualExtractor for NPU-accelerated inference.

        Returns True if Hailo is ready, False if falling back to CPU.
        Falls back silently if hailo_platform unavailable or HEF missing —
        the caller continues with the CPU path without raising.
        """
        if not self.hailo_enabled or not self.hailo_hef_path:
            return False
        if not self.hailo_hef_path.exists():
            logger.warning(
                "Hailo enabled but HEF not found at %s — falling back to CPU.",
                self.hailo_hef_path,
            )
            return False
        try:
            from src.vision.hailo_extractor import HailoVisualExtractor  # noqa: PLC0415

            self._hailo_extractor = HailoVisualExtractor(
                str(self.hailo_hef_path),
                shared_vdevice=self._shared_vdevice,
            )
            self._hailo_extractor.open()
            logger.info("Hailo inference active — HEF loaded from %s", self.hailo_hef_path)
            return True
        except Exception as exc:
            logger.warning("Hailo load failed (%s) — falling back to CPU.", exc)
            self._hailo_extractor = None
            return False

    def predict(
        self,
        frame: np.ndarray,
        camera_index: int | None = None,
    ) -> ClassificationResult:
        """
        Run inference on a preprocessed frame and return the top species prediction.

        Feature extraction runs on Hailo NPU if enabled and available, otherwise
        falls back to CPU PyTorch. The sklearn LogReg head always runs on CPU.

        Pipeline (CPU path):
            frame (224, 224, 3) HWC float32
                → CHW transpose + batch dim → EfficientNet → 1280-dim feature
                → StandardScaler → LogisticRegression.predict_proba()
                → top species + confidence → ClassificationResult

        Pipeline (Hailo path):
            frame (224, 224, 3) HWC float32
                → uint8 conversion → HailoVisualExtractor.extract()
                → 1280-dim feature (dequantized float32)
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
            ValueError:   If frame has unexpected shape.
            RuntimeError: If artifacts have not been saved yet.
        """
        if self._extractor is None:
            self._load()

        if frame.ndim != 3 or frame.shape != (224, 224, 3):
            raise ValueError(
                f"Expected frame of shape (224, 224, 3), got {frame.shape}. "
                "Ensure preprocess_frame() was called before predict()."
            )

        # ── Feature extraction — Hailo fast path or CPU fallback ──────────────
        # Attempt Hailo on first call when enabled; subsequent calls reuse extractor
        if self._hailo_extractor is None and self.hailo_enabled:
            self._load_hailo()

        if self._hailo_extractor is not None and self._hailo_extractor.is_open:
            # Hailo path: convert float32 [0,1] normalized frame to uint8 [0,255]
            # HailoRT expects uint8 input — quantization handled internally by chip
            frame_uint8 = (frame * 255).clip(0, 255).astype(np.uint8)
            features = self._hailo_extractor.extract(frame_uint8)  # (1, 1280) float32
        else:
            # CPU path: HWC → CHW → batch tensor → EfficientNet forward pass
            tensor = (
                torch.from_numpy(frame.astype(np.float32))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                features = self._extractor(tensor).cpu().numpy()  # (1, 1280)

        # ── Sklearn classification (always CPU) ───────────────────────────────
        features_scaled = self._scaler.transform(features)  # (1, 1280)
        probas = self._clf.predict_proba(features_scaled)[0]  # (n_classes,)
        pred_idx = int(probas.argmax())
        confidence = float(probas[pred_idx])
        species_code = self._label_map[pred_idx]
        meta = self._species_meta.get(species_code, {})

        logger.debug(
            "Visual predict: %s (%.3f) camera=%s backend=%s",
            species_code,
            confidence,
            camera_index,
            "hailo" if (self._hailo_extractor and self._hailo_extractor.is_open) else "cpu",
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
