"""
src/vision/classify.py

Wraps a fine-tuned EfficientNet-B0 model to produce a ClassificationResult.

Architecture:
    - Backbone: EfficientNet-B0 pretrained on ImageNet (via timm).
    - Head: replaced linear layer fine-tuned on our 19 SD species subset
      of NABirds. Fine-tuning done in notebooks/visual_efficientnet.ipynb.
    - Saved checkpoint: models/visual/finetuned_sdbirds.pt

Input:  preprocessed frame — float32 array of shape (224, 224, 3), HWC,
        ImageNet-normalized, as returned by src.vision.preprocess.
Output: ClassificationResult (src.data.schema)

Design notes:
    - HWC → CHW transpose applied here (not in preprocess). Preprocess outputs
      (224, 224, 3); model expects (1, 3, 224, 224) after batch dim is added.
    - Model loaded lazily on first predict() call — __init__ stays fast.
    - label_map.json is shared with AudioClassifier: same file, same index
      convention. Visual classes may differ (19 vs 18 species) so the map
      contains the union; each classifier uses only its trained indices.
    - species_list_path always resolves to configs/species.yaml — single
      source of truth for common_name and scientific_name lookups.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import yaml

from src.data.schema import ClassificationResult, Modality

logger = logging.getLogger(__name__)


class VisualClassifier:
    """
    Wrapper around fine-tuned EfficientNet-B0 for SD bird species classification.

    The ImageNet pretrained backbone is loaded via timm, then the classification
    head is replaced and fine-tuned on NABirds SD species images.

    Usage:
        classifier = VisualClassifier.from_config("configs/paths.yaml")
        frame = FramePreprocessor().preprocess(raw_frame)
        result = classifier.predict(frame)
    """

    MODEL_VERSION = "efficientnet-b0-sdbirds-v1"

    def __init__(
        self,
        model_path: str,
        label_map_path: str,
        species_list_path: str,
        device: str | None = None,
    ) -> None:
        """
        Args:
            model_path: Path to fine-tuned checkpoint (.pt).
                        Saved by notebooks/visual_efficientnet.ipynb.
            label_map_path: Path to label_map.json (int index → species_code).
            species_list_path: Path to configs/species.yaml (code → names).
            device: Torch device string ('cpu', 'cuda'). Auto-detected if None.
        """
        self.model_path = Path(model_path)
        self.label_map_path = Path(label_map_path)
        self.species_list_path = Path(species_list_path)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: nn.Module | None = None
        self._label_map: dict[int, str] = {}  # int index → species_code
        self._species_meta: dict[str, dict] = {}  # code → {common_name, scientific_name}

        logger.info(
            "VisualClassifier initialized | model=%s device=%s",
            self.model_path,
            self.device,
        )

    @classmethod
    def from_config(cls, config_path: str) -> VisualClassifier:
        """
        Construct a VisualClassifier from configs/paths.yaml.

        Args:
            config_path: Path to configs/paths.yaml.

        Returns:
            Configured VisualClassifier instance (model not yet loaded).
        """
        config_path = Path(config_path)
        with config_path.open() as f:
            cfg = yaml.safe_load(f)

        model_path = cfg["models"]["visual_finetuned"]
        label_map_path = cfg["models"]["label_map"]
        species_list_path = config_path.parent / "species.yaml"

        logger.info("VisualClassifier.from_config | paths.yaml=%s", config_path)
        return cls(
            model_path=model_path,
            label_map_path=label_map_path,
            species_list_path=str(species_list_path),
        )

    def _load(self) -> None:
        """
        Lazily load model weights, label map, and species metadata.

        Called automatically on first predict(). Separated from __init__ so
        constructing a classifier in tests does not require weights on disk.
        """
        # Species metadata from species.yaml
        with self.species_list_path.open() as f:
            species_cfg = yaml.safe_load(f)
        self._species_meta = {
            s["code"]: {
                "common_name": s["common_name"],
                "scientific_name": s["scientific_name"],
            }
            for s in species_cfg["species"]
        }

        # Label map: {"0": "HOFI", "1": "MODO", ...}
        with self.label_map_path.open() as f:
            raw = json.load(f)
        self._label_map = {int(k): v for k, v in raw.items()}

        # EfficientNet-B0: load pretrained backbone, replace head
        n_classes = len(self._label_map)
        self._model = _build_efficientnet(n_classes=n_classes)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self.device)
        self._model.eval()

        logger.info(
            "VisualClassifier loaded | classes=%d device=%s weights=%s",
            n_classes,
            self.device,
            self.model_path,
        )

    def predict(self, frame: np.ndarray) -> ClassificationResult:
        """
        Run inference on a preprocessed frame and return the top species prediction.

        Args:
            frame: Float32 array of shape (224, 224, 3), HWC, ImageNet-normalized,
                   as returned by src.vision.preprocess.FramePreprocessor.

        Returns:
            ClassificationResult for the highest-confidence species prediction.

        Raises:
            ValueError: If frame has unexpected shape or dtype.
            RuntimeError: If model weights have not been trained yet.
        """
        if not self.model_path.exists():
            raise RuntimeError(
                f"Model weights not found at {self.model_path}. "
                "Run notebooks/visual_efficientnet.ipynb to train the model first."
            )

        if self._model is None:
            self._load()

        if frame.ndim != 3 or frame.shape != (224, 224, 3):
            raise ValueError(f"Expected frame of shape (224, 224, 3), got {frame.shape}")

        # HWC (224, 224, 3) → CHW (3, 224, 224) → batch (1, 3, 224, 224)
        tensor = (
            torch.from_numpy(frame.astype(np.float32))
            .permute(2, 0, 1)  # HWC → CHW
            .unsqueeze(0)  # add batch dim
            .to(self.device)
        )

        with torch.no_grad():
            logits = self._model(tensor)  # (1, n_classes)
            probs = torch.softmax(logits, dim=1)  # (1, n_classes)
            confidence, idx = probs.max(dim=1)

        species_code = self._label_map[idx.item()]
        meta = self._species_meta.get(species_code, {})

        return ClassificationResult(
            species_code=species_code,
            common_name=meta.get("common_name", species_code),
            scientific_name=meta.get("scientific_name", ""),
            confidence=float(confidence.item()),
            modality=Modality.VISUAL,
            model_version=self.MODEL_VERSION,
        )


def _build_efficientnet(n_classes: int) -> nn.Module:
    """
    Load EfficientNet-B0 pretrained on ImageNet and replace the classifier head.

    The backbone weights are frozen during early fine-tuning epochs, then
    unfrozen for full fine-tuning (see notebooks/visual_efficientnet.ipynb).

    This function is the single source of truth for the model architecture —
    called here at inference and in the notebook at training time.

    Args:
        n_classes: Number of output classes. Must match label_map size.

    Returns:
        EfficientNet-B0 with replaced classification head. Weights not loaded —
        caller loads via load_state_dict().
    """
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False,  # weights loaded from checkpoint, not downloaded here
        num_classes=n_classes,
    )
    return model
