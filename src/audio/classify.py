"""
src/audio/classify.py

Wraps BirdNET pretrained inference to produce a ClassificationResult.

Architecture (Phase 5):
    - Model: BirdNET-Analyzer global model (birdnetlib 0.9.0, ~6000 species).
    - Interface: accepts a WAV file path. BirdNET reads the file directly —
      this is intentional. The capture module saves each 3-second window to
      data/captures/audio/ before classification, so the file naturally exists
      on disk. The saved path also populates BirdObservation.audio_path.
    - Species matching: BirdNET returns scientific names. We resolve these to
      our 4-letter AOU codes via configs/species.yaml, exactly as the notebook
      does. Detections for species outside our 20 SD species list are discarded.
    - min_conf: 0.1 — same threshold used in Phase 4 evaluation. BirdNET's
      internal confidence is separate from our agent's fused confidence threshold.

Why BirdNET instead of our CNN?
    Phase 4 evaluation: CNN from scratch macro F1=0.089 (below KNN baseline).
    BirdNET pretrained macro F1=0.776 (4x KNN baseline), zero fine-tuning.
    The CNN is retained in this file for reproducibility (training notebook
    references _build_audio_cnn) but is not used for inference.

Why a file path interface instead of a numpy array?
    birdnetlib.Recording operates on audio files. Converting an array to a
    temp file inside predict() would add disk I/O with no benefit — we already
    want the capture saved to disk for BirdObservation.audio_path. Passing the
    path through is cleaner and avoids any temp file lifecycle management.

Input:  path to a 3-second WAV file at 48kHz mono, as saved by
        src.audio.capture.AudioCapture.capture_window().
Output: ClassificationResult (src.data.schema)

Design notes:
    - Analyzer is loaded lazily on first predict() call (BirdNET model load
      takes ~2s — keeping __init__ fast matters for test startup time).
    - If BirdNET returns no detections or no SD-species match, predict()
      returns the best available result with its raw confidence, or raises
      NoBirdDetectedError if confidence is below min_conf entirely.
    - _build_audio_cnn() is preserved as-is from Phase 4 — training notebook
      imports it by reference and must remain importable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch.nn as nn
import yaml
from birdnetlib import Recording  # type: ignore[import]

from src.data.schema import ClassificationResult, Modality

logger = logging.getLogger(__name__)


class NoBirdDetectedError(Exception):
    """
    Raised by AudioClassifier.predict() when BirdNET finds no detections
    above min_conf in the provided audio clip.

    The agent handles this exception as a soft failure — the cycle continues
    with visual classification only (graceful single-modality degradation).
    """


class AudioClassifier:
    """
    Wrapper around BirdNET pretrained inference for SD bird species classification.

    BirdNET-Analyzer (via birdnetlib) is a global model trained on ~6000 species
    from Xeno-canto. No fine-tuning is needed — pretrained features transfer
    directly to our 20 San Diego species subset.

    Usage:
        classifier = AudioClassifier.from_config("configs/paths.yaml")
        audio_path = capture.capture_window()   # saves 3s WAV, returns path
        result = classifier.predict(audio_path)
    """

    MODEL_VERSION = "birdnet-global-6k-v2.4-birdnetlib-0.9.0"

    def __init__(
        self,
        species_list_path: str,
        min_conf: float = 0.1,
    ) -> None:
        """
        Args:
            species_list_path: Path to configs/species.yaml.
                               Used to build the scientific_name → species_code
                               lookup for matching BirdNET detections to our species.
            min_conf:          Minimum BirdNET confidence to accept a detection.
                               0.1 matches the threshold used in Phase 4 evaluation.
                               Lower = more detections, more false positives.
        """
        self.species_list_path = Path(species_list_path)
        self.min_conf = min_conf

        # Lazy-loaded on first predict() call
        self._analyzer = None  # birdnetlib.analyzer.Analyzer instance
        self._sci_to_code: dict[str, str] = {}  # scientific_name → species_code
        self._species_meta: dict[str, dict] = {}  # code → {common_name, scientific_name}

        # CNN path kept for backward compatibility (training notebook imports this)
        self._model: nn.Module | None = None

        logger.info(
            "AudioClassifier initialized | species=%s min_conf=%.2f",
            self.species_list_path,
            self.min_conf,
        )

    @classmethod
    def from_config(cls, config_path: str) -> AudioClassifier:
        """
        Construct an AudioClassifier from configs/paths.yaml.

        BirdNET loads its own weights from the birdnetlib package installation —
        no model path is needed in paths.yaml for the audio classifier.
        Species list path is resolved relative to the config directory.

        Args:
            config_path: Path to configs/paths.yaml.

        Returns:
            Configured AudioClassifier instance (BirdNET not yet loaded).
        """
        config_path = Path(config_path)
        species_list_path = config_path.parent / "species.yaml"

        logger.info("AudioClassifier.from_config | paths.yaml=%s", config_path)
        return cls(species_list_path=str(species_list_path))

    def _load(self) -> None:
        """
        Lazily load BirdNET Analyzer and build species lookup tables.

        BirdNET model load takes approximately 2 seconds. Deferring to first
        predict() call keeps from_config() and __init__ fast, which matters
        for test startup time and agent initialization.

        Called automatically on first predict().
        """
        # Import here so birdnetlib is only required when actually running
        # audio classification — not during schema imports, test collection, etc.
        from birdnetlib.analyzer import Analyzer  # type: ignore[import]

        logger.info("Loading BirdNET Analyzer (first call — ~2s)...")
        self._analyzer = Analyzer()
        logger.info("BirdNET Analyzer loaded.")

        # Build species lookup tables from configs/species.yaml
        with self.species_list_path.open() as f:
            species_cfg = yaml.safe_load(f)

        for s in species_cfg["species"]:
            code = s["code"]
            sci = s["scientific_name"]
            self._sci_to_code[sci] = code
            self._species_meta[code] = {
                "common_name": s["common_name"],
                "scientific_name": sci,
            }

        logger.info(
            "AudioClassifier species lookup built | %d species",
            len(self._sci_to_code),
        )

    def predict(self, audio_path: str | Path) -> ClassificationResult:
        """
        Run BirdNET inference on a WAV file and return the top SD species prediction.

        BirdNET analyzes the full file and returns detections sorted by time segment.
        We take the highest-confidence detection that matches one of our 20 SD species.
        If no SD species is detected above min_conf, NoBirdDetectedError is raised.

        Args:
            audio_path: Path to a WAV file. Expected: 3 seconds, 48kHz, mono.
                        As saved by src.audio.capture.AudioCapture.capture_window().

        Returns:
            ClassificationResult for the highest-confidence SD species detection.

        Raises:
            FileNotFoundError:    If audio_path does not exist.
            NoBirdDetectedError:  If BirdNET finds no SD species above min_conf.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self._analyzer is None:
            self._load()

        # Run BirdNET inference

        try:
            recording = Recording(
                self._analyzer,
                str(audio_path),
                min_conf=self.min_conf,
            )
            recording.analyze()
            detections = recording.detections
        except Exception as exc:
            logger.exception("BirdNET inference failed on %s: %s", audio_path, exc)
            raise NoBirdDetectedError(f"BirdNET inference failed on {audio_path}: {exc}") from exc

        if not detections:
            logger.debug(
                "BirdNET: no detections above min_conf=%.2f in %s", self.min_conf, audio_path
            )
            raise NoBirdDetectedError(
                f"No detections above min_conf={self.min_conf} in {audio_path.name}"
            )

        # Filter to SD species and pick highest confidence
        # BirdNET returns scientific names — resolve to our AOU codes
        sd_detections = []
        for d in detections:
            sci = d.get("scientific_name", "")
            code = self._sci_to_code.get(sci)
            if code is not None:
                sd_detections.append((code, float(d["confidence"]), sci))

        if not sd_detections:
            # BirdNET found birds but none are in our SD species list
            top = max(detections, key=lambda d: d["confidence"])
            logger.debug(
                "BirdNET: no SD species match — top detection was '%s' (%.3f)",
                top.get("scientific_name", "unknown"),
                top.get("confidence", 0.0),
            )
            raise NoBirdDetectedError(
                f"BirdNET detected species outside SD list in {audio_path.name}. "
                f"Top: {top.get('scientific_name', 'unknown')} ({top.get('confidence', 0.0):.3f})"
            )

        # Winner: highest confidence among SD species matches
        best_code, best_conf, best_sci = max(sd_detections, key=lambda x: x[1])
        meta = self._species_meta.get(best_code, {})

        logger.debug(
            "Audio predict: %s (%.3f) from %s",
            best_code,
            best_conf,
            audio_path.name,
        )

        return ClassificationResult(
            species_code=best_code,
            common_name=meta.get("common_name", best_code),
            scientific_name=meta.get("scientific_name", best_sci),
            confidence=best_conf,
            modality=Modality.AUDIO,
            model_version=self.MODEL_VERSION,
        )


# ── CNN from scratch — Phase 4 training artifact ──────────────────────────────
# Retained for reproducibility. notebooks/audio_birdnet.ipynb imports
# _build_audio_cnn() to train and evaluate the CNN baseline.
# NOT used for inference in Phase 5 — BirdNET is the production audio model.


def _build_audio_cnn(n_classes: int) -> nn.Module:
    """
    Build the lightweight CNN used for mel spectrogram classification.

    Phase 4 training-only. Not used for inference in Phase 5.
    Macro F1=0.089 on test set — substantially below BirdNET (F1=0.776).
    Retained so notebooks/audio_birdnet.ipynb remains fully reproducible.

    Architecture: three conv blocks + global average pooling + linear head.
    Input: (1, n_mels, time_frames) — channel dim added by training loop.

    Args:
        n_classes: Number of output classes. Must match audio label_map size.

    Returns:
        Untrained nn.Module. Caller loads weights via load_state_dict().
    """
    return nn.Sequential(
        # Block 1: (1, n_mels, T) → (32, n_mels/2, T/2)
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 2: (32, n_mels/2, T/2) → (64, n_mels/4, T/4)
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 3: (64, n_mels/4, T/4) → (128, n_mels/8, T/8)
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Global average pool → (128,)
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(128, n_classes),
        # No softmax — CrossEntropyLoss expects raw logits during training.
    )
