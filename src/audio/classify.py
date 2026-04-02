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
        Build species lookup tables from configs/species.yaml.

        On the Pi, BirdNET inference runs in a Python 3.11 subprocess
        (scripts/audio_inference.py) because tflite_runtime has no Python
        3.13 wheel. _load() no longer instantiates the Analyzer directly —
        it only builds the lookup tables needed to validate subprocess output.
        """
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
        Run BirdNET inference on a WAV file via Python 3.11 subprocess.

        Spawns scripts/audio_inference.py under the pyenv 3.11 interpreter
        which has tflite_runtime installed. Result is returned as JSON on
        stdout and parsed back into a ClassificationResult.

        Args:
            audio_path: Path to a WAV file. Expected: 3 seconds, 48kHz, mono.

        Returns:
            ClassificationResult for the highest-confidence SD species detection.

        Raises:
            FileNotFoundError:   If audio_path does not exist.
            NoBirdDetectedError: If BirdNET finds no SD species above min_conf.
        """
        import json
        import subprocess

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not self._sci_to_code:
            self._load()

        # Resolve paths
        project_root = Path(__file__).resolve().parent.parent.parent
        inference_script = project_root / "scripts" / "audio_inference.py"
        python_311 = Path("/home/birdfeeder01/.pyenv/versions/3.11.9/bin/python")

        # Fall back to direct birdnetlib call if 3.11 not available (dev/CI)
        if not python_311.exists():
            return self._predict_direct(audio_path)

        try:
            result = subprocess.run(
                [
                    str(python_311),
                    str(inference_script),
                    str(audio_path),
                    str(self.species_list_path),
                    str(self.min_conf),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Parse JSON from stdout — ignore stderr (numpy warnings etc.)
            output = result.stdout.strip()
            # Find the last line that looks like JSON
            for line in reversed(output.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    data = json.loads(line)
                    break
            else:
                raise NoBirdDetectedError(f"audio_inference.py produced no JSON output: {output}")

        except subprocess.TimeoutExpired as exc:
            raise NoBirdDetectedError("BirdNET inference subprocess timed out") from exc
        except Exception as exc:
            raise NoBirdDetectedError(f"BirdNET subprocess failed: {exc}") from exc

        if data.get("error"):
            if data["error"] == "NO_BIRD_DETECTED":
                raise NoBirdDetectedError(f"No SD species detected above min_conf={self.min_conf}")
            raise NoBirdDetectedError(f"BirdNET error: {data['error']}")

        code = data["species_code"]

        logger.debug(
            "Audio predict: %s (%.3f) from %s",
            code,
            data["confidence"],
            audio_path.name,
        )

        return ClassificationResult(
            species_code=code,
            common_name=data.get("common_name", code),
            scientific_name=data.get("scientific_name", ""),
            confidence=data["confidence"],
            modality=Modality.AUDIO,
            model_version=self.MODEL_VERSION,
        )

    def _predict_direct(self, audio_path: Path) -> ClassificationResult:
        """
        Direct birdnetlib call — used on dev machines (not Pi) where
        tflite_runtime or tensorflow is available in the main venv.
        """
        from birdnetlib.analyzer import Analyzer  # type: ignore[import]

        if self._analyzer is None:
            logger.info("Loading BirdNET Analyzer (first call — ~2s)...")
            self._analyzer = Analyzer()
            logger.info("BirdNET Analyzer loaded.")

        try:
            recording = Recording(
                self._analyzer,
                str(audio_path),
                min_conf=self.min_conf,
            )
            recording.analyze()
            detections = recording.detections
        except Exception as exc:
            raise NoBirdDetectedError(f"BirdNET inference failed on {audio_path}: {exc}") from exc

        if not detections:
            raise NoBirdDetectedError(
                f"No detections above min_conf={self.min_conf} in {audio_path.name}"
            )

        sd_detections = []
        for d in detections:
            sci = d.get("scientific_name", "")
            code = self._sci_to_code.get(sci)
            if code is not None:
                sd_detections.append((code, float(d["confidence"]), sci))

        if not sd_detections:
            raise NoBirdDetectedError(
                f"BirdNET detected species outside SD list in {audio_path.name}"
            )

        best_code, best_conf, best_sci = max(sd_detections, key=lambda x: x[1])
        meta = self._species_meta.get(best_code, {})

        return ClassificationResult(
            species_code=best_code,
            common_name=meta.get("common_name", best_code),
            scientific_name=meta.get("scientific_name", best_sci),
            confidence=best_conf,
            modality=Modality.AUDIO,
            model_version=self.MODEL_VERSION,
        )
