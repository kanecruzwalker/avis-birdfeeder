"""
src/audio/capture.py

Records audio from the Fifine USB microphone in fixed-duration windows
and saves them to disk for BirdNET classification.

Hardware:
    Device:      Fifine USB (sounddevice index 1, hw:2,0)
    Sample rate: 48000 Hz native — no resampling needed
    Channels:    1 (mono)
    Window:      3 seconds (BirdNET training convention — do not change)

Pipeline:
    sounddevice.rec(window_samples)
        → RMS energy gate (skip silent windows — wind, empty feeder)
        → save WAV to data/captures/audio/{timestamp}.wav
        → return file path to agent

Why save to disk before classifying?
    BirdNET (via birdnetlib) operates on audio files, not numpy arrays.
    Saving the capture window before classification is therefore required.
    It also gives us the audio_path for BirdObservation at no extra cost —
    the saved clip is the observation record.

Why RMS energy gate?
    A feeder microphone records continuously — most windows contain only
    wind, ambient noise, or silence. Running BirdNET on every window would
    waste compute and produce spurious detections. The energy gate discards
    windows where mean RMS is below a configurable threshold, so BirdNET
    only runs when something is vocalizing.
    Threshold is tunable in configs/thresholds.yaml: audio.energy_threshold.

Config keys consumed:
    hardware.yaml:    microphone.device_index, microphone.sample_rate,
                      microphone.channels, microphone.window_seconds,
                      microphone.dtype
    thresholds.yaml:  audio.energy_threshold
    paths.yaml:       captures.audio

Dependencies:
    sounddevice  — cross-platform audio I/O (wraps PortAudio)
    scipy.io.wavfile — WAV writing (already in scipy, no extra install)
    numpy        — array operations
"""

from __future__ import annotations

import logging
import wave
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class AudioCapture:
    """
    Captures fixed-duration audio windows from the Fifine USB microphone.

    Each call to capture_window() blocks for window_seconds, then returns
    the path to the saved WAV file (or None if the window was below the
    energy threshold and should be skipped).

    Usage:
        capture = AudioCapture.from_config("configs/")
        audio_path = capture.capture_window()
        if audio_path is not None:
            result = classifier.predict(audio_path)
    """

    def __init__(
        self,
        device_index: int,
        sample_rate: int,
        channels: int,
        window_seconds: float,
        energy_threshold: float,
        output_dir: str | Path,
        dtype: str = "float32",
    ) -> None:
        """
        Args:
            device_index:     sounddevice device index for the Fifine USB mic.
                              Confirmed as index 1 on the deployed Pi.
            sample_rate:      Capture sample rate in Hz. Must be 48000 for BirdNET.
            channels:         Number of input channels. Fifine reports 1 (mono).
            window_seconds:   Duration of each capture window in seconds.
                              Must be 3.0 — BirdNET was trained on 3-second clips.
            energy_threshold: Minimum RMS energy to accept a window.
                              Windows below this are discarded (not saved, not classified).
                              Range [0.0, 1.0] on float32 audio.
                              Tune higher in noisy environments.
            output_dir:       Directory where WAV files are saved.
                              Corresponds to paths.yaml: captures.audio.
            dtype:            sounddevice capture dtype. float32 throughout the pipeline.
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_seconds = window_seconds
        self.energy_threshold = energy_threshold
        self.output_dir = Path(output_dir)
        self.dtype = dtype
        self._window_samples = int(sample_rate * window_seconds)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "AudioCapture initialized | device=%d sr=%d ch=%d window=%.1fs "
            "energy_threshold=%.4f output=%s",
            device_index,
            sample_rate,
            channels,
            window_seconds,
            energy_threshold,
            self.output_dir,
        )

    @classmethod
    def from_config(cls, config_dir: str | Path) -> AudioCapture:
        """
        Construct an AudioCapture from the configs/ directory.

        Reads hardware.yaml, thresholds.yaml, and paths.yaml.

        Args:
            config_dir: Path to the configs/ directory.

        Returns:
            Configured AudioCapture instance.
        """
        config_dir = Path(config_dir)

        with (config_dir / "hardware.yaml").open() as f:
            hw = yaml.safe_load(f)
        with (config_dir / "thresholds.yaml").open() as f:
            thr = yaml.safe_load(f)
        with (config_dir / "paths.yaml").open() as f:
            paths = yaml.safe_load(f)

        mic = hw["microphone"]
        output_dir = paths["captures"]["audio"]

        return cls(
            device_index=mic["device_index"],
            sample_rate=mic["sample_rate"],
            channels=mic["channels"],
            window_seconds=mic["window_seconds"],
            energy_threshold=thr["audio"]["energy_threshold"],
            output_dir=output_dir,
            dtype=mic.get("dtype", "float32"),
        )

    def capture_window(self) -> Path | None:
        """
        Record one audio window and save it to disk if above the energy threshold.

        Blocks for window_seconds while recording. This is by design — the
        agent loop calls capture_window() synchronously. Phase 5 runs audio
        and visual capture sequentially. A future phase may parallelize them
        using threading, but sequential capture is simpler and sufficient for
        the 1-second loop interval.

        Returns:
            Path to the saved WAV file, or None if the window was below the
            energy threshold (caller should skip audio classification this cycle).

        Raises:
            RuntimeError: If sounddevice fails to open the device.
                          Check that device_index is correct and mic is connected.
        """
        # Import here so sounddevice is only required on the Pi — laptop
        # tests can run without it by mocking this method.
        try:
            import sounddevice as sd  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is not installed. On Pi: pip install sounddevice. "
                "On laptop, mock AudioCapture.capture_window() in tests."
            ) from exc

        logger.debug(
            "Recording %.1fs audio window (device=%d, sr=%d)",
            self.window_seconds,
            self.device_index,
            self.sample_rate,
        )

        try:
            # sd.rec blocks until recording is complete.
            # Returns shape (window_samples, channels) — we squeeze to 1D mono.
            audio = sd.rec(
                frames=self._window_samples,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                device=self.device_index,
                blocking=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"sounddevice failed to record from device {self.device_index}: {exc}. "
                "Verify mic is connected and device index is correct."
            ) from exc

        # Squeeze (window_samples, 1) → (window_samples,) for mono
        audio_mono = audio.squeeze()

        # ── RMS energy gate ───────────────────────────────────────────────────
        rms = float(np.sqrt(np.mean(audio_mono**2)))
        if rms < self.energy_threshold:
            logger.debug(
                "Audio window below energy threshold (rms=%.4f < %.4f) — skipping",
                rms,
                self.energy_threshold,
            )
            return None

        logger.debug("Audio window accepted (rms=%.4f >= %.4f)", rms, self.energy_threshold)

        # ── Save to WAV ───────────────────────────────────────────────────────
        output_path = self._wav_path()
        self._save_wav(audio_mono, output_path)

        logger.info("Captured audio window → %s (rms=%.4f)", output_path.name, rms)
        return output_path

    def _wav_path(self) -> Path:
        """
        Generate a timestamped output path for the current capture window.

        Format: data/captures/audio/YYYYMMDD_HHMMSS_ffffff.wav
        Microsecond precision prevents collisions at high capture rates.

        Returns:
            Path object for the output WAV file. Parent directory is guaranteed
            to exist (created in __init__).
        """
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ts}.wav"
        return self.output_dir / filename

    def _save_wav(self, audio: np.ndarray, path: Path) -> None:
        """
        Save a mono float32 audio array to a WAV file.

        Uses Python's built-in wave module to avoid a scipy dependency on the Pi.
        float32 is written as 32-bit PCM — compatible with BirdNET and librosa.

        Args:
            audio: 1-D float32 numpy array of audio samples.
            path:  Output file path. Parent directory must exist.
        """
        # Convert float32 → int16 for standard WAV compatibility.
        # BirdNET and librosa both handle int16 WAV correctly.
        # Clip to [-1, 1] first to prevent int16 overflow on loud captures.
        audio_clipped = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_clipped * 32767).astype(np.int16)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)  # 2 bytes = int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

        logger.debug("Saved WAV: %s (%d samples)", path.name, len(audio_int16))
