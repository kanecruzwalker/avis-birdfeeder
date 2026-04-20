"""
src/audio/capture.py

Records audio from the Fifine USB microphone in fixed-duration windows
and saves them to disk for BirdNET classification.

Hardware:
    Device:      Fifine USB (resolved by name: "USB PnP Audio Device")
    Sample rate: 48000 Hz native — no resampling needed
    Channels:    1 (mono)
    Window:      3 seconds (BirdNET training convention — do not change)

Device resolution (new in PR-B):
    sounddevice enumerates audio devices in an order that is not stable
    across reboots or USB replug events. On the deployed Pi, the Fifine
    has been observed at index 0 and index 1 in different sessions.
    Hardcoding a device_index in hardware.yaml led to silent audio failure
    for hours when the Pi rebooted and the Fifine shifted slots.

    We now resolve the device by name substring match. A distinctive
    fragment of the device name (e.g. "USB PnP Audio Device") is matched
    against sd.query_devices() output at first capture. The resulting
    index is cached for subsequent captures.

    Fallback chain:
        1. If device_name is set and matches a device → use that index
        2. If device_name is set but no match → log warning, use device_index
        3. If device_name is None → use device_index directly
        4. If nothing resolves to a valid input device → RuntimeError

    This scheme tolerates:
        - USB replug / reboot shuffling indices
        - Different hardware (Dan's Pi, dev laptops)
        - Adding new audio devices (future USB webcams with built-in mics)

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
    hardware.yaml:    microphone.device_name (optional, preferred),
                      microphone.device_index (fallback),
                      microphone.sample_rate, microphone.channels,
                      microphone.window_seconds, microphone.dtype
    thresholds.yaml:  audio.energy_threshold
    paths.yaml:       captures.audio

Dependencies:
    sounddevice  — cross-platform audio I/O (wraps PortAudio)
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

    Device resolution is lazy — sounddevice is only imported and queried
    on the first capture_window() call. This keeps tests and dev environments
    free of a hard sounddevice dependency.

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
        device_name: str | None = None,
    ) -> None:
        """
        Args:
            device_index:     sounddevice device index to use as a fallback if
                              device_name is not set or cannot be matched.
                              Held as the committed default for the deployed
                              Pi; a misconfigured value is recoverable via
                              device_name lookup.
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
            device_name:      Optional substring to match against sounddevice device
                              names. When set, the first matching input device is used
                              regardless of device_index. Preferred over device_index
                              for robustness against USB enumeration changes.
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_seconds = window_seconds
        self.energy_threshold = energy_threshold
        self.output_dir = Path(output_dir)
        self.dtype = dtype
        self.device_name = device_name
        self._window_samples = int(sample_rate * window_seconds)

        # Resolved index is set lazily on first capture_window() call so
        # sounddevice import doesn't happen at construction time.
        self._resolved_index: int | None = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "AudioCapture initialized | device_name=%s device_index=%d sr=%d ch=%d "
            "window=%.1fs energy_threshold=%.4f output=%s",
            repr(device_name) if device_name else "(none)",
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
            device_name=mic.get("device_name"),
        )

    def _resolve_device_index(self) -> int:
        """
        Resolve which sounddevice index to use for capture.

        Called lazily on first capture_window(). Result is cached in
        self._resolved_index so subsequent captures reuse the same device
        without re-querying sounddevice.

        Resolution strategy:
            1. If self.device_name is set, scan sounddevice.query_devices()
               for an input device whose name contains device_name as a
               substring. Use the first match.
            2. If no name match (either device_name unset or no device matches),
               fall back to self.device_index.
            3. Validate the resulting index points to a device with at least
               one input channel. If not, raise RuntimeError with a listing
               of available devices for diagnosis.

        Returns:
            The resolved sounddevice index, guaranteed to be a valid input device.

        Raises:
            RuntimeError: If no valid input device can be resolved.
        """
        try:
            import sounddevice as sd  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is not installed. On Pi: pip install sounddevice. "
                "On laptop, mock AudioCapture.capture_window() in tests."
            ) from exc

        devices = sd.query_devices()

        # Try name-based lookup first
        if self.device_name:
            for i, dev in enumerate(devices):
                name = dev.get("name", "")
                inputs = dev.get("max_input_channels", 0)
                if self.device_name in name and inputs > 0:
                    logger.info(
                        "Audio device resolved by name: %r → index %d (%r)",
                        self.device_name,
                        i,
                        name,
                    )
                    return i

            # Name was set but didn't match — log a warning and fall through
            available = [
                f"{i}={d.get('name', '?')!r}(inputs={d.get('max_input_channels', 0)})"
                for i, d in enumerate(devices)
            ]
            logger.warning(
                "Audio device name %r not found. Available: [%s]. "
                "Falling back to device_index=%d.",
                self.device_name,
                ", ".join(available),
                self.device_index,
            )
        else:
            logger.info(
                "No device_name configured — using device_index=%d directly.",
                self.device_index,
            )

        # Validate the fallback index points to a real input device
        if self.device_index < 0 or self.device_index >= len(devices):
            available = [
                f"{i}={d.get('name', '?')!r}(inputs={d.get('max_input_channels', 0)})"
                for i, d in enumerate(devices)
            ]
            raise RuntimeError(
                f"Audio device_index={self.device_index} is out of range. "
                f"Available devices: [{', '.join(available)}]. "
                f"Set microphone.device_name in hardware.yaml for robust resolution."
            )

        fallback_dev = devices[self.device_index]
        if fallback_dev.get("max_input_channels", 0) < 1:
            available = [
                f"{i}={d.get('name', '?')!r}(inputs={d.get('max_input_channels', 0)})"
                for i, d in enumerate(devices)
            ]
            raise RuntimeError(
                f"Audio device_index={self.device_index} "
                f"({fallback_dev.get('name', '?')!r}) has no input channels. "
                f"Available input devices: [{', '.join(available)}]. "
                f"Set microphone.device_name in hardware.yaml for robust resolution."
            )

        logger.info(
            "Audio device fallback to index %d (%r)",
            self.device_index,
            fallback_dev.get("name", "?"),
        )
        return self.device_index

    def capture_window(self) -> Path | None:
        """
        Record one audio window and save it to disk if above the energy threshold.

        Blocks for window_seconds while recording. This is by design — the
        agent loop calls capture_window() synchronously. Phase 5 runs audio
        and visual capture sequentially. A future phase may parallelize them
        using threading, but sequential capture is simpler and sufficient for
        the 1-second loop interval.

        Device index is resolved on first call and cached.

        Returns:
            Path to the saved WAV file, or None if the window was below the
            energy threshold (caller should skip audio classification this cycle).

        Raises:
            RuntimeError: If sounddevice fails to open the device, or if no
                          valid input device can be resolved by name or index.
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

        # Resolve device index lazily on first call
        if self._resolved_index is None:
            self._resolved_index = self._resolve_device_index()

        logger.debug(
            "Recording %.1fs audio window (device=%d, sr=%d)",
            self.window_seconds,
            self._resolved_index,
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
                device=self._resolved_index,
                blocking=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"sounddevice failed to record from device {self._resolved_index}: {exc}. "
                "Verify mic is connected and device_name/device_index are correct."
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
