"""
tests/audio/test_capture.py

Unit tests for AudioCapture.

AudioCapture records audio windows from a USB microphone and writes them
to disk for BirdNET classification. PR-B introduces device resolution by
name substring (microphone.device_name in hardware.yaml) with fallback
to microphone.device_index, which is what most of these tests exercise.

Strategy:
    - sounddevice is never imported during tests — patch("...capture.sd", ...)
      would require the import to have already happened. Instead, patch the
      module inside capture.py via patch.dict on sys.modules.
    - WAV output is verified on a real tempdir, no mocking of the filesystem.
    - from_config() uses the real configs/ directory as source of truth —
      if that config file drifts, tests surface it immediately.
    - Every test is isolated; no shared state between them.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio.capture import AudioCapture

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def capture(tmp_path: Path) -> AudioCapture:
    """Default AudioCapture instance with output_dir in tmp_path."""
    return AudioCapture(
        device_index=0,
        sample_rate=48000,
        channels=1,
        window_seconds=3.0,
        energy_threshold=0.01,
        output_dir=tmp_path / "audio",
        device_name="USB PnP Audio Device",
    )


@pytest.fixture()
def capture_no_name(tmp_path: Path) -> AudioCapture:
    """AudioCapture with device_name unset — exercises index-only fallback."""
    return AudioCapture(
        device_index=0,
        sample_rate=48000,
        channels=1,
        window_seconds=3.0,
        energy_threshold=0.01,
        output_dir=tmp_path / "audio",
    )


def _fake_sounddevice(
    devices: list[dict] | None = None,
    rec_return: np.ndarray | None = None,
    rec_raises: Exception | None = None,
) -> MagicMock:
    """
    Build a mock sounddevice module with configurable query_devices() and rec().

    Used by patch.dict(sys.modules, {"sounddevice": fake_sd}) to intercept
    the import inside capture.py without polluting the test environment.
    """
    fake_sd = MagicMock()
    fake_sd.query_devices.return_value = devices or [
        {"name": "USB PnP Audio Device: Audio (hw:2,0)", "max_input_channels": 1},
    ]
    if rec_raises is not None:
        fake_sd.rec.side_effect = rec_raises
    else:
        # Default to 3 seconds of low-amplitude noise — below energy gate
        samples = 48000 * 3
        default = (np.random.randn(samples, 1).astype(np.float32) * 0.001)
        fake_sd.rec.return_value = rec_return if rec_return is not None else default
    return fake_sd


# ── __init__ ──────────────────────────────────────────────────────────────────


class TestAudioCaptureInit:
    def test_stores_all_params(self, tmp_path: Path) -> None:
        cap = AudioCapture(
            device_index=2,
            sample_rate=48000,
            channels=1,
            window_seconds=3.0,
            energy_threshold=0.05,
            output_dir=tmp_path / "out",
            dtype="float32",
            device_name="TestMic",
        )
        assert cap.device_index == 2
        assert cap.sample_rate == 48000
        assert cap.channels == 1
        assert cap.window_seconds == 3.0
        assert cap.energy_threshold == 0.05
        assert cap.output_dir == tmp_path / "out"
        assert cap.dtype == "float32"
        assert cap.device_name == "TestMic"

    def test_default_dtype(self, tmp_path: Path) -> None:
        cap = AudioCapture(
            device_index=0,
            sample_rate=48000,
            channels=1,
            window_seconds=3.0,
            energy_threshold=0.01,
            output_dir=tmp_path,
        )
        assert cap.dtype == "float32"

    def test_default_device_name_is_none(self, tmp_path: Path) -> None:
        cap = AudioCapture(
            device_index=0,
            sample_rate=48000,
            channels=1,
            window_seconds=3.0,
            energy_threshold=0.01,
            output_dir=tmp_path,
        )
        assert cap.device_name is None

    def test_resolved_index_starts_none(self, capture: AudioCapture) -> None:
        """Device index is resolved lazily on first capture, not at construction."""
        assert capture._resolved_index is None

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "deep" / "nested" / "audio"
        AudioCapture(
            device_index=0,
            sample_rate=48000,
            channels=1,
            window_seconds=3.0,
            energy_threshold=0.01,
            output_dir=output_dir,
        )
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_window_samples_computed(self, capture: AudioCapture) -> None:
        assert capture._window_samples == 48000 * 3


# ── from_config ───────────────────────────────────────────────────────────────


class TestFromConfig:
    def test_constructs_without_error(self) -> None:
        cap = AudioCapture.from_config("configs")
        assert cap is not None

    def test_reads_device_name(self) -> None:
        """device_name is read from hardware.yaml microphone.device_name."""
        cap = AudioCapture.from_config("configs")
        assert cap.device_name == "USB PnP Audio Device"

    def test_reads_device_index(self) -> None:
        cap = AudioCapture.from_config("configs")
        assert cap.device_index == 0

    def test_missing_device_name_tolerated(self, tmp_path: Path) -> None:
        """If device_name is absent from hardware.yaml, AudioCapture still works."""
        hw = tmp_path / "hardware.yaml"
        hw.write_text(
            "microphone:\n"
            "  device_index: 0\n"
            "  sample_rate: 48000\n"
            "  channels: 1\n"
            "  window_seconds: 3.0\n"
            "  dtype: float32\n"
        )
        thr = tmp_path / "thresholds.yaml"
        thr.write_text("audio:\n  energy_threshold: 0.01\n")
        paths = tmp_path / "paths.yaml"
        paths.write_text(f"captures:\n  audio: {tmp_path / 'audio'}\n")

        cap = AudioCapture.from_config(tmp_path)
        assert cap.device_name is None
        assert cap.device_index == 0

    def test_wires_energy_threshold(self) -> None:
        cap = AudioCapture.from_config("configs")
        assert cap.energy_threshold == 0.01

    def test_wires_sample_rate(self) -> None:
        cap = AudioCapture.from_config("configs")
        assert cap.sample_rate == 48000

    def test_wires_window_seconds(self) -> None:
        cap = AudioCapture.from_config("configs")
        assert cap.window_seconds == 3.0


# ── _resolve_device_index ─────────────────────────────────────────────────────


class TestResolveDeviceIndex:
    def test_finds_by_name(self, capture: AudioCapture) -> None:
        devices = [
            {"name": "bcm2835 Headphones", "max_input_channels": 0},
            {"name": "USB PnP Audio Device: Audio (hw:2,0)", "max_input_channels": 1},
        ]
        fake_sd = _fake_sounddevice(devices=devices)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            idx = capture._resolve_device_index()
        assert idx == 1

    def test_substring_match(self, capture: AudioCapture) -> None:
        """Match is a substring, not exact — 'USB PnP Audio Device' finds longer names."""
        devices = [
            {"name": "USB PnP Audio Device: Audio (hw:99,0)", "max_input_channels": 1},
        ]
        fake_sd = _fake_sounddevice(devices=devices)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            idx = capture._resolve_device_index()
        assert idx == 0

    def test_first_match_wins(self, capture: AudioCapture) -> None:
        """When two devices match, the lower index is chosen."""
        devices = [
            {"name": "bcm2835 Headphones", "max_input_channels": 0},
            {"name": "USB PnP Audio Device A", "max_input_channels": 1},
            {"name": "USB PnP Audio Device B", "max_input_channels": 1},
        ]
        fake_sd = _fake_sounddevice(devices=devices)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            idx = capture._resolve_device_index()
        assert idx == 1

    def test_name_match_requires_input_channels(self, capture: AudioCapture) -> None:
        """A device whose name matches but has 0 input channels is skipped."""
        devices = [
            {"name": "USB PnP Audio Device (output only)", "max_input_channels": 0},
            {"name": "Some Other Mic", "max_input_channels": 1},
            {"name": "USB PnP Audio Device (input)", "max_input_channels": 1},
        ]
        fake_sd = _fake_sounddevice(devices=devices)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            idx = capture._resolve_device_index()
        # Index 0 matches the name but has 0 inputs — skipped.
        # Index 2 matches the name AND has inputs — chosen.
        assert idx == 2

    def test_fallback_to_index_when_name_unset(
        self, capture_no_name: AudioCapture
    ) -> None:
        devices = [
            {"name": "USB PnP Audio Device: Audio (hw:2,0)", "max_input_channels": 1},
            {"name": "Something Else", "max_input_channels": 1},
        ]
        fake_sd = _fake_sounddevice(devices=devices)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            idx = capture_no_name._resolve_device_index()
        # device_name is None → falls through to device_index=0
        assert idx == 0

    def test_fallback_to_index_when_name_no_match(self, tmp_path: Path) -> None:
        cap = AudioCapture(
            device_index=0,
            sample_rate=48000,
            channels=1,
            window_seconds=3.0,
            energy_threshold=0.01,
            output_dir=tmp_path,
            device_name="NonexistentMic",
        )
        devices = [
            {"name": "USB PnP Audio Device", "max_input_channels": 1},
        ]
        fake_sd = _fake_sounddevice(devices=devices)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            idx = cap._resolve_device_index()
        assert idx == 0

    def test_raises_when_index_out_of_range(self, tmp_path: Path) -> None:
        cap = AudioCapture(
            device_index=99,
            sample_rate=48000,
            channels=1,
            window_seconds=3.0,
            energy_threshold=0.01,
            output_dir=tmp_path,
        )
        devices = [{"name": "OnlyDevice", "max_input_channels": 1}]
        fake_sd = _fake_sounddevice(devices=devices)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            with pytest.raises(RuntimeError, match="out of range"):
                cap._resolve_device_index()

    def test_raises_when_fallback_has_no_inputs(self, tmp_path: Path) -> None:
        """Fallback index points to an output-only device — helpful error."""
        cap = AudioCapture(
            device_index=0,
            sample_rate=48000,
            channels=1,
            window_seconds=3.0,
            energy_threshold=0.01,
            output_dir=tmp_path,
        )
        devices = [{"name": "OutputOnlyDevice", "max_input_channels": 0}]
        fake_sd = _fake_sounddevice(devices=devices)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            with pytest.raises(RuntimeError, match="no input channels"):
                cap._resolve_device_index()


# ── capture_window — energy gate ──────────────────────────────────────────────


class TestCaptureWindowEnergyGate:
    def test_returns_none_when_below_threshold(self, capture: AudioCapture) -> None:
        """A silent window (low RMS) is discarded."""
        silent = np.zeros((48000 * 3, 1), dtype=np.float32)
        fake_sd = _fake_sounddevice(rec_return=silent)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            result = capture.capture_window()
        assert result is None

    def test_returns_path_when_above_threshold(self, capture: AudioCapture) -> None:
        """A loud window (high RMS) is saved and its path returned."""
        # 0.3 amplitude sine wave — RMS ≈ 0.21, well above default 0.01 threshold
        t = np.linspace(0, 3, 48000 * 3, dtype=np.float32)
        loud = (0.3 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
        fake_sd = _fake_sounddevice(rec_return=loud)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            result = capture.capture_window()
        assert result is not None
        assert result.exists()

    def test_just_above_threshold_accepted(self, capture: AudioCapture) -> None:
        """RMS slightly above threshold is accepted."""
        # Constant signal at 0.015 — RMS = 0.015, comfortably above 0.01 threshold
        # even after float32 precision loss.
        n = 48000 * 3
        audio = np.full((n, 1), 0.015, dtype=np.float32)
        fake_sd = _fake_sounddevice(rec_return=audio)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            result = capture.capture_window()
        assert result is not None

    def test_just_below_threshold_rejected(self, capture: AudioCapture) -> None:
        """RMS slightly below threshold is rejected."""
        # Constant signal at 0.005 — RMS = 0.005, well below 0.01 threshold.
        n = 48000 * 3
        audio = np.full((n, 1), 0.005, dtype=np.float32)
        fake_sd = _fake_sounddevice(rec_return=audio)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            result = capture.capture_window()
        assert result is None

# ── capture_window — WAV output ───────────────────────────────────────────────


class TestCaptureWindowWavWriting:
    def test_wav_file_created(self, capture: AudioCapture, tmp_path: Path) -> None:
        t = np.linspace(0, 3, 48000 * 3, dtype=np.float32)
        loud = (0.3 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
        fake_sd = _fake_sounddevice(rec_return=loud)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            result = capture.capture_window()
        assert result is not None
        assert result.suffix == ".wav"

    def test_timestamp_filename(self, capture: AudioCapture) -> None:
        """Filename format YYYYMMDD_HHMMSS_ffffff.wav — all digits + underscores."""
        t = np.linspace(0, 3, 48000 * 3, dtype=np.float32)
        loud = (0.3 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
        fake_sd = _fake_sounddevice(rec_return=loud)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            result = capture.capture_window()
        assert result is not None
        stem = result.stem  # without .wav
        parts = stem.split("_")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts), f"expected all-digit parts, got {parts}"

    def test_wav_is_mono_int16(self, capture: AudioCapture) -> None:
        """Output WAV is mono, 16-bit PCM, 48000 Hz."""
        t = np.linspace(0, 3, 48000 * 3, dtype=np.float32)
        loud = (0.3 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
        fake_sd = _fake_sounddevice(rec_return=loud)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            result = capture.capture_window()
        assert result is not None
        with wave.open(str(result), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 2 bytes = int16
            assert wf.getframerate() == 48000


# ── capture_window — device resolution caching ────────────────────────────────


class TestCaptureWindowDeviceResolution:
    def test_resolves_on_first_call(self, capture: AudioCapture) -> None:
        assert capture._resolved_index is None
        t = np.linspace(0, 3, 48000 * 3, dtype=np.float32)
        loud = (0.3 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
        devices = [
            {"name": "USB PnP Audio Device: Audio (hw:2,0)", "max_input_channels": 1},
        ]
        fake_sd = _fake_sounddevice(devices=devices, rec_return=loud)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            capture.capture_window()
        assert capture._resolved_index == 0

    def test_caches_resolved_index(self, capture: AudioCapture) -> None:
        """Second capture_window() does not re-query devices."""
        t = np.linspace(0, 3, 48000 * 3, dtype=np.float32)
        loud = (0.3 * np.sin(2 * np.pi * 440 * t)).reshape(-1, 1)
        devices = [
            {"name": "USB PnP Audio Device: Audio (hw:2,0)", "max_input_channels": 1},
        ]
        fake_sd = _fake_sounddevice(devices=devices, rec_return=loud)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            capture.capture_window()
            capture.capture_window()
        # query_devices should have been called exactly once
        assert fake_sd.query_devices.call_count == 1


# ── capture_window — error handling ───────────────────────────────────────────


class TestCaptureWindowErrorHandling:
    def test_sounddevice_not_installed(self, capture: AudioCapture) -> None:
        """Clean RuntimeError when sounddevice cannot be imported."""
        # Remove sounddevice if present in sys.modules, and patch to raise on import
        with patch.dict(sys.modules, {"sounddevice": None}):
            with pytest.raises(RuntimeError, match="sounddevice is not installed"):
                capture.capture_window()

    def test_recording_failure_wrapped(self, capture: AudioCapture) -> None:
        """sd.rec() failure is wrapped in a descriptive RuntimeError."""
        fake_sd = _fake_sounddevice(rec_raises=OSError("PortAudio device open failed"))
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            with pytest.raises(RuntimeError, match="sounddevice failed to record"):
                capture.capture_window()