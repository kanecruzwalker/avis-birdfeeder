"""
tests/audio/test_preprocess.py

Unit tests for src.audio.preprocess.

Design principles:
    - Zero hardware dependencies — all audio is generated synthetically with NumPy.
    - No real WAV files required — file-based tests write to a temporary directory
      via pytest's `tmp_path` fixture so nothing is left on disk after the run.
    - Tests validate behavior and output shape/range, not exact numeric values,
      so they remain valid as librosa versions evolve.
    - Each test covers one clearly-named contract, making CI failure messages
      immediately actionable.

Synthetic audio strategy:
    We generate pure sine waves at a known frequency (e.g., 1 kHz). This gives us
    predictable amplitude, easy silence/noise construction, and avoids any copyright
    or dataset-availability concerns. BirdNET-style preprocessing works identically
    on synthetic sine waves as on real bird audio for the purposes of shape/range testing.

Markers:
    No special markers needed — all tests run on any machine with the venv active.
    Hardware-dependent tests (real mic, real files) are deferred to Phase 5 and
    will be marked @pytest.mark.hardware.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from src.audio.preprocess import (
    load_wav,
    normalize,
    preprocess_array,
    preprocess_file,
    to_mel_spectrogram,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic audio generation
# ---------------------------------------------------------------------------

_SR = 48_000  # match the BirdNET standard used throughout the module


def _sine_wave(
    frequency: float = 1000.0,
    duration_s: float = 3.0,
    sample_rate: int = _SR,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate a mono sine wave as a float32 NumPy array.

    Args:
        frequency:   Tone frequency in Hz.
        duration_s:  Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude:   Peak amplitude in [0.0, 1.0].

    Returns:
        1-D float32 array of length ``int(duration_s * sample_rate)``.
    """
    t = np.linspace(0.0, duration_s, int(duration_s * sample_rate), endpoint=False)
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


def _silent_array(duration_s: float = 3.0, sample_rate: int = _SR) -> np.ndarray:
    """Return an array of zeros — a perfectly silent clip."""
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = _SR) -> None:
    """
    Write a float32 NumPy array to a 16-bit PCM WAV file.

    We write 16-bit PCM (the most universally supported WAV format) by scaling
    the float32 array to int16 range. librosa.load() handles PCM transparently
    and returns float32 regardless.
    """
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Tests: load_wav
# ---------------------------------------------------------------------------


class TestLoadWav:
    """Tests for load_wav() — file loading, resampling, mono conversion."""

    def test_loads_wav_and_returns_correct_sample_rate(self, tmp_path: Path) -> None:
        """load_wav returns a tuple whose second element equals target_sr."""
        wav_path = tmp_path / "tone.wav"
        _write_wav(wav_path, _sine_wave())
        _, sr = load_wav(wav_path, target_sr=_SR)
        assert sr == _SR

    def test_returns_float32_array(self, tmp_path: Path) -> None:
        """load_wav always returns float32 regardless of source bit depth."""
        wav_path = tmp_path / "tone.wav"
        _write_wav(wav_path, _sine_wave())
        audio, _ = load_wav(wav_path, target_sr=_SR)
        assert audio.dtype == np.float32

    def test_returns_1d_mono_array(self, tmp_path: Path) -> None:
        """load_wav converts multi-channel audio to 1-D mono."""
        wav_path = tmp_path / "tone.wav"
        _write_wav(wav_path, _sine_wave())
        audio, _ = load_wav(wav_path, target_sr=_SR)
        assert audio.ndim == 1

    def test_resamples_to_target_sr(self, tmp_path: Path) -> None:
        """
        load_wav resamples audio written at a different rate.

        We write at 22 050 Hz and request 48 000 Hz. The returned array
        should have approximately (original_samples / 22050 * 48000) samples,
        confirming resampling occurred.
        """
        original_sr = 22_050
        duration_s = 1.0
        expected_samples = int(duration_s * _SR)

        wav_path = tmp_path / "tone_22k.wav"
        _write_wav(
            wav_path,
            _sine_wave(duration_s=duration_s, sample_rate=original_sr),
            sample_rate=original_sr,
        )

        audio, sr = load_wav(wav_path, target_sr=_SR)

        assert sr == _SR
        # Allow ±5 samples tolerance for rounding during resampling.
        assert abs(len(audio) - expected_samples) <= 5

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        """load_wav raises FileNotFoundError for a non-existent path."""
        missing = tmp_path / "does_not_exist.wav"
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            load_wav(missing)

    def test_audio_values_in_valid_range(self, tmp_path: Path) -> None:
        """
        Loaded audio should be in [-1, 1] (librosa normalizes on load).

        librosa.load() returns float32 in [-1.0, 1.0] by default.
        This test ensures we haven't accidentally disabled that behavior.
        """
        wav_path = tmp_path / "tone.wav"
        _write_wav(wav_path, _sine_wave(amplitude=0.9))
        audio, _ = load_wav(wav_path, target_sr=_SR)
        assert np.all(audio >= -1.0)
        assert np.all(audio <= 1.0)


# ---------------------------------------------------------------------------
# Tests: normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    """Tests for normalize() — peak normalization to [-1, 1]."""

    def test_peak_is_one_after_normalize(self) -> None:
        """After normalization, max(abs(audio)) == 1.0."""
        audio = _sine_wave(amplitude=0.3)
        result = normalize(audio)
        assert np.isclose(np.max(np.abs(result)), 1.0, atol=1e-5)

    def test_output_dtype_is_float32(self) -> None:
        """normalize always returns float32."""
        audio = _sine_wave().astype(np.float64)
        result = normalize(audio)
        assert result.dtype == np.float32

    def test_output_shape_unchanged(self) -> None:
        """normalize does not change the array length."""
        audio = _sine_wave()
        result = normalize(audio)
        assert result.shape == audio.shape

    def test_already_normalized_is_unchanged(self) -> None:
        """Audio already at peak 1.0 should pass through essentially unchanged."""
        audio = _sine_wave(amplitude=1.0)
        result = normalize(audio)
        assert np.allclose(result, audio, atol=1e-5)

    def test_silent_clip_returns_zeros(self) -> None:
        """
        A silent (all-zero) clip should return all zeros, not NaN or Inf.

        Without this guard, dividing by a near-zero peak would produce NaN,
        which would silently propagate through the spectrogram and crash the model.
        """
        audio = _silent_array()
        result = normalize(audio)
        assert np.all(result == 0.0)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_values_within_minus_one_to_one(self) -> None:
        """All output values are in [-1.0, 1.0]."""
        audio = _sine_wave(amplitude=0.7)
        result = normalize(audio)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# Tests: to_mel_spectrogram
# ---------------------------------------------------------------------------


class TestToMelSpectrogram:
    """Tests for to_mel_spectrogram() — output shape, range, and error handling."""

    def test_output_shape_rows_equals_n_mels(self) -> None:
        """Output has exactly n_mels rows (frequency axis)."""
        audio = _sine_wave()
        spec = to_mel_spectrogram(audio, sample_rate=_SR, n_mels=128)
        assert spec.shape[0] == 128

    def test_output_shape_cols_match_hop_length(self) -> None:
        """
        Number of time frames is approximately len(audio) // hop_length + 1.

        We allow ±1 tolerance because librosa's STFT may add a single frame
        of center-padding at the boundaries.
        """
        audio = _sine_wave(duration_s=3.0)
        hop_length = 512
        spec = to_mel_spectrogram(audio, sample_rate=_SR, hop_length=hop_length)
        expected_frames = 1 + len(audio) // hop_length
        assert abs(spec.shape[1] - expected_frames) <= 1

    def test_output_dtype_is_float32(self) -> None:
        """Spectrogram output is always float32."""
        audio = _sine_wave()
        spec = to_mel_spectrogram(audio, sample_rate=_SR)
        assert spec.dtype == np.float32

    def test_output_is_2d(self) -> None:
        """Spectrogram is always 2-D (n_mels × time_frames)."""
        audio = _sine_wave()
        spec = to_mel_spectrogram(audio, sample_rate=_SR)
        assert spec.ndim == 2

    def test_values_are_in_db_range(self) -> None:
        """
        dB spectrogram values should be in [-top_db, 0].

        ``power_to_db(ref=np.max)`` maps the loudest frame to 0 dB and clips
        quiet frames at -top_db. Values outside this window indicate a bug
        in the dB conversion step.
        """
        audio = _sine_wave()
        top_db = 80.0
        spec = to_mel_spectrogram(audio, sample_rate=_SR, top_db=top_db)
        # Allow a small floating-point margin above 0.
        assert spec.max() <= 0.5
        assert spec.min() >= -(top_db + 0.5)

    def test_different_n_mels_produces_different_row_count(self) -> None:
        """n_mels parameter controls the number of frequency bins."""
        audio = _sine_wave()
        for n_mels in (64, 128):
            spec = to_mel_spectrogram(audio, sample_rate=_SR, n_mels=n_mels)
            assert spec.shape[0] == n_mels

    def test_no_nan_or_inf_in_output(self) -> None:
        """Spectrogram must be finite — NaN/Inf would crash model inference."""
        audio = _sine_wave()
        spec = to_mel_spectrogram(audio, sample_rate=_SR)
        assert not np.any(np.isnan(spec))
        assert not np.any(np.isinf(spec))

    def test_raises_on_2d_input(self) -> None:
        """Passing a 2-D array (e.g., stereo) raises ValueError with a helpful message."""
        stereo = np.zeros((2, _SR * 3), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            to_mel_spectrogram(stereo, sample_rate=_SR)

    def test_raises_when_audio_shorter_than_n_fft(self) -> None:
        """
        Audio shorter than n_fft should raise ValueError.

        This prevents silent failures where librosa would produce a single-frame
        spectrogram from too-short input, which the model would reject anyway.
        """
        n_fft = 2048
        short_audio = np.zeros(n_fft - 1, dtype=np.float32)
        with pytest.raises(ValueError, match="too short"):
            to_mel_spectrogram(short_audio, sample_rate=_SR, n_fft=n_fft)

    def test_silent_audio_produces_uniform_low_db_spectrogram(self) -> None:
        """
        A silent clip (all-zeros after normalize) produces a uniform spectrogram.

        normalize() returns all-zeros for a silent clip. librosa's power_to_db
        on an all-zero array uses 0.0 as the reference (since max is 0), which
        results in a uniform 0.0 dB spectrogram rather than -top_db. This is
        librosa's defined behavior for zero-power input and is handled correctly
        downstream — the energy_threshold check in the agent will reject this
        clip before it reaches the classifier.
        """

    audio = normalize(_silent_array())
    spec = to_mel_spectrogram(audio, sample_rate=_SR)
    # All values should be uniform (either all 0.0 or all -top_db).
    assert spec.min() == spec.max()
    assert not np.any(np.isnan(spec))
    assert not np.any(np.isinf(spec))


# ---------------------------------------------------------------------------
# Tests: preprocess_file (integration — exercises the full pipeline via disk)
# ---------------------------------------------------------------------------


class TestPreprocessFile:
    """Integration tests for preprocess_file() — exercises load → normalize → spectrogram."""

    def test_returns_2d_float32_array(self, tmp_path: Path) -> None:
        """preprocess_file returns a 2-D float32 array."""
        wav_path = tmp_path / "tone.wav"
        _write_wav(wav_path, _sine_wave())
        result = preprocess_file(wav_path, target_sr=_SR)
        assert result.ndim == 2
        assert result.dtype == np.float32

    def test_output_rows_equals_default_n_mels(self, tmp_path: Path) -> None:
        """preprocess_file output has 128 rows by default (matching BirdNET convention)."""
        wav_path = tmp_path / "tone.wav"
        _write_wav(wav_path, _sine_wave())
        result = preprocess_file(wav_path, target_sr=_SR)
        assert result.shape[0] == 128

    def test_no_nan_or_inf(self, tmp_path: Path) -> None:
        """Full pipeline output must be finite."""
        wav_path = tmp_path / "tone.wav"
        _write_wav(wav_path, _sine_wave())
        result = preprocess_file(wav_path, target_sr=_SR)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        """preprocess_file propagates FileNotFoundError for missing input."""
        with pytest.raises(FileNotFoundError):
            preprocess_file(tmp_path / "ghost.wav")

    def test_accepts_path_object_and_string(self, tmp_path: Path) -> None:
        """preprocess_file accepts both Path objects and plain strings."""
        wav_path = tmp_path / "tone.wav"
        _write_wav(wav_path, _sine_wave())
        result_path = preprocess_file(wav_path, target_sr=_SR)
        result_str = preprocess_file(str(wav_path), target_sr=_SR)
        assert result_path.shape == result_str.shape


# ---------------------------------------------------------------------------
# Tests: preprocess_array (integration — exercises the live-capture path)
# ---------------------------------------------------------------------------


class TestPreprocessArray:
    """Tests for preprocess_array() — the in-memory preprocessing path for live capture."""

    def test_returns_2d_float32_array(self) -> None:
        """preprocess_array returns a 2-D float32 array."""
        audio = _sine_wave()
        result = preprocess_array(audio, sample_rate=_SR, target_sr=_SR)
        assert result.ndim == 2
        assert result.dtype == np.float32

    def test_resamples_when_rates_differ(self) -> None:
        """
        When sample_rate != target_sr, preprocess_array resamples before processing.

        We verify by comparing the output column count to what we'd expect
        from audio at target_sr: same duration, different frame count.
        """
        source_sr = 22_050
        target_sr = _SR
        duration_s = 3.0
        audio = _sine_wave(duration_s=duration_s, sample_rate=source_sr)

        result = preprocess_array(audio, sample_rate=source_sr, target_sr=target_sr, hop_length=512)

        # After resampling to 48 kHz, expected_frames ≈ (48000 * 3) / 512 + 1
        expected_frames = 1 + int(duration_s * target_sr) // 512
        assert abs(result.shape[1] - expected_frames) <= 2

    def test_no_nan_or_inf(self) -> None:
        """preprocess_array output must be finite."""
        audio = _sine_wave()
        result = preprocess_array(audio, sample_rate=_SR, target_sr=_SR)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_skips_resample_when_rates_match(self) -> None:
        """When sample_rate == target_sr, no resampling should occur (output matches direct call)."""
        audio = _sine_wave()
        # Both paths should produce identical output when rates already match.
        result_array = preprocess_array(audio, sample_rate=_SR, target_sr=_SR)
        from src.audio.preprocess import normalize, to_mel_spectrogram

        result_direct = to_mel_spectrogram(normalize(audio), sample_rate=_SR)
        np.testing.assert_array_equal(result_array, result_direct)
