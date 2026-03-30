"""
src/audio/preprocess.py

Converts raw audio into mel spectrograms suitable for BirdNET and our fine-tuned model.

Pipeline:
    WAV file or NumPy array
        → resample to target rate (48 kHz for BirdNET)
        → convert to mono
        → normalize amplitude
        → compute mel spectrogram
        → return as 2-D NumPy array (or save as PNG for visual inspection)

Why mel spectrograms?
    Bird vocalizations have rich frequency structure. Mel spectrograms compress raw
    audio into a 2-D image-like representation that captures time-frequency patterns
    at a perceptually meaningful scale — making image classification CNNs directly
    applicable to audio data. This is the standard approach in bird audio ML.

Phase 2 will implement using `librosa`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_wav(path: str | Path, target_sr: int = 48000) -> tuple[np.ndarray, int]:
    """
    Load a WAV file, resample to target_sr, and convert to mono.

    Args:
        path: Path to the WAV file.
        target_sr: Target sample rate in Hz.

    Returns:
        Tuple of (audio array float32, sample_rate).
    """
    raise NotImplementedError("Implement in Phase 2 using librosa.load().")


def normalize(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio amplitude to the range [-1, 1].

    Args:
        audio: 1-D float32 NumPy array.

    Returns:
        Normalized audio array of the same shape.
    """
    raise NotImplementedError("Implement in Phase 2.")


def to_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 48000,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """
    Convert a 1-D audio array to a 2-D mel spectrogram.

    Args:
        audio: 1-D float32 NumPy array.
        sample_rate: Sample rate of the input audio.
        n_mels: Number of mel filter banks (height of output).
        hop_length: Number of samples between successive frames.
        n_fft: FFT window size.

    Returns:
        2-D NumPy array of shape (n_mels, time_frames), values in dB scale.
    """
    raise NotImplementedError("Implement in Phase 2 using librosa.feature.melspectrogram().")


def preprocess_file(path: str | Path, target_sr: int = 48000) -> np.ndarray:
    """
    Full preprocessing pipeline for a single audio file.

    Convenience wrapper: load → normalize → mel spectrogram.

    Args:
        path: Path to input WAV file.
        target_sr: Target sample rate.

    Returns:
        2-D mel spectrogram array ready for model input.
    """
    raise NotImplementedError("Implement in Phase 2.")
