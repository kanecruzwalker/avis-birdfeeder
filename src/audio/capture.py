"""
src/audio/capture.py

Handles microphone input from the Fifine USB microphone.

Responsibilities:
    - Open the audio input device
    - Record a fixed-duration audio window (configurable via configs/thresholds.yaml)
    - Return raw audio as a NumPy array for downstream preprocessing

This module is intentionally hardware-only. It does NOT do any preprocessing.
Tests that require the physical microphone must be marked @pytest.mark.hardware.

Phase 2 will implement the full recording logic using `sounddevice`.
"""

from __future__ import annotations

import numpy as np


def record_window(
    duration_seconds: float,
    sample_rate: int = 22050,
    device: int | str | None = None,
) -> np.ndarray:
    """
    Record a single audio window from the microphone.

    Args:
        duration_seconds: How long to record, in seconds.
        sample_rate: Target sample rate in Hz. BirdNET expects 48000; adjust accordingly.
        device: sounddevice device index or name. None uses system default.

    Returns:
        1-D NumPy array of float32 audio samples, shape (duration_seconds * sample_rate,).

    Raises:
        RuntimeError: If the audio device cannot be opened.
    """
    # Phase 2: implement using sounddevice.rec()
    raise NotImplementedError("Audio capture will be implemented in Phase 2.")
