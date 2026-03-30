"""
src/vision/capture.py

Handles frame capture from the Raspberry Pi Camera Module 3.

Responsibilities:
    - Initialize the picamera2 interface for one or both cameras
    - Capture a single frame or a short burst on trigger
    - Return raw frame as a NumPy array for downstream preprocessing

Hardware note:
    Both cameras connect via CSI ribbon. `picamera2` manages them as
    separate camera indices (0 and 1). The primary camera (index 0) is
    the default for classification; the secondary may be used for wider
    field-of-view or stereo capture in a future phase.

Tests that require the physical camera must be marked @pytest.mark.hardware.

Phase 2 will implement capture logic using picamera2.
"""

from __future__ import annotations

import numpy as np


def capture_frame(camera_index: int = 0, resolution: tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """
    Capture a single frame from the specified camera.

    Args:
        camera_index: Which camera to use (0 = primary, 1 = secondary).
        resolution: Capture resolution as (width, height).

    Returns:
        NumPy array of shape (height, width, 3), dtype uint8, RGB color order.

    Raises:
        RuntimeError: If the camera cannot be initialized.
    """
    # Phase 2: implement using picamera2
    raise NotImplementedError("Camera capture will be implemented in Phase 2.")


def capture_burst(
    camera_index: int = 0,
    n_frames: int = 5,
    resolution: tuple[int, int] = (1920, 1080),
) -> list[np.ndarray]:
    """
    Capture a short burst of frames for motion analysis or best-frame selection.

    Args:
        camera_index: Which camera to use.
        n_frames: Number of frames to capture.
        resolution: Capture resolution as (width, height).

    Returns:
        List of NumPy arrays, each of shape (height, width, 3).
    """
    raise NotImplementedError("Burst capture will be implemented in Phase 2.")
