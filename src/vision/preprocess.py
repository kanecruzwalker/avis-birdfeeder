"""
src/vision/preprocess.py

Image preprocessing pipeline for bird classification.

Pipeline:
    Raw frame (NumPy uint8 RGB)
        → resize to model input size (224x224 for EfficientNet)
        → normalize pixel values to [0, 1] or ImageNet mean/std
        → optional: data augmentation (training only)
        → return as float32 tensor-ready array

Why this shape?
    EfficientNet-B0 expects 224x224 RGB input with ImageNet normalization.
    MobileNetV2 also expects 224x224. Keeping this consistent means we can
    swap backbones without changing the capture or agent code.

Phase 2 will implement using PIL / torchvision transforms.
"""

from __future__ import annotations

import numpy as np

# Target input size for EfficientNet-B0 and MobileNetV2
DEFAULT_INPUT_SIZE: tuple[int, int] = (224, 224)

# ImageNet normalization constants (standard for pretrained torchvision models)
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def resize(frame: np.ndarray, size: tuple[int, int] = DEFAULT_INPUT_SIZE) -> np.ndarray:
    """
    Resize a frame to the target size using bilinear interpolation.

    Args:
        frame: NumPy array of shape (H, W, 3), dtype uint8.
        size: Target (width, height).

    Returns:
        Resized NumPy array of shape (size[1], size[0], 3), dtype uint8.
    """
    raise NotImplementedError("Implement in Phase 2 using PIL.Image.resize().")


def normalize(
    frame: np.ndarray,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """
    Normalize a uint8 RGB frame to float32 with ImageNet mean/std.

    Args:
        frame: NumPy array of shape (H, W, 3), dtype uint8, values in [0, 255].
        mean: Per-channel mean for normalization.
        std: Per-channel standard deviation for normalization.

    Returns:
        Float32 array of shape (H, W, 3), approximately zero-centered.
    """
    raise NotImplementedError("Implement in Phase 2.")


def augment(frame: np.ndarray) -> np.ndarray:
    """
    Apply training-time data augmentation to a single frame.

    Augmentations (configurable, applied randomly):
        - Horizontal flip
        - Random brightness/contrast jitter
        - Random crop and resize

    Args:
        frame: NumPy array of shape (H, W, 3), dtype uint8.

    Returns:
        Augmented frame as NumPy array, same shape and dtype.

    Note:
        This should ONLY be called during training data preparation,
        never during live inference.
    """
    raise NotImplementedError("Implement in Phase 2.")


def preprocess_frame(frame: np.ndarray, augment_: bool = False) -> np.ndarray:
    """
    Full preprocessing pipeline for a single captured frame.

    Convenience wrapper: resize → normalize → (optionally) augment.

    Args:
        frame: Raw captured frame, shape (H, W, 3), dtype uint8.
        augment_: Whether to apply augmentation (True for training only).

    Returns:
        Preprocessed float32 array of shape (224, 224, 3), ready for model input.
    """
    raise NotImplementedError("Implement in Phase 2.")
