"""
src/vision/preprocess.py

Image preprocessing pipeline for bird species classification.

Pipeline:
    Raw frame (NumPy uint8 RGB, any resolution)
        → resize to model input size (224×224 for EfficientNet-B0)
        → convert to float32 and scale pixel values from [0, 255] → [0.0, 1.0]
        → normalize with ImageNet channel mean and std
        → return as (H, W, C) float32 NumPy array, ready for classify.py

Why (H, W, C) and not (C, H, W)?
    This module is framework-agnostic — it works with PIL and NumPy only,
    with no dependency on PyTorch. The CHW transpose that PyTorch requires
    is classify.py's responsibility, since classify.py is the only module
    that needs to know about tensor layout conventions. Keeping preprocess.py
    in HWC format also makes visual inspection and debugging easier, since
    PIL and matplotlib both expect HWC.

Why ImageNet normalization?
    EfficientNet-B0 and MobileNetV2 were pretrained on ImageNet with these
    specific channel statistics. Applying the same normalization at inference
    time ensures the input distribution matches what the model saw during
    training. Skipping normalization or using different values degrades
    accuracy significantly, even with fine-tuning.

Why PIL for resize (not OpenCV)?
    PIL is a pure Python dependency already in requirements.txt (Pillow).
    OpenCV would add a large binary dependency with no benefit for our use
    case. PIL's BILINEAR filter gives high-quality downsampling with
    acceptable speed for offline preprocessing.

Config keys consumed (configs/thresholds.yaml):
    vision.input_width    — model input width in pixels (default 224)
    vision.input_height   — model input height in pixels (default 224)

Dependencies:
    Pillow >= 10.0   — image resizing and format conversion
    numpy            — array operations and normalization math
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level defaults — all sourced from configs/thresholds.yaml.
# Callers should pass values loaded from YAML rather than relying on these.
# They are provided so individual functions remain usable in isolation
# (e.g., during testing with synthetic data).
# ---------------------------------------------------------------------------

# Target spatial dimensions. EfficientNet-B0 expects 224×224.
# Do not change without retraining — the model's first conv layer is
# sized for this resolution.
_DEFAULT_WIDTH: int = 224
_DEFAULT_HEIGHT: int = 224

# ImageNet channel statistics (RGB order).
# Computed across the full 1.2M-image ImageNet training set.
# mean[c] and std[c] are applied per channel after scaling to [0, 1].
# Source: torchvision documentation and PyTorch model hub conventions.
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resize(
    frame: np.ndarray,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
) -> np.ndarray:
    """
    Resize a raw frame to the target spatial dimensions using bilinear interpolation.

    Why bilinear interpolation?
        Bilinear (PIL.Image.BILINEAR) samples the weighted average of the four
        nearest pixels at each output location. This produces smooth edges and
        avoids the blocky artifacts of nearest-neighbor, at a compute cost
        that is negligible for 224×224 output. Bicubic would be marginally
        higher quality but slower — unnecessary for offline preprocessing.

    Args:
        frame:  NumPy array of shape (H, W, 3), dtype uint8, RGB color order.
                H and W may be any positive integers — the function handles
                both upsampling and downsampling.
        width:  Target width in pixels. Default 224 (EfficientNet-B0 standard).
                Pass ``configs['vision']['input_width']`` from thresholds.yaml.
        height: Target height in pixels. Default 224.
                Pass ``configs['vision']['input_height']`` from thresholds.yaml.

    Returns:
        NumPy array of shape (height, width, 3), dtype uint8.
        Note: output shape is (H, W, C) — rows first, then columns, then channels.

    Raises:
        ValueError: If ``frame`` is not a 3-D array with 3 channels.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"resize expects a (H, W, 3) array, got shape {frame.shape}. "
            "Ensure the frame is RGB with 3 channels."
        )

    logger.debug("Resizing frame %s → (%d, %d, 3)", frame.shape, height, width)

    # Convert NumPy array → PIL Image for resizing.
    # PIL expects uint8 input and handles the bilinear filter natively.
    pil_image = Image.fromarray(frame)

    # PIL.resize takes (width, height) — note the order is (W, H), not (H, W).
    # This is the opposite of NumPy's convention, which is why we pass (width, height)
    # explicitly rather than relying on the DEFAULT_INPUT_SIZE tuple directly.
    pil_resized = pil_image.resize((width, height), resample=Image.BILINEAR)

    # Convert back to NumPy. The result is (H, W, 3) uint8 — same layout as input,
    # just with different spatial dimensions.
    return np.array(pil_resized, dtype=np.uint8)


def normalize(
    frame: np.ndarray,
    mean: tuple[float, float, float] = _IMAGENET_MEAN,
    std: tuple[float, float, float] = _IMAGENET_STD,
) -> np.ndarray:
    """
    Normalize a uint8 RGB frame to float32 using ImageNet channel statistics.

    Math — two steps applied per channel c ∈ {R, G, B}:

        Step 1 — scale to [0, 1]:
            x[c] = pixel[c] / 255.0

        Step 2 — standardize with ImageNet statistics:
            y[c] = (x[c] - mean[c]) / std[c]

    After normalization, the output is approximately zero-centered with unit
    variance per channel, matching the distribution the pretrained model
    expects. In practice the range is roughly [-2.1, 2.6] depending on the
    channel — not exactly [-1, 1], because ImageNet statistics are not
    perfectly symmetric.

    Why divide by 255?
        Raw pixel values are uint8 in [0, 255]. Most ML frameworks and
        pretrained models expect float inputs in [0, 1] before applying
        channel normalization. Dividing by 255 is the standard first step.

    Why these specific mean/std values?
        (0.485, 0.456, 0.406) and (0.229, 0.224, 0.225) are the channel-wise
        mean and std computed across the full ImageNet training set. Using
        these exact values at inference time ensures the input distribution
        matches what EfficientNet-B0 saw during pretraining, which is critical
        for transfer learning to perform well on our SD bird species subset.

    Args:
        frame: NumPy array of shape (H, W, 3), dtype uint8, values in [0, 255].
               Typically the output of resize().
        mean:  Per-channel mean (R, G, B) for subtraction after scaling.
               Default: ImageNet mean (0.485, 0.456, 0.406).
        std:   Per-channel standard deviation (R, G, B) for division.
               Default: ImageNet std (0.229, 0.224, 0.225).

    Returns:
        Float32 array of shape (H, W, 3), approximately zero-centered per channel.
        Values are typically in the range [-2.2, 2.6] depending on channel.

    Raises:
        ValueError: If ``frame`` is not a 3-D array with 3 channels.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"normalize expects a (H, W, 3) array, got shape {frame.shape}.")

    # Step 1 — convert uint8 → float32 and scale to [0.0, 1.0].
    # We cast before dividing to prevent integer overflow: 255 / 255 in uint8
    # arithmetic would be fine, but intermediate values in other operations
    # could silently wrap. float32 has enough precision for 8-bit pixel data.
    frame_float = frame.astype(np.float32) / 255.0

    # Step 2 — subtract channel mean and divide by channel std.
    # np.array reshapes mean/std from (3,) to (1, 1, 3) implicitly via
    # broadcasting, so the operation applies per-channel across all spatial
    # positions simultaneously without a loop.
    # Formula: y[h, w, c] = (x[h, w, c] - mean[c]) / std[c]
    mean_arr = np.array(mean, dtype=np.float32)  # shape (3,)
    std_arr = np.array(std, dtype=np.float32)  # shape (3,)

    normalized = (frame_float - mean_arr) / std_arr

    logger.debug("Normalized frame: range [%.3f, %.3f]", normalized.min(), normalized.max())

    return normalized.astype(np.float32)


def augment(
    frame: np.ndarray,
    horizontal_flip: bool = True,
    color_jitter: bool = True,
) -> np.ndarray:
    """
    Apply training-time data augmentation to a single uint8 RGB frame.

    Augmentation artificially expands the training set by presenting the
    same bird in slightly different conditions. This reduces overfitting
    and improves generalization to real-world variation in lighting,
    orientation, and camera angle.

    Augmentations applied (each with 50% probability):
        - Horizontal flip: mirrors the frame left-right. Birds appear
          from either side of the feeder, so handedness is not meaningful.
        - Brightness jitter: randomly adjusts overall brightness by ±20%.
          Simulates variation in natural lighting conditions.
        - Contrast jitter: randomly adjusts contrast by ±20%.
          Simulates overcast vs direct sunlight conditions.

    IMPORTANT: Call this ONLY during training data preparation.
    Never call during live inference — augmentation would corrupt the
    model input with random transformations, making predictions unreliable.

    Args:
        frame:            NumPy array of shape (H, W, 3), dtype uint8.
                          Must be uint8 (pre-normalization) — augmentation
                          operates on pixel values, not normalized floats.
        horizontal_flip:  Whether to randomly flip horizontally (default True).
        color_jitter:     Whether to apply random brightness/contrast (default True).

    Returns:
        Augmented frame as NumPy array, same shape (H, W, 3) and dtype uint8.
        The specific augmentations applied are random — two calls with the
        same input may produce different outputs.
    """
    # Work on a copy — do not mutate the caller's array in place.
    result = frame.copy()

    # ── Horizontal flip ──────────────────────────────────────────────────────
    # np.fliplr reverses the column order of a 2D or 3D array.
    # For a (H, W, C) image this flips left-right while preserving channel order.
    # 50% probability — coin flip using random integer 0 or 1.
    if horizontal_flip and np.random.randint(0, 2):
        result = np.fliplr(result)
        logger.debug("augment: applied horizontal flip")

    # ── Color jitter ─────────────────────────────────────────────────────────
    if color_jitter:
        # Brightness: multiply all pixels by a factor in [0.8, 1.2].
        # np.random.uniform samples from a continuous uniform distribution.
        # Clip to [0, 255] prevents uint8 overflow (values wrapping around).
        if np.random.randint(0, 2):
            factor = np.random.uniform(0.8, 1.2)
            result = np.clip(result.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            logger.debug("augment: brightness factor=%.2f", factor)

        # Contrast: blend toward the channel mean.
        # formula: y = mean + factor * (x - mean)
        # factor > 1 increases contrast, factor < 1 reduces it.
        if np.random.randint(0, 2):
            factor = np.random.uniform(0.8, 1.2)
            mean_val = result.mean()
            result = np.clip(
                mean_val + factor * (result.astype(np.float32) - mean_val), 0, 255
            ).astype(np.uint8)
            logger.debug("augment: contrast factor=%.2f", factor)

    return result


def preprocess_frame(
    frame: np.ndarray,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    mean: tuple[float, float, float] = _IMAGENET_MEAN,
    std: tuple[float, float, float] = _IMAGENET_STD,
    augment_: bool = False,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single captured frame.

    Chains: (augment →) resize → normalize.

    Augmentation is applied before resize so that random crops and flips
    operate on the full-resolution image, preserving more detail before
    downsampling to 224×224.

    This is the primary entry point for the vision pipeline. The agent
    and dataset utilities should call this function rather than the
    individual steps, so that any future changes to the pipeline only
    require updating this function.

    Args:
        frame:    Raw captured frame, shape (H, W, 3), dtype uint8, RGB.
                  H and W may be any resolution — the function handles resizing.
        width:    Target width in pixels. Pass from configs/thresholds.yaml.
        height:   Target height in pixels. Pass from configs/thresholds.yaml.
        mean:     ImageNet channel mean for normalization.
        std:      ImageNet channel std for normalization.
        augment_: If True, apply random augmentation before resizing.
                  Set True only during training data preparation.

    Returns:
        Float32 array of shape (height, width, 3) in HWC layout, ImageNet-normalized.
        Ready to be transposed to CHW and converted to a PyTorch tensor in classify.py.

    Raises:
        ValueError: If ``frame`` is not a (H, W, 3) uint8 array.
    """
    logger.info("Preprocessing frame %s (augment=%s)", frame.shape, augment_)

    # Step 0 (optional) — augment at full resolution before downsampling.
    if augment_:
        frame = augment(frame)

    # Step 1 — resize to model input dimensions.
    frame = resize(frame, width=width, height=height)

    # Step 2 — scale to [0, 1] and apply ImageNet normalization.
    # Output is float32 (H, W, 3), approximately zero-centered per channel.
    result = normalize(frame, mean=mean, std=std)

    logger.info(
        "Preprocessed frame → shape %s, range [%.3f, %.3f]",
        result.shape,
        result.min(),
        result.max(),
    )
    return result


def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image file from disk as a uint8 RGB NumPy array.

    Analogous to load_wav() in the audio pipeline — this is the disk-based
    entry point. The live capture path (src.vision.capture) provides frames
    directly as NumPy arrays and skips this function.

    Handles any format PIL supports: JPEG, PNG, BMP, TIFF, WebP.
    Always converts to RGB, so grayscale and RGBA images are handled
    transparently — grayscale is replicated across 3 channels, alpha is dropped.

    Args:
        path: Path to the image file. Must exist and be readable.

    Returns:
        NumPy array of shape (H, W, 3), dtype uint8, RGB color order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        RuntimeError:      If PIL fails to decode the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    logger.debug("Loading image: %s", path)

    try:
        # PIL opens lazily — convert() forces decode and RGB conversion.
        # "RGB" mode guarantees exactly 3 uint8 channels regardless of source format.
        pil_image = Image.open(path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"PIL failed to load '{path}': {exc}") from exc

    # np.array copies the pixel data out of PIL's internal buffer into a
    # standard C-contiguous NumPy array of shape (H, W, 3), dtype uint8.
    array = np.array(pil_image, dtype=np.uint8)
    logger.debug("Loaded image: shape %s, dtype %s", array.shape, array.dtype)
    return array


def preprocess_file(
    path: str | Path,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    mean: tuple[float, float, float] = _IMAGENET_MEAN,
    std: tuple[float, float, float] = _IMAGENET_STD,
    augment_: bool = False,
) -> np.ndarray:
    """
    Full preprocessing pipeline for an image file on disk.

    Convenience wrapper that chains: load_image → preprocess_frame.

    Used by dataset preparation scripts to batch-process downloaded images
    into normalized arrays ready for model training or evaluation.

    Args:
        path:     Path to input image file (JPEG, PNG, etc.).
        width:    Target width in pixels.
        height:   Target height in pixels.
        mean:     ImageNet channel mean for normalization.
        std:      ImageNet channel std for normalization.
        augment_: Whether to apply augmentation (True for training only).

    Returns:
        Float32 array of shape (height, width, 3), ImageNet-normalized.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError:        If the loaded image has an unexpected shape.
    """
    logger.info("Preprocessing file: %s", path)
    frame = load_image(path)
    return preprocess_frame(
        frame, width=width, height=height, mean=mean, std=std, augment_=augment_
    )
