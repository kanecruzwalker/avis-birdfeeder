"""
src/data/downloader.py

Utilities for downloading and verifying dataset files.

Datasets are NOT committed to the repo. This module handles fetching them
from their original sources (Xeno-canto API, NABirds, etc.) and placing
them in the paths defined by configs/paths.yaml.

Usage:
    python scripts/download_datasets.py --config configs/paths.yaml

Phase 2 will implement the actual download logic per dataset.
"""

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """
    Create a directory if it does not exist and return the Path object.

    Args:
        path: Directory path to create.

    Returns:
        Resolved Path object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dataset_exists(path: str | Path) -> bool:
    """
    Check whether a dataset directory is non-empty (i.e., already downloaded).

    Args:
        path: Path to the dataset root directory.

    Returns:
        True if the path exists and contains at least one file.
    """
    p = Path(path)
    if not p.exists():
        return False
    return any(p.iterdir())
