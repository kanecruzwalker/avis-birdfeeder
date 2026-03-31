"""
scripts/generate_splits.py

CLI entry point for generating train/val/test split manifests.

Reads raw data from data/raw/ and writes CSV manifests to data/splits/.
All paths are read from configs/paths.yaml.
All species are read from configs/species.yaml.
Split ratio and random seed are read from configs/thresholds.yaml.

Usage:
    python scripts/generate_splits.py

    # Audio splits only
    python scripts/generate_splits.py --audio-only

    # Visual splits only
    python scripts/generate_splits.py --visual-only

    # Override split ratios (must sum to < 1.0, remainder goes to test)
    python scripts/generate_splits.py --train-ratio 0.7 --val-ratio 0.15

Prerequisites:
    - Audio: Run scripts/download_datasets.py first to populate
      data/raw/xeno_canto/
    - Visual: NABirds must be extracted to data/raw/nabirds/
      See docs/DATASETS.md for setup instructions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.splitter import (  # noqa: E402
    generate_audio_splits,
    generate_visual_splits,
    print_split_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        logger.error("Config not found: %s", path)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_splits.py",
        description="Generate train/val/test split CSVs for audio and visual data.",
    )
    parser.add_argument("--paths-config", default="configs/paths.yaml")
    parser.add_argument("--species-config", default="configs/species.yaml")
    parser.add_argument("--thresholds-config", default="configs/thresholds.yaml")
    parser.add_argument("--audio-only", action="store_true")
    parser.add_argument("--visual-only", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    paths_cfg = _load_yaml(Path(args.paths_config))
    species_cfg = _load_yaml(Path(args.species_config))
    thresholds_cfg = _load_yaml(Path(args.thresholds_config))

    species_codes = [s["code"] for s in species_cfg.get("species", [])]
    if not species_codes:
        logger.error("No species found in %s", args.species_config)
        return 1

    datasets = paths_cfg.get("datasets", {})
    xc_raw_dir = _PROJECT_ROOT / datasets.get("xeno_canto_raw", "data/raw/xeno_canto")
    nabirds_dir = _PROJECT_ROOT / datasets.get("nabirds_raw", "data/raw/nabirds")
    splits_dir = _PROJECT_ROOT / datasets.get("splits", "data/splits")

    # Read split config from thresholds.yaml, allow CLI overrides
    split_cfg = thresholds_cfg.get("splits", {})
    train_ratio = args.train_ratio or split_cfg.get("train_ratio", 0.6)
    val_ratio = args.val_ratio or split_cfg.get("val_ratio", 0.2)
    seed = split_cfg.get("random_seed", 42)

    if train_ratio + val_ratio >= 1.0:
        logger.error("train_ratio + val_ratio must be < 1.0 (got %.2f)", train_ratio + val_ratio)
        return 1

    logger.info(
        "Split ratios — train: %.0f%% val: %.0f%% test: %.0f%%",
        train_ratio * 100,
        val_ratio * 100,
        (1 - train_ratio - val_ratio) * 100,
    )
    logger.info("Random seed: %d", seed)
    logger.info("Species: %d", len(species_codes))

    audio_counts: dict[str, int] = {}
    visual_counts: dict[str, int] = {}

    if not args.visual_only:
        logger.info("═══ Audio Splits ══════════════════════════════════════")
        audio_counts = generate_audio_splits(
            xc_raw_dir=xc_raw_dir,
            splits_dir=splits_dir,
            species_codes=species_codes,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

    if not args.audio_only:
        logger.info("═══ Visual Splits ═════════════════════════════════════")
        visual_counts = generate_visual_splits(
            nabirds_dir=nabirds_dir,
            splits_dir=splits_dir,
            species_codes=species_codes,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

    print_split_summary(audio_counts, visual_counts)
    logger.info("Splits written to %s", splits_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
