"""
scripts/download_datasets.py

CLI entry point for downloading all Avis training datasets.

Downloads:
  - Xeno-canto audio recordings (SD species subset, MP3)
  - Verifies NABirds visual dataset is present (manual download required)

All output paths are read from configs/paths.yaml.
All species are read from configs/species.yaml.
The Xeno-canto API key is read from the XENO_CANTO_API_KEY environment variable.

Usage:
    # Activate venv first: .venv\\Scripts\\activate  (Windows)
    python scripts/download_datasets.py

    # Dry run — prints what would be downloaded without fetching anything
    python scripts/download_datasets.py --dry-run

    # Download only audio (skip NABirds verification)
    python scripts/download_datasets.py --audio-only

    # Download only a subset of species (useful for testing)
    python scripts/download_datasets.py --species ANHU HOFI AMRO

    # Limit recordings per species (default: 100)
    python scripts/download_datasets.py --max-per-species 50

Prerequisites:
    1. Register at https://xeno-canto.org and obtain an API key.
    2. Add XENO_CANTO_API_KEY=<your_key> to your .env file.
    3. NABirds must be downloaded manually from https://dl.allaboutbirds.org/nabirds
       and extracted to data/raw/nabirds/ (see docs/DATASETS.md).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.downloader import (  # noqa: E402
    dataset_exists,
    download_xc_species,
    ensure_directory,
    print_download_summary,
    verify_nabirds,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    """Load and parse a YAML file. Exits on file-not-found or parse error."""
    if not path.exists():
        logger.error("Config file not found: %s", path)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_datasets.py",
        description="Download Avis training datasets (Xeno-canto audio, NABirds visual).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--paths-config",
        default="configs/paths.yaml",
        help="Path to paths.yaml (default: configs/paths.yaml)",
    )
    parser.add_argument(
        "--species-config",
        default="configs/species.yaml",
        help="Path to species.yaml (default: configs/species.yaml)",
    )
    parser.add_argument(
        "--max-per-species",
        type=int,
        default=100,
        metavar="N",
        help="Maximum recordings to download per species (default: 100)",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        metavar="CODE",
        help="Download only these species codes, e.g. --species ANHU HOFI",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Skip NABirds verification and download audio only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without fetching anything",
    )
    return parser


# ---------------------------------------------------------------------------
# Download steps
# ---------------------------------------------------------------------------


def _run_xeno_canto(
    species_list: list[dict],
    xc_raw_dir: Path,
    api_key: str,
    max_per_species: int,
    dry_run: bool,
) -> dict[str, int]:
    """
    Download Xeno-canto recordings for all species in the list.

    Returns a dict of {species_code: file_count} for the summary table.
    """
    results: dict[str, int] = {}

    logger.info("═══ Xeno-canto Audio Download ═══════════════════════════")
    logger.info("Output directory : %s", xc_raw_dir)
    logger.info("Species          : %d", len(species_list))
    logger.info("Max per species  : %d", max_per_species)

    for species in species_list:
        code = species["code"]
        scientific = species["scientific_name"]
        common = species["common_name"]

        if dry_run:
            logger.info("[dry-run] Would download: %s (%s) → %s/", common, scientific, code)
            results[code] = 0
            continue

        files = download_xc_species(
            scientific_name=scientific,
            species_code=code,
            api_key=api_key,
            output_dir=xc_raw_dir,
            max_per_species=max_per_species,
        )
        results[code] = len(files)

    return results


def _run_nabirds_verify(nabirds_dir: Path, dry_run: bool) -> bool:
    """
    Verify NABirds dataset is present. Prints actionable instructions if not.

    Returns True if verified (or dry-run), False if the dataset is missing.
    """
    logger.info("═══ NABirds Visual Dataset ════════════════════════════════")
    logger.info("Expected location: %s", nabirds_dir)

    if dry_run:
        logger.info("[dry-run] Would verify NABirds at %s", nabirds_dir)
        return True

    if not dataset_exists(nabirds_dir):
        logger.error("NABirds not found at %s", nabirds_dir)
        logger.error("Manual download required — see docs/DATASETS.md for instructions.")
        logger.error("URL: https://dl.allaboutbirds.org/nabirds")
        return False

    return verify_nabirds(nabirds_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Entry point. Returns exit code (0 = success, 1 = error).
    """
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    # ── Load configs ──────────────────────────────────────────────────────
    paths_cfg = _load_yaml(Path(args.paths_config))
    species_cfg = _load_yaml(Path(args.species_config))

    all_species: list[dict] = species_cfg.get("species", [])
    if not all_species:
        logger.error("No species found in %s", args.species_config)
        return 1

    # Filter to --species subset if provided
    if args.species:
        requested = set(args.species)
        all_species = [s for s in all_species if s["code"] in requested]
        if not all_species:
            logger.error(
                "None of the requested species codes found in species.yaml: %s", args.species
            )
            return 1
        logger.info("Filtering to %d requested species: %s", len(all_species), args.species)

    # ── Resolve output paths ──────────────────────────────────────────────
    datasets_cfg = paths_cfg.get("datasets", {})
    xc_raw_dir = _PROJECT_ROOT / datasets_cfg.get("xeno_canto_raw", "data/raw/xeno_canto")
    nabirds_dir = _PROJECT_ROOT / datasets_cfg.get("nabirds_raw", "data/raw/nabirds")

    ensure_directory(xc_raw_dir)

    # ── Xeno-canto API key ────────────────────────────────────────────────
    api_key = os.environ.get("XENO_CANTO_API_KEY", "")
    if not api_key and not args.audio_only and not args.dry_run:
        logger.error(
            "XENO_CANTO_API_KEY not set. "
            "Register at https://xeno-canto.org and add the key to your .env file."
        )
        return 1
    if not api_key:
        api_key = "demo"  # XC demo key works for a handful of test requests

    # ── Run downloads ─────────────────────────────────────────────────────
    xc_results = _run_xeno_canto(
        species_list=all_species,
        xc_raw_dir=xc_raw_dir,
        api_key=api_key,
        max_per_species=args.max_per_species,
        dry_run=args.dry_run,
    )

    nabirds_ok = True
    if not args.audio_only:
        nabirds_ok = _run_nabirds_verify(nabirds_dir, dry_run=args.dry_run)

    # ── Summary ───────────────────────────────────────────────────────────
    print_download_summary(xc_results)

    if not nabirds_ok:
        logger.warning("NABirds verification failed — visual training data is incomplete.")
        return 1

    logger.info("All downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
