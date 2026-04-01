"""
scripts/generate_label_map.py

Generate models/label_map.json from the train split CSVs.

The label map is the single source of truth for integer index → species_code
mapping used by both AudioClassifier and VisualClassifier at inference time.
It is derived from the actual training splits (not species.yaml directly)
so that species excluded from training (CAVI, AMCR audio) are never assigned
an index — indices are always dense and contiguous starting from 0.

Outputs:
    models/label_map.json        — shared by both classifiers
    models/audio_label_map.json  — audio-only subset (18 species)
    models/visual_label_map.json — visual-only subset (19 species)

Usage:
    python scripts/generate_label_map.py
    python scripts/generate_label_map.py --splits-dir data/splits --output-dir models

The shared label_map.json is the union of audio and visual species, sorted
alphabetically by species code for determinism. Audio and visual maps use
the same integer keys as the shared map (not re-indexed) so a species code
always maps to the same integer regardless of which classifier is running.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_species_from_split(split_csv: Path) -> list[str]:
    """
    Read unique species codes from a split CSV file.

    Args:
        split_csv: Path to a train/val/test CSV with a 'species_code' or 'label'
                   column containing species codes (e.g. 'HOFI', 'MODO').

    Returns:
        Sorted list of unique species codes present in the CSV.
    """
    df = pd.read_csv(split_csv)
    label_col = "species_code" if "species_code" in df.columns else "label"
    if label_col not in df.columns:
        raise ValueError(
            f"Expected 'species_code' or 'label' column in {split_csv}, "
            f"got {list(df.columns)}"
        )
    codes = sorted(df[label_col].unique().tolist())
    logger.info("  %s → %d species", split_csv.name, len(codes))
    return codes


def _load_species_meta(species_yaml: Path) -> dict[str, dict]:
    """
    Load species metadata from configs/species.yaml.

    Returns:
        Dict mapping species_code → {common_name, scientific_name}.
    """
    with species_yaml.open() as f:
        cfg = yaml.safe_load(f)
    return {
        s["code"]: {
            "common_name": s["common_name"],
            "scientific_name": s["scientific_name"],
        }
        for s in cfg["species"]
    }


def generate_label_maps(
    splits_dir: Path,
    output_dir: Path,
    species_yaml: Path,
) -> dict[str, dict[int, str]]:
    """
    Generate label maps for audio, visual, and shared (union) classifiers.

    Reads the audio and visual train CSVs to determine which species actually
    made it into training. Species in species.yaml but absent from splits
    (e.g. CAVI, AMCR) are excluded automatically.

    Args:
        splits_dir: Directory containing audio/ and visual/ split CSVs.
        output_dir: Directory to write label_map JSON files.
        species_yaml: Path to configs/species.yaml for metadata validation.

    Returns:
        Dict with keys 'audio', 'visual', 'shared', each mapping
        int index → species_code.
    """
    audio_train = splits_dir / "audio_train.csv"
    visual_train = splits_dir / "visual_train.csv"

    if not audio_train.exists():
        raise FileNotFoundError(
            f"Audio train split not found: {audio_train}\n"
            "Run scripts/generate_splits.py first."
        )
    if not visual_train.exists():
        raise FileNotFoundError(
            f"Visual train split not found: {visual_train}\n"
            "Run scripts/generate_splits.py first."
        )

    logger.info("Loading species from splits...")
    audio_codes = _load_species_from_split(audio_train)
    visual_codes = _load_species_from_split(visual_train)

    # Union sorted alphabetically — determines shared index assignment
    shared_codes = sorted(set(audio_codes) | set(visual_codes))

    logger.info(
        "Species counts | audio=%d visual=%d shared=%d",
        len(audio_codes),
        len(visual_codes),
        len(shared_codes),
    )

    # Validate against species.yaml — warn if a training species is missing
    meta = _load_species_meta(species_yaml)
    for code in shared_codes:
        if code not in meta:
            logger.warning(
                "Species %s in splits but not in species.yaml — "
                "common_name/scientific_name will be missing at inference.",
                code,
            )

    # Build maps — shared indices are the reference
    # Audio and visual maps use the SAME indices as shared (not re-indexed)
    # so HOFI is always index 2 (for example) regardless of modality
    shared_map = {i: code for i, code in enumerate(shared_codes)}
    code_to_idx = {code: i for i, code in shared_map.items()}

    audio_map = {code_to_idx[c]: c for c in audio_codes}
    visual_map = {code_to_idx[c]: c for c in visual_codes}

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    maps = {
        "label_map": shared_map,        # configs/paths.yaml → models/label_map.json
        "audio_label_map": audio_map,
        "visual_label_map": visual_map,
    }

    for filename, label_map in maps.items():
        out_path = output_dir / f"{filename}.json"
        # JSON keys must be strings
        serializable = {str(k): v for k, v in label_map.items()}
        with out_path.open("w") as f:
            json.dump(serializable, f, indent=2, sort_keys=False)
        logger.info("Wrote %s (%d entries) → %s", filename, len(label_map), out_path)

    # Summary
    logger.info("")
    logger.info("Label map summary:")
    logger.info("  Shared (%d species): %s",
                len(shared_map),
                list(shared_map.values()))
    logger.info("  Audio  (%d species): %s",
                len(audio_map),
                [shared_map[i] for i in sorted(audio_map)])
    logger.info("  Visual (%d species): %s",
                len(visual_map),
                [shared_map[i] for i in sorted(visual_map)])
    logger.info("")
    logger.info(
        "Audio-only species (visual split only):  %s",
        sorted(set(visual_codes) - set(audio_codes)) or "none",
    )
    logger.info(
        "Visual-only species (audio split only):  %s",
        sorted(set(audio_codes) - set(visual_codes)) or "none",
    )

    return {"audio": audio_map, "visual": visual_map, "shared": shared_map}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate label maps from split CSVs.")
    parser.add_argument(
        "--splits-dir",
        default="data/splits",
        help="Directory containing audio/ and visual/ split CSVs (default: data/splits)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to write label_map JSON files (default: models/)",
    )
    parser.add_argument(
        "--species-yaml",
        default="configs/species.yaml",
        help="Path to species.yaml for metadata validation (default: configs/species.yaml)",
    )
    args = parser.parse_args()

    generate_label_maps(
        splits_dir=Path(args.splits_dir),
        output_dir=Path(args.output_dir),
        species_yaml=Path(args.species_yaml),
    )


if __name__ == "__main__":
    main()
