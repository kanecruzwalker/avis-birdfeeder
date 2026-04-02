"""
src/data/splitter.py

Generates stratified train/val/test split manifests for audio and visual data.

Reads raw data from:
  - data/raw/xeno_canto/<CODE>/<CODE>_*.mp3  (audio)
  - data/raw/nabirds/                         (visual, via NABirds index files)

Writes CSV manifests to data/splits/:
  - audio_train.csv, audio_val.csv, audio_test.csv
  - visual_train.csv, visual_val.csv, visual_test.csv

CSV schema (same for all six files):
    file_path,species_code,split

Design rules:
  - Split ratio and random seed come from configs/thresholds.yaml — never
    hardcoded here.
  - Splits are stratified by species so every species appears in every split
    at the correct ratio.
  - Splits are deterministic: same seed always produces the same split.
  - Re-running is idempotent: existing CSVs are overwritten cleanly.
  - Species list comes from configs/species.yaml — only species in that list
    are included, everything else in the raw data is ignored.
  - NABirds class IDs are mapped to species codes via NABIRDS_CLASS_MAP below.
    A species may have multiple class IDs (plumage variants) — all are folded
    into the same species code label.

Usage:
    Called by scripts/generate_splits.py — not intended to be run directly.
"""

from __future__ import annotations

import csv
import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NABirds class ID → species code mapping
#
# Each species may have multiple class IDs covering plumage variants
# (adult male, female/immature, juvenile, breeding, nonbreeding).
# All variants are folded into the same species code — the model learns
# the species regardless of plumage.
#
# Class IDs verified against data/raw/nabirds/classes.txt.
# ---------------------------------------------------------------------------

NABIRDS_CLASS_MAP: dict[int, str] = {
    # HOFI — House Finch
    419: "HOFI",
    790: "HOFI",  # Adult Male
    997: "HOFI",  # Female/immature
    # MODO — Mourning Dove
    171: "MODO",
    529: "MODO",
    # ANHU — Anna's Hummingbird
    152: "ANHU",
    375: "ANHU",  # Adult Male
    674: "ANHU",  # Female/immature
    # CAVI — California Scrub-Jay (listed as Western Scrub-Jay in NABirds)
    # NABirds predates the AOU split; Western Scrub-Jay covers CAVI
    # Class IDs confirmed via classes.txt search
    301: "CAVI",  # Western Scrub-Jay
    # MOCH — Northern Mockingbird
    389: "MOCH",
    852: "MOCH",
    # AMRO — American Robin
    718: "AMRO",
    753: "AMRO",  # Adult
    960: "AMRO",  # Juvenile
    # SOSP — Song Sparrow
    462: "SOSP",
    # LEGO — Lesser Goldfinch
    715: "LEGO",
    793: "LEGO",  # Adult Male
    1000: "LEGO",  # Female/juvenile
    # DOWO — Downy Woodpecker
    337: "DOWO",
    # WREN — House Wren
    572: "WREN",
    832: "WREN",
    # AMCR — American Crow
    683: "AMCR",
    957: "AMCR",
    # SPTO — Spotted Towhee
    258: "SPTO",
    888: "SPTO",
    # BLPH — Black Phoebe
    409: "BLPH",
    928: "BLPH",
    # HOSP — House Sparrow
    445: "HOSP",
    796: "HOSP",  # Male
    1003: "HOSP",  # Female/Juvenile
    # EUST — European Starling
    439: "EUST",
    748: "EUST",  # Breeding Adult
    856: "EUST",  # Nonbreeding Adult
    1005: "EUST",  # Juvenile
    # WCSP — White-crowned Sparrow
    729: "WCSP",
    766: "WCSP",  # Adult
    973: "WCSP",  # Immature
    # HOORI — Hooded Oriole
    719: "HOORI",
    784: "HOORI",  # Adult Male
    991: "HOORI",  # Female/Immature male
    # WBNU — White-breasted Nuthatch
    413: "WBNU",
    824: "WBNU",
    # OCWA — Orange-crowned Warbler
    289: "OCWA",
    867: "OCWA",
    # YRUM — Yellow-rumped Warbler (all subspecies/plumages)
    691: "YRUM",
    747: "YRUM",  # Breeding Myrtle
    798: "YRUM",  # Winter/juvenile Myrtle
    958: "YRUM",  # Breeding Audubon's
    1009: "YRUM",  # Winter/juvenile Audubon's
}


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------


def _stratified_split(
    items: list,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    """
    Split a list into train/val/test subsets using stratified sampling.

    Items are shuffled deterministically using the given seed, then divided
    at the computed indices. The test set receives all remaining items after
    train and val are allocated.

    Args:
        items:       List of items to split (file paths, tuples, etc.).
        train_ratio: Fraction for training set, e.g. 0.6.
        val_ratio:   Fraction for validation set, e.g. 0.2.
        seed:        Random seed for reproducible shuffling.

    Returns:
        Tuple of (train_items, val_items, test_items).
    """
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    # test gets the remainder — guarantees no items are lost
    n_train = min(n_train, n - 2)  # leave room for at least 1 val + 1 test
    n_val = min(n_val, n - n_train - 1)  # leave room for at least 1 test

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def _write_split_csv(
    rows: list[tuple[str, str, str]],
    output_path: Path,
) -> None:
    """
    Write split manifest rows to a CSV file.

    Args:
        rows:        List of (file_path, species_code, split) tuples.
        output_path: Destination CSV path. Parent directory must exist.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "species_code", "split"])
        writer.writerows(rows)
    logger.info("  wrote %d rows → %s", len(rows), output_path)


# ---------------------------------------------------------------------------
# Audio split generation
# ---------------------------------------------------------------------------


def generate_audio_splits(
    xc_raw_dir: Path,
    splits_dir: Path,
    species_codes: list[str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, int]:
    """
    Generate train/val/test split CSVs for Xeno-canto audio recordings.

    Scans data/raw/xeno_canto/<CODE>/ for MP3 files, applies stratified
    60/20/20 split per species, and writes three CSV manifests.

    Args:
        xc_raw_dir:    Path to data/raw/xeno_canto/.
        splits_dir:    Path to data/splits/ (created if absent).
        species_codes: List of 4-letter species codes to include.
        train_ratio:   Fraction for training set (default 0.6).
        val_ratio:     Fraction for validation set (default 0.2).
        seed:          Random seed for reproducible splits (default 42).

    Returns:
        Dict mapping split name → row count, e.g.
        {"train": 720, "val": 240, "test": 240}
    """
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[tuple[str, str, str]] = []
    val_rows: list[tuple[str, str, str]] = []
    test_rows: list[tuple[str, str, str]] = []

    skipped: list[str] = []

    logger.info("Generating audio splits from %s", xc_raw_dir)

    for code in sorted(species_codes):
        species_dir = xc_raw_dir / code
        if not species_dir.exists():
            logger.warning("  %s: directory not found — skipping", code)
            skipped.append(code)
            continue

        mp3_files = sorted(species_dir.glob("*.mp3"))
        if not mp3_files:
            logger.warning("  %s: no MP3 files found — skipping", code)
            skipped.append(code)
            continue

        if len(mp3_files) < 3:
            logger.warning(
                "  %s: only %d file(s) — too few for a 3-way split, skipping",
                code,
                len(mp3_files),
            )
            skipped.append(code)
            continue

        train, val, test = _stratified_split(mp3_files, train_ratio, val_ratio, seed)
        train_rows.extend((str(p), code, "train") for p in train)
        val_rows.extend((str(p), code, "val") for p in val)
        test_rows.extend((str(p), code, "test") for p in test)

        logger.info(
            "  %s: %d files → train=%d val=%d test=%d",
            code,
            len(mp3_files),
            len(train),
            len(val),
            len(test),
        )

    if skipped:
        logger.warning("Skipped %d species with no audio data: %s", len(skipped), skipped)

    _write_split_csv(train_rows, splits_dir / "audio_train.csv")
    _write_split_csv(val_rows, splits_dir / "audio_val.csv")
    _write_split_csv(test_rows, splits_dir / "audio_test.csv")

    return {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)}


# ---------------------------------------------------------------------------
# Visual split generation
# ---------------------------------------------------------------------------


def _load_nabirds_index(nabirds_dir: Path) -> list[tuple[str, int, str, int]]:
    """
    Load and join NABirds index files into a flat list of image records.

    Joins image_class_labels.txt, images.txt, and train_test_split.txt
    on image UUID.

    Args:
        nabirds_dir: Path to extracted NABirds root directory.

    Returns:
        List of (image_uuid, class_id, relative_image_path, original_split)
        where original_split is 0=test, 1=train from NABirds' own split.

    Raises:
        FileNotFoundError: If any required index file is missing.
    """
    # Load image_class_labels.txt: uuid → class_id
    class_labels: dict[str, int] = {}
    with open(nabirds_dir / "image_class_labels.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                class_labels[parts[0]] = int(parts[1])

    # Load images.txt: uuid → relative path
    image_paths: dict[str, str] = {}
    with open(nabirds_dir / "images.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                image_paths[parts[0]] = parts[1]

    # Load train_test_split.txt: uuid → 0 or 1
    split_flags: dict[str, int] = {}
    with open(nabirds_dir / "train_test_split.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                split_flags[parts[0]] = int(parts[1])

    # Join on UUID — only include images present in all three files
    records: list[tuple[str, int, str, int]] = []
    for uuid, class_id in class_labels.items():
        if uuid in image_paths and uuid in split_flags:
            records.append((uuid, class_id, image_paths[uuid], split_flags[uuid]))

    logger.info("Loaded %d NABirds image records", len(records))
    return records


def generate_visual_splits(
    nabirds_dir: Path,
    splits_dir: Path,
    species_codes: list[str],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, int]:
    """
    Generate train/val/test split CSVs for NABirds visual data.

    Loads NABirds index files, filters to species in species_codes via
    NABIRDS_CLASS_MAP, pools all plumage variants under the same species
    code, then applies stratified 60/20/20 split per species.

    Note: NABirds' own train/test split is intentionally ignored — we
    generate our own split so the ratio and seed are consistent with the
    audio splits.

    Args:
        nabirds_dir:   Path to extracted NABirds root directory.
        splits_dir:    Path to data/splits/ (created if absent).
        species_codes: List of 4-letter species codes to include.
        train_ratio:   Fraction for training set (default 0.6).
        val_ratio:     Fraction for validation set (default 0.2).
        seed:          Random seed for reproducible splits (default 42).

    Returns:
        Dict mapping split name → row count.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    codes_set = set(species_codes)

    # Load and join NABirds index files
    records = _load_nabirds_index(nabirds_dir)

    # Group image paths by species code using NABIRDS_CLASS_MAP
    # Multiple class IDs per species are pooled together
    species_images: dict[str, list[str]] = {code: [] for code in species_codes}

    images_base = nabirds_dir / "images"
    unmatched_classes: set[int] = set()

    for _uuid, class_id, rel_path, _orig_split in records:
        code = NABIRDS_CLASS_MAP.get(class_id)
        if code is None:
            unmatched_classes.add(class_id)
            continue
        if code not in codes_set:
            continue
        full_path = images_base / rel_path
        species_images[code].append(str(full_path))

    logger.debug("NABirds class IDs with no mapping: %d", len(unmatched_classes))

    train_rows: list[tuple[str, str, str]] = []
    val_rows: list[tuple[str, str, str]] = []
    test_rows: list[tuple[str, str, str]] = []
    skipped: list[str] = []

    for code in sorted(species_codes):
        images = species_images.get(code, [])

        if not images:
            logger.warning("  %s: no NABirds images found — skipping", code)
            skipped.append(code)
            continue

        if len(images) < 3:
            logger.warning(
                "  %s: only %d image(s) — too few for a 3-way split, skipping",
                code,
                len(images),
            )
            skipped.append(code)
            continue

        train, val, test = _stratified_split(images, train_ratio, val_ratio, seed)
        train_rows.extend((p, code, "train") for p in train)
        val_rows.extend((p, code, "val") for p in val)
        test_rows.extend((p, code, "test") for p in test)

        logger.info(
            "  %s: %d images → train=%d val=%d test=%d",
            code,
            len(images),
            len(train),
            len(val),
            len(test),
        )

    if skipped:
        logger.warning("Skipped %d species with no visual data: %s", len(skipped), skipped)

    _write_split_csv(train_rows, splits_dir / "visual_train.csv")
    _write_split_csv(val_rows, splits_dir / "visual_val.csv")
    _write_split_csv(test_rows, splits_dir / "visual_test.csv")

    return {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)}


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------


def print_split_summary(audio_counts: dict[str, int], visual_counts: dict[str, int]) -> None:
    """
    Print a formatted summary table of split sizes for both modalities.

    Args:
        audio_counts:  Dict of split → count for audio.
        visual_counts: Dict of split → count for visual.
    """
    print("\n── Split Summary ─────────────────────────────────────")
    print(f"  {'Split':<8} {'Audio':>8} {'Visual':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*8}")
    for split in ("train", "val", "test"):
        a = audio_counts.get(split, 0)
        v = visual_counts.get(split, 0)
        print(f"  {split:<8} {a:>8} {v:>8}")
    total_a = sum(audio_counts.values())
    total_v = sum(visual_counts.values())
    print(f"  {'TOTAL':<8} {total_a:>8} {total_v:>8}")
    print("──────────────────────────────────────────────────────\n")
