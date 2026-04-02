"""
src/data/downloader.py

Utilities for downloading, caching, and verifying dataset files.

Datasets are NOT committed to the repo. This module handles fetching them
from their original sources (Xeno-canto API v3, NABirds, etc.) and placing
them in the paths defined by configs/paths.yaml.

Design rules:
  - All paths come from configs/paths.yaml via the caller — never hardcoded here.
  - API keys come from environment variables — never hardcoded or logged.
  - Downloads are idempotent: re-running skips already-present files.
  - Rate limiting is always applied to respect upstream servers.
  - Every public function has a corresponding test in tests/data/test_downloader.py.

Usage:
    Called by scripts/download_datasets.py — not intended to be run directly.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Xeno-canto API constants
# ---------------------------------------------------------------------------

# API v3 base URL. Key is appended as a query parameter per XC documentation.
_XC_API_BASE = "https://xeno-canto.org/api/3/recordings"

# Results per page. XC maximum is 500; we use 100 to keep responses manageable.
_XC_PAGE_SIZE = 100

# Seconds to wait between API requests to respect XC's fair-use policy.
# XC explicitly warns about IP blocks for aggressive scrapers.
_XC_REQUEST_DELAY = 1.0

# Seconds to wait between individual audio file downloads.
_XC_DOWNLOAD_DELAY = 0.5

# Minimum recording quality to accept. XC grades A–E; A is highest.
# Using "A" or "B" for training keeps noise low.
_XC_MIN_QUALITY = "B"


# ---------------------------------------------------------------------------
# Generic filesystem helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Xeno-canto API helpers
# ---------------------------------------------------------------------------


def _build_xc_query(scientific_name: str, country: str = "United States") -> str:
    """
    Build a Xeno-canto search query string for a single species.

    Filters to:
      - The given species by scientific name (genus + species).
      - Recordings from the specified country.
      - Song and call types only (excludes drums, alarms, etc.).

    Args:
        scientific_name: Binomial Latin name, e.g. "Calypte anna".
        country:         Country filter. Defaults to "United States".

    Returns:
        URL-encoded query string, ready to append to the API URL.
    """
    genus, species = scientific_name.split(" ", 1)
    # XC query syntax: gen:Genus sp:species cnt:"Country" type:song
    parts = [
        f"gen:{genus}",
        f"sp:{species}",
        f'cnt:"{country}"',
        "type:song",
        "type:call",
    ]
    return " ".join(parts)


def _fetch_xc_page(
    query: str,
    api_key: str,
    page: int = 1,
) -> dict:
    """
    Fetch a single page of Xeno-canto search results.

    Args:
        query:   XC search query string (built by _build_xc_query).
        api_key: Xeno-canto API v3 key from environment.
        page:    Page number (1-indexed).

    Returns:
        Parsed JSON response dict from XC API.

    Raises:
        requests.HTTPError: On non-2xx responses.
        requests.ConnectionError: On network failure.
    """
    params = {
        "query": query,
        "key": api_key,
        "per_page": _XC_PAGE_SIZE,
        "page": page,
    }
    url = f"{_XC_API_BASE}?{urlencode(params)}"
    logger.debug("GET %s", url.replace(api_key, "***"))  # never log the key

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def iter_xc_recordings(
    scientific_name: str,
    api_key: str,
    min_quality: str = _XC_MIN_QUALITY,
) -> Iterator[dict]:
    """
    Yield Xeno-canto recording metadata dicts for a single species.

    Paginates automatically through all result pages.
    Filters to recordings meeting the minimum quality grade.
    Applies rate limiting between page requests.

    Args:
        scientific_name: Binomial Latin name, e.g. "Turdus migratorius".
        api_key:         Xeno-canto API v3 key.
        min_quality:     Minimum quality grade to include ("A"–"E").
                         Grade "A" is highest quality. Defaults to "B".

    Yields:
        Recording metadata dict as returned by the XC API, filtered to
        recordings at or above min_quality.
    """
    # Quality grades ordered best → worst
    quality_order = ["A", "B", "C", "D", "E"]
    if min_quality not in quality_order:
        raise ValueError(f"min_quality must be one of {quality_order}, got {min_quality!r}")

    acceptable = set(quality_order[: quality_order.index(min_quality) + 1])
    query = _build_xc_query(scientific_name)

    page = 1
    num_pages = None  # populated after first response

    while num_pages is None or page <= num_pages:
        data = _fetch_xc_page(query, api_key, page=page)

        if num_pages is None:
            num_pages = int(data.get("numPages", 1))
            total = data.get("numRecordings", "?")
            logger.info(
                "  %s: %s recordings across %d pages",
                scientific_name,
                total,
                num_pages,
            )

        for rec in data.get("recordings", []):
            if rec.get("q", "") in acceptable:
                yield rec

        page += 1
        if page <= num_pages:
            time.sleep(_XC_REQUEST_DELAY)


def download_xc_species(
    scientific_name: str,
    species_code: str,
    api_key: str,
    output_dir: Path,
    max_per_species: int = 100,
) -> list[Path]:
    """
    Download Xeno-canto audio recordings for a single species.

    Files are saved as WAV — XC serves MP3, which is converted on download.
    Each file is named: {species_code}_{xc_id}.mp3

    Skips files that already exist on disk (idempotent).
    Saves a metadata sidecar: {species_code}_metadata.json

    Args:
        scientific_name: Binomial Latin name, e.g. "Haemorhous mexicanus".
        species_code:    4-letter AOU banding code, e.g. "HOFI".
        api_key:         Xeno-canto API v3 key.
        output_dir:      Directory to save files into. Created if absent.
        max_per_species: Maximum number of recordings to download.
                         Keeps dataset balanced across species.

    Returns:
        List of Paths for downloaded (or already-present) files.
    """
    species_dir = ensure_directory(output_dir / species_code)
    metadata_path = species_dir / f"{species_code}_metadata.json"
    downloaded: list[Path] = []
    metadata_records: list[dict] = []

    logger.info("Downloading XC audio: %s (%s)", species_code, scientific_name)

    for rec in iter_xc_recordings(scientific_name, api_key):
        if len(downloaded) >= max_per_species:
            logger.info("  %s: reached limit of %d recordings", species_code, max_per_species)
            break

        xc_id = rec.get("id", "unknown")
        filename = f"{species_code}_{xc_id}.mp3"
        dest = species_dir / filename

        if dest.exists():
            logger.debug("  skip (exists): %s", filename)
            downloaded.append(dest)
            metadata_records.append(rec)
            continue

        # XC audio file URL: the 'file' field is a direct download link
        audio_url = rec.get("file", "")
        if not audio_url:
            logger.warning("  no file URL for recording %s — skipping", xc_id)
            continue

        try:
            logger.debug("  downloading %s", filename)
            audio_response = requests.get(audio_url, timeout=60, stream=True)
            audio_response.raise_for_status()

            with open(dest, "wb") as f:
                for chunk in audio_response.iter_content(chunk_size=8192):
                    f.write(chunk)

            downloaded.append(dest)
            metadata_records.append(rec)
            time.sleep(_XC_DOWNLOAD_DELAY)

        except requests.RequestException as exc:
            logger.warning("  failed to download %s: %s", filename, exc)
            # Don't abort the whole species — skip and continue
            continue

    # Write metadata sidecar so we can trace every file back to its XC record
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2, ensure_ascii=False)
    logger.info("  %s: %d files saved, metadata → %s", species_code, len(downloaded), metadata_path)

    return downloaded


# ---------------------------------------------------------------------------
# NABirds helpers
# ---------------------------------------------------------------------------


def verify_nabirds(nabirds_dir: Path) -> bool:
    """
    Verify that the NABirds dataset is present and structurally complete.

    Checks for the key index files that must exist in a valid NABirds extract.
    Does not validate every image — just the manifest files.

    Args:
        nabirds_dir: Path to the extracted NABirds root directory.

    Returns:
        True if all expected index files are present.
    """
    # These files must exist in any valid NABirds extraction
    required_files = [
        "classes.txt",
        "hierarchy.txt",
        "image_class_labels.txt",
        "images.txt",
        "train_test_split.txt",
        "bounding_boxes.txt",
        "sizes.txt",
        "nabirds.py",
    ]
    missing = [f for f in required_files if not (nabirds_dir / f).exists()]
    if missing:
        logger.warning("NABirds missing files: %s", missing)
        return False

    images_dir = nabirds_dir / "images"
    if not images_dir.is_dir():
        logger.warning("NABirds images/ directory not found at %s", images_dir)
        return False

    logger.info("NABirds verified at %s", nabirds_dir)
    return True


def load_nabirds_class_map(nabirds_dir: Path) -> dict[str, str]:
    """
    Parse NABirds classes.txt into a {class_id: class_name} mapping.

    classes.txt format (space-separated):
        <class_id> <class_name>

    Args:
        nabirds_dir: Path to the NABirds root directory.

    Returns:
        Dict mapping string class IDs to class name strings.

    Raises:
        FileNotFoundError: If classes.txt is not present.
    """
    classes_path = nabirds_dir / "classes.txt"
    class_map: dict[str, str] = {}

    with open(classes_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                class_map[parts[0]] = parts[1]

    logger.info("Loaded %d NABirds classes from %s", len(class_map), classes_path)
    return class_map


# ---------------------------------------------------------------------------
# Download summary helper
# ---------------------------------------------------------------------------


def print_download_summary(results: dict[str, int]) -> None:
    """
    Print a formatted summary table of download counts per species.

    Args:
        results: Dict mapping species_code → number of files downloaded.
    """
    print("\n── Download Summary ──────────────────────────────────")
    total = 0
    for code, count in sorted(results.items()):
        print(f"  {code:<8} {count:>4} files")
        total += count
    print(f"  {'TOTAL':<8} {total:>4} files")
    print("──────────────────────────────────────────────────────\n")
