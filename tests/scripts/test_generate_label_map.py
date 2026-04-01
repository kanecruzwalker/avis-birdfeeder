"""
tests/scripts/test_generate_label_map.py

Unit tests for scripts/generate_label_map.py.

All tests use synthetic split CSVs in tmp_path — no real dataset files needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.generate_label_map import _load_species_from_split, generate_label_maps

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def species_yaml(tmp_path: Path) -> Path:
    data = {
        "species": [
            {
                "code": "HOFI",
                "common_name": "House Finch",
                "scientific_name": "Haemorhous mexicanus",
            },
            {"code": "MODO", "common_name": "Mourning Dove", "scientific_name": "Zenaida macroura"},
            {
                "code": "DOWO",
                "common_name": "Downy Woodpecker",
                "scientific_name": "Dryobates pubescens",
            },
            {
                "code": "ANHU",
                "common_name": "Anna's Hummingbird",
                "scientific_name": "Calypte anna",
            },
        ]
    }
    p = tmp_path / "species.yaml"
    p.write_text(yaml.dump(data))
    return p


@pytest.fixture()
def splits_dir(tmp_path: Path) -> Path:
    """Write minimal audio and visual train CSVs to tmp_path/splits/."""
    splits = tmp_path / "splits"
    splits.mkdir(parents=True)

    pd.DataFrame(
        {
            "file_path": ["a.wav", "b.wav", "c.wav"],
            "species_code": ["HOFI", "MODO", "ANHU"],
        }
    ).to_csv(splits / "audio_train.csv", index=False)

    pd.DataFrame(
        {
            "file_path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            "species_code": ["HOFI", "MODO", "DOWO", "ANHU"],
        }
    ).to_csv(splits / "visual_train.csv", index=False)

    return splits


# ── _load_species_from_split ──────────────────────────────────────────────────


class TestLoadSpeciesFromSplit:
    def test_returns_sorted_unique_codes(self, tmp_path: Path) -> None:
        csv = tmp_path / "train.csv"
        pd.DataFrame({"label": ["MODO", "HOFI", "HOFI", "ANHU"]}).to_csv(csv, index=False)
        codes = _load_species_from_split(csv)
        assert codes == ["ANHU", "HOFI", "MODO"]

    def test_raises_on_missing_label_column(self, tmp_path: Path) -> None:
        csv = tmp_path / "train.csv"
        pd.DataFrame({"file_path": ["a.wav"]}).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="Expected 'species_code' or 'label' column"):
            _load_species_from_split(csv)


# ── generate_label_maps ───────────────────────────────────────────────────────


class TestGenerateLabelMaps:
    def test_shared_map_is_union(
        self, splits_dir: Path, tmp_path: Path, species_yaml: Path
    ) -> None:
        maps = generate_label_maps(splits_dir, tmp_path / "models", species_yaml)
        shared_codes = set(maps["shared"].values())
        assert shared_codes == {"HOFI", "MODO", "ANHU", "DOWO"}

    def test_audio_map_excludes_visual_only_species(
        self, splits_dir: Path, tmp_path: Path, species_yaml: Path
    ) -> None:
        maps = generate_label_maps(splits_dir, tmp_path / "models", species_yaml)
        # DOWO only in visual — should not appear in audio map
        assert "DOWO" not in maps["audio"].values()

    def test_visual_map_includes_all_four(
        self, splits_dir: Path, tmp_path: Path, species_yaml: Path
    ) -> None:
        maps = generate_label_maps(splits_dir, tmp_path / "models", species_yaml)
        assert set(maps["visual"].values()) == {"HOFI", "MODO", "DOWO", "ANHU"}

    def test_shared_indices_consistent_across_maps(
        self, splits_dir: Path, tmp_path: Path, species_yaml: Path
    ) -> None:
        """A species that appears in both maps must have the same integer index."""
        maps = generate_label_maps(splits_dir, tmp_path / "models", species_yaml)
        shared_inv = {v: k for k, v in maps["shared"].items()}
        audio_inv = {v: k for k, v in maps["audio"].items()}
        for code in maps["audio"].values():
            assert audio_inv[code] == shared_inv[code], (
                f"{code} has index {audio_inv[code]} in audio but "
                f"{shared_inv[code]} in shared — must match"
            )

    def test_writes_three_json_files(
        self, splits_dir: Path, tmp_path: Path, species_yaml: Path
    ) -> None:
        out_dir = tmp_path / "models"
        generate_label_maps(splits_dir, out_dir, species_yaml)
        assert (out_dir / "label_map.json").exists()
        assert (out_dir / "audio_label_map.json").exists()
        assert (out_dir / "visual_label_map.json").exists()

    def test_json_keys_are_strings(
        self, splits_dir: Path, tmp_path: Path, species_yaml: Path
    ) -> None:
        out_dir = tmp_path / "models"
        generate_label_maps(splits_dir, out_dir, species_yaml)
        with (out_dir / "label_map.json").open() as f:
            data = json.load(f)
        assert all(isinstance(k, str) for k in data.keys())

    def test_raises_if_audio_split_missing(self, tmp_path: Path, species_yaml: Path) -> None:
        empty_splits = tmp_path / "empty_splits"
        (empty_splits / "visual" / "audio").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Audio train split not found"):
            generate_label_maps(empty_splits, tmp_path / "models", species_yaml)

    def test_indices_are_contiguous_from_zero(
        self, splits_dir: Path, tmp_path: Path, species_yaml: Path
    ) -> None:
        maps = generate_label_maps(splits_dir, tmp_path / "models", species_yaml)
        indices = sorted(maps["shared"].keys())
        assert indices == list(range(len(indices)))
