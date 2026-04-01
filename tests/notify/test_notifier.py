"""
tests/notify/test_notifier.py

Unit tests for src/notify/notifier.py.

All tests are fully synthetic — no network access, no real files beyond
temporary directories, no hardware.

Test groups:
    TestNotifierInit       — constructor stores parameters correctly
    TestNotifierLog        — _log() writes valid JSONL to disk
    TestNotifierPrint      — _print() formats and outputs message correctly
    TestNotifierDispatch   — dispatch() calls the right channels
    TestFromConfig         — from_config() reads notify.yaml + paths.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.data.schema import BirdObservation
from src.notify.notifier import Notifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_observation(
    species_code: str = "HOFI",
    common_name: str = "House Finch",
    scientific_name: str = "Haemorhous mexicanus",
    fused_confidence: float = 0.82,
) -> BirdObservation:
    """Build a minimal BirdObservation for testing."""
    return BirdObservation(
        species_code=species_code,
        common_name=common_name,
        scientific_name=scientific_name,
        fused_confidence=fused_confidence,
    )


def _make_notify_yaml(tmp_path: Path, print_on: bool = True) -> Path:
    cfg = {
        "channels": {"log": True, "print": print_on, "push": False, "email": False},
        "display": {
            "message_template": (
                "🐦 {common_name} ({scientific_name}) detected! " "Confidence: {confidence:.0%}"
            )
        },
    }
    p = tmp_path / "notify.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def _make_paths_yaml(tmp_path: Path) -> Path:
    cfg = {"logs": {"observations": str(tmp_path / "logs" / "observations.jsonl")}}
    p = tmp_path / "paths.yaml"
    p.write_text(yaml.dump(cfg))
    return p


# ---------------------------------------------------------------------------
# TestNotifierInit
# ---------------------------------------------------------------------------


class TestNotifierInit:
    def test_stores_log_path(self, tmp_path: Path):
        n = Notifier(log_path=tmp_path / "obs.jsonl")
        assert n.log_path == tmp_path / "obs.jsonl"

    def test_default_enable_print_true(self, tmp_path: Path):
        n = Notifier(log_path=tmp_path / "obs.jsonl")
        assert n.enable_print is True

    def test_default_enable_push_false(self, tmp_path: Path):
        n = Notifier(log_path=tmp_path / "obs.jsonl")
        assert n.enable_push is False

    def test_default_enable_email_false(self, tmp_path: Path):
        n = Notifier(log_path=tmp_path / "obs.jsonl")
        assert n.enable_email is False

    def test_accepts_string_log_path(self, tmp_path: Path):
        n = Notifier(log_path=str(tmp_path / "obs.jsonl"))
        assert isinstance(n.log_path, Path)

    def test_custom_flags(self, tmp_path: Path):
        n = Notifier(log_path=tmp_path / "obs.jsonl", enable_print=False, enable_push=True)
        assert n.enable_print is False
        assert n.enable_push is True


# ---------------------------------------------------------------------------
# TestNotifierLog
# ---------------------------------------------------------------------------


class TestNotifierLog:
    def test_creates_log_file(self, tmp_path: Path):
        n = Notifier(log_path=tmp_path / "logs" / "obs.jsonl")
        obs = _make_observation()
        n._log(obs)
        assert (tmp_path / "logs" / "obs.jsonl").exists()

    def test_creates_parent_directory(self, tmp_path: Path):
        log_path = tmp_path / "deep" / "nested" / "obs.jsonl"
        n = Notifier(log_path=log_path)
        n._log(_make_observation())
        assert log_path.exists()

    def test_writes_valid_json(self, tmp_path: Path):
        log_path = tmp_path / "obs.jsonl"
        n = Notifier(log_path=log_path)
        n._log(_make_observation())
        line = log_path.read_text(encoding="utf-8").strip()
        record = json.loads(line)
        assert isinstance(record, dict)

    def test_log_contains_species_code(self, tmp_path: Path):
        log_path = tmp_path / "obs.jsonl"
        n = Notifier(log_path=log_path)
        n._log(_make_observation(species_code="WCSP"))
        record = json.loads(log_path.read_text())
        assert record["species_code"] == "WCSP"

    def test_log_contains_confidence(self, tmp_path: Path):
        log_path = tmp_path / "obs.jsonl"
        n = Notifier(log_path=log_path)
        n._log(_make_observation(fused_confidence=0.75))
        record = json.loads(log_path.read_text())
        assert abs(record["fused_confidence"] - 0.75) < 1e-6

    def test_appends_multiple_observations(self, tmp_path: Path):
        log_path = tmp_path / "obs.jsonl"
        n = Notifier(log_path=log_path)
        n._log(_make_observation(species_code="HOFI"))
        n._log(_make_observation(species_code="WCSP"))
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["species_code"] == "HOFI"
        assert json.loads(lines[1])["species_code"] == "WCSP"

    def test_log_is_idempotent_on_rerun(self, tmp_path: Path):
        """Calling _log twice appends two lines, not overwrites."""
        log_path = tmp_path / "obs.jsonl"
        n = Notifier(log_path=log_path)
        obs = _make_observation()
        n._log(obs)
        n._log(obs)
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# TestNotifierPrint
# ---------------------------------------------------------------------------


class TestNotifierPrint:
    def test_prints_to_stdout(self, tmp_path: Path, capsys):
        n = Notifier(log_path=tmp_path / "obs.jsonl")
        n._print(_make_observation())
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_contains_common_name(self, tmp_path: Path, capsys):
        n = Notifier(log_path=tmp_path / "obs.jsonl")
        n._print(_make_observation(common_name="House Finch"))
        captured = capsys.readouterr()
        assert "House Finch" in captured.out

    def test_print_contains_scientific_name(self, tmp_path: Path, capsys):
        n = Notifier(log_path=tmp_path / "obs.jsonl")
        n._print(_make_observation(scientific_name="Haemorhous mexicanus"))
        captured = capsys.readouterr()
        assert "Haemorhous mexicanus" in captured.out

    def test_custom_template(self, tmp_path: Path, capsys):
        n = Notifier(
            log_path=tmp_path / "obs.jsonl",
            message_template="Spotted: {species_code}",
        )
        n._print(_make_observation(species_code="DOWO"))
        captured = capsys.readouterr()
        assert "Spotted: DOWO" in captured.out


# ---------------------------------------------------------------------------
# TestNotifierDispatch
# ---------------------------------------------------------------------------


class TestNotifierDispatch:
    def test_dispatch_always_calls_log(self, tmp_path: Path):
        log_path = tmp_path / "obs.jsonl"
        n = Notifier(log_path=log_path, enable_print=False)
        n.dispatch(_make_observation())
        assert log_path.exists()

    def test_dispatch_calls_print_when_enabled(self, tmp_path: Path, capsys):
        n = Notifier(log_path=tmp_path / "obs.jsonl", enable_print=True)
        n.dispatch(_make_observation(common_name="House Finch"))
        assert "House Finch" in capsys.readouterr().out

    def test_dispatch_skips_print_when_disabled(self, tmp_path: Path, capsys):
        n = Notifier(log_path=tmp_path / "obs.jsonl", enable_print=False)
        n.dispatch(_make_observation())
        assert capsys.readouterr().out == ""

    def test_dispatch_skips_push_when_disabled(self, tmp_path: Path):
        """Push channel disabled — should not raise NotImplementedError."""
        n = Notifier(log_path=tmp_path / "obs.jsonl", enable_push=False)
        n.dispatch(_make_observation())  # should not raise

    def test_dispatch_push_enabled_raises_not_implemented(self, tmp_path: Path):
        """Push channel enabled but not implemented — should raise."""
        n = Notifier(log_path=tmp_path / "obs.jsonl", enable_push=True)
        with pytest.raises(NotImplementedError):
            n.dispatch(_make_observation())


# ---------------------------------------------------------------------------
# TestFromConfig
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_loads_log_path_from_yaml(self, tmp_path: Path):
        notify_yaml = _make_notify_yaml(tmp_path)
        paths_yaml = _make_paths_yaml(tmp_path)
        n = Notifier.from_config(str(notify_yaml), str(paths_yaml))
        assert "observations.jsonl" in str(n.log_path)

    def test_loads_print_channel_true(self, tmp_path: Path):
        notify_yaml = _make_notify_yaml(tmp_path, print_on=True)
        paths_yaml = _make_paths_yaml(tmp_path)
        n = Notifier.from_config(str(notify_yaml), str(paths_yaml))
        assert n.enable_print is True

    def test_loads_print_channel_false(self, tmp_path: Path):
        notify_yaml = _make_notify_yaml(tmp_path, print_on=False)
        paths_yaml = _make_paths_yaml(tmp_path)
        n = Notifier.from_config(str(notify_yaml), str(paths_yaml))
        assert n.enable_print is False

    def test_push_disabled_by_default(self, tmp_path: Path):
        notify_yaml = _make_notify_yaml(tmp_path)
        paths_yaml = _make_paths_yaml(tmp_path)
        n = Notifier.from_config(str(notify_yaml), str(paths_yaml))
        assert n.enable_push is False

    def test_raises_on_missing_notify_config(self, tmp_path: Path):
        paths_yaml = _make_paths_yaml(tmp_path)
        with pytest.raises(FileNotFoundError):
            Notifier.from_config(str(tmp_path / "missing.yaml"), str(paths_yaml))

    def test_raises_on_missing_paths_config(self, tmp_path: Path):
        notify_yaml = _make_notify_yaml(tmp_path)
        with pytest.raises(FileNotFoundError):
            Notifier.from_config(str(notify_yaml), str(tmp_path / "missing.yaml"))

    def test_returns_notifier_instance(self, tmp_path: Path):
        notify_yaml = _make_notify_yaml(tmp_path)
        paths_yaml = _make_paths_yaml(tmp_path)
        n = Notifier.from_config(str(notify_yaml), str(paths_yaml))
        assert isinstance(n, Notifier)
