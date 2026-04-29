"""Tests for BaselineOptimizer stub."""

import pytest

from src.agent.baseline_optimizer import BaselineOptimizer


def test_from_config_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        BaselineOptimizer.from_config("configs/")


def test_optimize_raises_not_implemented(tmp_path):
    opt = BaselineOptimizer(
        observations_path="logs/observations.jsonl",
        splits_dir="data/splits/",
        models_dir="models/",
        experiments_csv="notebooks/results/experiments.csv",
    )
    with pytest.raises(NotImplementedError):
        opt.optimize("audio")


def test_init_stores_params(tmp_path):
    opt = BaselineOptimizer(
        observations_path="logs/observations.jsonl",
        splits_dir="data/splits/",
        models_dir="models/",
        experiments_csv="notebooks/results/experiments.csv",
        max_trials=10,
        max_hours=2.0,
    )
    assert opt.max_trials == 10
    assert opt.max_hours == 2.0
