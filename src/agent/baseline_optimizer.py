"""
src/agent/baseline_optimizer.py

AutoBaselineAgent — agentic hyperparameter and feature search for
audio/visual baseline classifiers.

STATUS: Stub — architecture defined, implementation deferred to Phase 8.

Architecture:
    Uses an agentic loop (OpenClaw or similar long-running framework) to
    autonomously explore feature extraction configurations and classifier
    hyperparameters. Logs every attempt to experiments.csv via the shared
    tool library.

Perceive → Reason → Act → Memory loop:
    Perceive:  read experiments.csv — what has already been tried?
    Reason:    LLM decides which configuration to try next based on
               results so far (Bayesian-style reasoning over the search space)
    Act:       extract features, train classifier, evaluate on val set
    Memory:    append result to experiments.csv, update entity store

Search space (audio):
    features:    MFCC, MFCC+chroma, MFCC+spectral_contrast,
                 MFCC+chroma+spectral_contrast+zcr
    n_mfcc:      [20, 40, 80, 128]
    classifiers: KNN (k=1..15), SVM (C=0.1..100, rbf/linear),
                 LogisticRegression (C=0.1..10)
    normalization: per-utterance, global StandardScaler

Search space (visual):
    features:    HOG (varying orientations, cell sizes), color hist,
                 LBP, Gabor filters, combinations
    classifiers: SVM, LogReg, RandomForest, GradientBoosting
    input_size:  64x64, 128x128, 224x224

Stopping criteria:
    - Val F1 improvement < 0.005 for 3 consecutive trials (plateau)
    - Budget: max_trials parameter (default 20)
    - Time: max_hours parameter (default 4)

Framework: OpenClaw (planned) or custom loop
    OpenClaw suits this use case better than LangGraph because:
    - Long-running (hours not seconds)
    - No conversational state needed
    - Tool loop is the primary pattern, not state machine transitions
    - Native support for budget/stopping criteria

Usage (future):
    optimizer = BaselineOptimizer.from_config("configs/")
    result = optimizer.optimize(
        modality="audio",
        splits_dir="data/splits/",
        max_trials=20,
        max_hours=4,
    )
    print(f"Best config: {result.best_config}")
    print(f"Best val F1: {result.best_val_f1:.3f}")

Related:
    src/agent/tools/calibration_tools.py — proves the pattern works
    (run_fusion_weight_sweep is a working example of agent-callable
    parameter search that writes results to the observation log)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BaselineOptimizer:
    """
    Autonomous hyperparameter and feature search agent.

    STATUS: Stub — not yet implemented.
    See module docstring for architecture and planned implementation.
    """

    def __init__(
        self,
        observations_path: str,
        splits_dir: str,
        models_dir: str,
        experiments_csv: str,
        provider: str = "gemini",
        model: str = "gemini-2.5-flash",
        max_trials: int = 20,
        max_hours: float = 4.0,
    ) -> None:
        self.observations_path = observations_path
        self.splits_dir = Path(splits_dir)
        self.models_dir = Path(models_dir)
        self.experiments_csv = Path(experiments_csv)
        self.provider = provider
        self.model = model
        self.max_trials = max_trials
        self.max_hours = max_hours
        logger.info(
            "BaselineOptimizer initialized (stub) | modality=TBD "
            "max_trials=%d max_hours=%.1f",
            max_trials, max_hours,
        )

    @classmethod
    def from_config(cls, config_dir: str | Path) -> BaselineOptimizer:
        """Construct from configs/ directory. Not yet implemented."""
        raise NotImplementedError(
            "BaselineOptimizer.from_config() is not yet implemented. "
            "See module docstring for planned architecture. "
            "Current workaround: run audio_baseline.ipynb manually with "
            "different N_MFCC and feature combinations."
        )

    def optimize(
        self,
        modality: str,
        max_trials: int | None = None,
        max_hours: float | None = None,
    ) -> Any:
        """
        Run autonomous feature/hyperparameter search. Not yet implemented.

        Planned behavior:
            1. Read experiments.csv to find already-tried configurations
            2. LLM reasons about which configuration to try next
            3. Extract features, train, evaluate on val set
            4. Append result to experiments.csv
            5. Repeat until stopping criteria met
            6. Return best configuration found

        Args:
            modality:   "audio" or "visual"
            max_trials: Override instance max_trials
            max_hours:  Override instance max_hours

        Returns:
            OptimizationResult with best_config, best_val_f1, all_trials

        Raises:
            NotImplementedError: Always, until implemented.
        """
        raise NotImplementedError(
            "BaselineOptimizer.optimize() is not yet implemented. "
            "Planned framework: OpenClaw for long-running agentic loops. "
            "See module docstring for full architecture."
        )
