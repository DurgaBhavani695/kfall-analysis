"""
config.py
=========
Centralised configuration for the KFall-Analysis project.

All hyper-parameters, path defaults and environment toggles live here.
Import this module wherever you need a setting rather than hard-coding
magic numbers throughout the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Project-level paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
RESULTS_DIR: Path = OUTPUTS_DIR / "results"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"

# Auto-create output directories so the user doesn't have to
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Sensor / dataset constants
# ---------------------------------------------------------------------------

#: Columns produced after merging label + sensor data
FEATURE_COLUMNS: List[str] = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "EulerX", "EulerY", "EulerZ"]

#: Metadata columns (not used as model inputs)
META_COLUMNS: List[str] = ["SubjectId", "TaskId", "TaskCode", "Description", "Class"]

#: Name of the merged/processed CSV produced by the data-prep pipeline
PROCESSED_CSV: Path = PROCESSED_DATA_DIR / "data.csv"

#: KFall activity-catalogue CSV (must be placed in data/raw)
KFALL_CATALOGUE_CSV: Path = RAW_DATA_DIR / "k_fall.csv"

#: Sub-directories inside data/raw that the loader expects
SENSOR_DATA_SUBDIR: str = "sensor_data"
LABEL_DATA_SUBDIR: str = "label_data"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 1


# ---------------------------------------------------------------------------
# Data-splitting
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    """Configuration for train / test splitting."""

    test_size: float = 0.30
    """Fraction of data held out for testing."""

    shuffle: bool = True
    """Whether to shuffle before splitting."""


# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Architecture hyper-parameters for the CRHNN model."""

    cnn_filters: int = 128
    """Number of filters in the Conv1D stem layer."""

    cnn_kernel_size: int = 3
    """Kernel width for the Conv1D stem."""

    lstm_units: List[int] = field(default_factory=lambda: [64, 32, 16])
    """Hidden-unit counts for the three stacked Bi-LSTM layers."""

    hopfield_units: int = 8
    """Output dimension of the HopField attention layer."""

    dropout_rate: float = 0.0
    """Optional dropout between LSTM layers (0 = disabled)."""


# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training loop configuration."""

    epochs: int = 800
    batch_size: int = 1_000
    learning_rate: float = 0.001
    rho: float = 0.9
    momentum: float = 0.0
    epsilon: float = 1e-7
    centered: bool = False

    monitor_metric: str = "accuracy"
    """Metric watched by ModelCheckpoint (save best only)."""

    validation_split: float = 0.0
    """If > 0, use a fraction of *training* data for validation.
    When explicit val data is supplied this is ignored."""


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """Options for the evaluation / reporting stage."""

    splits: List[str] = field(default_factory=lambda: ["Train", "Test"])
    """Which data-splits to evaluate."""

    save_plots: bool = True
    """Whether to write confusion-matrix / PR / ROC figures to disk."""


# ---------------------------------------------------------------------------
# Master config object
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """
    Master configuration object.

    Usage
    -----
    >>> from kfall.config import Config
    >>> cfg = Config()
    >>> print(cfg.training.epochs)
    800
    """

    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Paths (re-exported for convenience)
    data_dir: Path = DATA_DIR
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    models_dir: Path = MODELS_DIR
    results_dir: Path = RESULTS_DIR
    logs_dir: Path = LOGS_DIR


# Singleton-style default config
default_config = Config()
