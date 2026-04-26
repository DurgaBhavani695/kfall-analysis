"""
conftest.py
===========
Shared pytest fixtures and session-level configuration.

Fixtures defined here are automatically available to all test modules
without any explicit import.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Silence TF/Keras verbosity during test runs ──────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
warnings.filterwarnings("ignore")

# ── Make src/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


# ---------------------------------------------------------------------------
# Session-scoped fixtures (built once per test session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def feature_columns():
    """Return the ordered list of IMU sensor feature column names."""
    from kfall.config import FEATURE_COLUMNS
    return FEATURE_COLUMNS


@pytest.fixture(scope="session")
def n_features(feature_columns):
    return len(feature_columns)


@pytest.fixture(scope="session")
def n_classes():
    """Default number of activity classes used in toy tests."""
    return 5


@pytest.fixture(scope="session")
def sample_dataframe(feature_columns, n_classes):
    """
    A synthetic DataFrame that mirrors the KFall merged schema.
    Built once per session for speed.
    """
    rng = np.random.default_rng(42)
    n = 400

    data = {
        "SubjectId": [f"S{i % 8}" for i in range(n)],
        "TaskId": rng.integers(1, n_classes + 1, size=n).tolist(),
        "TaskCode": rng.integers(1, n_classes + 1, size=n).tolist(),
        "Description": [f"Activity_{i % n_classes}" for i in range(n)],
        "Class": rng.integers(0, n_classes, size=n).tolist(),
    }
    for col in feature_columns:
        data[col] = rng.standard_normal(n).tolist()

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def tiny_model_session(n_classes):
    """
    A very small CRHNN model shared across the whole test session.
    Scope='session' avoids rebuilding the TF graph for every test class.
    """
    from kfall.config import ModelConfig, TrainingConfig
    from kfall.models.crhnn import build_crhnn

    model_cfg = ModelConfig(
        cnn_filters=8,
        cnn_kernel_size=3,
        lstm_units=[4, 4],
        hopfield_units=4,
    )
    training_cfg = TrainingConfig(learning_rate=0.001)
    return build_crhnn((1, 9), n_classes, model_cfg=model_cfg, training_cfg=training_cfg)


@pytest.fixture(scope="session")
def random_batch_session(n_classes):
    """Random (X, y_onehot) batch, built once per session."""
    from keras.utils import to_categorical
    rng = np.random.default_rng(0)
    X = rng.standard_normal((32, 1, 9)).astype("float32")
    y_int = rng.integers(0, n_classes, size=32)
    y = to_categorical(y_int, num_classes=n_classes).astype("float32")
    return X, y


# ---------------------------------------------------------------------------
# Function-scoped fixtures (fresh per test)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Provide a fresh temporary directory for output artefacts."""
    out = tmp_path / "outputs"
    out.mkdir()
    return out
