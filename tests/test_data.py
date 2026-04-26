"""
tests/test_data.py
==================
Unit tests for the data loading and preprocessing pipeline.

These tests use synthetic / in-memory data so they run without requiring
the actual KFall dataset to be downloaded.

Run
---
.. code-block:: bash

    pytest tests/test_data.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path bootstrap (works whether pytest is run from project root or tests/) ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kfall.data.preprocessor import DataPreprocessor
from kfall.config import SplitConfig, FEATURE_COLUMNS, META_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a minimal synthetic DataFrame matching the KFall schema."""
    rng = np.random.default_rng(42)
    n = 500
    n_classes = 5

    data = {
        "SubjectId": [f"S{i % 10}" for i in range(n)],
        "TaskId": rng.integers(1, n_classes + 1, size=n).tolist(),
        "TaskCode": rng.integers(1, n_classes + 1, size=n).tolist(),
        "Description": [f"Activity_{i % n_classes}" for i in range(n)],
        "Class": rng.integers(0, n_classes, size=n).tolist(),
    }
    for col in FEATURE_COLUMNS:
        data[col] = rng.standard_normal(n).tolist()

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# DataPreprocessor tests
# ---------------------------------------------------------------------------

class TestDataPreprocessor:
    """Tests for :class:`kfall.data.preprocessor.DataPreprocessor`."""

    def test_output_shapes(self, sample_df: pd.DataFrame) -> None:
        """fit_transform should return correctly shaped arrays."""
        n_classes = int(sample_df["Class"].nunique())
        prep = DataPreprocessor(split_config=SplitConfig(test_size=0.3))
        train_x, test_x, train_y, test_y = prep.fit_transform(sample_df, n_classes)

        n = len(sample_df)
        n_features = len(FEATURE_COLUMNS)

        assert train_x.ndim == 3, "train_x should be 3-D (N, 1, F)"
        assert train_x.shape[1] == 1, "Time-step dimension should be 1"
        assert train_x.shape[2] == n_features

        assert test_x.ndim == 3
        assert train_x.shape[0] + test_x.shape[0] == n

        assert train_y.ndim == 2
        assert train_y.shape[1] == n_classes

    def test_one_hot_rows_sum_to_one(self, sample_df: pd.DataFrame) -> None:
        """Each one-hot row must sum to exactly 1.0."""
        n_classes = int(sample_df["Class"].nunique())
        prep = DataPreprocessor()
        _, _, train_y, test_y = prep.fit_transform(sample_df, n_classes)

        np.testing.assert_allclose(
            train_y.sum(axis=1),
            np.ones(len(train_y)),
            atol=1e-6,
            err_msg="One-hot rows must sum to 1",
        )

    def test_scaler_zero_mean(self, sample_df: pd.DataFrame) -> None:
        """After StandardScaler the feature means should be ≈ 0."""
        n_classes = int(sample_df["Class"].nunique())
        prep = DataPreprocessor()
        train_x, _, _, _ = prep.fit_transform(sample_df, n_classes)

        means = train_x[:, 0, :].mean(axis=0)
        # Mean won't be exactly 0 on the split subset but should be close
        assert np.all(np.abs(means) < 1.0), "Feature means after scaling are unexpectedly large"

    def test_transform_matches_fit_transform(self, sample_df: pd.DataFrame) -> None:
        """transform() should produce identical values to those from fit_transform."""
        n_classes = int(sample_df["Class"].nunique())
        prep = DataPreprocessor()
        train_x, _, _, _ = prep.fit_transform(sample_df, n_classes)

        raw_X = sample_df[FEATURE_COLUMNS].values.astype("float32")
        # transform() includes the expand_dims step
        # We test on the first 10 rows of the original data
        first10_raw = raw_X[:10]
        transformed = prep.transform(first10_raw)
        assert transformed.shape == (10, 1, len(FEATURE_COLUMNS))

    def test_custom_test_size(self, sample_df: pd.DataFrame) -> None:
        """test_size parameter should be respected."""
        n_classes = int(sample_df["Class"].nunique())
        prep = DataPreprocessor(split_config=SplitConfig(test_size=0.2))
        train_x, test_x, _, _ = prep.fit_transform(sample_df, n_classes)

        expected_test = int(len(sample_df) * 0.2)
        # Allow ±1 for rounding
        assert abs(len(test_x) - expected_test) <= 1

    def test_reproducibility(self, sample_df: pd.DataFrame) -> None:
        """Two preprocessors with the same seed must produce identical splits."""
        n_classes = int(sample_df["Class"].nunique())

        prep1 = DataPreprocessor()
        train_x1, test_x1, _, _ = prep1.fit_transform(sample_df, n_classes)

        prep2 = DataPreprocessor()
        train_x2, test_x2, _, _ = prep2.fit_transform(sample_df, n_classes)

        np.testing.assert_array_equal(train_x1, train_x2)
        np.testing.assert_array_equal(test_x1, test_x2)


# ---------------------------------------------------------------------------
# DataLoader smoke tests (path-validation branch only, no real files needed)
# ---------------------------------------------------------------------------

class TestDataLoaderValidation:
    """Smoke tests for DataLoader path validation (no real dataset needed)."""

    def test_missing_raw_dir_raises(self) -> None:
        """DataLoader.load() must raise FileNotFoundError for a missing dir."""
        from kfall.data.loader import DataLoader

        loader = DataLoader(raw_data_dir=Path("/nonexistent/path"))
        with pytest.raises(FileNotFoundError, match="Raw data directory"):
            loader.load()

    def test_missing_catalogue_raises(self) -> None:
        """DataLoader.load() must raise FileNotFoundError when k_fall.csv is absent."""
        from kfall.data.loader import DataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(
                raw_data_dir=Path(tmpdir),
                catalogue_csv=Path(tmpdir) / "nonexistent.csv",
            )
            with pytest.raises(FileNotFoundError, match="catalogue CSV"):
                loader.load()
