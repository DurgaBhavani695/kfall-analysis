"""
data/preprocessor.py
====================
Transforms the raw merged DataFrame into train/test numpy arrays that are
ready to feed directly into the CRHNN model.

Pipeline
--------
1. Extract feature matrix ``X`` and integer label vector ``y``.
2. Apply :class:`sklearn.preprocessing.StandardScaler` column-wise.
3. Stratified train / test split (default 70 / 30).
4. Expand a time-step dimension (``shape [N, 9] → [N, 1, 9]``).
5. One-hot encode labels with :func:`keras.utils.to_categorical`.

Usage
-----
>>> from kfall.data.preprocessor import DataPreprocessor
>>> prep = DataPreprocessor()
>>> splits = prep.fit_transform(df, n_classes=20)
>>> train_x, test_x, train_y_cat, test_y_cat = splits
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kfall.config import FEATURE_COLUMNS, RANDOM_SEED, SplitConfig, default_config

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Stateful preprocessor: fits a :class:`StandardScaler` on training data and
    applies it to both splits.  Call :meth:`fit_transform` for a fresh run or
    :meth:`transform` to reuse a fitted scaler on new data.

    Parameters
    ----------
    split_config : SplitConfig, optional
        Controls test fraction and shuffle behaviour.

    Attributes
    ----------
    scaler : StandardScaler
        Fitted scaler; only available after :meth:`fit_transform` is called.
    n_classes : int
        Number of output classes; set during :meth:`fit_transform`.

    Examples
    --------
    >>> prep = DataPreprocessor()
    >>> train_x, test_x, train_y, test_y = prep.fit_transform(df, n_classes=20)
    >>> train_x.shape
    (N_train, 1, 9)
    """

    def __init__(self, split_config: Optional[SplitConfig] = None) -> None:
        self.split_config: SplitConfig = split_config or default_config.split
        self.scaler: StandardScaler = StandardScaler()
        self.n_classes: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame, n_classes: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline on a raw DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Output of :class:`~kfall.data.loader.DataLoader`.
        n_classes : int
            Total number of activity / fall classes.

        Returns
        -------
        train_x : np.ndarray  shape (N_train, 1, n_features)
        test_x  : np.ndarray  shape (N_test,  1, n_features)
        train_y : np.ndarray  shape (N_train, n_classes)   one-hot
        test_y  : np.ndarray  shape (N_test,  n_classes)   one-hot
        """
        self.n_classes = n_classes

        X, y = self._extract_arrays(df)

        logger.info("Fitting StandardScaler on full dataset (%d samples).", len(X))
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        logger.info(
            "Splitting data — test_size=%.0f%%.", self.split_config.test_size * 100
        )
        train_x_raw, test_x_raw, train_y_raw, test_y_raw = train_test_split(
            X_scaled,
            y,
            test_size=self.split_config.test_size,
            shuffle=self.split_config.shuffle,
            random_state=RANDOM_SEED,
        )

        train_x = self._add_timestep_dim(train_x_raw)
        test_x = self._add_timestep_dim(test_x_raw)

        train_y = self._one_hot(train_y_raw)
        test_y = self._one_hot(test_y_raw)

        logger.info(
            "Preprocessing complete — train: %s | test: %s",
            train_x.shape,
            test_x.shape,
        )
        return train_x, test_x, train_y, test_y

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the fitted scaler and add the time-step dimension to new data.

        Parameters
        ----------
        X : np.ndarray  shape (N, n_features)
            Raw sensor features (same columns as training data).

        Returns
        -------
        np.ndarray  shape (N, 1, n_features)
        """
        return self._add_timestep_dim(self.scaler.transform(X))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_arrays(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and integer label vector from ``df``."""
        X = df[FEATURE_COLUMNS].values.astype(np.float32)
        y = df["Class"].values.astype(int)
        return X, y

    @staticmethod
    def _add_timestep_dim(X: np.ndarray) -> np.ndarray:
        """Insert a singleton time-step axis: ``(N, F) → (N, 1, F)``."""
        return np.expand_dims(X, axis=1)

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        """Convert integer labels to one-hot encoding."""
        # Avoid importing keras at module level to keep this file lightweight
        from keras.utils import to_categorical  # type: ignore

        return to_categorical(y, num_classes=self.n_classes)
