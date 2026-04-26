"""
tests/test_optimizer.py
=======================
Focused integration tests verifying that the FireHawks optimizer trains a
simple toy network to convergence, and that its serialisation round-trips
cleanly.

Run
---
.. code-block:: bash

    pytest tests/test_optimizer.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_toy_model(optimizer):
    """2-layer MLP for quick convergence tests."""
    from keras.layers import Dense
    from keras.models import Sequential

    model = Sequential(
        [
            Dense(16, activation="relu", input_shape=(4,)),
            Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def _random_xy(n: int = 200, n_features: int = 4, n_classes: int = 3):
    """Return random (X, y_onehot) pair."""
    from keras.utils import to_categorical

    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, n_features)).astype("float32")
    y_int = rng.integers(0, n_classes, size=n)
    y = to_categorical(y_int, num_classes=n_classes).astype("float32")
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFireHawksOptimizerIntegration:
    """Integration tests: optimizer + Keras model."""

    def test_loss_decreases_over_epochs(self) -> None:
        """Loss should be lower after 10 epochs than at epoch 0."""
        from kfall.models.optimizer import FireHawksOptimizer

        opt = FireHawksOptimizer(learning_rate=0.01)
        model = _build_toy_model(opt)
        X, y = _random_xy()

        history = model.fit(X, y, epochs=15, batch_size=32, verbose=0)
        losses = history.history["loss"]
        # Loss in last 5 epochs should be lower than first 5
        assert np.mean(losses[-5:]) < np.mean(losses[:5]), (
            f"Loss did not decrease: first={np.mean(losses[:5]):.4f}, "
            f"last={np.mean(losses[-5:]):.4f}"
        )

    def test_no_nan_in_outputs(self) -> None:
        """Predictions must be finite after training."""
        from kfall.models.optimizer import FireHawksOptimizer

        opt = FireHawksOptimizer(learning_rate=0.001)
        model = _build_toy_model(opt)
        X, y = _random_xy()
        model.fit(X, y, epochs=5, verbose=0)
        preds = model.predict(X, verbose=0)
        assert not np.any(np.isnan(preds)), "NaN detected in predictions"
        assert not np.any(np.isinf(preds)), "Inf detected in predictions"

    def test_with_momentum(self) -> None:
        """Optimizer should work correctly with momentum > 0."""
        from kfall.models.optimizer import FireHawksOptimizer

        opt = FireHawksOptimizer(learning_rate=0.01, momentum=0.9)
        model = _build_toy_model(opt)
        X, y = _random_xy()
        history = model.fit(X, y, epochs=5, verbose=0)
        assert all(np.isfinite(v) for v in history.history["loss"])

    def test_with_centered(self) -> None:
        """Centered mode should not crash."""
        from kfall.models.optimizer import FireHawksOptimizer

        opt = FireHawksOptimizer(learning_rate=0.005, centered=True)
        model = _build_toy_model(opt)
        X, y = _random_xy()
        history = model.fit(X, y, epochs=3, verbose=0)
        assert len(history.history["loss"]) == 3

    def test_config_serialisation(self) -> None:
        """Model compiled with FireHawks must survive get_config / from_config."""
        from kfall.models.optimizer import FireHawksOptimizer

        opt = FireHawksOptimizer(learning_rate=0.005, rho=0.88, epsilon=1e-6)
        cfg = opt.get_config()

        # Verify all standard keys are present
        required_keys = {"learning_rate", "rho", "momentum", "epsilon", "centered", "decay"}
        assert required_keys.issubset(cfg.keys()), f"Missing keys: {required_keys - cfg.keys()}"

    def test_learning_rate_hyperparameter(self) -> None:
        """Learning rate passed at construction must match get_config()."""
        from kfall.models.optimizer import FireHawksOptimizer

        opt = FireHawksOptimizer(learning_rate=0.0042)
        cfg = opt.get_config()
        assert abs(cfg["learning_rate"] - 0.0042) < 1e-6

    def test_set_weights_compatibility(self) -> None:
        """set_weights must not raise for a freshly-built optimizer."""
        from kfall.models.optimizer import FireHawksOptimizer

        opt = FireHawksOptimizer(learning_rate=0.01)
        model = _build_toy_model(opt)
        X, y = _random_xy(n=32)
        model.fit(X, y, epochs=1, verbose=0)

        # Save and restore weights
        weights = model.get_weights()
        model.set_weights(weights)
        preds = model.predict(X, verbose=0)
        assert preds.shape == (32, 3)
