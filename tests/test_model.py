"""
tests/test_model.py
===================
Unit tests for the CRHNN model, HopField layer and training callback.

All tests build toy models on CPU in eager mode — no GPU or dataset required.

Run
---
.. code-block:: bash

    pytest tests/test_model.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Path bootstrap ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Silence TF/Keras output during tests
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_CLASSES = 5
INPUT_SHAPE = (1, 9)  # (timesteps, features)
BATCH = 16


@pytest.fixture(scope="module")
def tiny_model():
    """Build a minimal CRHNN for fast CPU tests."""
    from kfall.config import ModelConfig, TrainingConfig
    from kfall.models.crhnn import build_crhnn

    model_cfg = ModelConfig(
        cnn_filters=8,
        cnn_kernel_size=3,
        lstm_units=[4, 4],
        hopfield_units=4,
    )
    training_cfg = TrainingConfig(learning_rate=0.001, epochs=2, batch_size=8)
    return build_crhnn(INPUT_SHAPE, N_CLASSES, model_cfg=model_cfg, training_cfg=training_cfg)


@pytest.fixture
def random_batch():
    """Return (X, y_onehot) tensors of shape (BATCH, 1, 9) and (BATCH, N_CLASSES)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((BATCH, *INPUT_SHAPE)).astype("float32")
    y_int = rng.integers(0, N_CLASSES, size=BATCH)
    from keras.utils import to_categorical
    y = to_categorical(y_int, num_classes=N_CLASSES).astype("float32")
    return X, y


# ---------------------------------------------------------------------------
# HopField tests
# ---------------------------------------------------------------------------

class TestHopField:
    """Tests for the HopField attention layer."""

    def test_output_shape(self) -> None:
        """HopField output must be (batch, units)."""
        from kfall.models.hopfield import HopField

        layer = HopField(units=16)
        x = np.random.rand(8, 10, 32).astype("float32")
        out = layer(x)
        assert out.shape == (8, 16), f"Expected (8,16) got {out.shape}"

    def test_serialisation_roundtrip(self) -> None:
        """get_config / from_config must reproduce the layer."""
        from kfall.models.hopfield import HopField

        layer = HopField(units=32, name="test_hop")
        config = layer.get_config()
        restored = HopField.from_config(config)
        assert restored.units == 32

    def test_different_seq_lengths(self) -> None:
        """HopField should handle any sequence length > 0."""
        from kfall.models.hopfield import HopField

        for seq_len in [1, 5, 20]:
            layer = HopField(units=8)
            x = np.random.rand(4, seq_len, 16).astype("float32")
            out = layer(x)
            assert out.shape[0] == 4 and out.shape[1] == 8


# ---------------------------------------------------------------------------
# CRHNN model tests
# ---------------------------------------------------------------------------

class TestBuildCRHNN:
    """Tests for the build_crhnn() factory function."""

    def test_model_name(self, tiny_model) -> None:
        assert tiny_model.name == "CRHNN"

    def test_input_shape(self, tiny_model) -> None:
        assert tiny_model.input_shape == (None, *INPUT_SHAPE)

    def test_output_shape(self, tiny_model) -> None:
        assert tiny_model.output_shape == (None, N_CLASSES)

    def test_output_sums_to_one(self, tiny_model, random_batch) -> None:
        """Softmax outputs must sum to 1 per sample."""
        X, _ = random_batch
        probs = tiny_model.predict(X, verbose=0)
        np.testing.assert_allclose(
            probs.sum(axis=1),
            np.ones(BATCH),
            atol=1e-5,
            err_msg="Softmax outputs don't sum to 1",
        )

    def test_output_probabilities_in_range(self, tiny_model, random_batch) -> None:
        """All predicted probabilities must be in [0, 1]."""
        X, _ = random_batch
        probs = tiny_model.predict(X, verbose=0)
        assert probs.min() >= -1e-6
        assert probs.max() <= 1.0 + 1e-6

    def test_forward_pass_no_nan(self, tiny_model, random_batch) -> None:
        """Forward pass must not produce NaN values."""
        X, _ = random_batch
        probs = tiny_model.predict(X, verbose=0)
        assert not np.any(np.isnan(probs)), "NaN detected in model output"

    def test_one_training_step(self, tiny_model, random_batch) -> None:
        """Model should be trainable (loss decreases or at least doesn't crash)."""
        X, y = random_batch
        history = tiny_model.fit(X, y, epochs=1, verbose=0)
        assert "loss" in history.history
        assert history.history["loss"][0] > 0

    def test_model_layer_count(self, tiny_model) -> None:
        """Sanity-check that the expected named layers are present."""
        layer_names = {l.name for l in tiny_model.layers}
        assert "conv1d_stem" in layer_names
        assert "hopfield_attention" in layer_names
        assert "class_output" in layer_names


# ---------------------------------------------------------------------------
# FireHawks optimizer tests
# ---------------------------------------------------------------------------

class TestFireHawksOptimizer:
    """Tests for the custom FireHawks gradient-descent optimizer."""

    def test_default_config(self) -> None:
        """Default hyperparameters should be set correctly."""
        from kfall.models.optimizer import FireHawksOptimizer

        opt = FireHawksOptimizer(learning_rate=0.005)
        cfg = opt.get_config()
        assert cfg["learning_rate"] == pytest.approx(0.005, rel=1e-4)
        assert cfg["rho"] == pytest.approx(0.9, rel=1e-4)

    def test_invalid_momentum_raises(self) -> None:
        """Momentum outside [0, 1] should raise ValueError."""
        from kfall.models.optimizer import FireHawksOptimizer

        with pytest.raises(ValueError, match="momentum"):
            FireHawksOptimizer(momentum=-0.1)

    def test_get_config_roundtrip(self) -> None:
        """Serialised config must contain all required keys."""
        from kfall.models.optimizer import FireHawksOptimizer

        opt = FireHawksOptimizer(learning_rate=0.01, rho=0.85, momentum=0.1)
        cfg = opt.get_config()
        for key in ("learning_rate", "rho", "momentum", "epsilon", "centered"):
            assert key in cfg, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# TrainingCallback tests
# ---------------------------------------------------------------------------

class TestTrainingCallback:
    """Tests for the per-epoch CSV logging callback."""

    def test_csv_created_on_init(self) -> None:
        from kfall.training.callbacks import TrainingCallback

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "acc_loss.csv"
            cb = TrainingCallback(csv_path)
            assert csv_path.exists()

    def test_epoch_appended_to_csv(self) -> None:
        from kfall.training.callbacks import TrainingCallback

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "acc_loss.csv"
            cb = TrainingCallback(csv_path)
            cb.on_epoch_end(0, logs={"accuracy": 0.5, "val_accuracy": 0.4, "loss": 1.2, "val_loss": 1.3})
            cb.on_epoch_end(1, logs={"accuracy": 0.6, "val_accuracy": 0.5, "loss": 1.0, "val_loss": 1.1})
            df = pd.read_csv(csv_path)
            assert len(df) == 2
            assert df["accuracy"].iloc[0] == pytest.approx(0.5)

    def test_resume_from_existing_csv(self) -> None:
        from kfall.training.callbacks import TrainingCallback

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "acc_loss.csv"

            # Pre-populate
            cb1 = TrainingCallback(csv_path)
            cb1.on_epoch_end(0, logs={"accuracy": 0.7, "val_accuracy": 0.65, "loss": 0.9, "val_loss": 1.0})

            # Resume
            cb2 = TrainingCallback(csv_path)
            assert len(cb2.df) == 1, "Resume should load existing rows"
