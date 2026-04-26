"""
models/crhnn.py
===============
**Conv-Recurrent HopField Neural Network (CRHNN)**

Architecture overview
---------------------

.. code-block:: text

    Input (batch, 1, 9)
         │
    Conv1D(128, k=3, pad='same', relu)      ← temporal feature extraction
         │
    BiLSTM(64, return_sequences=True)        ┐
    BiLSTM(32, return_sequences=True)        │ stacked recurrent layers
    BiLSTM(16, return_sequences=True)        ┘
         │
    HopField(units=8)                        ← content-addressable attention
         │
    Dense(n_classes, softmax)               ← classification head

Design rationale
----------------
* **Conv1D stem**: A single 1-D convolutional layer captures short-range
  temporal correlations across the 9 IMU channels before the recurrent stack.

* **Bidirectional LSTM stack**: Three stacked Bi-LSTMs with decreasing hidden
  sizes (64 → 32 → 16) gradually compress the representation while retaining
  both past and future context.  Using ``return_sequences=True`` lets the
  HopField layer attend over *all* timesteps rather than just the last one.

* **HopField attention**: The Modern Hopfield layer replaces a naive global
  average or last-state pooling with a content-addressable retrieval that
  highlights the most *relevant* hidden states for each sample.

* **FireHawks Optimizer**: An RMSProp variant with bio-inspired naming,
  providing adaptive learning-rate scaling per weight.

Usage
-----
>>> from kfall.models.crhnn import build_crhnn
>>> model = build_crhnn(input_shape=(1, 9), n_classes=20)
>>> model.summary()
"""

from __future__ import annotations

import logging
from typing import Tuple

from keras import Input
from keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    LSTM,
)
from keras.models import Model

from kfall.config import ModelConfig, TrainingConfig, default_config
from kfall.models.hopfield import HopField
from kfall.models.optimizer import FireHawksOptimizer

logger = logging.getLogger(__name__)


def build_crhnn(
    input_shape: Tuple[int, int],
    n_classes: int,
    model_cfg: ModelConfig | None = None,
    training_cfg: TrainingConfig | None = None,
) -> Model:
    """
    Construct, compile and return a CRHNN Keras model.

    Parameters
    ----------
    input_shape : tuple of int
        ``(timesteps, features)`` — typically ``(1, 9)`` for the KFall data
        with a singleton time-step dimension.
    n_classes : int
        Number of output classes (fall / ADL categories).
    model_cfg : ModelConfig, optional
        Architecture hyper-parameters.  Defaults to ``default_config.model``.
    training_cfg : TrainingConfig, optional
        Optimiser hyper-parameters.  Defaults to ``default_config.training``.

    Returns
    -------
    keras.Model
        Compiled Keras model ready for ``.fit()``.

    Examples
    --------
    >>> model = build_crhnn(input_shape=(1, 9), n_classes=20)
    >>> model.input_shape
    (None, 1, 9)
    """
    model_cfg = model_cfg or default_config.model
    training_cfg = training_cfg or default_config.training

    logger.info("Building CRHNN — input_shape=%s, n_classes=%d", input_shape, n_classes)

    # ── Input ──────────────────────────────────────────────────────────────
    inputs = Input(shape=input_shape, name="sensor_input")

    # ── Conv1D stem ────────────────────────────────────────────────────────
    x = Conv1D(
        filters=model_cfg.cnn_filters,
        kernel_size=model_cfg.cnn_kernel_size,
        padding="same",
        activation="relu",
        name="conv1d_stem",
    )(inputs)

    # ── Stacked Bidirectional LSTMs ────────────────────────────────────────
    for i, units in enumerate(model_cfg.lstm_units):
        x = Bidirectional(
            LSTM(units, return_sequences=True),
            name=f"bi_lstm_{i + 1}",
        )(x)
        if model_cfg.dropout_rate > 0:
            x = Dropout(model_cfg.dropout_rate, name=f"dropout_{i + 1}")(x)

    # ── HopField Attention ─────────────────────────────────────────────────
    x = HopField(units=model_cfg.hopfield_units, name="hopfield_attention")(x)

    # ── Classification head ────────────────────────────────────────────────
    outputs = Dense(n_classes, activation="softmax", name="class_output")(x)

    # ── Compile ────────────────────────────────────────────────────────────
    model = Model(inputs=inputs, outputs=outputs, name="CRHNN")

    optimizer = FireHawksOptimizer(
        learning_rate=training_cfg.learning_rate,
        rho=training_cfg.rho,
        momentum=training_cfg.momentum,
        epsilon=training_cfg.epsilon,
        centered=training_cfg.centered,
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("CRHNN compiled successfully.")
    model.summary(print_fn=logger.info)
    return model
