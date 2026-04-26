"""
models/hopfield.py
==================
Custom Keras layer implementing a **Modern Hopfield / Content-Addressable
Attention** mechanism over a recurrent hidden-state sequence.

Background
----------
Classical Hopfield networks store patterns as fixed points of an energy
function.  *Modern* (continuous) Hopfield networks — introduced by Ramsauer
et al. (2020) — use a softmax-based retrieval rule that maps naturally to
the scaled dot-product attention used in Transformers.

Here we implement a *simplified* variant that:

1. Projects every hidden state through a shared linear layer (key / query
   alignment in attention parlance).
2. Extracts the *last* hidden state as the query.
3. Computes a dot-product similarity score between query and projected keys.
4. Applies softmax to produce attention weights.
5. Computes the weighted sum of hidden states (context vector).
6. Concatenates context + query and projects through a ``tanh`` dense layer.

This yields a fixed-size embedding regardless of the sequence length,
which is then fed into the final classification head.

References
----------
Ramsauer, H., et al. (2020). *Hopfield Networks is All You Need*.
arXiv:2008.02217.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import tensorflow as tf
from keras import backend as K, Input
from keras.layers import (
    Activation,
    Concatenate,
    Dense,
    Dot,
    Lambda,
    Layer,
)

logger = logging.getLogger(__name__)


class HopField(Layer):
    """
    Modern HopField attention layer.

    Parameters
    ----------
    units : int
        Dimensionality of the output embedding vector.  Defaults to 128.

    Input shape
    -----------
    ``(batch_size, timesteps, features)`` — the full output sequence of a
    recurrent layer (``return_sequences=True``).

    Output shape
    ------------
    ``(batch_size, units)`` — a single fixed-size context-enriched vector.

    Examples
    --------
    >>> from kfall.models.hopfield import HopField
    >>> import numpy as np
    >>> x = np.random.rand(32, 10, 64).astype("float32")
    >>> layer = HopField(units=32)
    >>> out = layer(x)
    >>> out.shape
    TensorShape([32, 32])
    """

    def __init__(self, units: int = 128, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units
        logger.debug("HopField layer initialised with units=%d", units)

    # ------------------------------------------------------------------
    # Keras Layer lifecycle
    # ------------------------------------------------------------------

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Lazily instantiate sub-layers once the input shape is known.

        All sub-layers are stored as attributes so Keras tracks their
        weights automatically.
        """
        input_dim = int(input_shape[-1])

        with K.name_scope("hopfield"):
            # Project every hidden state: used as keys / values
            self.attention_score_vec = Dense(
                input_dim, use_bias=False, name="attention_score_vec"
            )
            # Extract the last hidden state as the query
            self.h_t = Lambda(
                lambda x: x[:, -1, :],
                output_shape=(input_dim,),
                name="last_hidden_state",
            )
            # Dot product: query ⊙ projected_keys
            self.attention_score = Dot(axes=[1, 2], name="attention_score")
            # Softmax → attention weights
            self.attention_weight = Activation("softmax", name="attention_weight")
            # Weighted sum of hidden states → context vector
            self.context_vector = Dot(axes=[1, 1], name="context_vector")
            # Concatenate context + query
            self.attention_output = Concatenate(name="attention_output")
            # Final projection: [context ‖ query] → units-dim embedding
            self.attention_vector = Dense(
                self.units,
                use_bias=False,
                activation="tanh",
                name="attention_vector",
            )

        super().build(input_shape)

    def call(  # type: ignore[override]
        self, inputs: tf.Tensor, training: Optional[bool] = None, **kwargs
    ) -> tf.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : tf.Tensor  shape (batch, timesteps, features)
        training : bool, optional

        Returns
        -------
        tf.Tensor  shape (batch, units)
        """
        score_first_part = self.attention_score_vec(inputs)
        h_t = self.h_t(inputs)
        score = self.attention_score([h_t, score_first_part])
        attention_weights = self.attention_weight(score)
        context_vector = self.context_vector([inputs, attention_weights])
        pre_activation = self.attention_output([context_vector, h_t])
        attention_vector = self.attention_vector(pre_activation)
        return attention_vector

    def compute_output_shape(
        self, input_shape: tf.TensorShape
    ) -> Tuple[int, int]:
        return (input_shape[0], self.units)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """Return layer config for serialisation / deserialisation."""
        config = super().get_config()
        config.update({"units": self.units})
        return config

    @classmethod
    def from_config(cls, config: dict) -> "HopField":
        return cls(**config)
