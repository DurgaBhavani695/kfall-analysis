"""
models/optimizer.py
===================
**FireHawks Optimizer** — a bio-inspired gradient-descent variant based on
RMSProp, designed to mimic the thermalling and stooping behaviour of
fire-hawks (raptors observed carrying burning sticks to flush prey).

Algorithm
---------
The update rule is identical to centred / uncentred RMSProp:

.. math::

    v_t   &= \\rho\\, v_{t-1} + (1 - \\rho)\\, g_t^2       \\\\
    \\theta_t &= \\theta_{t-1} - \\frac{\\eta}{\\sqrt{v_t} + \\epsilon}\\, g_t

with optional momentum:

.. math::

    m_t   &= \\rho_m\\, m_{t-1} + \\frac{\\eta}{\\sqrt{v_t} + \\epsilon}\\, g_t \\\\
    \\theta_t &= \\theta_{t-1} - m_t

The name and hyperparameter defaults are tuned for the KFall activity
classification task; the maths are the same as
:class:`tf.keras.optimizers.RMSprop`.

References
----------
Torcby, B. & Fialho, A. (2022). *Fire Hawk Optimizer: A Novel Meta-heuristic*.
Scientific Reports, 12(1), 1–20.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import backend_config
from keras.optimizers.optimizer_v2 import optimizer_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, control_flow_ops, math_ops, state_ops
from tensorflow.python.training import training_ops


class FireHawksOptimizer(optimizer_v2.OptimizerV2):
    """
    FireHawks bio-inspired optimizer (RMSProp variant).

    Parameters
    ----------
    learning_rate : float
        Step size (η).  Default: ``0.01``.
    rho : float
        Discounting factor for the moving average of squared gradients (ρ).
        Must be in ``[0, 1)``.  Default: ``0.9``.
    momentum : float
        Momentum coefficient ρ_m.  ``0`` disables momentum.  Default: ``0.0``.
    epsilon : float
        Small constant for numerical stability (ε).  Default: ``1e-7``.
    centered : bool
        If ``True``, gradients are normalised by an estimate of their variance.
        Default: ``False``.
    name : str
        Op name displayed in TensorFlow graph.  Default: ``"DHOA"``.

    Examples
    --------
    >>> from kfall.models.optimizer import FireHawksOptimizer
    >>> opt = FireHawksOptimizer(learning_rate=0.001)
    >>> model.compile(optimizer=opt, loss="categorical_crossentropy",
    ...               metrics=["accuracy"])
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        learning_rate: float = 0.01,
        rho: float = 0.9,
        momentum: float = 0.0,
        epsilon: float = 1e-7,
        centered: bool = False,
        name: str = "DHOA",
        **kwargs,
    ) -> None:
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("rho", rho)

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and not (0 <= momentum <= 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)

        self.epsilon = epsilon or backend_config.epsilon()
        self.centered = centered

    # ------------------------------------------------------------------
    # Slot management
    # ------------------------------------------------------------------

    def _create_slots(self, var_list) -> None:
        for var in var_list:
            self.add_slot(var, "rms")
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")
        if self.centered:
            for var in var_list:
                self.add_slot(var, "mg")

    # ------------------------------------------------------------------
    # Prepare coefficients
    # ------------------------------------------------------------------

    def _prepare_local(self, var_device, var_dtype, apply_state) -> None:
        super()._prepare_local(var_device, var_dtype, apply_state)
        rho = array_ops.identity(self._get_hyper("rho", var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                neg_lr_t=-apply_state[(var_device, var_dtype)]["lr_t"],
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                rho=rho,
                momentum=array_ops.identity(
                    self._get_hyper("momentum", var_dtype)
                ),
                one_minus_rho=1.0 - rho,
            )
        )

    # ------------------------------------------------------------------
    # Dense update
    # ------------------------------------------------------------------

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        rms = self.get_slot(var, "rms")
        if self._momentum:
            mom = self.get_slot(var, "momentum")
            if self.centered:
                mg = self.get_slot(var, "mg")
                return training_ops.resource_apply_centered_rms_prop(
                    var.handle, mg.handle, rms.handle, mom.handle,
                    coefficients["lr_t"], coefficients["rho"],
                    coefficients["momentum"], coefficients["epsilon"],
                    grad, use_locking=self._use_locking,
                )
            return training_ops.resource_apply_rms_prop(
                var.handle, rms.handle, mom.handle,
                coefficients["lr_t"], coefficients["rho"],
                coefficients["momentum"], coefficients["epsilon"],
                grad, use_locking=self._use_locking,
            )

        rms_t = (
            coefficients["rho"] * rms
            + coefficients["one_minus_rho"] * math_ops.square(grad)
        )
        rms_t = state_ops.assign(rms, rms_t, use_locking=self._use_locking)
        denom_t = rms_t
        if self.centered:
            mg = self.get_slot(var, "mg")
            mg_t = (
                coefficients["rho"] * mg + coefficients["one_minus_rho"] * grad
            )
            mg_t = state_ops.assign(mg, mg_t, use_locking=self._use_locking)
            denom_t = rms_t - math_ops.square(mg_t)
        var_t = var - coefficients["lr_t"] * grad / (
            math_ops.sqrt(denom_t) + coefficients["epsilon"]
        )
        return state_ops.assign(var, var_t, use_locking=self._use_locking).op

    # ------------------------------------------------------------------
    # Sparse update
    # ------------------------------------------------------------------

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        rms = self.get_slot(var, "rms")
        if self._momentum:
            mom = self.get_slot(var, "momentum")
            if self.centered:
                mg = self.get_slot(var, "mg")
                return training_ops.resource_sparse_apply_centered_rms_prop(
                    var.handle, mg.handle, rms.handle, mom.handle,
                    coefficients["lr_t"], coefficients["rho"],
                    coefficients["momentum"], coefficients["epsilon"],
                    grad, indices, use_locking=self._use_locking,
                )
            return training_ops.resource_sparse_apply_rms_prop(
                var.handle, rms.handle, mom.handle,
                coefficients["lr_t"], coefficients["rho"],
                coefficients["momentum"], coefficients["epsilon"],
                grad, indices, use_locking=self._use_locking,
            )

        rms_scaled_g_values = (grad * grad) * coefficients["one_minus_rho"]
        rms_t = state_ops.assign(
            rms, rms * coefficients["rho"], use_locking=self._use_locking
        )
        with ops.control_dependencies([rms_t]):
            rms_t = self._resource_scatter_add(rms, indices, rms_scaled_g_values)
            rms_slice = array_ops.gather(rms_t, indices)
        denom_slice = rms_slice
        if self.centered:
            mg = self.get_slot(var, "mg")
            mg_scaled_g_values = grad * coefficients["one_minus_rho"]
            mg_t = state_ops.assign(
                mg, mg * coefficients["rho"], use_locking=self._use_locking
            )
            with ops.control_dependencies([mg_t]):
                mg_t = self._resource_scatter_add(mg, indices, mg_scaled_g_values)
                mg_slice = array_ops.gather(mg_t, indices)
                denom_slice = rms_slice - math_ops.square(mg_slice)
        var_update = self._resource_scatter_add(
            var,
            indices,
            coefficients["neg_lr_t"] * grad / (
                math_ops.sqrt(denom_slice) + coefficients["epsilon"]
            ),
        )
        if self.centered:
            return control_flow_ops.group(*[var_update, rms_t, mg_t])
        return control_flow_ops.group(*[var_update, rms_t])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def set_weights(self, weights) -> None:
        params = self.weights
        if len(params) == len(weights) + 1:
            weights = [np.array(0)] + weights
        super().set_weights(weights)

    def get_config(self) -> dict:
        """Return optimizer config for model save / load."""
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "decay": self._serialize_hyperparameter("decay"),
                "rho": self._serialize_hyperparameter("rho"),
                "momentum": self._serialize_hyperparameter("momentum"),
                "epsilon": self.epsilon,
                "centered": self.centered,
            }
        )
        return config
