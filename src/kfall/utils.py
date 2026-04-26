"""
utils.py
========
Shared utility functions used across the kfall package.

Importing this module as a side-effect sets the global random seed for
Python, NumPy and TensorFlow so that experiments are reproducible.
"""

from __future__ import annotations

import logging
import os
import random
import warnings

import numpy as np

from kfall.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def reset_random(seed: int = RANDOM_SEED) -> None:
    """
    Set global random seeds for Python, NumPy and TensorFlow.

    Parameters
    ----------
    seed : int
        Seed value.  Defaults to ``kfall.config.RANDOM_SEED`` (1).

    Notes
    -----
    * TensorFlow's ``tf.compat.v1.disable_eager_execution()`` is **not**
      called here because eager mode is required for modern Keras 2 / TF2
      workflows.  If you need graph-mode behaviour for an older pipeline,
      call it manually before building the model.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    warnings.filterwarnings("ignore", category=Warning)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
        logger.debug("TensorFlow random seed set to %d", seed)
    except ImportError:
        logger.warning("TensorFlow not available — skipping TF seed.")

    logger.debug("Random seed reset to %d", seed)


# Apply seeds on import so any script that does `from kfall import utils`
# automatically gets a reproducible environment.
reset_random()
