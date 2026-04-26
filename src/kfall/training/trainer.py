"""
training/trainer.py
===================
High-level :class:`Trainer` that orchestrates the full training loop:

1. Builds / loads the CRHNN model.
2. Attaches :class:`~kfall.training.callbacks.TrainingCallback` and
   :class:`keras.callbacks.ModelCheckpoint`.
3. Calls ``model.fit()`` with resume support (initial_epoch from CSV).
4. Returns the trained model and training history.

Usage
-----
>>> from kfall.training.trainer import Trainer
>>> trainer = Trainer(n_classes=20)
>>> model = trainer.fit(train_x, train_y, val_x, val_y)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.models import Model

from kfall.config import MODELS_DIR, TrainingConfig, ModelConfig, default_config
from kfall.models.crhnn import build_crhnn
from kfall.training.callbacks import TrainingCallback

logger = logging.getLogger(__name__)


class Trainer:
    """
    Manages the CRHNN training lifecycle.

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    model_cfg : ModelConfig, optional
        Architecture config.  Defaults to ``default_config.model``.
    training_cfg : TrainingConfig, optional
        Training hyper-parameters.  Defaults to ``default_config.training``.
    models_dir : Path, optional
        Directory where ``model.h5`` and ``acc_loss.csv`` are saved.

    Examples
    --------
    >>> trainer = Trainer(n_classes=20)
    >>> model = trainer.fit(train_x, train_y_cat, val_x=test_x, val_y=test_y_cat)
    """

    def __init__(
        self,
        n_classes: int,
        model_cfg: Optional[ModelConfig] = None,
        training_cfg: Optional[TrainingConfig] = None,
        models_dir: Path = MODELS_DIR,
    ) -> None:
        self.n_classes = n_classes
        self.model_cfg = model_cfg or default_config.model
        self.training_cfg = training_cfg or default_config.training
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.models_dir / "model.h5"
        self.acc_loss_csv = self.models_dir / "acc_loss.csv"
        self.model: Optional[Model] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> Model:
        """
        Train (or resume training) the CRHNN model.

        Parameters
        ----------
        train_x : np.ndarray  shape (N, 1, 9)
        train_y : np.ndarray  shape (N, n_classes)  — one-hot
        val_x   : np.ndarray, optional  shape (M, 1, 9)
        val_y   : np.ndarray, optional  shape (M, n_classes)

        Returns
        -------
        keras.Model
            The trained Keras model.
        """
        input_shape = train_x.shape[1:]
        self.model = build_crhnn(
            input_shape=input_shape,
            n_classes=self.n_classes,
            model_cfg=self.model_cfg,
            training_cfg=self.training_cfg,
        )

        initial_epoch = self._maybe_resume()

        callbacks = [
            TrainingCallback(self.acc_loss_csv),
            ModelCheckpoint(
                str(self.model_path),
                save_best_only=True,
                save_weights_only=True,
                monitor=self.training_cfg.monitor_metric,
                mode="max",
                verbose=0,
            ),
        ]

        validation_data = (val_x, val_y) if val_x is not None else None

        logger.info(
            "Starting training — epochs=%d, batch_size=%d, initial_epoch=%d",
            self.training_cfg.epochs,
            self.training_cfg.batch_size,
            initial_epoch,
        )

        self.model.fit(
            train_x,
            train_y,
            validation_data=validation_data,
            epochs=self.training_cfg.epochs,
            batch_size=self.training_cfg.batch_size,
            verbose=0,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
        )

        logger.info("Training complete.  Best model saved to %s", self.model_path)
        return self.model

    def load(self) -> Model:
        """
        Load the best saved weights into a freshly built model.

        Returns
        -------
        keras.Model
            Model with weights loaded from ``model.h5``.

        Raises
        ------
        FileNotFoundError
            If no saved model exists at ``models_dir/model.h5``.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {self.model_path}. "
                "Run `scripts/train.py` first."
            )
        # Build a dummy model to infer input_shape (requires data shape metadata)
        raise NotImplementedError(
            "Use Trainer.fit() to train, then call model.predict() directly. "
            "For inference-only loading, use scripts/predict.py which supplies "
            "the correct input_shape."
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_resume(self) -> int:
        """
        If a checkpoint exists, load weights and return the epoch to resume from.

        Returns
        -------
        int
            ``initial_epoch`` to pass to ``model.fit()``.
        """
        if self.model_path.exists() and self.acc_loss_csv.exists():
            logger.info("Resuming from checkpoint: %s", self.model_path)
            self.model.load_weights(str(self.model_path))
            completed = len(pd.read_csv(self.acc_loss_csv))
            logger.info("Resuming at epoch %d.", completed)
            return completed
        return 0
