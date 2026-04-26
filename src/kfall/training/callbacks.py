"""
training/callbacks.py
=====================
Custom Keras callbacks for logging epoch metrics to CSV and live-plotting
training / validation accuracy and loss curves.

Classes
-------
TrainingCallback
    Persists per-epoch metrics to ``acc_loss.csv`` and regenerates the
    accuracy / loss figure after every epoch.

Functions
---------
plot_acc_loss(df, save_dir)
    Render and save the 2-panel (accuracy | loss) figure from a metrics
    DataFrame.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import Callback

matplotlib.use("Agg")  # non-interactive backend safe for background threads

logger = logging.getLogger(__name__)

# Column order in the persisted CSV
_METRIC_COLS = ["epoch", "accuracy", "val_accuracy", "loss", "val_loss"]


class TrainingCallback(Callback):
    """
    Epoch-level callback that:

    1. Appends metrics to a CSV file after each epoch.
    2. Saves a two-panel accuracy / loss figure (``acc_loss.png``) alongside
       the CSV.

    Parameters
    ----------
    acc_loss_path : str or Path
        Full path to the CSV file where metrics are appended.  If the file
        already exists (e.g. resuming training) it is loaded and the curves
        are pre-populated before training starts.

    Examples
    --------
    >>> cb = TrainingCallback("outputs/models/acc_loss.csv")
    >>> model.fit(X, y, callbacks=[cb], epochs=50)
    """

    def __init__(self, acc_loss_path: str | Path) -> None:
        super().__init__()
        self.acc_loss_path = Path(acc_loss_path)
        self.save_dir = self.acc_loss_path.parent

        if self.acc_loss_path.is_file():
            logger.info("Resuming from existing metrics: %s", self.acc_loss_path)
            self.df = pd.read_csv(self.acc_loss_path)
            plot_acc_loss(self.df, self.save_dir)
        else:
            self.df = pd.DataFrame([], columns=_METRIC_COLS)
            self.df.to_csv(self.acc_loss_path, index=False)

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        """Append current epoch metrics, persist CSV and refresh the plot."""
        logs = logs or {}
        row = [
            int(epoch + 1),
            round(logs.get("accuracy", 0.0), 4),
            round(logs.get("val_accuracy", 0.0), 4),
            round(logs.get("loss", 0.0), 4),
            round(logs.get("val_loss", 0.0), 4),
        ]
        self.df.loc[len(self.df)] = row
        self.df.to_csv(self.acc_loss_path, index=False)

        acc, val_acc, loss, val_loss = row[1:]
        logger.info(
            "[Epoch %4d] Acc: %.4f | Val_Acc: %.4f | Loss: %.4f | Val_Loss: %.4f",
            epoch + 1,
            acc,
            val_acc,
            loss,
            val_loss,
        )
        plot_acc_loss(self.df, self.save_dir)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_acc_loss(df: pd.DataFrame, save_dir: str | Path) -> None:
    """
    Generate and save a two-panel figure showing training & validation
    accuracy (left) and loss (right) over epochs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``epoch``, ``accuracy``, ``val_accuracy``,
        ``loss``, ``val_loss``.
    save_dir : str or Path
        Directory where ``acc_loss.png`` is written.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = len(df)
    if epochs == 0:
        return

    fig, (acc_ax, loss_ax) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    _plot_metric(acc_ax, df["accuracy"].values, df["val_accuracy"].values, epochs, "Accuracy")
    _plot_metric(loss_ax, df["loss"].values, df["val_loss"].values, epochs, "Loss")

    fig.tight_layout()
    fig.savefig(save_dir / "acc_loss.png", dpi=150)
    plt.close(fig)


def _plot_metric(
    ax: plt.Axes,
    train_vals,
    val_vals,
    epochs: int,
    label: str,
) -> None:
    """Render a single accuracy-or-loss subplot."""
    x = range(1, epochs + 1)
    ax.plot(x, train_vals, label="Training", color="#1f77b4", linewidth=1.5)
    ax.plot(x, val_vals, label="Validation", color="#ff7f0e", linewidth=1.5, linestyle="--")
    ax.set_title(f"Training vs Validation {label}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(True, alpha=0.3)
