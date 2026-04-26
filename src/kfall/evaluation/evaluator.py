"""
evaluation/evaluator.py
=======================
End-to-end evaluation of a trained CRHNN model.

For each split (Train / Test) the :class:`Evaluator` produces:

* **metrics.csv** — per-class precision, recall, F1, plus macro/weighted averages.
* **conf_mat.png** — confusion matrix heatmap.
* **pr_curve.png** — per-class precision-recall curves.
* **roc_curve.png** — per-class ROC curves with AUC.

The class intentionally avoids a hard dependency on the original
``performance_evaluator`` package so the project is self-contained.  All
metrics are computed with :mod:`sklearn.metrics`.

Usage
-----
>>> from kfall.evaluation.evaluator import Evaluator
>>> ev = Evaluator(class_names=class_names)
>>> ev.evaluate(model, train_x, train_y, split_name="Train")
>>> ev.evaluate(model, test_x,  test_y,  split_name="Test")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import Model
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from kfall.config import RESULTS_DIR

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluate a trained CRHNN model and persist diagnostic artefacts.

    Parameters
    ----------
    class_names : list of str
        Human-readable name for each output class, in label-index order.
    results_dir : Path, optional
        Root directory under which per-split subdirectories are created.
        Defaults to ``outputs/results``.

    Examples
    --------
    >>> ev = Evaluator(class_names=["Fall-Front", "Fall-Back", ...])
    >>> metrics_df = ev.evaluate(model, test_x, test_y_int, split_name="Test")
    """

    def __init__(
        self,
        class_names: List[str],
        results_dir: Path = RESULTS_DIR,
    ) -> None:
        self.class_names = class_names
        self.results_dir = Path(results_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: Model,
        X: np.ndarray,
        y_true: np.ndarray,
        split_name: str = "Test",
    ) -> pd.DataFrame:
        """
        Run inference, compute metrics and save all diagnostic figures.

        Parameters
        ----------
        model : keras.Model
            Trained CRHNN model.
        X : np.ndarray  shape (N, 1, n_features)
            Preprocessed input features.
        y_true : np.ndarray  shape (N,)
            Integer ground-truth class labels.
        split_name : str
            Label for this split (used as sub-directory name and plot titles).

        Returns
        -------
        pd.DataFrame
            Classification report as a DataFrame (rows = classes + averages).
        """
        split_dir = self.results_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Evaluating split: %s (%d samples)", split_name, len(X))

        y_prob = model.predict(X, verbose=0)
        y_pred = np.argmax(y_prob, axis=1).astype(int)
        y_true = y_true.astype(int)

        metrics_df = self._compute_metrics(y_true, y_pred, split_dir)
        self._plot_confusion_matrix(y_true, y_pred, split_name, split_dir)
        self._plot_pr_curves(y_true, y_prob, split_name, split_dir)
        self._plot_roc_curves(y_true, y_prob, split_name, split_dir)

        logger.info("Results saved to %s", split_dir)
        return metrics_df

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_dir: Path
    ) -> pd.DataFrame:
        """Compute and persist the classification report."""
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Class"})
        csv_path = save_dir / "metrics.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Metrics saved → %s", csv_path)
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=self.class_names,
                zero_division=0,
            )
        )
        return df

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        split_name: str,
        save_dir: Path,
    ) -> None:
        """Render and save the confusion-matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(max(8, len(self.class_names)), max(8, len(self.class_names))))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_title(f"Confusion Matrix — {split_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        fig.tight_layout()
        fig.savefig(save_dir / "conf_mat.png", dpi=150)
        plt.close(fig)

    def _plot_pr_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        split_name: str,
        save_dir: Path,
    ) -> None:
        """Render per-class precision-recall curves."""
        n_classes = len(self.class_names)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(9, 7))
        for i, name in enumerate(self.class_names):
            if y_bin.shape[1] <= i:
                continue
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
            ax.plot(rec, prec, lw=1.2, label=name)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curves — {split_name}", fontweight="bold")
        ax.legend(ncol=2, fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "pr_curve.png", dpi=150)
        plt.close(fig)

    def _plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        split_name: str,
        save_dir: Path,
    ) -> None:
        """Render per-class ROC curves with AUC scores."""
        n_classes = len(self.class_names)
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(9, 7))
        for i, name in enumerate(self.class_names):
            if y_bin.shape[1] <= i:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.2, label=f"{name} (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {split_name}", fontweight="bold")
        ax.legend(ncol=2, fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / "roc_curve.png", dpi=150)
        plt.close(fig)
