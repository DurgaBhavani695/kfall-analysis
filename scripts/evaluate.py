"""
scripts/evaluate.py
===================
Standalone evaluation script: load a trained CRHNN checkpoint and regenerate
all diagnostic artefacts (metrics, confusion matrix, PR / ROC curves).

Run
---
.. code-block:: bash

    python scripts/evaluate.py [--split Train] [--split Test]

Requires:
    • ``data/processed/data.csv``   (run train.py once, or train.py --skip-load)
    • ``outputs/models/model.h5``   (saved by train.py)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from kfall.config import MODELS_DIR, PROCESSED_CSV, RESULTS_DIR
from kfall.data.preprocessor import DataPreprocessor
from kfall.evaluation.evaluator import Evaluator
from kfall.models.crhnn import build_crhnn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kfall.evaluate")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved CRHNN model on Train / Test splits."
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        choices=["Train", "Test"],
        help="Split(s) to evaluate (can be specified multiple times). "
             "Default: both Train and Test.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    splits = args.split or ["Train", "Test"]

    model_path = MODELS_DIR / "model.h5"
    if not model_path.exists():
        logger.error("No model found at %s — run train.py first.", model_path)
        sys.exit(1)

    if not PROCESSED_CSV.exists():
        logger.error("Processed CSV not found at %s — run train.py first.", PROCESSED_CSV)
        sys.exit(1)

    df = pd.read_csv(PROCESSED_CSV)
    n_classes = int(df["Class"].nunique())
    class_names = (
        df.drop_duplicates("Class")
        .sort_values("Class")["Description"]
        .tolist()
    )

    prep = DataPreprocessor()
    train_x, test_x, train_y, test_y = prep.fit_transform(df, n_classes)

    input_shape = train_x.shape[1:]
    model = build_crhnn(input_shape=input_shape, n_classes=n_classes)
    model.load_weights(str(model_path))
    logger.info("Loaded weights from %s", model_path)

    data_map = {
        "Train": (train_x, np.argmax(train_y, axis=1)),
        "Test": (test_x, np.argmax(test_y, axis=1)),
    }

    evaluator = Evaluator(class_names=class_names, results_dir=RESULTS_DIR)
    for split in splits:
        X, y_int = data_map[split]
        evaluator.evaluate(model, X, y_int, split_name=split)

    logger.info("✓ Evaluation complete.  Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
