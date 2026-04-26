"""
scripts/train.py
================
Entry-point script to run the full KFall training pipeline:

    1. Load & merge raw sensor + label data
    2. Preprocess (scale → split → one-hot)
    3. Build CRHNN model
    4. Train with resume support
    5. Evaluate on Train and Test splits

Run
---
.. code-block:: bash

    python scripts/train.py [--epochs 800] [--batch-size 1000] [--lr 0.001]

Environment
-----------
Expects KFall raw data under ``data/raw/`` — see ``data/README.md``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

# ── Silence TF / Keras verbosity before import ──────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
warnings.filterwarnings("ignore")

# ── Path bootstrap ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from kfall.config import (
    Config,
    ModelConfig,
    PROCESSED_CSV,
    RAW_DATA_DIR,
    SplitConfig,
    TrainingConfig,
)
from kfall.data.loader import DataLoader
from kfall.data.preprocessor import DataPreprocessor
from kfall.evaluation.evaluator import Evaluator
from kfall.training.trainer import Trainer
from kfall import utils  # noqa: F401  (seeds random state)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kfall.train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CRHNN fall-detection model on the KFall dataset."
    )
    parser.add_argument(
        "--epochs", type=int, default=800, help="Total training epochs (default: 800)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Mini-batch size (default: 1000)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of data held out for testing (default: 0.3)",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip data loading step and use existing processed CSV.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    cfg = Config(
        split=SplitConfig(test_size=args.test_size),
        training=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        ),
    )

    # ── 1. Load data ─────────────────────────────────────────────────────
    if args.skip_load and PROCESSED_CSV.exists():
        logger.info("Loading pre-processed CSV from %s", PROCESSED_CSV)
        df = pd.read_csv(PROCESSED_CSV)
    else:
        logger.info("Loading raw KFall data from %s", RAW_DATA_DIR)
        loader = DataLoader(raw_data_dir=cfg.raw_data_dir)
        df = loader.load()
        loader.save(df)

    n_classes = int(df["Class"].nunique())
    class_names = (
        df.drop_duplicates("Class")
        .sort_values("Class")["Description"]
        .tolist()
    )
    logger.info("Dataset: %d samples | %d classes", len(df), n_classes)

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    prep = DataPreprocessor(split_config=cfg.split)
    train_x, test_x, train_y, test_y = prep.fit_transform(df, n_classes)

    train_y_int = np.argmax(train_y, axis=1)
    test_y_int = np.argmax(test_y, axis=1)

    # ── 3. Train ──────────────────────────────────────────────────────────
    trainer = Trainer(
        n_classes=n_classes,
        training_cfg=cfg.training,
        models_dir=cfg.models_dir,
    )
    model = trainer.fit(train_x, train_y, val_x=test_x, val_y=test_y)

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    evaluator = Evaluator(class_names=class_names, results_dir=cfg.results_dir)
    evaluator.evaluate(model, train_x, train_y_int, split_name="Train")
    evaluator.evaluate(model, test_x, test_y_int, split_name="Test")

    logger.info("✓ Pipeline finished.  Outputs in %s", cfg.results_dir)


if __name__ == "__main__":
    main()
